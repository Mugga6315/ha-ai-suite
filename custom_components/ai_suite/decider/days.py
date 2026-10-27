"""The days a request names, from the day words of every language (languages/).

Ahead: today, tomorrow, the day after, the next days, the weekend, a weekday, a weekday of next week. Back: yesterday,
the day before. No HA imports.
"""

from datetime import date, timedelta
import re
from typing import NoReturn

from ..languages import LANGUAGES, pattern
from .turn import HandOff

NEXT_DAYS = 3  # "the next days"
_NEAR_DAYS = {-2: "the day before yesterday", -1: "yesterday", 0: "today", 1: "tomorrow", 2: "the day after tomorrow"}

_WEEKDAYS = {
    day: re.compile(
        rf"\b(?:{'|'.join(language.days.weekdays[day] for language in LANGUAGES.values())})\b", re.IGNORECASE
    )
    for day in range(7)
}
_DAY_AFTER = pattern(lambda language: language.days.day_after_tomorrow)
_TOMORROW = pattern(lambda language: language.days.tomorrow)
_TODAY = pattern(lambda language: language.days.today)
_NEXT_DAYS = pattern(lambda language: language.days.next_days)
_WEEKEND = pattern(lambda language: language.days.weekend)
_NEXT_WEEK = pattern(lambda language: language.days.next_week)
_DAY_BEFORE_YESTERDAY = pattern(lambda language: language.days.day_before_yesterday)
_YESTERDAY = pattern(lambda language: language.days.yesterday)


def days_asked(text: str, today: date) -> list[date]:
    """The days the question is about; HandOff when it names none this can read."""
    if _DAY_AFTER.search(text):
        return [today + timedelta(days=2)]
    if _NEXT_DAYS.search(text):
        return [today + timedelta(days=n) for n in range(1, NEXT_DAYS + 1)]
    next_week = bool(_NEXT_WEEK.search(text))
    if _WEEKEND.search(text):
        if next_week:  # the weekend ahead or the one after it: people mean both
            return _no_day(text)
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        return [saturday, saturday + timedelta(days=1)]
    if named := [day for day, weekday in _WEEKDAYS.items() if weekday.search(text)]:
        if len(named) > 1:
            return _no_day(text)
        if next_week:  # the week after this one, from its Monday
            return [today + timedelta(days=7 - today.weekday() + named[0])]
        return [today + timedelta(days=(named[0] - today.weekday()) % 7)]
    if next_week:  # a week, not a day
        return _no_day(text)
    if _TOMORROW.search(text):
        return [today + timedelta(days=1)]
    if _TODAY.search(text):
        return [today]
    return _no_day(text)


def next_days_asked(text: str) -> bool:
    """Whether the text asks about the next days or this week, without naming the days."""
    return bool(_NEXT_DAYS.search(text))


def past_day_asked(text: str, today: date) -> date:
    """The earlier day the question is about; HandOff when it names none this can read."""
    if _DAY_BEFORE_YESTERDAY.search(text):
        return today - timedelta(days=2)
    if _YESTERDAY.search(text):
        return today - timedelta(days=1)
    return _no_day(text)


def day_named(day: date, today: date) -> str:
    """A day as a result note names it: its date, with the word for today or a day next to it.

    With the date alone the answer step does not know which day it is: it names today by its weekday as if it were a
    later day, or says yesterday's total as today's.
    """
    near = _NEAR_DAYS.get((day - today).days)
    return f"{near} ({day:%A %Y-%m-%d})" if near else f"{day:%A %Y-%m-%d}"


def _no_day(text: str) -> NoReturn:
    raise HandOff(f"no day this reads in {text!r}")
