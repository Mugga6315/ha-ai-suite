"""Calendar questions.

By day: the entries of the days the text names (days.py). By name: the decider picks one of the calendars' own
entries of the coming year, or of the last 30 days for a question about the past. HA's calendar.get_events gives the
entries (home.read_calendars), the answer step words the facts. No HA imports.
"""

from dataclasses import dataclass
from datetime import date, datetime
import re

from ..languages import pattern
from .days import day_named
from .turn import HandOff

CALENDAR = "calendar"
CALENDAR_DESCRIPTION = "ask what is in the calendar on a day, or when an appointment, birthday or event is or was"
AHEAD_DAYS = 366  # a yearly entry (a birthday) comes once in this
BACK_DAYS = 30
MAX_CHOICES = 30  # more entries than this: only those sharing a word with the question are offered
NO_EVENT = "none"

_BACK = pattern(lambda language: f"{language.time.past_verbs}|{language.time.past}")
_WORD = re.compile(r"\w{4,}")


@dataclass(frozen=True, slots=True)
class Event:
    """One calendar entry in the home's local time (home.calendar_event): summary, start, location and end."""
    summary: str
    start: datetime | date  # a date: all day
    location: str | None = None
    end: datetime | date | None = None


def looks_back(text: str) -> bool:
    """Whether the question asks about the past."""
    return bool(_BACK.search(text))


def day_facts(events: list[Event], days: list[date], today: date) -> str:
    """The entries of each asked day, in order, each day after its entries as in `event_fact`.

    With the day first, small answer models answer a question about a day's calendar with the date alone.
    """
    lines = []
    for day in days:
        on_day = sorted((e for e in events if _day(e) == day), key=event_order)
        lines.append((", ".join(_shown(e) for e in on_day) or "nothing") + f" on {day_named(day, today)}")
    return "; ".join(lines)


def event_question(events: list[Event], text: str) -> tuple[dict, dict[str, Event]]:
    """The question which entry the user means, and each option's entry (the first of a recurring one)."""
    by_summary: dict[str, Event] = {}
    for event in events:
        by_summary.setdefault(event.summary, event)
    if len(by_summary) > MAX_CHOICES:
        asked = {word[:5].lower() for word in _WORD.findall(text)}
        by_summary = {s: e for s, e in by_summary.items() if asked & {w[:5].lower() for w in _WORD.findall(s)}}
    if not by_summary or len(by_summary) > MAX_CHOICES:
        raise HandOff(f"{len(by_summary)} calendar entries to choose from")
    question = {"type": "choice", "instructions": "Which calendar entry does the user mean?",
                "criteria": {**dict.fromkeys(by_summary, ""), NO_EVENT: "none of these entries"}}
    return question, by_summary


def event_fact(event: Event, today: date) -> str:
    """One entry as a fact for the answer step, its day named from `today`."""
    return f"{_shown(event)} on {day_named(_day(event), today)}"


def _shown(event: Event) -> str:
    when = "all day" if not isinstance(event.start, datetime) else f"at {event.start:%H:%M}"
    where = f" ({event.location})" if event.location else ""
    return f"{event.summary} {when}{where}"


def _day(event: Event) -> date:
    return event.start.date() if isinstance(event.start, datetime) else event.start


def event_order(event: Event) -> tuple[date, str]:
    """The entries' order: by day, an all-day entry before the day's timed ones."""
    return (_day(event), f"{event.start:%H:%M}" if isinstance(event.start, datetime) else "")
