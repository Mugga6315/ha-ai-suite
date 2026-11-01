"""Calendar questions.

By day: the entries of the days the text names (days.py). Next or last: the entries coming next, or the most recent
ones, for a question that asks for those and names none. By name: the decider picks one of the calendars' own
entries of the coming year, or of the past year for a question about the past. HA's calendar.get_events gives the
entries (home.read_calendars), the answer step words the facts. No HA imports.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
import re

from ..languages import LANGUAGES, alternatives, merged, module_code, pattern
from .days import day_named
from .turn import HandOff

CALENDAR = "calendar"
CALENDAR_DESCRIPTION = "ask what is in the calendar on a day, or when an appointment, birthday or event is or was"
AHEAD_DAYS = 366  # a yearly entry (a birthday) comes once in this
BACK_DAYS = 366  # the last of an entry may be months ago
MAX_CHOICES = 30  # more entries than this: only those sharing a word with the question are offered
ENTRY_COUNT = 3  # the entries a question for what comes next, or for the last ones, gets
# A question for the last entry gets the most recent alone: of several listed, in either order, an answer model named
# another one as the last.
LAST_COUNT = 1
NO_EVENT = "none"
NOTHING_AHEAD = "nothing in the coming year"
NOTHING_BACK = f"nothing in the last {BACK_DAYS} days"

_BACK = pattern(lambda language: f"{language.time.past_verbs}|{language.time.past}")
_BACK_IN = {code: re.compile(f"{language.time.past_verbs}|{language.time.past}", re.IGNORECASE)
            for code, language in LANGUAGES.items()}
_NEXT = pattern(lambda language: language.calendar.next_up)
_LAST = pattern(lambda language: language.calendar.last_up)
_SEVERAL = pattern(lambda language: language.calendar.several)
_LETTERS = re.compile(r"[^\W\d_]+")
# What a question for the next or the last entries may hold besides those words: words that name nothing.
_NAMES_NOTHING = merged(lambda language: language.calendar.filler) | merged(lambda language: language.calendar.words)
# A count the question asks for, by its number word (two to nine): right after the next or last word, or right before
# an entry word ("my last two appointments", "zwei Termine"). Elsewhere a number word may be a time ("at eight").
_NUMBERS = {word: value for language in LANGUAGES.values() for word, value in language.numbers.units.items()}
_NUMBER = "|".join(sorted(_NUMBERS, key=len, reverse=True))
_ENTRY_WORDS = "|".join(sorted(merged(lambda language: language.calendar.words), key=len, reverse=True))
_COUNT = re.compile(
    rf"(?:{alternatives(lambda language: language.calendar.next_up)}|"
    rf"{alternatives(lambda language: language.calendar.last_up)})\s+(?P<after>{_NUMBER})\b"
    rf"|\b(?P<before>{_NUMBER})\s+(?:{_ENTRY_WORDS})\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w{4,}")
# Words are compared by their first five letters, so "Termine" is "Termin".
_STEM = 5
_CALENDAR_WORDS = frozenset(word[:_STEM] for word in merged(lambda language: language.calendar.words))


@dataclass(frozen=True, slots=True)
class Event:
    """One calendar entry in the home's local time (home.calendar_event): summary, start, location and end."""
    summary: str
    start: datetime | date  # a date: all day
    location: str | None = None
    end: datetime | date | None = None


def looks_back(text: str, language: str | None = None) -> bool:
    """Whether the question asks about the past; one asking what comes next does not, whatever else it holds.

    `language`: the request's language tag. Only its past words count: the German "Was" asks what, the English "was"
    is the past. For a language without a module, every language's past words count.
    """
    past = _BACK_IN[code] if (code := module_code(language)) else _BACK
    return bool(past.search(text)) and not _NEXT.search(text)


def day_facts(events: list[Event], days: list[date], today: date) -> str:
    """The entries of each asked day, in order, each day after its entries as in `event_fact`.

    With the day first, small answer models answer a question about a day's calendar with the date alone.
    """
    lines = []
    for day in days:
        on_day = sorted((e for e in events if _day(e) == day), key=event_order)
        lines.append((", ".join(_shown(e) for e in on_day) or "nothing") + f" on {day_named(day, today)}")
    return "; ".join(lines)


def coming_next(events: list[Event], text: str) -> list[Event] | None:
    """The next entries, each name once, for a question for what comes next that names nothing else.

    "What's next on my calendar?" gets ENTRY_COUNT, "my next two appointments" two. Any other word may name an entry,
    a kind of entry or a time ("the next Visit 1", "next week", "at eight"): the decider takes such a question by
    name. A recurring entry counts once, or it would fill the list. None for any other question.
    """
    if not _asks_only(text, _NEXT):
        return None
    return _first_of_each(sorted(events, key=event_order))[:_count(text, ENTRY_COUNT)]


def last_ones(events: list[Event], text: str, now: datetime) -> list[Event] | None:
    """The most recent entries that have ended, each name once, for a question for the last one or ones.

    "What was my last appointment?" gets LAST_COUNT, "my last appointments" ENTRY_COUNT, "my last two appointments"
    two; naming nothing else (as in `coming_next`). A past verb alone does not ask for them: "Was I at Visit 1?" asks
    about one, by name. An entry still running (today's all-day one) is no last one. In time order, the most recent
    last. None for any other question.
    """
    if not _asks_only(text, _LAST):
        return None
    ended = [event for event in sorted(events, key=event_order) if _ended(event, now)]
    count = _count(text, ENTRY_COUNT if _SEVERAL.search(text) else LAST_COUNT)
    return _first_of_each(reversed(ended))[:count][::-1]


def entry_facts(events: list[Event], today: date, nothing: str) -> str:
    """Entries as facts for the answer step, each as in `event_fact`; `nothing` when there are none."""
    return "; ".join(event_fact(event, today) for event in events) or nothing


def event_question(events: list[Event], text: str) -> tuple[dict, dict[str, Event]]:
    """The question which entry the user means, and each option's entry (the first of a recurring one)."""
    by_summary: dict[str, Event] = {}
    for event in events:
        by_summary.setdefault(event.summary, event)
    if len(by_summary) > MAX_CHOICES:
        by_summary = {s: e for s, e in by_summary.items() if _shares_a_word(s, text)}
    if not by_summary or len(by_summary) > MAX_CHOICES:
        raise HandOff(f"{len(by_summary)} calendar entries to choose from")
    question = {"type": "choice", "instructions": "Which calendar entry does the user mean?",
                "criteria": {**dict.fromkeys(by_summary, ""), NO_EVENT: "none of these entries"}}
    return question, by_summary


def _asks_only(text: str, asks: re.Pattern[str]) -> bool:
    """Whether the text has words of `asks` and besides them only words that name nothing.

    A digit may be a time or a year ("my last appointment at 9", "in 2025"): such a question is the decider's.
    """
    if not asks.search(text) or any(char.isdigit() for char in text):
        return False
    return all(word.lower() in _NAMES_NOTHING for word in _LETTERS.findall(asks.sub(" ", _COUNT.sub(" ", text))))


def _count(text: str, otherwise: int) -> int:
    """The number of entries the text asks for by a number word ("my next two appointments"), else `otherwise`."""
    if (match := _COUNT.search(text)) is None:
        return otherwise
    return _NUMBERS[(match.group("after") or match.group("before")).lower()]


def _first_of_each(events: Iterable[Event]) -> list[Event]:
    """The events in their order, each summary at its first."""
    first: dict[str, Event] = {}
    for event in events:
        first.setdefault(event.summary, event)
    return list(first.values())


def _ended(event: Event, now: datetime) -> bool:
    """Whether the entry is over at `now`; an all-day entry's end is the day after its last."""
    end = event.end if event.end is not None else event.start
    if isinstance(end, datetime):
        return end <= now
    return end <= now.date() if event.end is not None else end < now.date()


def _shares_a_word(summary: str, text: str) -> bool:
    """Whether the text has a word of the entry's summary; the calendar's own words (appointment) do not count."""
    return bool(_stems(text) & _stems(summary))


def _stems(text: str) -> set[str]:
    return {word[:_STEM].lower() for word in _WORD.findall(text)} - _CALENDAR_WORDS


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
