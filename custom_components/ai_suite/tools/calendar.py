"""Calendar: the entries of every exposed calendar at once, each with its weekday, and a new entry added to one."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import logging
from typing import Any

import probatio

from homeassistant.components.calendar import CalendarEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_SUPPORTED_FEATURES,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir, llm, selector
from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from ..decider.calendar import Event, event_order
from ..home import entity_aliases, entity_named, exposed_states, read_calendars
from .spec import READ_ONLY, RecordFailure, ToolArgs, ToolError, ToolSpec, spec_tools
from .when import BARE_DATE, NAMED_DAYS, WEEKDAYS, fmt_day, parse_offset, parse_when

_LOGGER = logging.getLogger(__name__)

MAX_CALENDAR_EVENTS = 40
NOT_EXPOSED_ISSUE = "calendars_not_exposed"

CONF_DEFAULT_CALENDAR = "default_calendar"

ADDS = llm.ToolAnnotations(read_only=False, destructive=False, idempotent=False, open_world=False)

NO_END = (
    "A timed entry needs its end. Ask the user how long it lasts or when it ends, then call again with end as a "
    "date and time (2030-08-01T15:30) or a length (90m, 2h). Nothing was added."
)
TIMED_END_IS_A_DAY = (
    "For a timed entry give end as a date and time (2030-08-01T15:30) or a length (90m, 2h), not a date alone. If "
    "the user did not say when it ends, ask. Nothing was added."
)
START_IS_A_LENGTH = (
    "Give start as a date (2030-08-01) for an all-day entry, or a date and time (2030-08-01T14:00), not a length. "
    "Nothing was added."
)
ALL_DAY_END_IN_HOURS = (
    "An all-day entry covers whole days: give end as its last day (2030-08-03) or a number of days (3d). Nothing "
    "was added."
)
NO_SUMMARY = "Give the entry's title as summary. Nothing was added."
NOT_LOCAL_TIME = (
    "Give start and end in local time without an offset (2030-08-01T14:00), as the user said them. Nothing was added."
)


@dataclass(frozen=True, slots=True)
class Span:
    """When a new entry lasts, in local time: `end` is exclusive, an all-day entry's the midnight after its last day."""

    start: datetime
    end: datetime
    all_day: bool


async def get_calendar_events(
    hass: HomeAssistant, _default_calendar: str | None, args: ToolArgs, _context: llm.LLMContext
) -> str:
    """List the entries of the asked range, oldest first."""
    try:
        start, end = _range(args)
    except ValueError as err:
        raise ToolError(str(err), learn=True) from err

    wanted = (args.get("calendar") or "").strip().lower()
    entities = _calendars(hass, wanted)
    if not entities:
        known = ", ".join(_name(hass, entity_id) for entity_id in _calendars(hass, "")) or "none"
        raise ToolError(
            f"No calendar matching '{args.get('calendar')}'. Available: {known}. Leave 'calendar' out to search all of "
            "them."
        )

    read = await read_calendars(hass, entities, start, end)
    for entity_id, error in read.failed.items():  # a broken calendar hides no other
        _LOGGER.warning("get_calendar_events: %s failed: %s", entity_id, error)
    events = sorted(
        ((event, _name(hass, entity_id)) for entity_id, found in read.events.items() for event in found),
        key=lambda item: event_order(item[0]),
    )

    header = (
        f"{fmt_day(start, 'day')} -> {fmt_day(end, 'day')} (local) | "
        f"calendars: {', '.join(_name(hass, entity_id) for entity_id in entities)}"
    )
    if not events:
        return f"{header}\nNo appointments in this period."
    lines = [f"{header} | {len(events)} appointments"]
    lines += [f"  {_fmt_event(event)} [{calendar_name}]" for event, calendar_name in events[:MAX_CALENDAR_EVENTS]]
    if len(events) > MAX_CALENDAR_EVENTS:
        lines.append(f"  ... {len(events) - MAX_CALENDAR_EVENTS} more")
    return "\n".join(lines)


async def add_calendar_event(
    hass: HomeAssistant, default_calendar: str | None, args: ToolArgs, llm_context: llm.LLMContext
) -> str:
    """Add one entry to the named calendar, else `default_calendar`, else the only one that can take it.

    Writes with the request's context, so the entry is the user's change; HA's own failure raises ToolError.
    """
    summary = args["summary"].strip()
    if not summary:
        raise ToolError(NO_SUMMARY)
    span = _new_span(args)
    entity_id = _calendar_for_new_entry(hass, (args.get("calendar") or "").strip(), default_calendar)
    details = {key: args[key] for key in ("description", "location") if args.get(key)}
    try:
        await hass.services.async_call(
            "calendar",
            "create_event",
            {"entity_id": entity_id, "summary": summary, **_service_times(span), **details},
            blocking=True,
            context=llm_context.context,
        )
    except (HomeAssistantError, probatio.Invalid) as err:  # a calendar's own failure, or HA's check of the entry
        raise ToolError(f"{_name(hass, entity_id)} did not take the entry: {err}. Nothing was added.") from err
    return f"Added to {_name(hass, entity_id)}: {_fmt_span(span)}: {summary}"


SPECS: tuple[ToolSpec[str | None], ...] = (
    ToolSpec(
        name="get_calendar_events",
        description=(
            "Use this for EVERY question about appointments, regardless of period: "
            "today, this week, the weekend, next week, this month, a named date, or "
            "any range. It searches ALL calendars at once - private appointments, "
            "birthdays, holidays - and returns them merged in date order, each "
            "labelled with its calendar. Prefer it over any single-calendar tool, "
            "which sees only one calendar and would silently miss birthdays and "
            "holidays. Give start, plus end for anything longer than one day; dates "
            "are inclusive, so start=2030-08-01 end=2030-08-31 covers all of August. "
            "end may also be a length after start: 7d, 2w, 1y. "
            "For 'when is the next ...' questions, ask start=today end=1y "
            "and take the first match - a short window can miss it entirely."
        ),
        parameters=probatio.Schema(
            {
                probatio.Required("start"): str,
                probatio.Optional("end"): str,
                probatio.Optional("calendar"): str,
            }
        ),
        run=get_calendar_events,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="add_calendar_event",
        description=(
            "Adds one entry to a calendar. For an all-day entry - a birthday, a holiday, an anniversary - give start "
            "as a date alone: start=2030-08-01; for several days add end, its last day: end=2030-08-03. For an "
            "appointment give start as a date with a time, start=2030-08-01T14:00, and its end, "
            "end=2030-08-01T15:30, or its length, end=90m or end=2h. If the user gave no end or length for an "
            "appointment, ask the user how long it lasts instead of guessing. Times are local time, no offset. Leave "
            "calendar out unless the user names one."
        ),
        parameters=probatio.Schema(
            {
                probatio.Required("summary"): str,
                probatio.Required("start"): str,
                probatio.Optional("end"): str,
                probatio.Optional("calendar"): str,
                probatio.Optional("description"): str,
                probatio.Optional("location"): str,
            }
        ),
        run=add_calendar_event,
        annotations=ADDS,
    ),
)


def build(
    _hass: HomeAssistant, _entry: ConfigEntry, data: Mapping[str, Any], record_failure: RecordFailure | None
) -> list[llm.Tool]:
    """The Calendar tools; a new entry goes to the set's default calendar when the request names none."""
    return spec_tools(SPECS, data.get(CONF_DEFAULT_CALENDAR) or None, record_failure)


def subentry_schema(data: dict) -> dict:
    """The Calendar set's form: the calendar new entries go to when the request names none."""
    # suggested_value, not default: a cleared field is left out of the answer, and a default would bring it back.
    return {
        probatio.Optional(
            CONF_DEFAULT_CALENDAR, description={"suggested_value": data.get(CONF_DEFAULT_CALENDAR)}
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                filter={"domain": "calendar", "supported_features": ["calendar.CalendarEntityFeature.CREATE_EVENT"]}
            )
        ),
    }


async def async_check_default_calendar(hass: HomeAssistant, data: dict) -> str | None:
    """Why the form's default calendar cannot be used, or None: the tools see exposed calendars only."""
    chosen = data.get(CONF_DEFAULT_CALENDAR)
    if not chosen or entity_named(hass, chosen, domain="calendar") is not None:
        return None
    return (
        f"{_name(hass, chosen)} is not exposed to voice assistants, so no entry could be added to it. Expose it "
        "(Settings > Voice assistants > Expose) or choose another calendar."
    )


def check_exposed(hass: HomeAssistant, present: bool) -> None:
    """Raise a repair issue while the home has calendars and exposes none of them to the assistant.

    `present`: whether the Calendar set is added.
    """
    # Calendars are not among HA's default exposed domains, and nothing else would tell the home.
    hidden = bool(hass.states.async_entity_ids("calendar")) and not exposed_states(hass, "calendar")
    if present and hidden:
        ir.async_create_issue(
            hass, DOMAIN, NOT_EXPOSED_ISSUE, is_fixable=False, severity=ir.IssueSeverity.WARNING,
            translation_key=NOT_EXPOSED_ISSUE,
        )
    else:
        ir.async_delete_issue(hass, DOMAIN, NOT_EXPOSED_ISSUE)


def _range(args: ToolArgs) -> tuple[datetime, datetime]:
    """The asked window; a ValueError names what is wrong with it.

    An end like '1y' is a length after the start, so the model can ask a year ahead without knowing the year.
    """
    raw_end = (args.get("end") or "").strip()
    start = parse_when(args["start"], dt_util.now())
    end = parse_when(raw_end, None) if (length := parse_offset(raw_end)) is None else start + length
    if end is None:
        end = start + timedelta(days=1)
    elif BARE_DATE.match(raw_end):
        # Calendar ends are exclusive (RFC 5545), but a user's end date includes that day. Without this, month-end
        # events vanish.
        end += timedelta(days=1)
    if start >= end:
        raise ValueError(f"'start' ({start}) must be before 'end' ({end}).")
    return start, end


def _new_span(args: ToolArgs) -> Span:
    """When the new entry lasts; ToolError tells the model what to give instead."""
    raw_start, raw_end = args["start"].strip(), (args.get("end") or "").strip()
    if parse_offset(raw_start) is not None:
        raise ToolError(START_IS_A_LENGTH)
    all_day = bool(BARE_DATE.match(raw_start)) or raw_start.lower() in NAMED_DAYS
    if not all_day and not raw_end:
        raise ToolError(NO_END)
    if not all_day and BARE_DATE.match(raw_end):
        raise ToolError(TIMED_END_IS_A_DAY)
    try:
        if not _in_local_time(raw_start) or not _in_local_time(raw_end):
            raise ToolError(NOT_LOCAL_TIME)
        start, end = _range(args)
    except ValueError as err:
        raise ToolError(f"{str(err).rstrip('.')}. Nothing was added.") from err
    if not all_day and (length := parse_offset(raw_end)) is not None:
        # Counted in real time: across a change of the clock, the local clock would give the entry an hour more or less.
        end = dt_util.as_utc(start) + length
    start, end = dt_util.as_local(start), dt_util.as_local(end)
    if all_day and end.time() != time():
        raise ToolError(ALL_DAY_END_IN_HOURS)
    return Span(start, end, all_day)


def _in_local_time(raw: str) -> bool:
    """Whether a time the model wrote is local: without an offset, or with the local one at that time.

    Raises ValueError for a time that is no date.
    """
    written = dt_util.parse_datetime(raw) if raw else None
    return written is None or written.tzinfo is None or written.utcoffset() == dt_util.as_local(written).utcoffset()


def _calendar_for_new_entry(hass: HomeAssistant, named: str, default_calendar: str | None) -> str:
    """The exposed calendar the entry goes to: the named one, else the default, else the only one that can take it."""
    taking = _calendars_taking_entries(hass)
    wanted = named or default_calendar
    if not wanted:
        if len(taking) == 1:
            return taking[0]
        raise ToolError(f"No calendar was named. {_choose_from(hass, taking)}")
    if len(alike := _named_alike(hass, wanted)) > 1:
        ids = ", ".join(alike)
        raise ToolError(f"Several calendars are named '{wanted}': {ids}. Ask the user which one, then give its id as "
                        "calendar. Nothing was added.")
    entity_id = entity_named(hass, wanted, domain="calendar")
    if entity_id in taking:
        return entity_id
    # No other calendar instead: the entry must not land in one the user did not mean.
    which = f"The calendar '{named}'" if named else f"The default calendar for new entries ({_name(hass, wanted)})"
    if entity_id is not None and not _available(hass, entity_id):
        raise ToolError(f"{which} is unavailable right now. Nothing was added; tell the user.")
    why = "is not an exposed calendar" if entity_id is None else "cannot take new entries"
    raise ToolError(f"{which} {why}. {_choose_from(hass, taking)}")


def _calendars_taking_entries(hass: HomeAssistant) -> list[str]:
    """The exposed calendars that can take a new entry now (HA's CREATE_EVENT feature)."""
    # An unavailable calendar keeps its features in its state, and HA's action leaves it out without an error.
    return [
        state.entity_id
        for state in exposed_states(hass, "calendar")
        if state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) & CalendarEntityFeature.CREATE_EVENT
        and _available(hass, state.entity_id)
    ]


def _available(hass: HomeAssistant, entity_id: str) -> bool:
    state = hass.states.get(entity_id)
    return state is not None and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)


def _named_alike(hass: HomeAssistant, wanted: str) -> list[str]:
    """The exposed calendars whose name or alias is `wanted`, in any case: more than one, and the name picks none."""
    needle = wanted.strip().lower()
    return [
        state.entity_id
        for state in exposed_states(hass, "calendar")
        if needle in {name.strip().lower() for name in (state.name, *entity_aliases(hass, state.entity_id))}
    ]


def _choose_from(hass: HomeAssistant, taking: list[str]) -> str:
    if not taking:
        return ("No exposed calendar can take new entries right now: none can, or those that can are unavailable. "
                "Tell the user; nothing was added.")
    names = ", ".join(_name(hass, entity_id) for entity_id in taking)
    return f"Calendars that can take new entries: {names}. Ask the user which one, then give it as calendar."


def _service_times(span: Span) -> dict[str, date | datetime]:
    """The span as calendar.create_event takes it; an all-day entry's end date is exclusive, as HA's."""
    if span.all_day:
        return {"start_date": span.start.date(), "end_date": span.end.date()}
    # Datetimes, not text: across a change to or from summer time the text would carry two offsets, which HA refuses.
    return {"start_date_time": span.start, "end_date_time": span.end}


def _fmt_span(span: Span) -> str:
    """The new entry's days with their weekdays, and its times or "all day", as the read tool writes an entry.

    Holds every date and time the reply may say: the reply check compares the reply's numbers with what the model saw.
    """
    first = span.start.date()
    if span.all_day:
        last = span.end.date() - timedelta(days=1)
        days = _day(first) if last == first else f"{_day(first)} - {_day(last)}"
        return f"{days} (all day)"
    if span.end.date() == first:
        return f"{_day(first)} {span.start:%H:%M}-{span.end:%H:%M}"
    return f"{_day(first)} {span.start:%H:%M} - {_day(span.end.date())} {span.end:%H:%M}"


def _day(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()]} {day.isoformat()}"


def _calendars(hass: HomeAssistant, wanted: str) -> list[str]:
    """The exposed calendars; only the one named `wanted` (its id, name or an alias), when given."""
    if not wanted:
        return [state.entity_id for state in exposed_states(hass, "calendar")]
    named = entity_named(hass, wanted, domain="calendar")
    return [named] if named else []


def _name(hass: HomeAssistant, entity_id: str) -> str:
    state = hass.states.get(entity_id)
    return (state and state.attributes.get("friendly_name")) or entity_id


def _fmt_event(event: Event) -> str:
    """One line per appointment, in local time, its weekday written by code: models get it wrong from the date."""
    summary = event.summary or "(no title)"
    day = WEEKDAYS[event.start.weekday()]
    if not isinstance(event.start, datetime):
        return f"{day} {event.start.isoformat()} (all day): {summary}"
    until = f"-{event.end:%H:%M}" if isinstance(event.end, datetime) else ""
    return f"{day} {event.start:%Y-%m-%d %H:%M}{until}: {summary}"
