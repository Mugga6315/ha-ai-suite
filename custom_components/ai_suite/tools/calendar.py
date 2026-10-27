"""Calendar: the entries of every exposed calendar at once, each with its weekday.

HA's own calendar tool reads one calendar for today or the next seven days and gives ISO dates only, from which models
name wrong weekdays. This one takes any range, merges every exposed calendar in date order, and code writes each
entry's weekday.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
from typing import Any

import probatio

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir, llm
from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from ..decider.calendar import Event, event_order
from ..home import entity_named, exposed_states, read_calendars
from .spec import READ_ONLY, RecordFailure, ToolArgs, ToolError, ToolSpec, spec_tools
from .when import BARE_DATE, WEEKDAYS, fmt_day, parse_offset, parse_when

_LOGGER = logging.getLogger(__name__)

MAX_CALENDAR_EVENTS = 40
NOT_EXPOSED_ISSUE = "calendars_not_exposed"


async def get_calendar_events(hass: HomeAssistant, _deps: None, args: ToolArgs, _context: llm.LLMContext) -> str:
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


SPECS: tuple[ToolSpec[None], ...] = (
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
)


def build(_hass: HomeAssistant, _data: Mapping[str, Any], record_failure: RecordFailure | None) -> list[llm.Tool]:
    """The Calendar tool. The set has no settings."""
    return spec_tools(SPECS, None, record_failure)


def check_exposed(hass: HomeAssistant, present: bool) -> None:
    """Raise a repair issue while the home has calendars and exposes none of them to the assistant.

    Calendars are not among HA's default exposed domains, so get_calendar_events sees none until they are exposed;
    nothing else would say so. `present`: whether the Calendar set is added.
    """
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

    An end like '1y' is a length after the start: the model asks a year ahead without knowing the year.
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
    """One line per appointment, in local time.

    The weekday is written by code because models get it wrong when left to work it out from the date.
    """
    summary = event.summary or "(no title)"
    day = WEEKDAYS[event.start.weekday()]
    if not isinstance(event.start, datetime):
        return f"{day} {event.start.isoformat()} (all day): {summary}"
    until = f"-{event.end:%H:%M}" if isinstance(event.end, datetime) else ""
    return f"{day} {event.start:%Y-%m-%d %H:%M}{until}: {summary}"
