"""History: a sensor's long-term statistics and an entity's state history, read from HA's recorder.

The model is a tool caller, not a data analyst. It picks the tool, the entity, the time range, and whether the user
asked for a breakdown. Every deterministic decision is made here:

- **Which statistic applies** comes from recorder metadata, not from the model. `state_class: measurement` produces
  `has_mean` (answer = average/min/max); `total`/`total_increasing` produces `has_sum` (answer = consumption).
  For a sensor HA sets one of the two, never both (sensor/recorder.py, DEFAULT_STATISTICS).
- **`sum` is a cumulative counter**, so consumption over a window is a difference, never an addition. We ask HA for
  `change`, which is that difference computed by the recorder itself.
- **Row size** (HA's `period`) is derived from the span so a two-year question returns months and a six-hour question
  returns 5-minute slots. The model never sees or chooses a period.
- **Answers, not data.** A total is one number. A breakdown is the handful of numbers that *is* the answer. Neither
  is a dump of recorder rows.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta
import logging
from typing import Any, Literal

import probatio

from homeassistant.components import recorder
from homeassistant.components.recorder import history, statistics
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, selector
from homeassistant.util import dt as dt_util

from ..home import entity_named
from .spec import READ_ONLY, RecordFailure, ToolArgs, ToolError, ToolSpec, spec_tools
from .when import BARE_DATE, NAMED_DAYS, fmt_day, parse_when

_LOGGER = logging.getLogger(__name__)

# The History tools may read entities that are not exposed to voice assistants (the set's setting).
CONF_UNEXPOSED = "unexposed_entities"
DEFAULT_UNEXPOSED = True

# A breakdown is an answer, so it stays readable. Beyond this many lines the question was really about a coarser unit,
# and the code coarsens rather than truncating (see _period_for_breakdown).
MAX_BREAKDOWN_ROWS = 40
MAX_HISTORY_ROWS = 40

_PERIODS = ("5minute", "hour", "day", "week", "month")
_PERIOD_HOURS = {"5minute": 1 / 12, "hour": 1, "day": 24, "week": 24 * 7, "month": 24 * 30}

type Period = Literal["5minute", "hour", "day", "week", "month", "year"]


async def get_statistics(hass: HomeAssistant, unexposed: bool, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Answer from the sensor's statistics over the asked range; `unexposed`: unexposed entities may be read."""
    entity_id = _entity(hass, args["entity_id"], unexposed)
    try:
        start, end = _range(args, calendar_day=True)
    except ValueError as err:
        raise ToolError(str(err), learn=True) from err

    instance = recorder.get_instance(hass)
    metadata = await instance.async_add_executor_job(_metadata_for, hass, entity_id)
    if not metadata:
        raise ToolError(
            f"No long-term statistics for {entity_id}. Only entities with a state_class "
            "(measurement/total/total_increasing) are recorded; template and SQL sensors usually are not. Pick the "
            "underlying measuring entity, or use get_history for recent state changes.",
            learn=True,
        )

    has_sum = bool(metadata.get("has_sum"))
    breakdown = args.get("breakdown", "none")
    span = end - start
    period = _period_for_total(span) if breakdown == "none" else _period_for_breakdown(span, breakdown)
    types = {"change"} if has_sum else {"mean", "min", "max"}
    _LOGGER.debug(
        "get_statistics: %s args=%s -> %s .. %s period=%s types=%s", entity_id, args, start, end, period, sorted(types)
    )
    # Fine-grained rows may simply not exist: HA purges 5-minute statistics with the states, and externally imported
    # statistics often carry daily rows only. Escalate instead of reporting "no data": a few hours a month back have no
    # 5-minute rows but hourly ones.
    rows: list = []
    used = period
    for candidate in _PERIODS[_PERIODS.index(period):]:
        rows = await instance.async_add_executor_job(_statistics, hass, start, end, entity_id, candidate, types)
        if rows:
            used = candidate
            break
    coarser = used != period
    period = used
    if not rows:
        raise ToolError(
            f"{entity_id} has statistics metadata but no recorded rows between {fmt_day(start, period)} and "
            f"{fmt_day(end, period)}. The entity may have been added after that period, or the recorder purged it."
        )

    unit = metadata.get("unit") or ""
    lines = [f"{entity_id} | {_fmt_range(start, end)} local | unit: {unit or 'unknown'}"]
    if coarser:
        # Say it plainly: coarse rows can reach past the window that was asked for, and an unqualified number would
        # look exact.
        lines.append(
            f"note: no data at finer resolution, so this uses whole {period} values, which may extend past the "
            "requested range"
        )
    lines += _totals(rows, has_sum, unit)
    if breakdown != "none":
        lines += _breakdown(rows, has_sum, unit, period, "day" if breakdown == "day" else "month")
    return "\n".join(lines)


async def get_history(hass: HomeAssistant, unexposed: bool, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Answer from the entity's recorded state changes; `unexposed`: unexposed entities may be read."""
    entity_id = _entity(hass, args["entity_id"], unexposed)
    try:
        start, end = _range(args)
    except ValueError as err:
        raise ToolError(str(err), learn=True) from err

    instance = recorder.get_instance(hass)
    keep_days = getattr(instance, "keep_days", None)
    if keep_days and start < dt_util.now() - timedelta(days=keep_days):
        raise ToolError(
            f"State history only goes back {keep_days} days on this system; older changes are purged. For longer "
            "ranges use get_statistics, which reads long-term statistics."
        )

    states = await instance.async_add_executor_job(_history, hass, start, end, entity_id)
    if not states:
        return f"{entity_id} | {_fmt_range(start, end)} local: no recorded state changes."
    lines = [f"{entity_id} | {_fmt_range(start, end)} local (local) | {len(states)} state changes"]
    shown = states[-MAX_HISTORY_ROWS:]
    if len(states) > len(shown):
        lines.append(f"showing the most recent {len(shown)}:")
    lines += [
        f"  {dt_util.as_local(state.last_changed).strftime('%Y-%m-%d %H:%M:%S')}: {state.state}" for state in shown
    ]
    return "\n".join(lines)


SPECS: tuple[ToolSpec[bool], ...] = (
    ToolSpec(
        name="get_statistics",
        description=(
            "Answer questions about measured values over time: energy or water "
            "consumed, average temperature or power, highest/lowest. Works for any "
            "range from minutes to years. Give the entity_id and the time range; "
            "the answer comes back computed, with its unit. For yesterday pass "
            "start='yesterday'; for a named date pass that date (2030-01-15). Either "
            "way leave end out - it means that whole calendar day, not the last 24 "
            "hours. Never compute a date yourself. Set breakdown='day' or "
            "'month' only when the user wants the values split per day or month, "
            "otherwise leave it out and get a single total or average. This is the "
            "only way to answer 'how much', 'how often', 'on average' or 'compared "
            "to last week' - current readings come from the live context instead."
        ),
        parameters=probatio.Schema(
            {
                probatio.Required("entity_id"): str,
                probatio.Required("start"): str,
                probatio.Optional("end"): str,
                probatio.Optional("breakdown"): probatio.In(["none", "day", "month"]),
            }
        ),
        run=get_statistics,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="get_history",
        description=(
            "Look up when an entity changed state: when a door was last opened, how "
            "long a light was on, what the thermostat was set to this morning. Give "
            "the entity_id and the time range. Only recent history is kept (the "
            "recorder purges older data), so for anything beyond that use "
            "get_statistics instead."
        ),
        parameters=probatio.Schema(
            {
                probatio.Required("entity_id"): str,
                probatio.Required("start"): str,
                probatio.Optional("end"): str,
            }
        ),
        run=get_history,
        annotations=READ_ONLY,
    ),
)


def build(hass: HomeAssistant, data: Mapping[str, Any], record_failure: RecordFailure | None) -> list[llm.Tool]:
    """The History tools; none while the recorder does not run."""
    if "recorder" not in hass.config.components:
        _LOGGER.warning("the recorder is not running: the History tools are off")
        return []
    return spec_tools(SPECS, data.get(CONF_UNEXPOSED, DEFAULT_UNEXPOSED), record_failure)


def subentry_schema(data: dict) -> dict:
    """The History set's form: whether its tools may read entities not exposed to voice assistants."""
    return {
        probatio.Required(CONF_UNEXPOSED, default=data.get(CONF_UNEXPOSED, DEFAULT_UNEXPOSED)): selector.BooleanSelector()
    }


async def async_problem(hass: HomeAssistant, _data: Mapping[str, Any]) -> str | None:
    """What keeps the History tools from working, for the set's status light."""
    return None if "recorder" in hass.config.components else "the recorder is not running"


def _range(args: ToolArgs, calendar_day: bool = False) -> tuple[datetime, datetime]:
    """The asked window; a ValueError names what is wrong with it."""
    raw_start = (args.get("start") or "").strip()
    whole_day = bool(BARE_DATE.match(raw_start)) or raw_start.lower() in NAMED_DAYS
    # For consumption, a bare date means that calendar day. Without this, yesterday with no end becomes a rolling 24h
    # window reaching into today and reports today's figure as yesterday's. History is the opposite: a range from a
    # past day runs up to now, and a per-day reading made the model walk backwards one day per call.
    if calendar_day and whole_day and not args.get("end"):
        start = parse_when(raw_start, dt_util.now())
        end = min(start + timedelta(days=1), dt_util.now())
    else:
        end = _end_of(args.get("end"))
        start = parse_when(args.get("start"), end - timedelta(days=1), anchor=end)
    if start >= end:
        raise ValueError(f"'start' ({start}) must be before 'end' ({end}).")
    return start, end


def _end_of(value: str | None) -> datetime:
    """The range's end; an end that names a day (a bare date or a named day) includes that day, up to now.

    A model asking start=today end=today means today, not an empty range. The Calendar tool reads a date-only end the
    same way.
    """
    raw = (value or "").strip()
    end = parse_when(raw, dt_util.now())
    if BARE_DATE.match(raw) or raw.lower() in NAMED_DAYS:
        end = min(end + timedelta(days=1), dt_util.now())
    return end


def _period_for_total(span: timedelta) -> str:
    """Row size used to compute a total the model never sees.

    Rows are only summed here, so the only constraints are that they stay inside the requested range (coarse rows
    overshoot at the edges) and that the count stays sane. Sub-day precision for short ranges, days otherwise.
    """
    hours = span.total_seconds() / 3600
    if hours <= 6:
        return "5minute"
    if hours <= 48:
        return "hour"
    if hours <= 24 * 366 * 5:
        return "day"
    return "month"


def _period_for_breakdown(span: timedelta, asked: str) -> str:
    """Row size for a breakdown the model *does* see: coarsen before flooding."""
    hours = max(span.total_seconds() / 3600, 1)
    for period in _PERIODS[_PERIODS.index(asked):]:
        if hours / _PERIOD_HOURS[period] <= MAX_BREAKDOWN_ROWS:
            return period
    return "month"


def _totals(rows: list, has_sum: bool, unit: str) -> list[str]:
    """The total of a counter, or the average, minimum and maximum of a measurement."""
    if has_sum:
        return [f"total: {_fmt(sum(row.get('change') or 0 for row in rows))} {unit}".strip()]
    lines = []
    means = [row["mean"] for row in rows if row.get("mean") is not None]
    mins = [row["min"] for row in rows if row.get("min") is not None]
    maxes = [row["max"] for row in rows if row.get("max") is not None]
    if means:
        lines.append(f"average: {_fmt(sum(means) / len(means))} {unit}".strip())
    if mins:
        lines.append(f"minimum: {_fmt(min(mins))} {unit}".strip())
    if maxes:
        lines.append(f"maximum: {_fmt(max(maxes))} {unit}".strip())
    return lines


def _breakdown(rows: list, has_sum: bool, unit: str, period: str, asked: str) -> list[str]:
    """The rows per period, at most MAX_BREAKDOWN_ROWS, with the highest one of a counter."""
    if period != asked:
        lines = [f"breakdown per {period} (asked per {asked}; the range is too long for {asked} rows):"]
    else:
        lines = [f"breakdown per {period}:"]
    for row in rows[:MAX_BREAKDOWN_ROWS]:
        label = fmt_day(row["start"], period)
        if has_sum:
            lines.append(f"  {label}: {_fmt(row.get('change'))}")
        else:
            lines.append(
                f"  {label}: avg {_fmt(row.get('mean'))} (min {_fmt(row.get('min'))}, max {_fmt(row.get('max'))})"
            )
    if len(rows) > MAX_BREAKDOWN_ROWS:
        lines.append(f"  ... {len(rows) - MAX_BREAKDOWN_ROWS} more rows omitted")
    if has_sum:
        best = max(rows, key=lambda row: row.get("change") or 0)
        lines.append(f"highest {period}: {fmt_day(best['start'], period)} ({_fmt(best.get('change'))} {unit})".strip())
    return lines


def _fmt(value: float | None) -> str:
    """Trim float noise without hiding small values (0.00086666 -> 0.000867)."""
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _fmt_range(start: datetime, end: datetime) -> str:
    """Echo the resolved window with weekday names.

    The model computes dates and gets them wrong - for yesterday it may ask for the day before as well. Nothing
    server-side can tell a wrong day from an intended one, but a weekday in the answer makes the mistake visible to
    whoever is listening instead of leaving a plausible-looking number.
    """
    fmt = "%Y-%m-%d %H:%M (%a)"
    return f"{dt_util.as_local(start).strftime(fmt)} -> {dt_util.as_local(end).strftime(fmt)}"


def _entity(hass: HomeAssistant, given: str, unexposed: bool) -> str:
    """The entity the model means, by its id, name or an alias (home.entity_named).

    The model sends the entity_id most of the time but sometimes the friendly name instead, which then reads as "no
    statistics" for an entity that has them. `unexposed`: entities not exposed to voice assistants may be read too;
    external statistic ids (`<source>:<name>`) then pass through, and an unknown name is left to the recorder lookup,
    which errors honestly. Without it, only exposed entities are read.
    """
    given = (given or "").strip()
    if unexposed and ":" in given and "." not in given:  # external statistic id
        return given
    if (found := entity_named(hass, given, exposed_only=not unexposed)) is not None:
        return found
    if unexposed:
        return given
    raise ToolError(f"No entity exposed to the assistant is called '{given}'.", learn=True)


# --- executor-thread helpers (recorder calls are blocking) --------------------


def _metadata_for(hass: HomeAssistant, entity_id: str) -> dict | None:
    found = statistics.get_metadata(hass, statistic_ids={entity_id})
    if entity_id not in found:
        return None
    meta = found[entity_id][1]
    unit = meta.get("unit_of_measurement")
    if not unit:
        # list_statistic_ids can report a null unit; the live state knows it.
        state = hass.states.get(entity_id)
        if state:
            unit = state.attributes.get("unit_of_measurement")
    return {"has_sum": meta.get("has_sum"), "has_mean": meta.get("has_mean"), "unit": unit}


def _statistics(
    hass: HomeAssistant, start: datetime, end: datetime, entity_id: str, period: Period, types: set
) -> list:
    result = statistics.statistics_during_period(hass, start, end, {entity_id}, period, None, types)
    return result.get(entity_id, [])


def _history(hass: HomeAssistant, start: datetime, end: datetime, entity_id: str) -> list:
    # include_start_time_state=True prepends the state carried INTO the window. Without it an entity that never changed
    # inside the window returns zero rows and reads as "no recorded state changes".
    result = history.state_changes_during_period(
        hass, start, end, entity_id, no_attributes=True, include_start_time_state=True
    )
    return result.get(entity_id, [])
