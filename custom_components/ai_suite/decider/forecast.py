"""Weather forecasts for the days a question names.

Weather forecasts: the decider recognises a question about the weather of coming days, code works out which days
the text means (days.py) and reads HA's daily forecast (weather.get_forecasts) for them; the answer step words the
facts. No HA imports.
"""

from collections.abc import Mapping
from datetime import date, datetime

from .days import day_named, days_asked, next_days_asked
from .turn import HandOff

FORECAST = "forecast"
FORECAST_DESCRIPTION = "ask about the weather of a coming day or days (tomorrow, the weekend, the next days)"
# HA's conditions with rain. Said as a fact, so a small answer model need not work it out: from the condition alone
# it may answer no to a question whether it will rain.
RAIN = frozenset({"rainy", "pouring", "lightning-rainy", "snowy-rainy"})
# From this many days the note leads with the days grouped by condition: with many day lines the answer step merges
# days and moves the day the weather turns; two lines it words right.
SUMMARY_FROM_DAYS = 3


def forecast_days(text: str, entries: list[tuple[datetime, dict]], today: date) -> list[date]:
    """The days the question means.

    "The next days" are every day the forecast holds after today: a fixed few would leave out that the rain of the
    first days turns cloudy later. Named days stay as named.
    """
    if next_days_asked(text):
        return sorted({when.date() for when, _ in entries if when.date() > today})
    return days_asked(text, today)


def forecast_facts(
    entries: list[tuple[datetime, dict]],
    days: list[date],
    today: date,
    unit: str,
    conditions: Mapping[str, str] | None = None,
) -> str:
    """One line per asked day from HA's daily forecast entries (local time, entry); HandOff if a day is missing.

    `conditions`: HA's word for each condition in the user's language; small answer models misspell or drop the
    English one when they translate it.
    """
    by_day = {when.date(): entry for when, entry in entries}
    if missing := [day for day in days if day not in by_day]:
        raise HandOff(f"no forecast for {missing}")
    facts = "; ".join(_day_fact(day_named(day, today), by_day[day], unit, conditions or {}) for day in days)
    if len(days) < SUMMARY_FROM_DAYS:
        return facts
    return f"Summary by condition: {_summary(days, by_day, today, conditions or {})}. Each day: {facts}"


def _summary(days: list[date], by_day: Mapping[date, dict], today: date, conditions: Mapping[str, str]) -> str:
    """The days grouped by condition in a row.

    The answer step then words where the weather turns from code, not from its own reading of the day lines.
    """
    runs: list[tuple[str, list[date]]] = []
    for day in days:
        condition = by_day[day].get("condition") or "unknown"
        if runs and runs[-1][0] == condition:
            runs[-1][1].append(day)
        else:
            runs.append((condition, [day]))
    return "; ".join(
        f"{conditions.get(condition, condition)} on {day_named(run[0], today)}" if len(run) == 1
        else f"{conditions.get(condition, condition)} from {day_named(run[0], today)} to {day_named(run[-1], today)}"
        for condition, run in runs
    )


def _day_fact(day: str, entry: dict, unit: str, conditions: Mapping[str, str]) -> str:
    parts = ["unknown"]
    if condition := entry.get("condition"):
        parts = [conditions.get(condition, condition), f"rain: {'yes' if condition in RAIN else 'no'}"]
    if entry.get("temperature") is not None:
        parts.append(f"high {entry['temperature']} {unit}")
    if entry.get("templow") is not None:
        parts.append(f"low {entry['templow']} {unit}")
    if entry.get("precipitation_probability") is not None:
        parts.append(f"chance of rain {entry['precipitation_probability']} %")
    if entry.get("precipitation") is not None:
        parts.append(f"precipitation {entry['precipitation']} mm")
    return f"{day}: {', '.join(parts)}"
