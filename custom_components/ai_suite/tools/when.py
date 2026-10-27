"""Times as the model gives them and as the tools write them, for History and Calendar.

The model is a tool caller, not a calendar: it picks the day, code does the date maths and writes the weekday.
"""

from datetime import datetime, timedelta
import re
from typing import Any, overload

from homeassistant.util import dt as dt_util

BARE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Indexed by date.weekday(); strftime's %A would follow the process locale.
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Even with today's date from GetDateTime, a model may turn the user's yesterday into a two-day window starting the
# day before, whatever the tool description says, so the arithmetic is done here. English keywords only: the model
# translates the user's word for the day fine, it is the date maths it gets wrong.
NAMED_DAYS = {"today": 0, "yesterday": 1}


@overload
def parse_when(value: str | None, default: datetime, anchor: datetime | None = None) -> datetime: ...


@overload
def parse_when(value: str | None, default: None, anchor: datetime | None = None) -> datetime | None: ...


def parse_when(value: str | None, default: datetime | None, anchor: datetime | None = None) -> datetime | None:
    """Parse an ISO timestamp, 'today'/'yesterday', or a relative offset like '5d', '12h', '30m'.

    Relative offsets count back from `anchor` (the end of the range), so start='5d' with an explicit end means five
    days before that end, not five days before now.
    """
    if not value:
        return default
    text = value.strip()
    if (named := _named_day(text)) is not None:
        return named
    if (delta := parse_offset(text)) is not None:
        return (anchor or dt_util.now()) - delta
    parsed = dt_util.parse_datetime(text)
    if parsed is None:
        parsed = dt_util.parse_datetime(f"{text}T00:00:00")
    if parsed is None:
        raise ValueError(
            f"Could not read the time '{value}'. Use an ISO timestamp "
            "(2030-01-15 or 2030-01-15T14:00:00) or an offset like 5d, 12h, 30m."
        )
    return dt_util.as_local(parsed) if parsed.tzinfo is None else parsed


def parse_offset(value: str) -> timedelta | None:
    """A length like '30m', '12h', '5d', '2w' or '1y'; None for anything else."""
    text = value.strip()
    if not text or text[-1] not in "mhdwy" or not text[:-1].replace(".", "", 1).isdigit():
        return None
    amount = float(text[:-1])
    return {
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
        "w": timedelta(weeks=amount),
        "y": timedelta(days=amount * 365),
    }[text[-1]]


def fmt_day(ts: Any, period: str) -> str:
    """A row's or a range end's time, as fine as the period: a minute, a day or a month."""
    when = ts if isinstance(ts, datetime) else dt_util.utc_from_timestamp(float(ts))
    when = dt_util.as_local(when)
    if period in ("5minute", "hour"):
        return when.strftime("%Y-%m-%d %H:%M")
    if period == "month":
        return when.strftime("%Y-%m")
    return when.strftime("%Y-%m-%d")


def _named_day(value: str) -> datetime | None:
    """Resolve 'today'/'yesterday' to the start of that local calendar day."""
    days = NAMED_DAYS.get(value.strip().lower())
    if days is None:
        return None
    return dt_util.start_of_local_day(dt_util.now() - timedelta(days=days))
