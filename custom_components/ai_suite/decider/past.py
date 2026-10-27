"""A sensor's value over an earlier day, from HA's long-term statistics (recorder.get_statistics).

The value is the highest, the lowest, the average or the total. The decider says which one is asked, the statistics say
which the sensor has: a summing sensor (energy, water) has a total, a measuring one (temperature) the rest. No HA
imports.
"""

from .turn import HandOff

PAST = "past"
PAST_DESCRIPTION = "the highest, lowest, average or total value of a sensor on an earlier day"
NO_STATISTIC = "none"
TOTAL = "total"
STATISTICS = {
    "max": "the highest value",
    "min": "the lowest value",
    "mean": "the average",
    TOTAL: "the total amount (energy, water, rain)",
    NO_STATISTIC: "something else: when something happened, or which state it had",
}
# How a result note names each value. The question's description of a total would be spoken as part of the value,
# its examples read as what was measured.
NOTED = {"max": "the highest value", "min": "the lowest value", "mean": "the average", TOTAL: "the total"}
# How a quantity was over the day, no single value named: its average, lowest and highest
# (form.py decides it). No choice for the decider: its "none" also means "when did it happen", which statistics
# cannot answer. Handed to the text model, it may give another room's or today's value.
DAY = "day"
DAY_STATISTICS = ("mean", "min", "max")
STATISTIC_QUESTION = {"type": "choice", "instructions": "What does the user ask for?", "criteria": STATISTICS}
# What HA's statistics hold for each kind of sensor; the agent asks for all of them.
STATISTIC_TYPES = ["change", "max", "mean", "min"]


def past_note(statistic: str, rows: list[dict], unit: str) -> str:
    """The asked value for the result note ("the highest value 21.5 °C"); for DAY the average, lowest and highest."""
    asked = DAY_STATISTICS if statistic == DAY else (statistic,)
    return ", ".join(f"{NOTED[s]} {round(past_value(s, rows), 2)} {unit}".rstrip() for s in asked)


def past_value(statistic: str, rows: list[dict]) -> float:
    """The statistic over the day's rows (hourly).

    That is a total from a summing sensor's changes, or the highest, lowest or average from a measuring sensor's.
    HandOff when the sensor has no such value.
    """
    if statistic == TOTAL:
        changes = [row["change"] for row in rows if row.get("change") is not None]
        if not changes:
            raise HandOff("a total, but the sensor does not sum")
        return sum(changes)
    values = [row[statistic] for row in rows if row.get(statistic) is not None]
    if not values:
        raise HandOff(f"{statistic}, but the sensor has no such statistic")
    if statistic == "max":
        return max(values)
    if statistic == "min":
        return min(values)
    return sum(values) / len(values)
