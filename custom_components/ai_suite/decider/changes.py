"""When a device last changed.

The decider says which change is asked (opened, switched on, locked, ...), the domain table (domains.py) says which
devices can make it, and the device's recorded state history says when it last happened. No HA imports.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .decision import Candidate
from .domains import BINARY_SENSOR_CHANGES, CHANGE_DOMAINS, DETECTION, DOMAINS

CHANGED = "changed"
CHANGED_DESCRIPTION = (
    "when a device last changed: a door or window last opened or closed, a light last switched on or off, a lock last "
    "unlocked, or since when it is so"
)
ANY = "any"
# States that are no change a user means: the device was offline or not yet known.
_NOT_A_STATE = frozenset({"unavailable", "unknown"})


@dataclass(frozen=True, slots=True)
class Change:
    """A change a device can make: how the decider and the note name it, and the states it ends in."""
    description: str  # for the decider
    done: str  # for the result note
    states: dict[str, frozenset[str]]  # domain -> the states the device is in after the change


def _states(change: str) -> dict[str, frozenset[str]]:
    """The states each domain's devices end in after the change (domains.py)."""
    return {name: domain.changes[change] for name, domain in DOMAINS.items() if change in domain.changes}


CHANGES = {
    "opened": Change("a door, window or cover was opened", "opened", _states("opened")),
    "closed": Change("a door, window or cover was closed", "closed", _states("closed")),
    "switched_on": Change("a light or device was switched on", "switched on", _states("switched_on")),
    "switched_off": Change("a light or device was switched off", "switched off", _states("switched_off")),
    "locked": Change("a lock was locked", "locked", _states("locked")),
    "unlocked": Change("a lock was unlocked", "unlocked", _states("unlocked")),
    "detected": Change(
        "a sensor detected something (motion, presence, water, smoke)", "detected something", _states("detected")
    ),
}
OPENING_CHANGES = frozenset({"opened", "closed"})
LOCK_CHECK = {"type": "noul", "instructions": "Does the user ask about a lock: whether a door was locked or unlocked?"}
CHANGE_QUESTION = {
    "type": "choice",
    "instructions": "Which change does the user ask about?",
    "criteria": {name: change.description for name, change in CHANGES.items()}
    | {ANY: "any change, or since when the device is as it is now"},
}


@dataclass(frozen=True, slots=True)
class LastChange:
    """The state a device last changed into: since when, until when, and the state after it."""
    state: str
    since: datetime | None  # None: before the history begins
    until: datetime | None  # when it changed again; None: it still is so
    next_state: str | None


def can_make(change: str, device: Candidate) -> bool:
    """Whether the device can make the change.

    Its domain has the change's states, and a binary sensor's device class fits (a door sensor opens, a motion sensor
    detects).
    """
    if change == ANY:
        return device.domain in CHANGE_DOMAINS
    if device.domain not in CHANGES[change].states:
        return False
    if device.domain != "binary_sensor":
        return True
    device_class = (device.measures or "").split(" ")[0]
    return change in BINARY_SENSOR_CHANGES.get(device_class, DETECTION)


def last_change(history: Sequence[tuple[str, datetime | None]], change: str, domain: str) -> LastChange | None:
    """The latest change into the asked states in `history`, with the change after it.

    `history`: (state, since when), newest first; since None is the state the history begins with, not a change. A run
    of asked states is one change, from the oldest of them: a player that went from on to playing was switched on
    when it went on. For ANY, the state now and since when. None when the history holds no asked state.
    """
    changes = _real_changes(history)
    wanted = None if change == ANY else CHANGES[change].states[domain]
    for index, (state, _) in enumerate(changes):
        if wanted is None or state in wanted:
            start = index if wanted is None else _run_start(changes, index, wanted)
            after = changes[index - 1] if index else None
            into, since = changes[start]
            return LastChange(into, since, after[1] if after else None, after[0] if after else None)
    return None


def _run_start(changes: list[tuple[str, datetime | None]], index: int, wanted: frozenset[str]) -> int:
    """The index of the oldest entry in the run of wanted states that `index` begins, newest first."""
    while index + 1 < len(changes) and changes[index + 1][0] in wanted:
        index += 1
    return index


def _real_changes(history: Sequence[tuple[str, datetime | None]]) -> list[tuple[str, datetime | None]]:
    """The history's changes a user means, newest first.

    Offline and unknown states are left out, and a run of one state is counted from its oldest entry. Every HA start
    records each state again: without this, a door closed hours ago "was closed" at the restart.
    """
    changes: list[tuple[str, datetime | None]] = []
    for state, since in history:
        if state in _NOT_A_STATE:
            continue
        if changes and changes[-1][0] == state:
            changes[-1] = (state, since)
        else:
            changes.append((state, since))
    return changes


def change_fact(name: str, change: str, found: LastChange | None, history_start: datetime) -> str:
    """The result note for the answer step; times in the home's local time, as given.

    A model words a note that no change happened since a day as if the change happened on that day: no change and an
    unknown time are said so that no date reads as the change's.
    """
    begins = f"{history_start:%A %Y-%m-%d %H:%M}"
    if found is None:
        done = CHANGES[change].done if change != ANY else "changed"
        return f"{name}: was not {done} at any time in HA's recorded history (it goes back to {begins})"
    done = f"changed to {found.state}" if change == ANY else f"last {CHANGES[change].done}"
    when = f"on {found.since:%A %Y-%m-%d %H:%M}" if found.since else \
        f"at an unknown time before HA's recorded history begins ({begins})"
    then = f"; changed to {found.next_state} on {found.until:%A %Y-%m-%d %H:%M}" if found.until else "; still so now"
    return f"{name}: {done} {when}{then}"
