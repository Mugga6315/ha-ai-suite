"""What the decider does with each Home Assistant domain, in one table.

A domain's row says whether its state answers a question, whether HA's on/off intents switch it, whether its reading
carries a temperature, and the states each change (changes.py) ends in. The states are HA's, as its purpose-specific
triggers (door, light, lock, ...) have them. No HA imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Domain:
    """One domain as the decider handles it."""

    readable: bool = False  # its state answers a question itself (decision.read_answer)
    switched: bool = False  # HA's on/off intents switch it
    temperature: bool = False  # its reading carries a temperature without a temperature device class
    changes: Mapping[str, frozenset[str]] = field(default_factory=dict)  # change -> the states it ends in


_ON, _OFF = frozenset({"on"}), frozenset({"off"})
_SWITCHED = {"switched_on": _ON, "switched_off": _OFF}

# Left out of `readable` on purpose: a to-do list's state is its count of open items, a calendar's whether an event
# runs now, a script's whether it runs.
DOMAINS = {
    # Which of these changes a binary sensor makes is its device class's (BINARY_SENSOR_CHANGES).
    "binary_sensor": Domain(
        readable=True, changes={"opened": _ON, "closed": _OFF, "unlocked": _ON, "locked": _OFF, "detected": _ON}
    ),
    # Every HVAC mode but off is on.
    "climate": Domain(
        readable=True, switched=True, temperature=True,
        changes={"switched_on": frozenset({"auto", "cool", "dry", "fan_only", "heat", "heat_cool"}),
                 "switched_off": _OFF},
    ),
    # A cover is open in every state but closed, as HA's is_closed has it: a closing cover is still open.
    "cover": Domain(
        readable=True, changes={"opened": frozenset({"open", "opening", "closing"}), "closed": frozenset({"closed"})}
    ),
    "fan": Domain(readable=True, switched=True, changes=_SWITCHED),
    "light": Domain(readable=True, switched=True, changes=_SWITCHED),
    # An open lock (unlatched) is unlocked too.
    "lock": Domain(readable=True, changes={"locked": frozenset({"locked"}), "unlocked": frozenset({"unlocked", "open"})}),
    # A player's state alone says little, but it is read with what it plays. Every state but off is on.
    "media_player": Domain(
        readable=True, switched=True,
        changes={"switched_on": frozenset({"on", "playing", "paused", "idle", "buffering"}), "switched_off": _OFF},
    ),
    "sensor": Domain(readable=True),
    "switch": Domain(readable=True, switched=True, changes=_SWITCHED),
    "todo": Domain(),  # the lists (lists.py)
    "weather": Domain(readable=True, temperature=True),
}

# The changes a binary sensor makes, by device class: HA's classes that open and close, and its lock class (on:
# unlocked). A binary sensor of any other class detects something.
BINARY_SENSOR_CHANGES = {
    **dict.fromkeys(("door", "garage_door", "opening", "window"), frozenset({"opened", "closed"})),
    "lock": frozenset({"locked", "unlocked"}),
}
DETECTION = frozenset({"detected"})

READABLE = frozenset(name for name, domain in DOMAINS.items() if domain.readable)
SWITCHED = frozenset(name for name, domain in DOMAINS.items() if domain.switched)
TEMPERATURE = frozenset(name for name, domain in DOMAINS.items() if domain.temperature)
CHANGE_DOMAINS = frozenset(name for name, domain in DOMAINS.items() if domain.changes)
