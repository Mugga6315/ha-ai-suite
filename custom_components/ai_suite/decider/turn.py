"""What every decided turn shares: the hand-off, its timings, a target as HA intent slots. No HA imports."""

from dataclasses import dataclass, field
import time
from typing import Any

from .decision import RoomGroup, Target


class HandOff(Exception):
    """The decision path declines; the request goes to the fallback agent. The message says why."""


@dataclass(slots=True)
class Timings:
    """Milliseconds per stage, logged with every request."""

    start: float = field(default_factory=time.monotonic)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        """Note how long the turn took until this stage."""
        self.marks[stage] = (time.monotonic() - self.start) * 1000

    def __str__(self) -> str:
        """The stages, for the turn's log line."""
        return " ".join(f"{stage}={ms:.0f}ms" for stage, ms in self.marks.items())


def target_slots(target: Target) -> dict[str, Any]:
    """HA intent slots: a room group targets area + domain, an entity its name + domain."""
    if isinstance(target, RoomGroup):
        return {"area": target.area, "domain": target.domain}
    return {"name": target.name, "domain": target.domain}


def describe(target: Target) -> str:
    """The target, for a log line or a hand-off reason."""
    return target.label if isinstance(target, RoomGroup) else target.entity_id
