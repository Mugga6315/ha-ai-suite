"""What a media player's volume is after a volume intent: the intent only sends it, the player's state shows it.

HA's volume intents report success once the command is sent and answer with the players they targeted
(`matched_states`): HassSetVolumeRelative as they were before it, HassSetVolume as read once its action returned,
which can be before the device reported. The player shows its new volume when its device reports it, a moment
later; an unavailable player, or one that ignores the command, never does. A reply built on the intent alone would
confirm a change that did not happen.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.components.media_player import ATTR_MEDIA_VOLUME_LEVEL
from homeassistant.components.media_player.const import (
    INTENT_SET_VOLUME,
    INTENT_SET_VOLUME_RELATIVE,
)
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.event import async_track_state_change_event

VOLUME_INTENTS = frozenset({INTENT_SET_VOLUME, INTENT_SET_VOLUME_RELATIVE})
# How long a player gets to report its new volume; players report it well within a second.
VOLUME_TIMEOUT_S = 2.0
# A level this near the asked one counts as reached: players round it to their own volume steps.
_NEAR = 0.02


@dataclass(frozen=True, slots=True)
class Asked:
    """What a volume intent asks: a level from 0 to 1, or a direction (1 louder, -1 quieter)."""

    level: float | None = None
    direction: int = 0

    def possible(self, before: float) -> bool:
        """Whether the command can change a player at `before`: not one already there or at the end of the scale."""
        if self.level is not None:
            return abs(before - self.level) > _NEAR
        return before < 1 if self.direction > 0 else before > 0

    def shown(self, before: float, now: float | None) -> bool:
        """Whether a player that was at `before` shows the change at `now`: it moved the way asked.

        A level may land on the player's nearest step, short of the one asked or past it.
        """
        if now is None:
            return False
        if self.level is not None:
            return abs(now - self.level) <= _NEAR or (now - before) * (self.level - before) > 0
        return (now - before) * self.direction > 0


def asked_by(intent_type: str, slots: Mapping[str, Any]) -> Asked | None:
    """What a volume intent asks, read from the slot values as HA reads them (whole percent); None for another intent.

    A step of 0 asks nothing: None.
    """
    if intent_type == INTENT_SET_VOLUME:
        return Asked(level=int(slots["volume_level"]) / 100)
    if intent_type == INTENT_SET_VOLUME_RELATIVE:
        step = slots["volume_step"]
        if step in ("up", "down"):
            return Asked(direction=1 if step == "up" else -1)
        percent = int(step)
        return Asked(direction=1 if percent > 0 else -1) if percent else None
    return None


async def async_volume_problem(hass: HomeAssistant, players: list[State], asked: Asked) -> str | None:
    """Which of the intent's players do not show the volume asked, with the ones that do; None when none fails.

    Only players that showed a volume are checked: an off one shows none, and a command for an area still reaches the
    others. Each gets VOLUME_TIMEOUT_S to report its new volume, read from its state changes; one already where the
    command would take it is no failure. An unavailable player is named.
    """
    unavailable = [player.name for player in players if player.state == STATE_UNAVAILABLE]
    before = {
        player.entity_id: level
        for player in players
        if player.state != STATE_UNAVAILABLE and (level := _volume(player)) is not None and asked.possible(level)
    }

    def waiting() -> list[str]:
        return [entity_id for entity_id, level in before.items()
                if not asked.shown(level, _volume(hass.states.get(entity_id)))]

    done = asyncio.Event()

    @callback
    def check(_event: Event[EventStateChangedData]) -> None:
        if not waiting():
            done.set()

    unsubscribe = async_track_state_change_event(hass, list(before), check)
    try:
        if waiting():
            async with asyncio.timeout(VOLUME_TIMEOUT_S):
                await done.wait()
    except TimeoutError:
        pass
    finally:
        unsubscribe()
    stayed = waiting()
    if not unavailable and not stayed:
        return None
    facts = [f"{name} is unavailable" for name in unavailable]
    for entity_id in before:
        state = hass.states.get(entity_id)
        name, at = (state.name, _percent(_volume(state))) if state is not None else (entity_id, _percent(None))
        facts.append(f"the volume of {name} stayed at {at}" if entity_id in stayed else f"{name} is now at {at}")
    return "; ".join(facts)


def _volume(state: State | None) -> float | None:
    if state is None:
        return None
    level = state.attributes.get(ATTR_MEDIA_VOLUME_LEVEL)
    return float(level) if level is not None else None


def _percent(level: float | None) -> str:
    return "an unknown level" if level is None else f"{round(level * 100)} %"
