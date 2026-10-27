"""A turn's effects: the state changes in the home the turn caused, matched by context.

HA writes a service call's context on each entity the call changes (helpers/service.py) and keeps it there for 5 s
(CONTEXT_RECENT_TIME_SECONDS, helpers/entity.py). A change under a held turn's context is its own effect; a change
under a context the turn caused (a script or automation it started, whose context names the turn as parent) is a
child effect. Derived sensors (a device's power reading) change under their own context and so are not effects.
How long after the turn's close a change still counts is add_effect's window; a released turn is no longer matched.
"""

from typing import Literal

from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)

from .recorder import Effect, Recorder, TurnTrace, add_effect

# Written under the turn's context too, but the turn's own machinery, not what it did in the home: the agent's
# conversation entity (conversation/entity.py) and the satellite that carried the turn (voice.py watches its idle).
EXCLUDED_DOMAINS = frozenset({"conversation", "assist_satellite"})

type Via = Literal["own", "child"]


def async_listen_effects(hass: HomeAssistant, recorder: Recorder) -> CALLBACK_TYPE:
    """Keep every state change a held turn caused as that turn's effect; the returned callable stops listening."""

    @callback
    def caused_by_turn(data: EventStateChangedData) -> bool:
        # Runs on every state change in the home: dict lookups only.
        return _cause(recorder, data["new_state"]) is not None

    @callback
    def keep_effect(event: Event[EventStateChangedData]) -> None:
        # The bus runs this callback right after its filter, so the turn found there is still held.
        new_state = event.data["new_state"]
        if (cause := _cause(recorder, new_state)) is not None:
            trace, via = cause
            effect = Effect(event.time_fired_timestamp, event.data["entity_id"], event.data["old_state"], new_state, via)
            add_effect(trace, effect)

    return hass.bus.async_listen(EVENT_STATE_CHANGED, keep_effect, event_filter=caused_by_turn)


def _cause(recorder: Recorder, new_state: State | None) -> tuple[TurnTrace, Via] | None:
    """The held turn the state was written under, and how; None when it is no turn's effect.

    A removed entity (new_state None) is never an effect: the event's data carries no context to match it by.
    """
    if new_state is None:
        return None
    context = new_state.context
    via: Via
    if (trace := recorder.turns.get(context.id)) is not None:
        via = "own"
    elif context.parent_id is not None and (trace := recorder.turns.get(context.parent_id)) is not None:
        via = "child"
    else:
        return None
    if new_state.domain in EXCLUDED_DOMAINS or new_state.entity_id == trace.start.agent_entity:
        return None
    return trace, via
