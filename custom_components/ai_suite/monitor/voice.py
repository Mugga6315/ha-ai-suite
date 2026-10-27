"""A turn's voice part: the assist pipeline run that carried it, read from HA's pipeline debug store.

HA keeps the events of the last 10 runs of each pipeline (assist_pipeline/run.py STORED_PIPELINE_RUNS), the same
events a live run sends. At the turn's open, `find_run` looks for the run whose intent stage is running for the
turn's conversation; after the turn, `wait_voice` waits for that run's run-end, for the satellite's idle and for
the answer's playback on the media player of the satellite's device, then turns the events into the record's `voice`
block (times in ms from run-start).

The satellite's idle is not playback end: an ESPHome satellite goes idle when HA has streamed the answer, before its
media player starts playing it. The media player's change to `playing` after tts-start is when the first words sound,
its next change away from `playing` the playback's end.

The store is not public API: KEY_ASSIST_PIPELINE, PipelineData.pipeline_debug, PipelineRunDebug and DATA_CONFIG are
not in assist_pipeline's `__all__`. Every access to them is in this module and guarded: when HA changes them, turns
carry no voice part, one warning is logged per HA start, and the rest of the record still ships.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging
from time import monotonic, time
from typing import Any

from homeassistant.components.assist_pipeline import PipelineEvent, async_get_pipelines
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util.hass_dict import HassKey

from .built import VoicePart
from .const import IDLE_WAIT_SECONDS, POLL_SECONDS, RUN_END_WAIT_SECONDS

try:
    from homeassistant.components.assist_pipeline.const import (
        CONF_DEBUG_RECORDING_DIR,
        DATA_CONFIG,
    )
    from homeassistant.components.assist_pipeline.runtime import KEY_ASSIST_PIPELINE
except ImportError as err:  # HA moved them: no voice part, find_run warns
    _NOT_IMPORTED: ImportError | None = err
else:
    _NOT_IMPORTED = None

_LOGGER = logging.getLogger(__name__)

PIPELINE_DOMAIN = "assist_pipeline"
# What a changed store raises; ValueError: a timestamp no longer in ISO format.
_STORE_CHANGED = (ImportError, AttributeError, KeyError, TypeError, ValueError)
_WARNED: HassKey[bool] = HassKey("ai_suite_monitor_voice_store_warned")
SATELLITE_IDLE = "idle"  # AssistSatelliteState.IDLE: HA marked the satellite idle (not playback end, see above)
MEDIA_PLAYER_DOMAIN = "media_player"
MEDIA_PLAYING = "playing"  # MediaPlayerState.PLAYING


@dataclass(frozen=True, slots=True)
class RunRef:
    """The pipeline run behind a turn, as found at the turn's open."""

    pipeline_id: str
    pipeline_name: str | None  # None: a pipeline not in the store (a conversation entity's own)
    run_id: str
    debug: Any  # assist_pipeline's PipelineRunDebug: its `events` grow until run-end
    run_start: float  # wall clock of the run's run-start event
    device_id: str | None  # the run's device: part of its debug recording folder
    recording_dir: str | None  # assist_pipeline's debug_recording_dir; None when not set


def find_run(hass: HomeAssistant, conversation_id: str) -> RunRef | None:
    """The run whose intent stage is running for the conversation now; None for a turn no pipeline carries.

    Called on the event loop at the turn's open: reads at most 10 runs per pipeline, newest events first.
    """
    if PIPELINE_DOMAIN not in hass.config.components:
        return None
    try:
        return _find_run(hass, conversation_id)
    except _STORE_CHANGED as err:
        _warn_store_changed(hass, err)
        return None


async def wait_voice(hass: HomeAssistant, ref: RunRef, satellite_id: str | None) -> VoicePart | None:
    """The run's voice part once it ended, its satellite went idle and its answer played; None when HA's store changed.

    Waits up to RUN_END_WAIT_SECONDS for run-end and builds from what the run has by then. With a satellite and a
    tts-start, waits up to IDLE_WAIT_SECONDS after tts-start for the satellite's idle and the playback on its device's
    media players (null what is not seen).
    """
    try:
        await _run_end(ref.debug)
        events: list[PipelineEvent] = list(ref.debug.events)
        moments = _moments(events)
        playback = None
        if satellite_id is not None and (tts := moments.get("tts-start")) is not None:
            playback = await _playback_after(hass, satellite_id, _device_players(hass, satellite_id), _wall(tts))
        return VoicePart(ref.run_start, _block(ref, moments, satellite_id, playback))
    except _STORE_CHANGED as err:
        _warn_store_changed(hass, err)
        return None


def _find_run(hass: HomeAssistant, conversation_id: str) -> RunRef | None:
    """The newest run in the debug store whose open intent stage belongs to the conversation."""
    if _NOT_IMPORTED is not None:
        raise _NOT_IMPORTED
    found: tuple[str, str, Any, PipelineEvent] | None = None
    for pipeline_id, runs in hass.data[KEY_ASSIST_PIPELINE].pipeline_debug.items():
        for run_id, debug in runs.items():
            if (intent := _open_intent(debug.events, conversation_id)) is not None:
                found = pipeline_id, run_id, debug, intent
    if found is None:
        return None
    pipeline_id, run_id, debug, intent = found
    return RunRef(
        pipeline_id=pipeline_id,
        pipeline_name=next((p.name for p in async_get_pipelines(hass) if p.id == pipeline_id), None),
        run_id=run_id,
        debug=debug,
        run_start=_wall(debug.events[0]),  # run-start: the first event of every run (assist_pipeline/run.py)
        device_id=(intent.data or {}).get("device_id"),
        recording_dir=hass.data[DATA_CONFIG].get(CONF_DEBUG_RECORDING_DIR),
    )


def _open_intent(events: list[PipelineEvent], conversation_id: str) -> PipelineEvent | None:
    """The run's last intent-start when it is for the conversation and has not ended yet."""
    for event in reversed(events):
        if event.type in ("intent-end", "run-end"):
            return None
        if event.type == "intent-start":
            return event if (event.data or {}).get("conversation_id") == conversation_id else None
    return None


async def _run_end(debug: Any) -> None:
    """Until the run's last event is run-end, or RUN_END_WAIT_SECONDS passed."""
    deadline = monotonic() + RUN_END_WAIT_SECONDS
    while debug.events[-1].type != "run-end" and monotonic() < deadline:
        await asyncio.sleep(POLL_SECONDS)


def _device_players(hass: HomeAssistant, satellite_id: str) -> tuple[str, ...]:
    """The enabled media players on the satellite's device; none when the satellite has no registered device."""
    registry = er.async_get(hass)
    satellite = registry.async_get(satellite_id)
    if satellite is None or satellite.device_id is None:
        return ()
    entries = er.async_entries_for_device(registry, satellite.device_id)
    return tuple(entry.entity_id for entry in entries if entry.domain == MEDIA_PLAYER_DOMAIN)


@dataclass(slots=True)
class _Playback:
    """What the satellite and its device's media players did after tts-start, as wall times (epoch seconds).

    `player` is the first of `players` to start playing; `start` and `end` are its change to and away from playing.
    """

    satellite_id: str
    players: tuple[str, ...]
    idle: float | None = None
    player: str | None = None
    start: float | None = None
    end: float | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def note(self, entity_id: str, old: State | None, new: State | None, at: float) -> None:
        """Take one state change of a watched entity; `done` is set once nothing more is waited for."""
        before, after = _state_of(old), _state_of(new)
        if before == after:
            return  # an attribute change: no transition
        if entity_id == self.satellite_id and after == SATELLITE_IDLE and self.idle is None:
            self.idle = at
        elif self.player is None and entity_id in self.players and after == MEDIA_PLAYING:
            self.player, self.start = entity_id, at
        elif entity_id == self.player and self.end is None and before == MEDIA_PLAYING:
            self.end = at
        if self.idle is not None and (not self.players or self.end is not None):
            self.done.set()


async def _playback_after(hass: HomeAssistant, satellite_id: str, players: tuple[str, ...], since: float) -> _Playback:
    """The satellite's idle and its answer's playback after `since`; null what is not seen within IDLE_WAIT_SECONDS.

    A change before the watch began counts when the entity's last change was at or after `since`; a player already
    playing before it (music, the wake sound) has no change to playing and is ignored until it starts again.
    """
    playback = _Playback(satellite_id, players)
    for entity_id in (satellite_id, *players):
        state = hass.states.get(entity_id)
        if state is not None and state.last_changed_timestamp >= since:
            playback.note(entity_id, None, state, state.last_changed_timestamp)

    @callback
    def on_change(event: Event[EventStateChangedData]) -> None:
        data = event.data
        playback.note(data["entity_id"], data["old_state"], data["new_state"], event.time_fired_timestamp)

    stop = async_track_state_change_event(hass, [satellite_id, *players], on_change)
    try:
        async with asyncio.timeout(since + IDLE_WAIT_SECONDS - time()):
            await playback.done.wait()
    except TimeoutError:
        pass  # what was seen by then is the record's
    finally:
        stop()
    return playback


def _state_of(state: State | None) -> str | None:
    return None if state is None else state.state


def _moments(events: list[PipelineEvent]) -> dict[str, PipelineEvent]:
    """The first event of each type; for intent-progress the first chat-log delta and the TTS streaming start."""
    moments: dict[str, PipelineEvent] = {}
    for event in events:
        if event.type == "intent-progress":
            data = event.data or {}
            if "chat_log_delta" in data:
                moments.setdefault("first_delta", event)
            if data.get("tts_start_streaming"):
                moments.setdefault("tts_start_streaming", event)
        else:
            moments.setdefault(event.type, event)
    return moments


def _block(
    ref: RunRef, moments: dict[str, PipelineEvent], satellite_id: str | None, playback: _Playback | None
) -> dict[str, Any]:
    """The turn record's `voice` block without `audio`; a stage the run did not reach is left out.

    `speaker` only when the satellite's device has a media player, which is looked up only for a run with tts-start.
    """

    def ms(event: PipelineEvent | None) -> int | None:
        return None if event is None else _ms(_wall(event), ref.run_start)

    def wall_ms(wall: float | None) -> int | None:
        return None if wall is None else _ms(wall, ref.run_start)

    def data(kind: str) -> dict[str, Any]:
        return (event.data or {}) if (event := moments.get(kind)) is not None else {}

    block: dict[str, Any] = {"pipeline_id": ref.pipeline_id, "run_id": ref.run_id}
    if wake_word := data("wake_word-end").get("wake_word_output"):
        block["wake"] = {"word": wake_word.get("wake_word_id"), "t": ms(moments["wake_word-end"])}
    if "stt-start" in moments:
        block["stt"] = {
            "engine": data("stt-start").get("engine"),
            "metadata": data("stt-start").get("metadata"),
            "text": (data("stt-end").get("stt_output") or {}).get("text"),
            "vad_start": ms(moments.get("stt-vad-start")),
            "vad_end": ms(moments.get("stt-vad-end")),
            "end": ms(moments.get("stt-end")),
        }
    block["intent"] = {"start": ms(moments.get("intent-start")), "first_delta": ms(moments.get("first_delta"))}
    if "tts_start_streaming" in moments:
        block["intent"]["tts_start_streaming"] = ms(moments["tts_start_streaming"])
    block["intent"]["end"] = ms(moments.get("intent-end"))
    if "tts-start" in moments:
        tts = data("tts-start")
        block["tts"] = {
            "engine": tts.get("engine"),
            "language": tts.get("language"),
            "voice": tts.get("voice"),
            "input": tts.get("tts_input"),
            "start": ms(moments["tts-start"]),
            "streamed": "tts_start_streaming" in moments,  # the answer's text was streamed into TTS
        }
    if satellite_id is not None:
        # The satellite's states follow the run's events (assist_satellite/entity.py); only idle is its own.
        block["satellite"] = {
            "listening": ms(moments.get("stt-start")),
            "processing": ms(moments.get("intent-start")),
            "responding": ms(moments.get("tts-start")),
            "idle": None if playback is None else wall_ms(playback.idle),
        }
    if playback is not None and playback.players:
        block["speaker"] = {
            "entity_id": playback.player,
            "start": wall_ms(playback.start),
            "end": wall_ms(playback.end),
        }
    return block


def _wall(event: PipelineEvent) -> float:
    """The event's wall clock (epoch seconds) from its ISO timestamp."""
    return datetime.fromisoformat(event.timestamp).timestamp()


def _ms(wall: float, run_start: float) -> int:
    return round((wall - run_start) * 1000)


def _warn_store_changed(hass: HomeAssistant, err: Exception) -> None:
    """Warn once per HA start that turns carry no voice part."""
    if hass.data.get(_WARNED):
        return
    hass.data[_WARNED] = True
    _LOGGER.warning(
        "Turns are recorded without their voice part: Home Assistant's assist pipeline debug store could not be "
        "read (%s: %s). AI Suite needs an update for this Home Assistant version",
        type(err).__name__,
        err,
    )
