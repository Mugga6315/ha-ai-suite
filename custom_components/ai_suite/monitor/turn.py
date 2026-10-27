"""A turn of an AI Suite agent from its open to its record in the sender's queue.

`async_recorded_turn` runs around the agent's `async_process`: it opens the turn in the recorder, looks up the voice
pipeline run that carries it, records how it ended (the answer, or the error HA turns into one) and, once the turn
closed, hands it to a closing task the entry owns. That task waits for the turn's effects window and, for a voice
turn, for the run's end, the satellite's idle and the answer's playback; then it finds the run's STT audio (when the
option is on), builds the record and queues it for the sender. The turn is released in any case, so the recorder's
cap never fills with turns that will not be built. The turn itself only schedules that task.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
import json
import logging
from time import perf_counter_ns
from typing import Any

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.util.ulid import ulid_now

from ..agent.settings import LoopSettings
from ..const import (
    CONF_DECIDER_URL,
    CONF_LLM_URL,
    CONF_THINKING_OFF_KWARG,
    DEFAULT_THINKING_OFF_KWARG,
)
from ..decider.settings import DeciderSettings
from ..servers import without_credentials
from . import audio, record, recorder, voice
from .built import AudioClip, VoicePart
from .const import EFFECTS_WINDOW_SECONDS
from .recorder import TurnStart, TurnTrace
from .setup import Monitor
from .voice import RunRef

_LOGGER = logging.getLogger(__name__)

type Process = Callable[[], Awaitable[conversation.ConversationResult]]


async def async_recorded_turn(
    monitor: Monitor,
    user_input: conversation.ConversationInput,
    process: Process,
    *,
    agent_entity: str,
    settings: str,
) -> conversation.ConversationResult:
    """The turn as `process` answers it, recorded; built and queued by a task of the entry after it.

    `settings`: the agent's settings blob (settings_text). Not recorded, only answered: a turn of an AI Suite agent
    another one handed off to (its steps join the outer turn), and a turn past MAX_OPEN_TURNS.
    """
    start = TurnStart(
        id=ulid_now(),
        context_id=user_input.context.id,
        agent_entity=agent_entity,
        language=user_input.language,
        text=user_input.text,
        satellite_id=user_input.satellite_id,
        device_id=user_input.device_id,
    )
    trace: TurnTrace | None = None
    run: RunRef | None = None
    try:
        with recorder.open_turn(monitor.recorder, start) as trace:
            if trace is None:
                return await process()
            if user_input.conversation_id is not None:
                run = voice.find_run(monitor.hass, user_input.conversation_id)
            return await _answered(process)
    finally:
        if trace is not None:  # closed by now: the closing task starts at the turn's end
            monitor.entry.async_create_background_task(
                monitor.hass, _close(monitor, trace, run, settings), f"ai_suite monitor turn {start.id}"
            )


def settings_text(decider: DeciderSettings, loop: LoopSettings | None, servers: Mapping[str, Any]) -> str:
    """An agent's settings as the record's settings blob: sorted JSON, without any key, token or password.

    `servers`: the entry's data, of which the two server URLs (credentials in them removed) and the thinking switch
    are taken. The agent's settings hold no secret: models, prompt, tool APIs, sampling, the decider's bar.
    """
    settings = {
        "decider": asdict(decider),
        "loop": asdict(loop) if loop is not None else None,
        "servers": {
            CONF_DECIDER_URL: without_credentials(servers[CONF_DECIDER_URL]),
            CONF_LLM_URL: without_credentials(servers[CONF_LLM_URL]),
            CONF_THINKING_OFF_KWARG: servers.get(CONF_THINKING_OFF_KWARG, DEFAULT_THINKING_OFF_KWARG),
        },
    }
    return json.dumps(settings, sort_keys=True, indent=2, ensure_ascii=False)


async def _answered(process: Process) -> conversation.ConversationResult:
    """The turn's result, its answer recorded; an exception is recorded as the turn's error and raised on.

    HA speaks the error's text (conversation/agent_manager.py), so that is the turn's output. A cancelled turn (its
    pipeline stopped) is an error too: its record must still name a path.
    """
    try:
        result = await process()
    except BaseException as err:
        recorder.turn_path("error")
        recorder.turn_error("agent", err)
        recorder.turn_output(str(err), False)
        raise
    speech = result.response.speech.get("plain", {}).get("speech", "")
    recorder.turn_output(speech, result.continue_conversation)
    recorder.conversation(result.conversation_id)  # the last source: the chat session's id was given first
    return result


async def _close(monitor: Monitor, trace: TurnTrace, run: RunRef | None, settings: str) -> None:
    """Build the closed turn once its effects, its run and its satellite are done, and queue it; then release it."""
    try:
        voice_part = await _voice_after_effects(monitor.hass, trace, run)
        clips: tuple[AudioClip, ...] = ()
        if monitor.audio and run is not None and voice_part is not None:
            clips = await audio.find_clips(monitor.hass, run, trace)
        monitor.hook_ns.append(trace.hook_ns)
        began = perf_counter_ns()
        try:
            built = record.build(
                trace, voice=voice_part, audio=clips, settings=settings, ai_suite_version=monitor.ai_suite_version
            )
        except Exception as err:  # noqa: BLE001 - a record that cannot be built is only not sent; the turn was answered
            _build_failed(monitor, trace, err)
            return
        monitor.build_ns.append(perf_counter_ns() - began)
        monitor.exporter.submit(built)
    finally:
        recorder.release(monitor.recorder, trace)


async def _voice_after_effects(hass: HomeAssistant, trace: TurnTrace, run: RunRef | None) -> VoicePart | None:
    """The run's voice part, after the turn's effects window (started at its close); None for a turn without a run."""
    window = asyncio.sleep(EFFECTS_WINDOW_SECONDS)
    if run is None:
        await window
        return None
    async with asyncio.TaskGroup() as waits:
        waits.create_task(window)
        voice_part = waits.create_task(voice.wait_voice(hass, run, trace.start.satellite_id))
    return voice_part.result()


def _build_failed(monitor: Monitor, trace: TurnTrace, err: Exception) -> None:
    """Count a record that could not be built; warned about once per loaded entry, without the turn's content."""
    monitor.build_failed += 1
    level = logging.WARNING if monitor.build_failed == 1 else logging.DEBUG
    _LOGGER.log(level, "Turn %s could not be built for the trace monitor (%s); not sent", trace.start.id,
                type(err).__name__)
