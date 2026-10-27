"""The turn record built from a recorded turn, after the turn and off its path.

Of Home Assistant only its ULID helper is imported: the steps hold HA's objects as the hooks got them (states, intent responses, chat-log
contents), read here by their attributes. The record is JSON-ready for HA's json_bytes, which the exporter uses.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any

from homeassistant.util.ulid import ulid_at_time

from .built import AudioClip, Blob, BuiltTurn, VoicePart
from .const import SCHEMA_VERSION
from .recorder import (
    ActionStep,
    AnswerStep,
    DecideStep,
    Effect,
    ReplyCheckStep,
    RoundStep,
    Step,
    ToolStep,
    TurnError,
    TurnTrace,
)

# A round's call arguments that are fields of their own, not params: model, tools_ref, new_messages.
_NOT_PARAMS = frozenset({"model", "messages", "tools"})
# The answer model's request body: model and messages are fields of their own.
_NOT_ANSWER_PARAMS = frozenset({"model", "messages"})
# HA's date and time line as rendered (helpers/llm.py DATE_TIME_PROMPT), with the "\n" HA joins the prompt's parts
# with; found at the prompt's end when no API prompt locates it.
_TIME_LINE = re.compile(r"\nCurrent time is \d{2}:\d{2}:\d{2}\. Today's date is \d{4}-\d{2}-\d{2}\.\n\Z")


def build(
    trace: TurnTrace,
    *,
    voice: VoicePart | None,
    audio: Sequence[AudioClip],
    settings: str,
    ai_suite_version: str,
) -> BuiltTurn:
    """The turn as a turn record with the blobs it references.

    `voice`: the pipeline run behind the turn; its run-start is then the turn's t0, else the agent's start.
    `settings`: the agent's settings as sorted JSON text (blob kind settings).
    """
    clock = _Clock(voice.run_start if voice else trace.opened_wall, trace.opened_wall - trace.opened)
    blobs: dict[str, Blob] = {}
    settings_ref = _blob(blobs, "settings", settings)
    steps = _steps(trace, clock, blobs)
    turn: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": trace.start.id,
        "started_at": _iso(clock.t0),
        "ai_suite_version": ai_suite_version,
        "settings_ref": settings_ref,
        "conversation_id": trace.conversation_id or trace.start.id,
        "context_id": trace.start.context_id,
        "agent_entity": trace.start.agent_entity,
        **_given(satellite_id=trace.start.satellite_id, device_id=trace.start.device_id),
        "language": trace.start.language,
        "input": {"text": trace.start.text, "source": "voice" if voice and "stt" in voice.block else "text"},
        "path": trace.path,
        **_given(handoff_reason=trace.handoff_reason),
        "output": {"speech": trace.speech, "continue_conversation": trace.continue_conversation},
        "agent": {"start": clock.ms(trace.opened), "end": clock.ms(trace.closed)},
        **({"voice": _voice(voice, audio)} if voice else {}),
        "steps": steps,
        "effects": [_effect(effect, clock) for effect in trace.effects],
        "errors": [_turn_error(error) for error in trace.errors],
        "blobs_used": list(blobs),
    }
    return BuiltTurn(trace.start.id, turn, blobs, tuple(audio))


@dataclass(frozen=True, slots=True)
class _Clock:
    """The turn's times as integer ms from t0."""

    t0: float  # wall clock, epoch seconds
    offset: float  # wall clock minus monotonic clock, taken at the turn's open

    def wall(self, mono: float) -> float:
        return mono + self.offset

    def ms(self, mono: float | None) -> int | None:
        return None if mono is None else self.wall_ms(self.wall(mono))

    def wall_ms(self, wall: float) -> int:
        return round((wall - self.t0) * 1000)


def _steps(trace: TurnTrace, clock: _Clock, blobs: dict[str, Blob]) -> list[dict[str, Any]]:
    built = []
    earlier: list[dict[str, Any]] | None = None  # the messages of the round before
    for step in trace.steps:
        match step:
            case RoundStep():
                fields = _round(step, trace, blobs, earlier, ulid_at_time(clock.wall(step.start)))
                earlier = step.call.get("messages", [])
            case AnswerStep():
                fields = _answer(step, blobs, ulid_at_time(clock.wall(step.start)))
            case DecideStep():
                fields = _decide(step)
            case ToolStep():
                fields = _tool(step)
            case ActionStep():
                fields = _action(step)
            case ReplyCheckStep():
                fields = _reply_check(step)
        built.append(_step(step, clock, trace.closed, fields))
    return built


def _step(step: Step, clock: _Clock, closed: float | None, fields: dict[str, Any]) -> dict[str, Any]:
    """The fields every step has around its own; a step that never ended (an exception no hook saw) ends at close."""
    parent = {"parent": step.parent} if isinstance(step, ToolStep) and step.parent is not None else {}
    end = step.end if step.end is not None else closed
    error = {"error": _error(step.error)} if step.error is not None else {}
    return {"id": step.id, **parent, "kind": step.kind, "start": clock.ms(step.start), "end": clock.ms(end),
            **fields, **error}


def _decide(step: DecideStep) -> dict[str, Any]:
    answers = step.answers or {}
    questions = [
        _question(qid, question, answers.get(qid) or {}, step.bars.get(qid)) for qid, question in step.questions.items()
    ]
    return {"url": step.url, "state": step.state, "questions": questions}


def _question(qid: str, question: dict, answer: dict, bar: float | None) -> dict[str, Any]:
    """One question with its answer; a question the server did not answer has none."""
    if question.get("type") == "noul":
        return {"qid": qid, "noul": answer.get("noul"), "bar": bar}
    probs = answer.get("probabilities") or {}
    choice = answer.get("choice")
    return {"qid": qid, "choice": choice, "p": probs.get(choice), "probs": probs, "bar": bar}


def _answer(step: AnswerStep, blobs: dict[str, Blob], call_id: str) -> dict[str, Any]:
    """The answer model's request (the body as sent) and its answer (the parsed JSON)."""
    request = step.request or {}
    message = _dig(step.response, "choices", 0, "message")
    usage = _dig(step.response, "usage")
    return {
        "role": step.role,
        "model": step.model,
        "call_id": call_id,
        "params": {key: value for key, value in request.items() if key not in _NOT_ANSWER_PARAMS},
        "ttft": None,
        "finish": _dig(step.response, "choices", 0, "finish_reason"),
        "system_ref": _blob(blobs, "system", step.system),
        "system_vars": {},
        "tools_ref": None,
        "new_messages": [{"role": "user", "content": step.user}],
        # vLLM names the field `reasoning`, llama.cpp `reasoning_content`.
        "reasoning": _dig(message, "reasoning_content") or _dig(message, "reasoning") or "",
        "text": _dig(message, "content") or "",
        "tool_calls": [],
        **({"usage": _usage(usage)} if isinstance(usage, dict) else {}),
    }


def _round(
    step: RoundStep, trace: TurnTrace, blobs: dict[str, Blob], earlier: list[dict[str, Any]] | None, call_id: str
) -> dict[str, Any]:
    call = step.call
    messages = call.get("messages", [])
    system_ref, system_vars = None, {"time_line": None, "extra_prompt": None}
    if messages and messages[0].get("role") == "system":
        stable, system_vars = _system_split(messages[0]["content"], trace.api_prompt, trace.extra_prompt)
        system_ref = _blob(blobs, "system", stable)
    tools = call.get("tools")
    reasoning, text, tool_calls = _said(step.contents)
    return {
        "role": step.role,
        "model": call.get("model"),
        "call_id": call_id,
        # The loop passes `stream` beside the call's arguments.
        "params": {"stream": True, **{key: value for key, value in call.items() if key not in _NOT_PARAMS}},
        "ttft": None if step.first_token is None else round((step.first_token - step.start) * 1000),
        "finish": step.finish,
        "system_ref": system_ref,
        "system_vars": system_vars,
        "tools_ref": _blob(blobs, "tools", _tools_text(tools)) if tools else None,
        "new_messages": _new_messages(messages, earlier),
        "reasoning": reasoning,
        "text": text,
        "tool_calls": tool_calls,
        # The stream's usage is an openai CompletionUsage.
        **({"usage": _usage(step.usage.model_dump())} if step.usage is not None else {}),
    }


def _system_split(prompt: str, api_prompt: str | None, extra_prompt: str | None) -> tuple[str, dict[str, Any]]:
    r"""The loop prompt's stable part, and its varying parts (system_vars).

    HA joins `user_prompt \n api_prompt [\n time_line] [\n extra_prompt]` (conversation/chat_log.py); the time line
    is left out when a GetDateTime tool exists. A part not found stays in the stable part: the blob then repeats less
    often, the record is still whole.
    """
    extra = None
    if extra_prompt and prompt.endswith("\n" + extra_prompt):
        prompt, extra = prompt[: -len(extra_prompt) - 1], extra_prompt
    stable, time_line = _time_split(prompt, api_prompt)
    return stable, {"time_line": time_line, "extra_prompt": extra}


def _time_split(prompt: str, api_prompt: str | None) -> tuple[str, str | None]:
    if api_prompt:
        if prompt.endswith(api_prompt):
            return prompt, None
        if (at := prompt.rfind(api_prompt + "\n")) >= 0:
            end = at + len(api_prompt)
            return prompt[:end], prompt[end + 1 :]
    if found := _TIME_LINE.search(prompt):
        return prompt[: found.start()], found.group()[1:]
    return prompt, None


def _new_messages(messages: list[dict[str, Any]], earlier: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """What a round sends that the round before did not: the first round from the turn's user message on.

    Between rounds the history may be cut (agent/request.py recent()): then the messages kept reappear after the
    system prompt, and are the earlier round's last ones.
    """
    if earlier is None:
        users = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        return messages[users[-1] if users else 0 :]
    shared = 0
    while shared < min(len(messages), len(earlier)) and messages[shared] == earlier[shared]:
        shared += 1
    rest = messages[shared:]
    if shared == len(earlier):
        return rest
    for kept in range(min(len(rest), len(earlier) - shared), 0, -1):
        if rest[:kept] == earlier[-kept:]:
            return rest[kept:]
    return rest


def _said(contents: list[Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """The reasoning, reply and tool calls of the round's assistant contents (HA AssistantContent)."""
    reasoning, text, calls = [], [], []
    for content in contents:
        if getattr(content, "role", None) != "assistant":
            continue
        reasoning.append(content.thinking_content or "")
        text.append(content.content or "")
        calls += [{"id": call.id, "name": call.tool_name, "args": call.tool_args} for call in content.tool_calls or ()]
    return "".join(reasoning), "".join(text), calls


def _usage(usage: dict[str, Any]) -> dict[str, Any]:
    """The token counts of an OpenAI usage object; reasoning and cached only where the server gives them."""
    counts = {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens")}
    reasoning = _dig(usage, "completion_tokens_details", "reasoning_tokens")
    cached = _dig(usage, "prompt_tokens_details", "cached_tokens")
    return counts | _given(reasoning=reasoning, cached=cached)


def _tool(step: ToolStep) -> dict[str, Any]:
    result = getattr(step.result, "data", step.result)  # an llm.ToolResult holds the tool's JSON in `data`
    return {
        "call_id": step.call_id,
        "name": step.name,
        "args": step.args,
        "result": result,
        "success_targets": _targets(result, "success"),
        "failed_targets": _targets(result, "failed"),
    }


def _targets(result: Any, outcome: str) -> list[str]:
    """The entity ids an intent response lists as `success` or `failed`; none in any other result."""
    found = _dig(result, "data", outcome)
    if not isinstance(found, list):
        return []
    return [target["id"] for target in found if isinstance(target, dict) and "id" in target]


def _action(step: ActionStep) -> dict[str, Any]:
    what = {"intent": step.intent} if step.intent is not None else {"service": f"{step.domain}.{step.service}"}
    result = step.result.as_dict() if hasattr(step.result, "as_dict") else step.result  # an intent.IntentResponse
    return {**what, "target": step.target, "data": step.data, "result": result}


def _reply_check(step: ReplyCheckStep) -> dict[str, Any]:
    return {
        "language": step.language,
        "reply": step.reply,
        "acted": step.acted,
        "faults": [{"check": fault.check, "text": fault.text} for fault in step.faults],
        "retried": step.retried,
    }


def _effect(effect: Effect, clock: _Clock) -> dict[str, Any]:
    """A state change: the attributes new or changed with their new values, the removed ones as null."""
    old, new = effect.old_state, effect.new_state
    before = dict(old.attributes) if old is not None else {}
    after = dict(new.attributes) if new is not None else {}
    changed = {key: value for key, value in after.items() if key not in before or before[key] != value}
    changed |= {key: None for key in before if key not in after}
    return {
        "t": clock.wall_ms(effect.time_fired),
        "entity_id": effect.entity_id,
        "from": old.state if old is not None else None,
        "to": new.state if new is not None else None,
        "changed_attrs": changed,
        "via": effect.via,
    }


def _voice(voice: VoicePart, audio: Sequence[AudioClip]) -> dict[str, Any]:
    clips = [{"kind": clip.kind, "file": clip.file, "bytes": clip.bytes, "duration_ms": clip.duration_ms}
             for clip in audio]
    return voice.block | ({"audio": clips} if clips else {})


def _turn_error(error: TurnError) -> dict[str, Any]:
    return {"stage": error.stage, **_error(error.error)}


def _error(err: BaseException) -> dict[str, Any]:
    return {"type": type(err).__name__, "message": str(err)}


def _blob(blobs: dict[str, Blob], kind: str, text: str) -> str:
    """The text's hash (lowercase hex SHA-256 of its UTF-8), the text kept as a blob of the record."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    blobs.setdefault(digest, Blob(kind, text))
    return digest


def _tools_text(tools: list[dict[str, Any]]) -> str:
    """The tool list as stable text: the same tools give the same hash however their keys were ordered."""
    return json.dumps(tools, sort_keys=True, ensure_ascii=False, indent=2)


def _dig(value: Any, *path: str | int) -> Any:
    """The value at `path` in nested JSON; None where the path does not exist."""
    for key in path:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value


def _given(**fields: Any) -> dict[str, Any]:
    """The optional fields that have a value."""
    return {key: value for key, value in fields.items() if value is not None}


def _iso(wall: float) -> str:
    """UTC with ms and `Z`, as the monitor stores it."""
    return datetime.fromtimestamp(wall, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
