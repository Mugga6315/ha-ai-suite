"""The text model's streamed answer as HA chat-log deltas: the reply, the reasoning beside it, and the tool calls.

HA's chat log speaks the reply while it arrives, so reply text goes out chunk by chunk; tool calls go out once the
stream is complete, when their arguments are whole.
"""

from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
import json

import demoji
from openai.types.chat import ChatCompletionChunk

from homeassistant.components import conversation
from homeassistant.helpers import llm
from homeassistant.util.ulid import ulid_now

from ..monitor import recorder

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


async def deltas(
    stream: AsyncIterable[ChatCompletionChunk], strip_emojis: bool
) -> AsyncGenerator[conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict]:
    """The stream's chunks as chat-log deltas.

    One assistant message from the stream's chunks, then an error result for each tool call whose arguments
    are no JSON (the model reads it in the next round).
    """
    text, calls = _ReplyText(), _ToolCalls()
    started = False
    async for chunk in stream:
        if chunk.usage is not None:
            recorder.stream_usage(chunk.usage)
        if not chunk.choices:  # the usage chunk
            continue
        if finish := chunk.choices[0].finish_reason:
            recorder.stream_finish(finish)
        delta = chunk.choices[0].delta
        out: conversation.AssistantContentDeltaDict = {}
        if not started:
            out["role"] = "assistant"
            started = True
        # vLLM names the field `reasoning`, llama.cpp `reasoning_content`.
        thinking = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
        if thinking or delta.content or delta.tool_calls:
            recorder.stream_token()
        reply = ""
        if delta.content:
            content = delta.content
            if strip_emojis:
                # demoji loads its table on import; replace is a regex, nothing to move off the event loop.
                content = demoji.replace(content, "")
            tagged, reply = text.add(content)
            thinking += tagged
        _set(out, thinking, reply)
        for piece in delta.tool_calls or ():
            calls.add(piece)
        if out:
            yield out
    recorder.stream_end()
    if not started:
        return
    last: conversation.AssistantContentDeltaDict = _set({}, *text.flush())
    if tool_calls := calls.inputs():
        last["tool_calls"] = tool_calls
    if last:
        yield last
    for call in tool_calls:
        if call.external:
            yield _argument_error(call)


def _set(
    out: conversation.AssistantContentDeltaDict, thinking: str, reply: str
) -> conversation.AssistantContentDeltaDict:
    if thinking:
        out["thinking_content"] = thinking
    if reply:
        out["content"] = reply
    return out


class _ReplyText:
    """Splits the model's text into reasoning and the reply.

    Splits the text the model writes into reasoning (between think tags) and the reply, and drops the reply's
    leading blank space: a voice reads what it gets, and models often open with line breaks.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._reply_started = False
        self._held = ""  # a tag that may be cut by the chunk border

    def add(self, text: str) -> tuple[str, str]:
        """The reasoning and the reply in this piece of text."""
        text, self._held = self._held + text, ""
        thinking: list[str] = []
        reply: list[str] = []
        while text:
            tag = THINK_CLOSE if self._in_think else THINK_OPEN
            before, found, text = text.partition(tag)
            if not found:
                cut = len(before) - _partial_tag_length(before, tag)
                before, self._held = before[:cut], before[cut:]
            (thinking if self._in_think else reply).append(before)
            if found:
                self._in_think = not self._in_think
        return "".join(thinking), self._visible("".join(reply))

    def flush(self) -> tuple[str, str]:
        """What was held back at the end of the stream: it was no tag."""
        held, self._held = self._held, ""
        return (held, "") if self._in_think else ("", self._visible(held))

    def _visible(self, reply: str) -> str:
        if not self._reply_started:
            reply = reply.lstrip()
            self._reply_started = bool(reply)
        return reply


def _partial_tag_length(text: str, tag: str) -> int:
    """How many characters at the end of text begin the tag ("<thi" of "<think>")."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


class _ToolCalls:
    """Tool calls streamed in pieces.

    Servers mark a call's pieces differently: llama.cpp sends the id with the first piece only, Ollama gives parallel
    calls the same index, some OpenRouter engines repeat one id across parallel calls. So a piece with a new id or a new
    tool name starts a call, and any other piece continues the last one.
    """

    def __init__(self) -> None:
        self._calls: list[_Call] = []

    def add(self, piece) -> None:
        function = piece.function
        name = function.name if function else None
        last = self._calls[-1] if self._calls else None
        if last is None or (piece.id and piece.id != last.id) or (name and name != last.name):
            last = _Call(piece.id, name)
            self._calls.append(last)
        last.arguments += (function.arguments if function else None) or ""

    def inputs(self) -> list[llm.ToolInput]:
        """The calls for HA. A call whose arguments are no JSON is marked external, so HA does not run it."""
        used: set[str] = set()
        inputs = []
        for call in self._calls:
            # HA tracks a call by its id, so a repeated or missing one is replaced.
            call_id = call.id if call.id and call.id not in used else ulid_now()
            used.add(call_id)
            raw = call.arguments or "{}"
            try:
                inputs.append(llm.ToolInput(id=call_id, tool_name=call.name or "", tool_args=json.loads(raw)))
            except json.JSONDecodeError:
                inputs.append(llm.ToolInput(id=call_id, tool_name=call.name or "", tool_args={"invalid_json": raw},
                                            external=True))
        return inputs


@dataclass
class _Call:
    id: str | None
    name: str | None
    arguments: str = ""


def _argument_error(call: llm.ToolInput) -> conversation.ToolResultContentDeltaDict:
    arguments = call.tool_args["invalid_json"]
    message = f"The call was not run: its arguments are no valid JSON ({arguments}). Call the tool again with valid JSON."
    return {
        "role": "tool_result", "tool_call_id": call.id, "tool_name": call.tool_name,
        "result": llm.ToolResult(data={"error": message}, error=True),
    }
