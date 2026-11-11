"""What the text model is sent each round.

Each round holds the chat log as chat-completions messages, the agent's tools and its settings.
"""

import base64
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import probatio

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from . import system_prompt
from .settings import LoopSettings

_LOGGER = logging.getLogger(__name__)
# Tool schemas are plain JSON Schema objects to the servers; these keywords at the top are not accepted there.
_UNSUPPORTED_TOP_KEYS = ("allOf", "anyOf", "oneOf")
# OpenAI's mark on a content part: the server may keep the prompt up to it for other conversations.
_CACHE_BREAKPOINT = {"prompt_cache_breakpoint": {"mode": "explicit"}}


class AttachmentError(HomeAssistantError):
    """The user attached something the model cannot be sent."""


@dataclass(frozen=True, slots=True)
class TurnNote:
    """What the model is told about one turn: added to the request's message in that turn's rounds only."""

    asked: conversation.UserContent  # the chat log's entry of the turn's request
    text: str


async def build(
    settings: LoopSettings, chat_log: conversation.ChatLog, tools: list[dict[str, Any]], note: TurnNote | None = None
) -> dict[str, Any]:
    """The keyword arguments of one chat-completions call, built from the chat log as it is now.

    `tools`: the turn's tools as function definitions (tool_specs); `note`: the turn's note for the model.
    """
    chat = recent(await messages(chat_log.hass, chat_log.content, settings.send_reasoning, note), settings.max_history)
    if settings.cache_breakpoint:
        chat = _system_prompt_marked(chat, chat_log)
    request: dict[str, Any] = {
        "model": settings.model,
        "messages": chat,
        "parallel_tool_calls": settings.parallel_tool_calls,
        # Token counts in the stream's last chunk, for servers and proxies that report them.
        "stream_options": {"include_usage": True},
    }
    if tools:
        request["tools"] = tools
    # Left out when unset: some servers reject any temperature but their default.
    if settings.temperature is not None:
        request["temperature"] = settings.temperature
    if settings.reasoning_effort:
        request["reasoning_effort"] = settings.reasoning_effort
    if server_fields := settings.server_fields():
        request["extra_body"] = server_fields
    return request


async def messages(
    hass: HomeAssistant, contents: list[conversation.Content], send_reasoning: bool, note: TurnNote | None
) -> list[dict[str, Any]]:
    """The chat log's entries as chat messages, in order; an entry the API has no message for is left out.

    `note` is added in brackets to the message of its request.
    """
    chat = []
    for content in contents:
        entry = await message(hass, content, send_reasoning)
        if entry is None:
            continue
        chat.append(with_text(entry, f"[{note.text}]") if note is not None and content is note.asked else entry)
    return chat


def with_text(entry: dict[str, Any], text: str) -> dict[str, Any]:
    """The message with `text` added after its content.

    A system prompt in parts reads as one text, so the added part opens with the blank line a single text gets.
    """
    content = entry["content"]
    if not isinstance(content, list):
        return {**entry, "content": f"{content}\n\n{text}"}
    opening = "\n\n" if entry["role"] == "system" else ""
    return {**entry, "content": [*content, {"type": "text", "text": f"{opening}{text}"}]}


def _system_prompt_marked(chat: list[dict[str, Any]], chat_log: conversation.ChatLog) -> list[dict[str, Any]]:
    """The chat with its system prompt in two parts, the first marked for the server's prompt cache.

    The second part holds what a turn changes (system_prompt.split); the parts joined are the prompt as it was.
    """
    if not chat or chat[0]["role"] != "system":
        return chat
    prompt: str = chat[0]["content"]
    api_prompt = chat_log.llm_api.api_prompt if chat_log.llm_api else None
    stable = system_prompt.split(prompt, api_prompt, chat_log.extra_system_prompt).stable
    if not stable:
        return chat
    parts = [{"type": "text", "text": stable, **_CACHE_BREAKPOINT}]
    if varying := prompt[len(stable) :]:
        parts.append({"type": "text", "text": varying})
    return [{"role": "system", "content": parts}, *chat[1:]]


async def message(
    hass: HomeAssistant, content: conversation.Content, send_reasoning: bool
) -> dict[str, Any] | None:
    """One chat log entry as a chat message.

    `send_reasoning`: an assistant message carries the model's reasoning (`reasoning_content`, which vLLM, llama.cpp and read).
    """
    if isinstance(content, conversation.ToolResultContent):
        return {"role": "tool", "tool_call_id": content.tool_call_id, "content": _result_text(content)}
    if isinstance(content, conversation.AssistantContent):
        # A reply without text or tool calls (reasoning alone) says nothing; kept, it would put two assistant messages
        # in a row, which several chat templates refuse.
        if not (content.content or "").strip() and not content.tool_calls:
            return None
        return _assistant_message(content, send_reasoning)
    if isinstance(content, conversation.UserContent) and content.content:
        return {"role": "user", "content": await _user_content(hass, content)}
    if isinstance(content, conversation.SystemContent) and content.content:
        return {"role": "system", "content": content.content}
    return None


def recent(chat: list[dict[str, Any]], max_history: int | None) -> list[dict[str, Any]]:
    """The system prompt and the last `max_history` exchanges before the current one.

    An exchange starts at a user message and keeps its tool calls and results whole. None (or below 0): all of them.
    """
    if max_history is None or max_history < 0:
        return chat
    starts = [index for index, entry in enumerate(chat) if entry["role"] == "user"]
    if len(starts) <= max_history + 1:
        return chat
    cut = starts[-(max_history + 1)]
    return [entry for entry in chat[:cut] if entry["role"] == "system"] + chat[cut:]


def tool_specs(llm_api: llm.APIInstance | None) -> list[dict[str, Any]]:
    """The tools as function definitions, sorted by name.

    HA lists intent tools in registration order, which changes with every start, and the tools open the prompt: a new
    order would void a server's whole prompt cache.
    """
    if llm_api is None:
        return []
    tools = sorted(llm_api.tools, key=lambda tool: tool.name)
    return [_tool_spec(tool, llm_api.custom_serializer) for tool in tools]


def _tool_spec(tool: llm.Tool, serializer) -> dict[str, Any]:
    parameters = probatio.to_openapi(tool.parameters, custom_serializer=serializer)
    for key in _UNSUPPORTED_TOP_KEYS:
        parameters.pop(key, None)
    function: dict[str, Any] = {"name": tool.name, "parameters": parameters}
    if tool.description and tool.description.strip():
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _assistant_message(content: conversation.AssistantContent, send_reasoning: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": content.content}
    if send_reasoning and content.thinking_content:
        entry["reasoning_content"] = content.thinking_content
    if content.tool_calls:
        entry["tool_calls"] = [
            {"type": "function", "id": call.id,
             "function": {"name": call.tool_name, "arguments": json.dumps(call.tool_args)}}
            for call in content.tool_calls
        ]
    return entry


async def _user_content(hass: HomeAssistant, content: conversation.UserContent) -> str | list[dict[str, Any]]:
    """The text alone, or with images the text first: the API's two shapes of a user message."""
    if not content.attachments:
        return content.content
    parts: list[dict[str, Any]] = [{"type": "text", "text": content.content}]
    for attachment in content.attachments:
        if not attachment.mime_type.startswith("image/"):
            raise AttachmentError(f"Only images can be sent to the model, not {attachment.mime_type}")
        encoded = await hass.async_add_executor_job(_read_base64, attachment.path)
        parts.append({"type": "image_url", "image_url": {"url": f"data:{attachment.mime_type};base64,{encoded}"}})
    return parts


def _read_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _result_text(content: conversation.ToolResultContent) -> str:
    """The tool's result as JSON; a value JSON has no type for becomes its text."""

    def as_text(value: Any) -> str:
        _LOGGER.warning("%s returned a value that is no JSON, sent as text: %r", content.tool_name, value)
        return str(value)

    return json.dumps(content.result.data, default=as_text)
