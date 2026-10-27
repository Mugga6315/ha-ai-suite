"""The agent loop: the text model answers in rounds.

A round's tool calls run in HA's chat log, and the next round reads their results, until a round ends without tool
calls.
"""

import logging
from typing import Any

import openai

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError

from ..const import DOMAIN
from ..monitor import recorder
from . import request
from .reply import deltas
from .settings import LoopSettings

_LOGGER = logging.getLogger(__name__)

# Rounds per turn. The last one may not call tools, so a model that keeps calling them still answers.
MAX_ROUNDS = 10
# Added to the newest tool result in the last round. Not a system message of its own: chat templates of several
# models accept a system message only at the start.
LAST_ROUND_NOTE = "\n\n[No tool calls are left in this turn. Answer the user now from the results you have.]"


class LLMServerError(HomeAssistantError):
    """The LLM server failed or answered with something the loop cannot use."""


async def async_answer(
    client: openai.AsyncOpenAI, settings: LoopSettings, chat_log: conversation.ChatLog, agent_id: str
) -> None:
    """Add the model's answer to the chat log.

    Each round is built from the chat log as it is then, so it holds the rounds before and anything added between (the
    reply check's retry request). The tools stay the same for the turn, so they are converted once, not per round.
    """
    tools = request.tool_specs(chat_log.llm_api)
    for round_number in range(1, MAX_ROUNDS + 1):
        call = await request.build(settings, chat_log, tools)
        if round_number == MAX_ROUNDS:
            _make_last_round(call)
        recorder.round_opened(call)
        await _run_round(client, call, settings, chat_log, agent_id)
        if not chat_log.unresponded_tool_results:
            return


async def _run_round(
    client: openai.AsyncOpenAI, call: dict[str, Any], settings: LoopSettings, chat_log: conversation.ChatLog,
    agent_id: str,
) -> None:
    try:
        stream = await client.chat.completions.create(**call, stream=True)
        async for content in chat_log.async_add_delta_content_stream(agent_id, deltas(stream, settings.strip_emojis)):
            recorder.round_content(content)
    except openai.OpenAIError as err:
        _LOGGER.exception("The LLM server returned an error")
        recorder.round_failed(err)
        server_error = LLMServerError(translation_domain=DOMAIN, translation_key="llm_server_error")
        recorder.turn_error("llm", server_error)
        raise server_error from err


def _make_last_round(call: dict[str, Any]) -> None:
    call["tool_choice"] = "none"
    newest = call["messages"][-1]
    if newest["role"] == "tool":
        newest["content"] += LAST_ROUND_NOTE
