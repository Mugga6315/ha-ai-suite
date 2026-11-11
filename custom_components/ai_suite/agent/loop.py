"""The agent loop: the text model answers in rounds.

A round's tool calls run in HA's chat log, and the next round reads their results, until a round ends without tool
calls. A turn that ends without a reply that has text is asked once more, without tools; still empty, it fails
with an error HA speaks instead of ending in silence.
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
# Asks again for a reply that held no text, in that request only: in the chat log it would count as the user's next
# question (request.recent) and stay in later turns. Without tools: a round with them could repeat an action.
EMPTY_REPLY_NOTE = "[Your reply was empty, so the user heard nothing. Answer the user now from the results you have.]"


class LLMServerError(HomeAssistantError):
    """The LLM server failed or answered with something the loop cannot use."""


async def async_answer(
    client: openai.AsyncOpenAI, settings: LoopSettings, chat_log: conversation.ChatLog, agent_id: str,
    note: request.TurnNote | None,
) -> None:
    """Add the model's answer to the chat log.

    Each round is built from the chat log as it is then, so it holds the rounds before and anything added between (the
    reply check's retry request). The tools stay the same for the turn, so they are converted once, not per round.
    `note`: what every round of the turn tells the model with the request.
    """
    tools = request.tool_specs(chat_log.llm_api)
    for round_number in range(1, MAX_ROUNDS + 1):
        call = await request.build(settings, chat_log, tools, note)
        if round_number == MAX_ROUNDS:
            _make_last_round(call)
        recorder.round_opened(call)
        await _run_round(client, call, settings, chat_log, agent_id)
        if not chat_log.unresponded_tool_results:
            break
    if _empty_reply(chat_log):
        await _answer_again(client, settings, chat_log, agent_id, tools, note)


async def _answer_again(
    client: openai.AsyncOpenAI, settings: LoopSettings, chat_log: conversation.ChatLog, agent_id: str,
    tools: list[dict[str, Any]], note: request.TurnNote | None,
) -> None:
    """One more round without tools for a reply that held no text; LLMServerError when it is empty again."""
    _LOGGER.info("%s gave an empty reply, answering once more", agent_id)
    recorder.retrying()
    call = await request.build(settings, chat_log, tools, note)
    _ask_again(call)
    recorder.round_opened(call)
    await _run_round(client, call, settings, chat_log, agent_id)
    if _empty_reply(chat_log):
        error = LLMServerError(translation_domain=DOMAIN, translation_key="empty_reply")
        recorder.turn_error("llm", error)
        raise error


def _ask_again(call: dict[str, Any]) -> None:
    """Make the request one without tools that tells the model its reply was empty.

    The note joins the newest message, a user message or tool result: several chat templates refuse two user messages
    in a row. After an earlier reply (a request without text leaves one newest) it is a user message of its own.
    """
    call["tool_choice"] = "none"
    newest = call["messages"][-1]
    if newest["role"] == "assistant":
        call["messages"].append({"role": "user", "content": EMPTY_REPLY_NOTE})
    else:
        call["messages"][-1] = request.with_text(newest, EMPTY_REPLY_NOTE)


def _empty_reply(chat_log: conversation.ChatLog) -> bool:
    """Whether the turn ends without a reply that has text.

    A round that streamed nothing adds no reply (HA then fails the turn with "Unable to get response"); one with
    reasoning only, or with emojis only that the loop strips, adds a reply without text, which HA speaks as silence.
    """
    last = chat_log.content[-1]
    return last.role != "assistant" or (not last.tool_calls and not (last.content or "").strip())


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
