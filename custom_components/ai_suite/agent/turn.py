"""A turn the text model answers: the agent's tools and prompt in the turn's chat log, the loop, and the reply check.

As an LLM agent of HA does it: the chat log runs the tools and streams the reply.
"""

import logging

import openai

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session

from ..const import AGENT_PLATFORM
from ..monitor import recorder
from . import reply_check
from .loop import async_answer
from .settings import LoopSettings
from .tools import AgentAPI

_LOGGER = logging.getLogger(__name__)


async def async_answer_turn(
    hass: HomeAssistant,
    client: openai.AsyncOpenAI,
    settings: LoopSettings,
    agent_id: str,
    entity_id: str,
    user_input: conversation.ConversationInput,
    language: str | None = None,
) -> conversation.ConversationResult:
    """The text model's answer to the turn.

    `agent_id`: the agent's subentry, which names its tools' API; `language`: the code of the reply language the decider
    named, the reply check's second source after the user's own words.
    """
    available = {api.id for api in llm.async_get_apis(hass)}
    sources = [api for api in settings.llm_apis if api in available]
    tools = AgentAPI(hass, agent_id, sources, settings.hidden_tools) if sources else None
    with (
        async_get_chat_session(hass, user_input.conversation_id) as session,
        conversation.async_get_chat_log(hass, session, user_input) as chat_log,
    ):
        recorder.conversation(session.conversation_id)
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(AGENT_PLATFORM), tools, settings.prompt, user_input.extra_system_prompt
            )
        except conversation.ConverseError as err:  # HA's answer to it, returned: the turn's wrapper never sees it
            recorder.turn_error("setup", err)
            return err.as_conversation_result()
        recorder.system_parts(chat_log.llm_api.api_prompt if chat_log.llm_api else None, chat_log.extra_system_prompt)
        await async_answer(client, settings, chat_log, entity_id)
        if settings.reply_check:
            await _check_reply(client, settings, chat_log, entity_id, user_input, language)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


async def _check_reply(
    client: openai.AsyncOpenAI,
    settings: LoopSettings,
    chat_log: conversation.ChatLog,
    entity_id: str,
    user_input: conversation.ConversationInput,
    language: str | None,
) -> None:
    """Check the reply before the user hears it.

    The reply before the user hears it: on a fault the model answers once more, with its tools, told what was
    wrong; only in a turn that changed nothing, since a retry after an action repeated it (reply_check.py).
    """
    step = recorder.reply_check_opened()
    turn = reply_check.turn_of(chat_log, user_input.text)
    checked_language = reply_check.language_of(user_input.text) or language
    found = reply_check.faults(turn, checked_language)
    recorder.reply_checked(step, language=checked_language, reply=turn.reply, acted=turn.acted, faults=found)
    if not found or turn.acted:
        return
    _LOGGER.info("%s reply check: %s, answering once more", entity_id, ", ".join(f.check for f in found))
    recorder.retrying()
    chat_log.async_add_user_content(conversation.UserContent(content=reply_check.retry_request(found)))
    await async_answer(client, settings, chat_log, entity_id)
