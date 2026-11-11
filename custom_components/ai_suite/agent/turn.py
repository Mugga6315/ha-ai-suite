"""A turn the text model answers: the agent's tools and prompt in the turn's chat log, the loop, and the reply check.

As an LLM agent of HA does it: the chat log runs the tools and streams the reply.
"""

from dataclasses import dataclass
import logging

import openai

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.chat_session import async_get_chat_session

from ..const import AGENT_PLATFORM
from ..languages import LANGUAGE_NAMES
from ..monitor import recorder
from . import reply_check
from .loop import async_answer
from .request import TurnNote
from .schedule import Schedule
from .settings import LoopSettings
from .tools import AgentAPI, forget_failed_calls

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandOffNote:
    """What the decider that handed a turn off tells whoever answers it."""

    text: str  # that a decider handed the request off, and why
    language: str | None = None  # the code of the reply language the decider named

    @property
    def reply_in(self) -> str | None:
        """The sentence that names the reply language; None when the decider named none."""
        return f"Reply in {LANGUAGE_NAMES[self.language]}." if self.language else None


async def async_answer_turn(
    hass: HomeAssistant,
    client: openai.AsyncOpenAI,
    settings: LoopSettings,
    agent_id: str,
    entity_id: str,
    user_input: conversation.ConversationInput,
    schedule: Schedule | None = None,
    hand_off: HandOffNote | None = None,
) -> conversation.ConversationResult:
    """The text model's answer to the turn.

    `agent_id`: the agent's subentry, which names its tools' API; `schedule`: where the agent's commands for later
    go, its tools added to the agent's; `hand_off`: the decider's note and reply language when it handed the turn off.
    """
    available = {api.id for api in llm.async_get_apis(hass)}
    sources = [api for api in settings.llm_apis if api in available]
    tools = AgentAPI(hass, agent_id, sources, settings.hidden_tools, schedule) if sources else None
    with (
        async_get_chat_session(hass, user_input.conversation_id, context=user_input.context) as session,
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
        if chat_log.llm_api:
            forget_failed_calls(chat_log.llm_api)
        recorder.system_parts(chat_log.llm_api.api_prompt if chat_log.llm_api else None, chat_log.extra_system_prompt)
        note = _turn_note(chat_log, hand_off) if hand_off else None
        await async_answer(client, settings, chat_log, entity_id, note)
        if settings.reply_check:
            language = hand_off.language if hand_off else None
            await _check_reply(client, settings, chat_log, entity_id, user_input, language, note)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


def _turn_note(chat_log: conversation.ChatLog, hand_off: HandOffNote) -> TurnNote:
    """The hand-off's note as the model reads it, bound to the turn's request."""
    asked = next(entry for entry in reversed(chat_log.content) if isinstance(entry, conversation.UserContent))
    return TurnNote(asked, " ".join(part for part in (hand_off.text, hand_off.reply_in) if part))


async def _check_reply(
    client: openai.AsyncOpenAI,
    settings: LoopSettings,
    chat_log: conversation.ChatLog,
    entity_id: str,
    user_input: conversation.ConversationInput,
    language: str | None,
    note: TurnNote | None,
) -> None:
    """Check the reply before the user hears it.

    The reply before the user hears it: on a fault the model answers once more, with its tools, told what was
    wrong; only in a turn that changed nothing, since a retry after an action repeated it (reply_check.py).
    """
    step = recorder.reply_check_opened()
    turn = reply_check.turn_of(chat_log, user_input.text, note.text if note else None)
    checked_language = reply_check.language_of(user_input.text) or language
    found = reply_check.faults(turn, checked_language)
    recorder.reply_checked(step, language=checked_language, reply=turn.reply, acted=turn.acted, faults=found)
    if not found or turn.acted:
        return
    _LOGGER.info("%s reply check: %s, answering once more", entity_id, ", ".join(f.check for f in found))
    recorder.retrying()
    chat_log.async_add_user_content(conversation.UserContent(content=reply_check.retry_request(found)))
    await async_answer(client, settings, chat_log, entity_id, note)
