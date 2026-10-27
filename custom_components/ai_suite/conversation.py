"""Conversation agent: the decision model first, the text model or a fallback agent for the rest.

Conversation agent: a decision model answers the form (form.decider_form), code acts (home_actions.py); the agent
loop or a configured fallback agent answers the rest.

The decided and handed-off lines this module logs (INFO, "<entity> decided ..." / "<entity> handoff (...)") are read
by log tools, so their shape stays.
"""

import asyncio
from dataclasses import dataclass, replace
from functools import partial
import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import AiSuiteConfigEntry
from .agent.settings import LoopSettings
from .agent.turn import async_answer_turn
from .brain.llm_api import BrainAPI
from .const import CONF_DECIDER_URL, DOMAIN, SUBENTRY_CONVERSATION
from .decider import command as decided, form
from .decider.api import (
    HAND_OFF_NOTE,
    DecisionClient,
    DecisionError,
    LLMClient,
    LLMError,
)
from .decider.decision import Candidate, previous_turn, state_text
from .decider.household import PENDING, Household, follow
from .decider.settings import DeciderSettings
from .decider.turn import HandOff, Timings
from .entity import subentry_device
from .home import area_aliases, exposed_candidates, satellite_area
from .home_actions import HomeActions
from .languages import LANGUAGE_NAMES
from .llm_client import LLMServer
from .monitor import recorder
from .monitor.turn import async_recorded_turn, settings_text

_LOGGER = logging.getLogger(__name__)
# Conversations whose last exchange is kept for follow-ups that refer to it; oldest dropped first.
MAX_REMEMBERED_CONVERSATIONS = 100
# Each turn is processed on its own; HA needs no limit on them.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant, entry: AiSuiteConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """One conversation agent per conversation subentry."""
    for subentry in entry.get_subentries_of_type(SUBENTRY_CONVERSATION):
        async_add_entities([ConversationAgentEntity(entry, subentry)], config_subentry_id=subentry.subentry_id)


@dataclass(slots=True)
class _Turn:
    """One request on the decision path, with what it reads once: the household's rules and the home."""

    user_input: conversation.ConversationInput
    timings: Timings
    language: asyncio.Task[str | None]  # the reply language's code, asked beside the form's first questions
    household: Household
    entities: list[Candidate]  # the exposed devices
    devices: dict[str, Candidate]  # the ones the form may name
    areas: dict[str, list[str]]


class ConversationAgentEntity(conversation.ConversationEntity):
    """An AI Suite agent: the decision model first (when on), then the text model or the fallback agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: AiSuiteConfigEntry, subentry: ConfigSubentry) -> None:
        """Read the agent's settings from its subentry."""
        self._entry = entry
        self._decider = DeciderSettings.from_subentry(subentry.data)
        self._text_model: str | None = subentry.data.get(CONF_MODEL)
        server = LLMServer.from_entry_data(entry.data)
        self._loop = (
            LoopSettings.from_subentry(subentry.data, thinking_switch=server.thinking_switch) if self._text_model else None
        )
        # The loop streams its reply into the chat log; HA's pipeline starts TTS from it after 60 characters, before
        # a long reply has ended.
        self._attr_supports_streaming = self._loop is not None
        self._attr_unique_id = subentry.subentry_id
        self._agent_id = subentry.subentry_id
        self._attr_device_info = subentry_device(subentry)
        self._previous: dict[str, str] = {}  # conversation_id -> previous_turn(...)
        self._monitor_settings = settings_text(self._decider, self._loop, entry.data)  # the trace monitor's blob

    @property
    def supported_languages(self) -> Literal["*"]:
        """Any language: the decider hands off what its word lists do not cover."""
        return "*"

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Decide the request, or hand it off; recorded for the trace monitor when its export is on."""
        if (monitor := self._entry.runtime_data.monitor) is None:
            return await self._async_process(user_input)
        return await async_recorded_turn(
            monitor, user_input, partial(self._async_process, user_input),
            agent_entity=self.entity_id, settings=self._monitor_settings,
        )

    async def _async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        if not self._decider.enabled:
            recorder.turn_path("loop")
            return await self._answer_with_loop(user_input)
        timings = Timings()
        household = await self._household_rules()
        # The language question runs beside the form's first questions: nothing before the results needs it.
        language = asyncio.create_task(self._reply_language(user_input))
        try:
            response = await self._decide(user_input, timings, language, household)
        except HandOff as reason:
            code = await language
            timings.mark("handoff")
            result = await self._hand_off(user_input, code, str(reason))
            recorder.turn_path("handoff", str(reason))  # after it: an AI Suite agent it went to set a path too
            timings.mark("total")
            _LOGGER.info("%s handoff (%s) %s: %r", self.entity_id, reason, timings, user_input.text)
        else:
            code = await language
            timings.mark("decided")
            await self._word_answer(user_input.text, response, code, household)
            timings.mark("answer")
            result = self._record(user_input, response)
            recorder.turn_path("decided")
            timings.mark("total")
            _LOGGER.info("%s decided %s: %r", self.entity_id, timings, user_input.text)
        finally:
            language.cancel()  # still running only when the turn failed otherwise
        self._remember(result, user_input.text)
        return result

    async def _reply_language(self, user_input: conversation.ConversationInput) -> str | None:
        """The code of the language the reply should be in.

        The decider's, the pipeline's when the decider is unsure; None when unknown.
        """
        question = form.LANGUAGE_CHECK
        try:
            code, p = await self._decision_client().choice(
                self._state(user_input), question["instructions"], question["criteria"]
            )
        except DecisionError:
            return None
        return form.reply_language(code, p, self._decider.threshold, user_input.language)

    async def _word_answer(
        self, user_text: str, response: intent.IntentResponse, language: str | None, household: Household
    ) -> None:
        """Replace the plain result sentence with the answer LLM's wording.

        On failure the result sentence stays: the action already ran, so a hand-off here would run it twice.
        """
        result = response.speech["plain"]["speech"]
        try:
            response.async_set_speech(
                await self._llm().say(user_text, result, LANGUAGE_NAMES.get(language or ""), household.answer_rules)
            )
        except LLMError as err:
            _LOGGER.warning("%s answer LLM failed, speaking the result sentence: %s", self.entity_id, err)

    def _remember(self, result: conversation.ConversationResult, text: str) -> None:
        if not result.conversation_id:
            return
        self._previous.pop(result.conversation_id, None)  # re-insert = newest
        self._previous[result.conversation_id] = previous_turn(text, result.response.speech["plain"]["speech"])
        while len(self._previous) > MAX_REMEMBERED_CONVERSATIONS:
            del self._previous[next(iter(self._previous))]

    def _record(
        self, user_input: conversation.ConversationInput, response: intent.IntentResponse
    ) -> conversation.ConversationResult:
        """Write the decided turn into HA's shared chat log, so a later hand-off sees it."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            recorder.conversation(session.conversation_id)
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=user_input.agent_id, content=response.speech["plain"]["speech"])
            )
            return conversation.ConversationResult(response=response, conversation_id=session.conversation_id)

    async def _decide(
        self,
        user_input: conversation.ConversationInput,
        timings: Timings,
        language: asyncio.Task[str | None],
        household: Household,
    ) -> intent.IntentResponse:
        """The decision model answers the form; code checks it and HA runs it, with results in the reply language.

        A household rule that covers the request by its words hands it to the text model, which has the rule, unless the
        rule's effect is in the decider's terms: then code follows it (household.follow).
        """
        if household.pending:
            raise HandOff(f"a household rule is not compiled yet: {household.pending[0].text}")
        rule = household.triggered(user_input.text)
        if rule is not None and rule.effect is None:
            raise HandOff(f"a household rule covers it: {rule.text}")
        entities = exposed_candidates(self.hass, household)
        turn = _Turn(user_input, timings, language, household, entities, form.devices_for_form(entities),
                     area_aliases(self.hass, household))
        filled = await self._form(turn, user_input)
        if filled.several_requests:
            return await self._run_parts(turn)
        if rule is not None:
            filled = follow(rule, user_input.text, await language, filled)
        return await self._run_form(turn, filled, _in_language(user_input, await language))

    async def _form(self, turn: _Turn, user_input: conversation.ConversationInput) -> form.Form:
        return await form.decider_form(
            self._decision_client(), self._state(user_input), user_input.text, turn.devices, turn.areas,
            self._decider.threshold, turn.language, turn.household.text or None,
        )

    async def _run_parts(self, turn: _Turn) -> intent.IntentResponse:
        """Decide a message with several requests part by part.

        A message the decider says holds several requests: the answer model splits it into complete single
        requests (api.SPLIT_SYSTEM_PROMPT), each is decided alone with every check of a single request, and they run
        one after the other only when all are decided and none overlap; anything else hands off the whole message.
        """
        user_input = turn.user_input
        try:
            parts = await self._llm().split(user_input.text)
        except LLMError as err:
            raise HandOff(f"several requests: {err}") from err
        turn.timings.mark(f"split:{len(parts)}")
        if len(parts) < 2:
            raise HandOff("several requests: the split found one")
        here = satellite_area(self.hass, user_input.device_id)
        previous = self._previous.get(user_input.conversation_id or "")
        language = await turn.language
        decided_parts = []
        followed_a_rule = False
        for part in parts:
            part_input = replace(user_input, text=part)
            filled = await self._form(turn, part_input)
            if filled.several_requests:
                raise HandOff(f"several requests: {part!r} is several again")
            if (rule := turn.household.triggered(part)) is not None:
                filled, followed_a_rule = follow(rule, part, language, filled), True
            filled = self._with_household_rules(turn, filled, part, language)  # before any part acts
            decided_parts.append((part_input, filled, decided.command(filled, turn.devices, part, here, previous)))
        if not followed_a_rule and (rule := turn.household.triggered(user_input.text)) is not None:
            raise HandOff(f"a household rule covers the message, none of its parts: {rule.text}")
        if decided.parts_overlap([command for _, _, command in decided_parts]):
            raise HandOff("several requests: two parts act on the same device")
        notes = []
        for part_input, filled, _ in decided_parts:
            response = await self._run_form(turn, filled, _in_language(part_input, language))
            notes.append(response.speech["plain"]["speech"])
        combined = intent.IntentResponse(language=user_input.language)
        combined.async_set_speech("\n".join(notes))
        return combined

    async def _run_form(
        self, turn: _Turn, filled: form.Form, user_input: conversation.ConversationInput
    ) -> intent.IntentResponse:
        """The checked command of one request, run (`user_input`: the request in the reply language)."""
        turn.timings.mark(f"form:{filled.action}")
        language = user_input.language if user_input.language in LANGUAGE_NAMES else None
        filled = self._with_household_rules(turn, filled, user_input.text, language)
        command = decided.command(
            filled, turn.devices, user_input.text, satellite_area(self.hass, user_input.device_id),
            self._previous.get(user_input.conversation_id or ""),
        )
        ask = partial(self._ask_sure, self._state(user_input))
        return await HomeActions(self.hass, user_input, turn.timings, turn.entities, ask).async_run(command)

    def _with_household_rules(self, turn: _Turn, filled: form.Form, text: str, language: str | None) -> form.Form:
        """The form under the household rule about this kind of request.

        The rule applies whatever the request's words are (a limit, a time, a default room). It is followed by code
        where its effect is in the decider's terms, else HandOff to the text model, which has the rule
        (household.follow).
        """
        device = turn.devices.get(filled.device)
        kind = Platform.LIGHT if filled.room_wide else device.domain if device else None
        if rule := turn.household.in_scope(filled.action, kind, filled.room != form.NO_ROOM):
            return follow(rule, text, language, filled)
        return filled

    async def _ask_sure(self, state: str, question: dict) -> str:
        return await form.ask_sure(self._decision_client(), state, question, self._decider.threshold)

    async def _hand_off(
        self, user_input: conversation.ConversationInput, language: str | None, reason: str
    ) -> conversation.ConversationResult:
        """Hand the request to the fallback agent, or to the agent loop without one.

        The fallback agent answers; told that a decider passed the request on and why, and the reply language
        when the decider named it (HA's extra system prompt, added to one the request already has).
        """
        reply_in = f"Reply in {LANGUAGE_NAMES[language]}." if language else None
        note = HAND_OFF_NOTE.format(reason=reason)
        extra = "\n".join(p for p in (user_input.extra_system_prompt, note, reply_in) if p)
        if self._decider.fallback_agent is None:
            return await self._answer_with_loop(replace(user_input, extra_system_prompt=extra), language)
        return await conversation.async_converse(
            self.hass,
            user_input.text,
            user_input.conversation_id,
            user_input.context,
            user_input.language,
            agent_id=self._decider.fallback_agent,
            device_id=user_input.device_id,
            satellite_id=user_input.satellite_id,
            extra_system_prompt=extra,
        )

    async def _answer_with_loop(
        self, user_input: conversation.ConversationInput, language: str | None = None
    ) -> conversation.ConversationResult:
        if self._loop is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="no_text_model",
                translation_placeholders={"entity_id": self.entity_id},
            )
        return await async_answer_turn(
            self.hass, self._entry.runtime_data.llm, self._loop, self._agent_id, self.entity_id, user_input, language
        )

    def _state(self, user_input: conversation.ConversationInput) -> str:
        return state_text(user_input.text, self._previous.get(user_input.conversation_id or ""))

    async def _household_rules(self) -> Household:
        """The household's rules as Second Brain compiles them for a decider (async_get_compiled_rules).

        It never reads the files: Second Brain decides what other integrations see. None without memory; a failed read
        is a rule not compiled, so the turn hands off rather than act against it.
        """
        brain = self._entry.runtime_data.brain
        if brain is None or not isinstance(brain.api, BrainAPI):
            return Household()
        try:
            return Household.parse(await brain.api.async_get_compiled_rules())
        except (OSError, HomeAssistantError) as err:
            _LOGGER.warning("%s could not read Second Brain's rules: %s", self.entity_id, err)
            return Household.parse([{"text": f"Second Brain's rules could not be read ({err})", "kind": PENDING}])

    def _decision_client(self) -> DecisionClient:
        return DecisionClient(
            async_get_clientsession(self.hass), self._entry.data[CONF_DECIDER_URL], self._decider.model
        )

    def _llm(self) -> LLMClient:
        model = self._decider.answer_model or self._text_model or ""
        return LLMClient(async_get_clientsession(self.hass), LLMServer.from_entry_data(self._entry.data), model)


def _in_language(
    user_input: conversation.ConversationInput, language: str | None
) -> conversation.ConversationInput:
    """The input in the reply language the decider named, the pipeline's when it named none.

    Results and HA's state words are then in it: in the pipeline's language an English question gets German state words,
    and the reply follows them into German.
    """
    return replace(user_input, language=language) if language else user_input
