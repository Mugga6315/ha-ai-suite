"""Config flow: the servers (the config entry) and the conversation agents (its subentries)."""

from http import HTTPStatus
import logging
from typing import Any

import aiohttp
import probatio

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_NAME, CONF_PROMPT
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers import config_validation as cv, entity_registry as er, llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .agent.settings import (
    CONF_HIDDEN_TOOLS,
    CONF_MAX_HISTORY,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_REASONING_EFFORT,
    CONF_SEND_REASONING,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    CONF_THINKING,
    CONF_THINKING_BUDGET,
    REASONING_EFFORTS,
    THINKING_OFF,
    THINKING_ON,
)
from .agent.tools import AgentAPI
from .brain import features
from .brain.const import SUBENTRY_MEMORY
from .brain.memory_flow import MemoryFlow
from .const import (
    CONF_ANSWER_MODEL,
    CONF_DECIDER,
    CONF_DECIDER_MODEL,
    CONF_DECIDER_URL,
    CONF_FALLBACK_AGENT,
    CONF_LLM_URL,
    CONF_REPLY_CHECK,
    CONF_THINKING_OFF_KWARG,
    CONF_THRESHOLD,
    DEFAULT_DECIDER,
    DEFAULT_REPLY_CHECK,
    DEFAULT_THINKING_OFF_KWARG,
    DEFAULT_THRESHOLD,
    DOMAIN,
    SUBENTRY_CONVERSATION,
)
from .decider.api import DecisionClient, DecisionError
from .monitor.const import (
    CONF_MONITOR_AUDIO,
    CONF_MONITOR_ENABLED,
    CONF_MONITOR_TOKEN,
    CONF_MONITOR_URL,
    HTTP_TIMEOUT_SECONDS,
    INGEST_TURNS_PATH,
)
from .servers import ServerError, decision_api_url, list_models, models_or_empty
from .tools import sets as tool_sets

_LOGGER = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "AI Suite"
CONVERSATION_DOMAIN = "conversation"
SECTION_MODELS, SECTION_REASONING, SECTION_TOOLS, SECTION_REPLIES = "models", "reasoning", "tools", "replies"
SECTIONS = (SECTION_MODELS, SECTION_REASONING, SECTION_TOOLS, SECTION_REPLIES)
# The monitor's ingest answers `{}` with 422 once the token is accepted: its bearer check runs before the body is read.
INGEST_ERRORS: dict[int, str] = {HTTPStatus.UNAUTHORIZED: "invalid_auth", HTTPStatus.NOT_FOUND: "not_a_monitor"}
MONITOR_SCHEMA = probatio.Schema(
    {
        probatio.Required(CONF_MONITOR_ENABLED, default=False): BooleanSelector(),
        probatio.Optional(CONF_MONITOR_URL): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
        probatio.Optional(CONF_MONITOR_TOKEN): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        probatio.Required(CONF_MONITOR_AUDIO, default=False): BooleanSelector(),
    }
)


class AiSuiteConfigFlow(ConfigFlow, domain=DOMAIN):
    """Set up AI Suite: the decision server and the LLM server."""
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Ask for the two servers."""
        errors = await _server_errors(self.hass, user_input) if user_input is not None else {}
        if user_input is not None and not errors:
            return self.async_create_entry(title="AI Suite", data=user_input, subentries=tool_sets.created_with_entry())
        return self.async_show_form(step_id="user", data_schema=_server_schema(user_input or {}), errors=errors)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Change the two servers."""
        entry = self._get_reconfigure_entry()
        errors = await _server_errors(self.hass, user_input) if user_input is not None else {}
        if user_input is not None and not errors:
            # The entry's update listener reloads it.
            return self.async_update_and_abort(entry, data=user_input)
        return self.async_show_form(
            step_id="reconfigure", data_schema=_server_schema(user_input or dict(entry.data)), errors=errors
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(cls, config_entry: ConfigEntry) -> dict[str, type[ConfigSubentryFlow]]:
        """The subentries an AI Suite entry offers: agents, the tool sets, Memory and its features."""
        return (
            {SUBENTRY_CONVERSATION: ConversationAgentFlow}
            | tool_sets.subentry_flows()
            | {SUBENTRY_MEMORY: MemoryFlow}
            | features.subentry_flows()
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """The entry's options: the trace monitor export."""
        return MonitorOptionsFlow()


class MonitorOptionsFlow(OptionsFlow):
    """Turn the export of each turn to the trace monitor on or off.

    A plain options flow: the entry's update listener reloads it on the change.
    """

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Ask for the monitor; when on, it must accept the token."""
        errors = await _monitor_errors(self.hass, user_input) if user_input is not None else {}
        if user_input is not None and not errors:
            return self.async_create_entry(data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(MONITOR_SCHEMA, user_input or self.config_entry.options),
            errors=errors,
        )


class ConversationAgentFlow(ConfigSubentryFlow):
    """One conversation agent: the loop's model, prompt and tools, and the decider in front of it.

    The form groups the fields in sections; the subentry stores them flat, as every reader expects.
    """

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Add a conversation agent."""
        values = _flatten(user_input or {})
        errors = await self._decider_errors(values) if user_input is not None else {}
        if user_input is not None and not errors:
            return self.async_create_entry(title=values.pop(CONF_NAME), data=values)
        name = {probatio.Required(CONF_NAME, default=values.get(CONF_NAME, DEFAULT_AGENT_NAME)): TextSelector()}
        return await self._show_form("user", name, values, errors)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Change a conversation agent's settings."""
        subentry = self._get_reconfigure_subentry()
        values = _flatten(user_input) if user_input is not None else dict(subentry.data)
        errors = await self._decider_errors(values) if user_input is not None else {}
        if user_input is not None and not errors:
            return self.async_update_and_abort(self._get_entry(), subentry, data=values)
        return await self._show_form("reconfigure", {}, values, errors)

    async def _show_form(
        self, step_id: str, first: dict, values: dict[str, Any], errors: dict[str, str]
    ) -> SubentryFlowResult:
        return self.async_show_form(
            step_id=step_id,
            data_schema=probatio.Schema(first | await self._agent_fields(values)),
            errors=errors,
        )

    async def _agent_fields(self, values: dict[str, Any]) -> dict:
        """Optional fields carry their value as a suggestion, not a default, so a cleared field stays cleared."""
        models = SelectSelector(
            SelectSelectorConfig(options=await models_or_empty(self.hass, self._get_entry().data[CONF_LLM_URL]),
                                 mode=SelectSelectorMode.DROPDOWN,
                                 custom_value=True)
        )

        def suggested(key: str) -> dict:
            return {"suggested_value": values.get(key)}

        def default(key: str, value: Any) -> Any:
            return values.get(key, value)

        return {
            probatio.Optional(CONF_PROMPT, description=suggested(CONF_PROMPT)): TemplateSelector(),
            probatio.Required(SECTION_MODELS): section(probatio.Schema({
                probatio.Required(CONF_MODEL, description=suggested(CONF_MODEL)): models,
                probatio.Optional(CONF_TEMPERATURE, description=suggested(CONF_TEMPERATURE)): NumberSelector(
                    NumberSelectorConfig(min=0, max=2, step=0.05, mode=NumberSelectorMode.BOX)
                ),
                probatio.Required(CONF_DECIDER, default=default(CONF_DECIDER, DEFAULT_DECIDER)): BooleanSelector(),
                probatio.Optional(CONF_DECIDER_MODEL, description=suggested(CONF_DECIDER_MODEL)): SelectSelector(
                    SelectSelectorConfig(options=await self._decider_models(),
                                         mode=SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                probatio.Required(CONF_THRESHOLD, default=default(CONF_THRESHOLD, DEFAULT_THRESHOLD)): NumberSelector(
                    NumberSelectorConfig(min=0, max=1, step=0.05, mode=NumberSelectorMode.BOX)
                ),
                probatio.Optional(CONF_ANSWER_MODEL, description=suggested(CONF_ANSWER_MODEL)): models,
                probatio.Optional(CONF_FALLBACK_AGENT, description=suggested(CONF_FALLBACK_AGENT)): SelectSelector(
                    SelectSelectorConfig(options=self._other_agents(), mode=SelectSelectorMode.DROPDOWN,
                                         custom_value=True)
                ),
            })),
            probatio.Required(SECTION_REASONING): section(probatio.Schema({
                probatio.Optional(CONF_THINKING, description=suggested(CONF_THINKING)): SelectSelector(
                    SelectSelectorConfig(options=[THINKING_ON, THINKING_OFF], translation_key=CONF_THINKING,
                                         mode=SelectSelectorMode.DROPDOWN)
                ),
                probatio.Optional(CONF_REASONING_EFFORT, description=suggested(CONF_REASONING_EFFORT)): SelectSelector(
                    SelectSelectorConfig(options=REASONING_EFFORTS, translation_key=CONF_REASONING_EFFORT,
                                         mode=SelectSelectorMode.DROPDOWN)
                ),
                probatio.Optional(CONF_THINKING_BUDGET, description=suggested(CONF_THINKING_BUDGET)): NumberSelector(
                    NumberSelectorConfig(min=0, max=32768, step=1, mode=NumberSelectorMode.BOX)
                ),
                probatio.Required(CONF_SEND_REASONING, default=default(CONF_SEND_REASONING, True)): BooleanSelector(),
            })),
            probatio.Required(SECTION_TOOLS): section(probatio.Schema({
                probatio.Optional(CONF_LLM_HASS_API, description=suggested(CONF_LLM_HASS_API)): SelectSelector(
                    SelectSelectorConfig(options=[SelectOptionDict(value=api.id, label=api.name)
                                                  for api in llm.async_get_apis(self.hass)],
                                         multiple=True, mode=SelectSelectorMode.LIST)
                ),
                probatio.Optional(CONF_HIDDEN_TOOLS, description=suggested(CONF_HIDDEN_TOOLS)): SelectSelector(
                    SelectSelectorConfig(options=await self._tool_names(values), multiple=True,
                                         mode=SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                probatio.Required(
                    CONF_PARALLEL_TOOL_CALLS, default=default(CONF_PARALLEL_TOOL_CALLS, True)
                ): BooleanSelector(),
            })),
            probatio.Required(SECTION_REPLIES): section(probatio.Schema({
                probatio.Required(CONF_REPLY_CHECK, default=default(CONF_REPLY_CHECK, DEFAULT_REPLY_CHECK)): BooleanSelector(),
                probatio.Required(CONF_STRIP_EMOJIS, default=default(CONF_STRIP_EMOJIS, False)): BooleanSelector(),
                probatio.Optional(CONF_MAX_HISTORY, description=suggested(CONF_MAX_HISTORY)): NumberSelector(
                    NumberSelectorConfig(min=0, max=100, step=1, mode=NumberSelectorMode.BOX)
                ),
            }), {"collapsed": True}),
        }

    def _other_agents(self) -> list[SelectOptionDict]:
        """The conversation agents a hand-off can go to: all but this one, which would hand off to itself.

        HA's agent picker shows no description, so the form lists them in a dropdown of its own.
        """
        own = self._own_entity_id()
        agents = sorted(self.hass.states.async_all(CONVERSATION_DOMAIN), key=lambda state: state.name)
        return [SelectOptionDict(value=state.entity_id, label=state.name) for state in agents if state.entity_id != own]

    def _own_entity_id(self) -> str | None:
        """This agent's entity; None while it is being added."""
        if self.source != SOURCE_RECONFIGURE:
            return None
        subentry_id = self._get_reconfigure_subentry().subentry_id
        return er.async_get(self.hass).async_get_entity_id(CONVERSATION_DOMAIN, DOMAIN, subentry_id)

    def _decider_url(self) -> str:
        return self._get_entry().data[CONF_DECIDER_URL]

    async def _decider_models(self) -> list[str]:
        """The decision server's models for the picker."""
        return await models_or_empty(self.hass, decision_api_url(self._decider_url()), decision=True)

    async def _decider_errors(self, values: dict[str, Any]) -> dict[str, str]:
        """The decision server must answer with the agent's model.

        Without one, a server that runs several models refuses, while one that serves a single model under several
        names answers.
        """
        if not values.get(CONF_DECIDER, DEFAULT_DECIDER):
            return {}
        url, model = self._decider_url(), values.get(CONF_DECIDER_MODEL)
        if await _decider_answers(self.hass, url, model):
            return {}
        if not model and len(await models_or_empty(self.hass, decision_api_url(url))) > 1:
            return {"base": "decider_model_needed"}
        return {"base": "decider_cannot_connect"}

    async def _tool_names(self, values: dict[str, Any]) -> list[str]:
        """The tool names the model sees with the agent's saved APIs, and the ones already left out.

        The APIs chosen in the open form are not known yet. The tools AI Suite replaces are absent already.
        """
        hidden = list(values.get(CONF_HIDDEN_TOOLS) or ())
        available = {api.id for api in llm.async_get_apis(self.hass)}
        sources = [api for api in values.get(CONF_LLM_HASS_API) or () if api in available]
        if not sources:
            return hidden
        context = llm.LLMContext(platform=DOMAIN, context=Context(), language=self.hass.config.language,
                                 assistant=CONVERSATION_DOMAIN, device_id=None)
        try:
            instance = await AgentAPI(self.hass, "settings", sources).async_get_api_instance(context)
        except Exception:  # any integration's API can raise anything here; the form must still open
            _LOGGER.warning("could not list the tools of %s", sources, exc_info=True)
            return hidden
        return sorted({tool.name for tool in instance.tools} | set(hidden))


def _flatten(user_input: dict[str, Any]) -> dict[str, Any]:
    """The form's sections as the flat data the subentry stores."""
    data = {key: value for key, value in user_input.items() if key not in SECTIONS}
    for name in SECTIONS:
        data.update(user_input.get(name) or {})
    return data


def _server_schema(values: dict[str, Any]) -> probatio.Schema:
    return probatio.Schema(
        {
            probatio.Required(CONF_DECIDER_URL, default=values.get(CONF_DECIDER_URL, probatio.UNDEFINED)): TextSelector(),
            probatio.Required(CONF_LLM_URL, default=values.get(CONF_LLM_URL, probatio.UNDEFINED)): TextSelector(),
            probatio.Required(
                CONF_THINKING_OFF_KWARG, default=values.get(CONF_THINKING_OFF_KWARG, DEFAULT_THINKING_OFF_KWARG)
            ): BooleanSelector(),
        }
    )


async def _server_errors(hass: HomeAssistant, servers: dict[str, Any]) -> dict[str, str]:
    """Which server does not list its models.

    The decision server is not asked a question here: with several models it needs a model name, which each agent sets.
    """
    session = async_get_clientsession(hass)
    errors = {}
    decider = decision_api_url(servers[CONF_DECIDER_URL])
    for key, base_url in ((CONF_DECIDER_URL, decider), (CONF_LLM_URL, servers[CONF_LLM_URL])):
        try:
            await list_models(session, base_url)
        except ServerError:
            errors[key] = "cannot_connect"
    return errors


async def _decider_answers(hass: HomeAssistant, url: str, model: str | None) -> bool:
    """Whether a decision server answers a ping question with this model."""
    client = DecisionClient(async_get_clientsession(hass), url, model)
    try:
        await client.choice("ping", "Is this a ping?", {"yes": None, "no": None})
    except DecisionError:
        return False
    return True


async def _monitor_errors(hass: HomeAssistant, options: dict[str, Any]) -> dict[str, str]:
    """Why the monitor cannot take the turns; nothing to check while the export is off."""
    if not options[CONF_MONITOR_ENABLED]:
        return {}
    url = options.get(CONF_MONITOR_URL, "")
    try:
        cv.url(url)
    except probatio.Invalid:
        return {CONF_MONITOR_URL: "invalid_url"}
    try:
        status = await _ingest_status(hass, url, options.get(CONF_MONITOR_TOKEN, ""))
    except (aiohttp.ClientError, TimeoutError):
        return {"base": "cannot_connect"}
    if status == HTTPStatus.UNPROCESSABLE_ENTITY:
        return {}
    return {"base": INGEST_ERRORS.get(status, "unknown")}


async def _ingest_status(hass: HomeAssistant, url: str, token: str) -> int:
    """The status the monitor's ingest answers an empty record with."""
    session = async_get_clientsession(hass)
    async with session.post(
        url.rstrip("/") + INGEST_TURNS_PATH,
        json={},
        headers={"Authorization": f"Bearer {token}"},
        timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS),
    ) as resp:
        return resp.status
