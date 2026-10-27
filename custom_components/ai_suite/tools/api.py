"""AI Suite's LLM API: the tools of every tool set the entry has, for any conversation agent that selects it.

The tools need nothing but HA. With Memory there, a call the model got wrong also goes into Memory's failure log
(`record_failure`), so self-improvement learns from it.
"""

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..const import DOMAIN
from . import calendar
from .sets import SUBENTRY_CALENDAR, TOOL_SETS
from .spec import RecordFailure

_LOGGER = logging.getLogger(__name__)

# What a home ticks under an agent's LLM APIs. HA's own merge, used by other integrations' agents, prefixes each tool
# with the API's slugged name; AI Suite's agents give the model the tool's own name.
TOOLS_API_ID = DOMAIN
TOOLS_API_NAME = "AI Suite"


class ToolsAPI(llm.API):
    """The tools of the entry's tool sets, built for each request from the subentries as they are then."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, record_failure: RecordFailure | None) -> None:
        """Serve the tool sets of `entry`."""
        super().__init__(hass=hass, id=TOOLS_API_ID, name=TOOLS_API_NAME)
        self._entry = entry
        self._record_failure = record_failure

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Every tool set's tools; a set that fails to build costs only its own tools."""
        tools: list[llm.Tool] = []
        for subentry in self._entry.subentries.values():
            if (tool_set := TOOL_SETS.get(subentry.subentry_type)) is None:
                continue
            try:
                tools += tool_set.build(self.hass, subentry.data, self._record_failure)
            except Exception:  # logged; another set's tools must survive a broken one
                _LOGGER.exception("tool set %s failed to build its tools, skipping it this turn", subentry.title)
        return llm.APIInstance(api=self, api_prompt="", llm_context=llm_context, tools=tools)


def async_setup_tools(hass: HomeAssistant, entry: ConfigEntry, record_failure: RecordFailure | None) -> None:
    """Serve AI Suite's LLM API for as long as the entry is loaded, and say when the calendars are not exposed."""
    entry.async_on_unload(llm.async_register_api(hass, ToolsAPI(hass, entry, record_failure)))
    calendar.check_exposed(hass, bool(entry.get_subentries_of_type(SUBENTRY_CALENDAR)))
