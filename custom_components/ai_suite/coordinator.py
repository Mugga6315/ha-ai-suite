"""The health of tool sets and features: one probe per one that has a status light, every SCAN_INTERVAL.

Each probe reuses the set's or feature's own check, so a light agrees with what its settings form would say: Emby
reachable, the recorder running for History, the librarian's LLM server reachable. One that turns unhealthy is logged
once, and once more when it is back.
"""

import asyncio
from datetime import timedelta
import logging
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .brain.const import SUBENTRY_LIBRARIAN
from .const import CONF_LLM_URL
from .servers import ServerError, list_models
from .tools.sets import TOOL_SETS

if TYPE_CHECKING:
    from . import AiSuiteConfigEntry

_LOGGER = logging.getLogger(__name__)

# Live network probes: often enough for a status light without loading the servers.
SCAN_INTERVAL = timedelta(minutes=5)
HEALTH_TYPES = (*(kind for kind, tool_set in TOOL_SETS.items() if tool_set.problem), SUBENTRY_LIBRARIAN)


class HealthCoordinator(DataUpdateCoordinator[dict[str, bool]]):
    """The health of each tool set and feature with a status light, by subentry id."""

    config_entry: AiSuiteConfigEntry

    def __init__(self, hass: HomeAssistant, entry: AiSuiteConfigEntry) -> None:
        """Probe the tool sets and features of `entry`."""
        super().__init__(hass, _LOGGER, name="AI Suite health", update_interval=SCAN_INTERVAL, config_entry=entry)

    async def _async_update_data(self) -> dict[str, bool]:
        # All at once: one unreachable server must not add its timeout to the others'.
        subentries = [s for s in self.config_entry.subentries.values() if s.subentry_type in HEALTH_TYPES]
        problems = await asyncio.gather(*(self._problem(s) for s in subentries))
        before = self.data or {}
        for subentry, problem in zip(subentries, problems, strict=True):
            was_healthy = before.get(subentry.subentry_id)
            if problem is not None and was_healthy is not False:
                _LOGGER.info("%s is unavailable: %s", subentry.title, problem)
            elif problem is None and was_healthy is False:
                _LOGGER.info("%s is available again", subentry.title)
        return {s.subentry_id: problem is None for s, problem in zip(subentries, problems, strict=True)}

    async def _problem(self, subentry: ConfigSubentry) -> str | None:
        """What keeps the tool set or feature from working, or None when it is healthy."""
        tool_set = TOOL_SETS.get(subentry.subentry_type)
        if tool_set is not None and tool_set.problem is not None:
            return await tool_set.problem(self.hass, dict(subentry.data))
        return await self._llm_server_problem()  # the librarian

    async def _llm_server_problem(self) -> str | None:
        try:
            await list_models(async_get_clientsession(self.hass), self.config_entry.data[CONF_LLM_URL])
        except ServerError as err:
            return f"the LLM server does not answer ({err})"
        return None
