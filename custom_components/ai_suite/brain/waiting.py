"""Memory while its store is offline (a network share that is down).

The agents keep Memory's API without its tools, with a prompt that says memory is off. The store is looked for again
only when a turn asks for memory, at most every CHECK_EVERY: no timer, no polling in the background. Once it is found
again, AI Suite reloads and memory comes up as usual.
"""

import asyncio
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util

from .const import BRAIN_API_ID, BRAIN_API_NAME, LOGGER
from .store import Store

CHECK_EVERY = timedelta(minutes=5)
CHECK_TIMEOUT_SECONDS = 2  # a hung network mount must not stall the turn
# Without it a model may still say it remembered something.
OFFLINE_PROMPT = (
    "Memory is unavailable right now: you cannot remember, recall or forget anything. "
    "Say so when the user asks for it."
)


class WaitingBrainAPI(llm.API):
    """Memory's API while the store is offline: no tools, and a prompt that says memory is off."""

    def __init__(self, hass: HomeAssistant, store: Store, entry: ConfigEntry) -> None:
        """Stand in for memory while `store` is offline."""
        super().__init__(hass=hass, id=BRAIN_API_ID, name=BRAIN_API_NAME)
        self._store = store
        self._entry = entry
        self._next_check = dt_util.utcnow()

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """The offline prompt; looks for the store again when due."""
        now = dt_util.utcnow()
        if now >= self._next_check:
            self._next_check = now + CHECK_EVERY
            if await self._store_is_back():
                LOGGER.info("memory: the store at %s is back, reloading AI Suite", self._store.root)
                self.hass.config_entries.async_schedule_reload(self._entry.entry_id)
        return llm.APIInstance(api=self, api_prompt=OFFLINE_PROMPT, llm_context=llm_context, tools=[])

    async def _store_is_back(self) -> bool:
        try:
            async with asyncio.timeout(CHECK_TIMEOUT_SECONDS):
                return await self._store.async_exists()
        except (TimeoutError, OSError):
            return False
