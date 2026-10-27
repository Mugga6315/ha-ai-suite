"""AI Suite: one integration for the home's voice assistant.

A decision model in front of an LLM agent, AI Suite's own tools, and Second Brain's memory.
"""

from dataclasses import dataclass
from typing import Any, cast

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .brain.llm_api import BrainAPI
from .brain.setup import Brain, async_setup_brain
from .const import CONF_LLM_URL, DOMAIN
from .monitor.setup import Monitor, async_setup_monitor
from .servers import ServerError, list_models
from .services import async_setup_services
from .tools.api import async_setup_tools

PLATFORMS = (Platform.CONVERSATION, Platform.BINARY_SENSOR)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
# Local servers need no key, but the client refuses to start without one.
NO_API_KEY = "ai-suite"


@dataclass(slots=True)
class RuntimeData:
    """What a loaded entry holds: the agent loop's client, memory and the trace monitor's exporter when they are on."""

    llm: openai.AsyncOpenAI
    brain: Brain | None
    monitor: Monitor | None = None


type AiSuiteConfigEntry = ConfigEntry[RuntimeData]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the actions, whether or not an entry is loaded (they say why they cannot run)."""
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: AiSuiteConfigEntry) -> bool:
    """Set up the servers' clients, memory, the tools, the trace monitor's exporter and the platforms.

    Only the LLM server is checked: without it no turn can be answered. A decision server that is down costs
    nothing but speed, since every turn then goes to the text model.
    """
    try:
        await list_models(async_get_clientsession(hass), entry.data[CONF_LLM_URL])
    except ServerError as err:
        raise ConfigEntryNotReady(
            translation_domain=DOMAIN, translation_key="llm_server_unreachable",
            translation_placeholders={"error": str(err)},
        ) from err
    # HA's client is an httpx client, openai's type names httpx2's; HA's own OpenAI integration passes it the same way.
    client = openai.AsyncOpenAI(
        base_url=entry.data[CONF_LLM_URL], api_key=NO_API_KEY, http_client=cast(Any, get_async_client(hass))
    )
    await hass.async_add_executor_job(_prepare_client, client)
    brain = await async_setup_brain(hass, entry)
    # With memory up, a tool call the model got wrong goes into its failure log; offline or without Memory, it does not.
    online = brain is not None and isinstance(brain.api, BrainAPI)
    async_setup_tools(hass, entry, brain.store.async_record_failure if brain is not None and online else None)
    entry.runtime_data = RuntimeData(llm=client, brain=brain, monitor=await async_setup_monitor(hass, entry))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # A changed server, an added, changed or removed subentry, or changed options reload the entry.
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: AiSuiteConfigEntry) -> bool:
    """Unload the platforms; the LLM APIs, timers and tasks go with the entry's unload callbacks."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_config_entry_device(
    hass: HomeAssistant, entry: AiSuiteConfigEntry, device: dr.DeviceEntry
) -> bool:
    """A device whose subentry is gone (a removed agent, tool set or feature) may be deleted; a live one's stays."""
    return not any(identifier[0] == DOMAIN and identifier[1] in entry.subentries for identifier in device.identifiers)


async def _async_reload(hass: HomeAssistant, entry: AiSuiteConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


def _prepare_client(client: openai.AsyncOpenAI) -> None:
    """Do what blocks once, in the executor, so no turn does it in the event loop.

    The client reads its platform headers from the system, and imports its chat resources on their first use
    (`AsyncOpenAI.chat`, a cached property).
    """
    client.platform_headers()
    _ = client.chat
