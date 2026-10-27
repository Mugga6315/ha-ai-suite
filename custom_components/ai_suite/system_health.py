"""System health panel: whether the store and the servers answer, and which subentries are added.

HA finds this platform by itself; nothing wires it in __init__.
"""

import os
from typing import Any

from homeassistant.components import system_health
from homeassistant.core import HomeAssistant, callback

from .brain.background import memory_subentry
from .brain.const import CONF_STORE_LOCATION
from .const import CONF_LLM_URL, DOMAIN
from .tools.emby import CONF_EMBY_URL
from .tools.sets import SUBENTRY_EMBY


@callback
def async_register(hass: HomeAssistant, register: system_health.SystemHealthRegistration) -> None:
    """Register the panel's info."""
    register.async_register_info(_system_health_info)


async def _system_health_info(hass: HomeAssistant) -> dict[str, Any]:
    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        return {"configured": False}
    entry = entries[0]  # AI Suite allows one entry (manifest: single_config_entry)
    active = sorted(subentry.subentry_type for subentry in entry.subentries.values())
    info: dict[str, Any] = {"active_features": ", ".join(active) or "none"}
    if memory := memory_subentry(entry):
        info["store_reachable"] = await hass.async_add_executor_job(os.path.isdir, memory.data[CONF_STORE_LOCATION])
    info["llm_reachable"] = system_health.async_check_can_reach_url(
        hass, f"{entry.data[CONF_LLM_URL].rstrip('/')}/models"
    )
    if (emby := next(iter(entry.get_subentries_of_type(SUBENTRY_EMBY)), None)) and emby.data.get(CONF_EMBY_URL):
        info["emby_reachable"] = system_health.async_check_can_reach_url(
            hass, f"{emby.data[CONF_EMBY_URL].rstrip('/')}/System/Info"
        )
    return info
