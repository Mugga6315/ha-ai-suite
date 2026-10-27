"""What Second Brain's background jobs run on.

What Second Brain's background jobs (the librarian, the self-improvement review and pass, the rule compiler) run
on: Memory's background model on AI Suite's LLM server.
"""

from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..llm_client import ChatClient, LLMServer
from .const import CONF_LLM_MODEL, NO_THINKING, SUBENTRY_MEMORY


class RunFailed(Exception):
    """A background run that changed nothing, and why."""


def subentry_of(entry: ConfigEntry, subentry_type: str) -> ConfigSubentry | None:
    """The entry's subentry of this type; each of Memory and its features exists at most once."""
    return next(iter(entry.get_subentries_of_type(subentry_type)), None)


def memory_subentry(entry: ConfigEntry) -> ConfigSubentry | None:
    """The Memory subentry, or None when Memory is not added."""
    return subentry_of(entry, SUBENTRY_MEMORY)


def background_client(hass: HomeAssistant, entry: ConfigEntry) -> ChatClient | None:
    """A client for Memory's background model; None while Memory has no model set."""
    memory = memory_subentry(entry)
    model = memory.data.get(CONF_LLM_MODEL) if memory else None
    if not model:
        return None
    return ChatClient(async_get_clientsession(hass), LLMServer.from_entry_data(entry.data), model)


def effort_options(effort: str) -> dict[str, Any]:
    """The request options for a job's thinking effort: thinking on with that effort, or off for NO_THINKING."""
    if effort == NO_THINKING:
        return {"thinking": False}
    return {"thinking": True, "effort": effort}
