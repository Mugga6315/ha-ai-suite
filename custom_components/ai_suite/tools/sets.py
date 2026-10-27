"""AI Suite's tool sets: each one a subentry a home adds or removes, with its tools, its form and its status check.

A new tool set is a module with its tools (ToolSpecs and a `build`) and one entry in TOOL_SETS.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.config_entries import ConfigSubentryData, ConfigSubentryFlow
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..settings_flow import SettingsForm, settings_flow
from . import calendar, emby, history
from .spec import RecordFailure

SUBENTRY_HISTORY = "history"
SUBENTRY_CALENDAR = "calendar"
SUBENTRY_EMBY = "emby"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSet:
    """One tool set: its subentry's form, its tools, and what keeps it from working (for its status light).

    `with_entry`: AI Suite adds the set when it is set up; the home can remove it.
    """

    form: SettingsForm
    build: Callable[[HomeAssistant, Mapping[str, Any], RecordFailure | None], list[llm.Tool]]
    problem: Callable[[HomeAssistant, dict], Awaitable[str | None]] | None = None
    with_entry: bool = False


def _no_fields(_data: dict) -> dict:
    return {}


async def _nothing_to_check(_hass: HomeAssistant, _data: dict) -> str | None:
    return None


TOOL_SETS = {
    SUBENTRY_HISTORY: ToolSet(
        form=SettingsForm(
            subentry_type=SUBENTRY_HISTORY, title="History", fields=history.subentry_schema, check=_nothing_to_check
        ),
        build=history.build,
        problem=history.async_problem,
        with_entry=True,
    ),
    SUBENTRY_CALENDAR: ToolSet(
        form=SettingsForm(
            subentry_type=SUBENTRY_CALENDAR, title="Calendar", fields=_no_fields, check=_nothing_to_check
        ),
        build=calendar.build,
        with_entry=True,
    ),
    SUBENTRY_EMBY: ToolSet(
        form=SettingsForm(
            subentry_type=SUBENTRY_EMBY, title="Emby", fields=emby.subentry_schema, check=emby.async_validate
        ),
        build=emby.build,
        problem=emby.async_validate,
    ),
}


def subentry_flows() -> dict[str, type[ConfigSubentryFlow]]:
    """The add and change dialogs of the tool sets, by subentry type."""
    return {subentry_type: settings_flow(tool_set.form) for subentry_type, tool_set in TOOL_SETS.items()}


def created_with_entry() -> list[ConfigSubentryData]:
    """The tool sets a new AI Suite entry starts with.

    Plain dicts, as HA's own integrations pass them: the UI's flow view turns the created entry into JSON, subentries
    included, and a MappingProxyType has no JSON form.
    """
    return [
        ConfigSubentryData(data={}, subentry_type=subentry_type, title=tool_set.form.title, unique_id=None)
        for subentry_type, tool_set in TOOL_SETS.items()
        if tool_set.with_entry
    ]
