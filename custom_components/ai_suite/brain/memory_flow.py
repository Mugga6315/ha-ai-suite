"""The Memory subentry flow: where the store lives and the background model.

The LLM server is AI Suite's, so only its model is asked.
"""
import os
import pathlib
from typing import Any

import probatio

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from ..const import CONF_LLM_URL
from ..servers import models_or_empty
from .background import memory_subentry
from .const import CONF_LLM_MODEL, CONF_STORE_LOCATION
from .store_location import default_store_location, unsafe_store_location

MEMORY_TITLE = "Memory"


class MemoryFlow(ConfigSubentryFlow):
    """Add Memory once, or change its settings."""

    def __init__(self) -> None:
        """Start with no input."""
        self._input: dict[str, Any] = {}
        self._existing: dict[str, list[str]] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Add Memory."""
        if memory_subentry(self._get_entry()) is not None:
            return self.async_abort(reason="already_configured")
        return await self._form("user", user_input, {})

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Change Memory's settings."""
        return await self._form("reconfigure", user_input, dict(self._get_reconfigure_subentry().data))

    async def async_step_store_info(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """The chosen folder already holds a store: say what was found before using it."""
        if user_input is not None:
            return self._finish()
        items = [f"{d}/" for d in self._existing["dirs"]] + self._existing["files"]
        return self.async_show_form(
            step_id="store_info", data_schema=probatio.Schema({}), description_placeholders={"found": ", ".join(items)}
        )

    async def _form(self, step_id: str, user_input: dict[str, Any] | None, current: dict) -> SubentryFlowResult:
        if user_input is None:
            return self.async_show_form(step_id=step_id, data_schema=await self._schema(current))
        unsafe = await self.hass.async_add_executor_job(
            unsafe_store_location, self.hass, user_input.get(CONF_STORE_LOCATION, "")
        )
        if unsafe:
            return self.async_show_form(
                step_id=step_id, data_schema=await self._schema(user_input),
                errors={"base": "unsafe_store_location"}, description_placeholders={"error": unsafe},
            )
        # The store's own flags (initialized, initialized_path) stay: they say where it was set up.
        self._input = {**current, **user_input}
        moved = user_input[CONF_STORE_LOCATION] != current.get(CONF_STORE_LOCATION)
        if moved:
            self._existing = await self.hass.async_add_executor_job(_existing_store, user_input[CONF_STORE_LOCATION])
            if self._existing["dirs"] or self._existing["files"]:
                return await self.async_step_store_info()
        return self._finish()

    def _finish(self) -> SubentryFlowResult:
        if self.source == SOURCE_RECONFIGURE:
            return self.async_update_and_abort(self._get_entry(), self._get_reconfigure_subentry(), data=self._input)
        return self.async_create_entry(title=MEMORY_TITLE, data=self._input)

    async def _schema(self, values: dict) -> probatio.Schema:
        locations = await self.hass.async_add_executor_job(_detect_locations, self.hass)
        models = await models_or_empty(self.hass, self._get_entry().data[CONF_LLM_URL])
        return probatio.Schema(
            {
                probatio.Required(
                    CONF_STORE_LOCATION, default=values.get(CONF_STORE_LOCATION) or default_store_location(self.hass)
                ): SelectSelector(
                    SelectSelectorConfig(options=locations, mode=SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                probatio.Required(CONF_LLM_MODEL, description={"suggested_value": values.get(CONF_LLM_MODEL)}): (
                    SelectSelector(SelectSelectorConfig(options=models, mode=SelectSelectorMode.DROPDOWN,
                                                        custom_value=True))
                ),
            }
        )


def _detect_locations(hass) -> list[SelectOptionDict]:
    """Candidate store locations: the own folder, and network storage mounts."""
    own = default_store_location(hass)
    options = [SelectOptionDict(value=own, label=f"{own} (local, created for you)")]
    for base in ("/share", "/media"):
        try:
            options.extend(
                SelectOptionDict(value=entry.path, label=entry.path) for entry in os.scandir(base) if entry.is_dir()
            )
        except OSError:
            continue
    return options


def _existing_store(path: str) -> dict[str, list[str]]:
    """What in this folder looks like an existing store."""
    root = pathlib.Path(path)
    return {
        "dirs": [d for d in ("memories", "wiki") if (root / d).is_dir()],
        "files": [f for f in ("CORE.md", "INDEX.md", "CONSOLIDATE.md") if (root / f).is_file()],
    }
