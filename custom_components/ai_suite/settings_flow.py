"""The add and change dialog of a subentry with one form: a tool set (tools/sets.py) or a Memory feature.

Each such subentry exists once: a second librarian would double the nightly schedule, a second tool set would
duplicate its tools.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import probatio

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingsForm:
    """A subentry's form: its fields, their check, and what must exist before it can be added.

    `fields`: the form's fields for the current values. `check`: why the values cannot be used, or None.
    `missing`: the abort reason while something the subentry needs is not there, or None.
    """

    subentry_type: str
    title: str
    fields: Callable[[dict], dict]
    check: Callable[[HomeAssistant, dict], Awaitable[str | None]]
    missing: Callable[[ConfigEntry], str | None] = lambda _entry: None


def settings_flow(form: SettingsForm) -> type[ConfigSubentryFlow]:
    """The dialog that adds the subentry once, and changes its settings."""

    class SettingsFlow(ConfigSubentryFlow):
        async def async_step_user(self, user_input: dict | None = None) -> SubentryFlowResult:
            entry = self._get_entry()
            if (reason := form.missing(entry)) is not None:
                return self.async_abort(reason=reason)
            if entry.get_subentries_of_type(form.subentry_type):
                return self.async_abort(reason="already_configured")
            return await self._form(user_input)

        async def async_step_reconfigure(self, user_input: dict | None = None) -> SubentryFlowResult:
            return await self._form(user_input)

        async def _form(self, user_input: dict | None) -> SubentryFlowResult:
            reconfigure = self.source == SOURCE_RECONFIGURE
            step_id = "reconfigure" if reconfigure else "user"
            if user_input is None:
                current = dict(self._get_reconfigure_subentry().data) if reconfigure else {}
                return self.async_show_form(step_id=step_id, data_schema=probatio.Schema(form.fields(current)))
            if error := await form.check(self.hass, user_input):
                return self.async_show_form(
                    step_id=step_id,
                    data_schema=probatio.Schema(form.fields(user_input)),
                    errors={"base": "cannot_connect"},
                    description_placeholders={"error": error},
                )
            if reconfigure:
                return self.async_update_and_abort(self._get_entry(), self._get_reconfigure_subentry(), data=user_input)
            return self.async_create_entry(title=form.title, data=user_input)

    return SettingsFlow
