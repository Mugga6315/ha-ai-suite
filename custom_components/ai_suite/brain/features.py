"""Memory's features: the Librarian and Self-improvement, each a subentry a home adds once Memory is there.

Both work on the store, so neither can be added without Memory.
"""

from homeassistant.config_entries import ConfigEntry, ConfigSubentryFlow

from ..settings_flow import SettingsForm, settings_flow
from . import analyzer, consolidator
from .background import memory_subentry
from .const import SUBENTRY_LIBRARIAN, SUBENTRY_SELF_IMPROVE


def _memory_missing(entry: ConfigEntry) -> str | None:
    return "memory_needed" if memory_subentry(entry) is None else None


FEATURES = (
    SettingsForm(
        subentry_type=SUBENTRY_LIBRARIAN,
        title="Librarian",
        fields=consolidator.subentry_schema,
        check=consolidator.async_validate,
        missing=_memory_missing,
    ),
    SettingsForm(
        subentry_type=SUBENTRY_SELF_IMPROVE,
        title="Self-improvement",
        fields=analyzer.subentry_schema,
        check=analyzer.async_validate,
        missing=_memory_missing,
    ),
)


def subentry_flows() -> dict[str, type[ConfigSubentryFlow]]:
    """The add and change dialogs of Memory's features, by subentry type."""
    return {form.subentry_type: settings_flow(form) for form in FEATURES}
