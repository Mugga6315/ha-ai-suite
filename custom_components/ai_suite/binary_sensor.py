"""A status light per tool set or feature that depends on something outside AI Suite.

Emby on its server, History on the recorder, the Librarian on the LLM server. Each light is added under its subentry,
so removing the tool set or feature removes its device and light. The store and the LLM server have no light of their
own: their status is in the System health panel.
"""

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import AiSuiteConfigEntry
from .coordinator import HEALTH_TYPES, HealthCoordinator
from .entity import subentry_device

# The coordinator does all updates.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant, entry: AiSuiteConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """One status light per tool set or feature with a health probe."""
    coordinator = HealthCoordinator(hass, entry)
    for subentry in entry.subentries.values():
        if subentry.subentry_type in HEALTH_TYPES:
            async_add_entities([HealthLight(coordinator, subentry)], config_subentry_id=subentry.subentry_id)
    # In the background: a slow or unreachable service must not delay the entry's setup. The lights read
    # unavailable until the first probe returns.
    entry.async_create_background_task(hass, coordinator.async_refresh(), "ai_suite_health_first_refresh")


class HealthLight(CoordinatorEntity[HealthCoordinator], BinarySensorEntity):
    """On while the dependency of a tool set or feature answers."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True
    _attr_translation_key = "status"

    def __init__(self, coordinator: HealthCoordinator, subentry: ConfigSubentry) -> None:
        """Show the health of what `subentry` adds."""
        super().__init__(coordinator)
        self._subentry_id = subentry.subentry_id
        self._attr_unique_id = f"{subentry.subentry_id}_status"
        self._attr_device_info = subentry_device(subentry)

    @property
    def available(self) -> bool:
        """Unavailable until the first probe."""
        return super().available and self._subentry_id in (self.coordinator.data or {})

    @property
    def is_on(self) -> bool:
        """Whether the dependency answered the last probe."""
        return bool((self.coordinator.data or {}).get(self._subentry_id))
