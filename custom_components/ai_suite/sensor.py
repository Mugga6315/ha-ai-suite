"""The Timers sensor: every running timer and command for later, for dashboards and wall displays.

Its state is how many there are; the attribute `entries` lists each, the next one first: id, kind (timer or command),
name, when it was set and when it ends (ISO, UTC), its length in seconds, the area of the device it was set on, and a
command's tool and arguments. A display counts down from `ends_at` itself: the sensor changes only when an entry is
added, cancelled or done. The list is left out of the recorder's history.
"""

from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import AiSuiteConfigEntry
from .agent.schedule import TIMER, Schedule, ScheduledEntry
from .entity import entry_device
from .home import satellite_area

# Only the schedule changes the sensor.
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant, entry: AiSuiteConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """The entry's one Timers sensor."""
    async_add_entities([TimersSensor(entry, entry.runtime_data.schedule)])


class TimersSensor(SensorEntity):
    """How many timers and commands for later are running, and each of them."""

    _attr_has_entity_name = True
    _attr_translation_key = "timers"
    _attr_should_poll = False
    _unrecorded_attributes = frozenset({"entries"})

    def __init__(self, entry: AiSuiteConfigEntry, schedule: Schedule) -> None:
        """Show the entries of `schedule`."""
        self._schedule = schedule
        self._attr_unique_id = f"{entry.entry_id}_timers"
        self._attr_device_info = entry_device(entry)

    async def async_added_to_hass(self) -> None:
        """Follow the schedule's changes."""
        self.async_on_remove(self._schedule.async_listen(self.async_write_ha_state))

    @property
    def native_value(self) -> int:
        """How many entries are running."""
        return len(self._schedule.pending())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Each entry, the next one first."""
        return {"entries": [_shown(self.hass, command) for command in self._schedule.pending()]}


def _shown(hass: HomeAssistant, command: ScheduledEntry) -> dict[str, Any]:
    shown: dict[str, Any] = {
        "id": command.id,
        "kind": command.kind,
        "name": command.name,
        "started_at": command.created.isoformat(),
        "ends_at": command.run_at.isoformat(),
        "seconds": round((command.run_at - command.created).total_seconds()),
        "area": satellite_area(hass, command.device_id),
    }
    if command.kind != TIMER:
        shown |= {"tool": command.tool, "arguments": dict(command.arguments)}
    return shown
