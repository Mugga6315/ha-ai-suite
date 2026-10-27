"""What AI Suite's entities share: each agent, tool set and feature is a subentry with a device of its own."""

from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import DOMAIN


def subentry_device(subentry: ConfigSubentry) -> DeviceInfo:
    """The service device of an agent, a tool set or a feature; removing the subentry removes it."""
    return DeviceInfo(identifiers={(DOMAIN, subentry.subentry_id)}, name=subentry.title, entry_type=DeviceEntryType.SERVICE)
