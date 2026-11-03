"""A timer that ends: said on the voice satellite it was set on, else a notification in Home Assistant."""

from homeassistant.core import HomeAssistant

from ..announce import async_tell
from ..const import DOMAIN
from ..languages import language_of_tag


async def async_ring(
    hass: HomeAssistant, timer_id: str, name: str, language: str | None, device_id: str | None
) -> None:
    """Say that the timer `name` is done where it was set."""
    speech = language_of_tag(language or hass.config.language).replies.speech
    message = speech["timer_done_named"].format(name=name) if name else speech["timer_done"]
    await async_tell(
        hass, device_id, message, title=speech["timer_title"], notification_id=f"{DOMAIN}_timer_{timer_id}"
    )
