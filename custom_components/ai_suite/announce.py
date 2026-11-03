"""Tell the user something later, where the request came from: on its voice satellite, else as a notification.

A message said after the turn (a timer that ends, a command that failed, a play that did not start) has no reply to
go into. On the satellite it was asked on, it is announced (`assist_satellite.announce`). From a chat, a device that
cannot announce, or an announcement that fails, it becomes a persistent notification: nobody would hear it otherwise.
"""

import asyncio
import logging

from homeassistant.components import persistent_notification
from homeassistant.components.assist_satellite import SatelliteBusyError
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

SATELLITE_DOMAIN = "assist_satellite"
# A satellite plays one announcement at a time: a second one (two timers ending together) waits this long, once.
BUSY_RETRY_S = 10
# An announcement that never ends (a satellite that stops answering) gives way to the notification.
ANNOUNCE_TIMEOUT_S = 60


async def async_tell(
    hass: HomeAssistant, device_id: str | None, message: str, *, title: str, notification_id: str | None = None
) -> None:
    """Say `message` on the satellite of `device_id`; a notification when there is none or the announcement fails."""
    if (satellite := _satellite(hass, device_id)) is not None:
        try:
            async with asyncio.timeout(ANNOUNCE_TIMEOUT_S):
                await _async_announce(hass, satellite, message)
        except (HomeAssistantError, TimeoutError) as err:
            _LOGGER.warning("Could not announce on %s, notifying instead: %s", satellite, err or type(err).__name__)
        except Exception:  # a bug in the satellite's integration: the message must still reach the user
            _LOGGER.warning("Could not announce on %s, notifying instead", satellite, exc_info=True)
        else:
            return
    persistent_notification.async_create(hass, message, title=title, notification_id=notification_id)


async def _async_announce(hass: HomeAssistant, satellite: str, message: str) -> None:
    """Announce `message`; once more after BUSY_RETRY_S when the satellite is still saying something else."""
    try:
        await _async_call_announce(hass, satellite, message)
    except SatelliteBusyError:
        await asyncio.sleep(BUSY_RETRY_S)
        await _async_call_announce(hass, satellite, message)


async def _async_call_announce(hass: HomeAssistant, satellite: str, message: str) -> None:
    await hass.services.async_call(
        SATELLITE_DOMAIN, "announce", {"message": message}, target={"entity_id": satellite}, blocking=True
    )


def _satellite(hass: HomeAssistant, device_id: str | None) -> str | None:
    """The voice satellite entity of the device a request came from, if it is one and reachable.

    HA leaves an unavailable entity out of a service call without an error: the announcement would go nowhere.
    """
    if device_id is None:
        return None
    entries = er.async_entries_for_device(er.async_get(hass), device_id)
    satellite = next((entry.entity_id for entry in entries if entry.domain == SATELLITE_DOMAIN), None)
    state = hass.states.get(satellite) if satellite else None
    return satellite if state is not None and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN) else None
