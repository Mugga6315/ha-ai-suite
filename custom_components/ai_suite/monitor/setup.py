"""The exporter of a loaded entry, set up when the entry's options turn it on: nothing exists while they do not.

On: one Recorder for the entry's turns, the effects listener (removed on unload), and the one sender task (the
entry's background task, cancelled on unload). The options flow's change reloads the entry, so a changed URL, token
or audio switch comes with a new exporter.
"""

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.loader import async_get_integration

from ..const import DOMAIN
from .const import (
    CONF_MONITOR_AUDIO,
    CONF_MONITOR_ENABLED,
    CONF_MONITOR_TOKEN,
    CONF_MONITOR_URL,
    TIMES_KEPT,
)
from .effects import async_listen_effects
from .exporter import Exporter
from .recorder import Recorder, release


@dataclass(slots=True)
class Monitor:
    """The exporter's parts of a loaded entry, and the times its diagnostics show."""

    hass: HomeAssistant
    entry: ConfigEntry  # owns each turn's closing task and the sender
    recorder: Recorder
    exporter: Exporter
    audio: bool  # the turns' STT audio is sent (CONF_MONITOR_AUDIO)
    ai_suite_version: str
    hook_ns: deque[int] = field(default_factory=lambda: deque(maxlen=TIMES_KEPT))  # per turn, all its hooks
    build_ns: deque[int] = field(default_factory=lambda: deque(maxlen=TIMES_KEPT))
    build_failed: int = 0


async def async_setup_monitor(hass: HomeAssistant, entry: ConfigEntry) -> Monitor | None:
    """The entry's exporter, listening and sending; None when its options leave the export off."""
    options = entry.options
    if not options.get(CONF_MONITOR_ENABLED, False):
        return None
    integration = await async_get_integration(hass, DOMAIN)
    recorder = Recorder()
    exporter = Exporter(
        hass, async_get_clientsession(hass), options.get(CONF_MONITOR_URL, ""), options.get(CONF_MONITOR_TOKEN, "")
    )
    entry.async_on_unload(async_listen_effects(hass, recorder))
    entry.async_on_unload(lambda: _release_all(recorder))
    entry.async_create_background_task(hass, exporter.run(), "ai_suite monitor sender")
    return Monitor(
        hass=hass,
        entry=entry,
        recorder=recorder,
        exporter=exporter,
        audio=options.get(CONF_MONITOR_AUDIO, False),
        ai_suite_version=str(integration.version),
    )


def monitor_diagnostics(monitor: Monitor) -> dict[str, Any]:
    """The turns recorded and skipped, the sender's counters, and the hook and build times in ms (p50, max)."""
    return {
        "recorded": monitor.recorder.recorded,
        "skipped": monitor.recorder.skipped,
        "held": len(monitor.recorder.turns),
        "build_failed": monitor.build_failed,
        "hook_ms": _spread(monitor.hook_ns),
        "build_ms": _spread(monitor.build_ns),
        **monitor.exporter.diagnostics(),
    }


def _release_all(recorder: Recorder) -> None:
    """At unload: no turn is held any more, neither one still running nor one waiting to be built."""
    for trace in list(recorder.turns.values()):
        release(recorder, trace)


def _spread(times_ns: deque[int]) -> dict[str, Any]:
    """The p50 and max of the latest turns' times, in ms; null before the first turn."""
    if not times_ns:
        return {"turns": 0, "p50": None, "max": None}
    return {"turns": len(times_ns), "p50": round(median(times_ns) / 1e6, 3), "max": round(max(times_ns) / 1e6, 3)}
