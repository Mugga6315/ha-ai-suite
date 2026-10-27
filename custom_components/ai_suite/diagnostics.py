"""Diagnostics: what the model can actually see, without touching log levels."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import AiSuiteConfigEntry
from .brain.background import memory_subentry
from .brain.const import CONF_STORE_LOCATION
from .brain.store import Store
from .monitor.const import CONF_MONITOR_TOKEN
from .monitor.setup import monitor_diagnostics
from .servers import without_credentials
from .tools.emby import CONF_EMBY_API_KEY

TO_REDACT = {CONF_EMBY_API_KEY, CONF_MONITOR_TOKEN}
# A value that is one absolute URL, which may hold a user and password; prose that mentions a URL is no match.
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$")


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: AiSuiteConfigEntry) -> dict[str, Any]:
    """The servers, the options, the store's files, the compiled household rules and every subentry's settings.

    Secrets are redacted and URLs shown without their credentials. With the trace monitor's export on, also its
    counters and times (monitor/setup.py).
    """
    memory = memory_subentry(entry)
    root = memory.data[CONF_STORE_LOCATION] if memory else None
    return {
        "servers": _redacted(entry.data),
        "options": _redacted(entry.options),
        "store": await hass.async_add_executor_job(_store_info, root) if root else None,
        "compiled_rules": (
            await hass.async_add_executor_job(_compiled_rules, Store(hass, Path(root)).compiled_rules_path)
            if root else None
        ),
        "subentries": [
            {
                "type": subentry.subentry_type,
                "title": subentry.title,
                "data": _redacted(subentry.data),
            }
            for subentry in entry.subentries.values()
        ],
        "monitor": monitor_diagnostics(monitor) if (monitor := entry.runtime_data.monitor) is not None else None,
    }


def _redacted(data: Mapping[str, Any]) -> dict[str, Any]:
    """The settings with their secrets redacted and every URL in them without its user and password."""
    redacted = async_redact_data(dict(data), TO_REDACT)
    return {
        key: without_credentials(value) if isinstance(value, str) and _URL.match(value) else value
        for key, value in redacted.items()
    }


def _raise(err: OSError) -> None:
    raise err


def _store_info(root: str) -> dict[str, Any]:
    try:
        files = sorted(
            os.path.relpath(Path(dirpath) / name, root)
            for dirpath, _, names in os.walk(root, onerror=_raise)  # os.walk skips errors by default
            if ".git" not in dirpath
            for name in names
            if name.endswith(".md")
        )
    except OSError as err:
        return {"path": root, "reachable": False, "error": str(err)}
    return {"path": root, "reachable": True, "files": files}


def _compiled_rules(path: Path) -> Any:
    """rules.md as decision agents get it: a wrong compile is otherwise silent."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        return {"error": str(err)}
