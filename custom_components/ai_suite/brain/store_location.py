"""Where the store may live.

The store seeds files and runs `git init` in whatever folder it is given, so a
folder Home Assistant owns is not a valid choice: pointed at the config dir it
commits `.storage/auth` and `secrets.yaml` into a git repo. Both the config
flow and setup check through here.
"""
import os
from pathlib import Path

from .const import STORE_FOLDER


def default_store_location(hass) -> str:
    """The store's own folder next to the HA config — created on first setup.

    Not inside `custom_components`: an integration update replaces the
    integration's directory, and the store would go with it.
    """
    return hass.config.path(STORE_FOLDER)


def unsafe_store_location(hass, path: str) -> str | None:
    """Why this folder must not hold the store, or None when it is fine.

    Does filesystem work (realpath), so call it from the executor.
    """
    if not (path or "").strip():
        return "Pick a folder for the store."
    config_dir = os.path.realpath(hass.config.config_dir)
    root = os.path.realpath(path)
    if root == config_dir or config_dir.startswith(root + os.sep):
        return (
            f"{path} holds Home Assistant's own configuration. The store would "
            f"seed files and a git repository over it — pick a subfolder such "
            f"as {default_store_location(hass)}."
        )
    components = str(Path(config_dir) / "custom_components")
    if root == components or root.startswith(components + os.sep):
        return (
            f"{path} is inside custom_components, which is replaced on every "
            f"integration update — the store would be deleted with it."
        )
    return None
