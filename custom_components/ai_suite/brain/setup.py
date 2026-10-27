"""Second Brain inside AI Suite: the store, its LLM API and its background jobs, set up from the Memory subentry.

Settings come from the Memory subentry. A store that cannot be used turns memory off alone (a repair issue says why),
so the conversation agents of the same entry keep working. An offline store is looked for again when a turn needs
memory (waiting.py) instead of having HA retry the whole entry.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from homeassistant.components.conversation.chat_log import async_subscribe_chat_logs
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import issue_registry as ir, llm
from homeassistant.helpers.event import async_track_time_change

from ..const import DOMAIN
from .analyzer import TurnAnalyzer
from .background import RunFailed, background_client, memory_subentry, subentry_of
from .consolidator import Consolidator
from .const import (
    BRAIN_API_ID,
    CONF_CONSOLIDATE_EFFORT,
    CONF_CONSOLIDATE_ENABLED,
    CONF_CONSOLIDATE_TIME,
    CONF_LEARN_TIME,
    CONF_SELF_IMPROVE_EFFORT,
    CONF_STORE_LOCATION,
    DEFAULT_CONSOLIDATE_TIME,
    DEFAULT_LEARN_TIME,
    DEFAULT_SELF_IMPROVE_EFFORT,
    LOGGER,
    SUBENTRY_LIBRARIAN,
    SUBENTRY_SELF_IMPROVE,
)
from .learner import RuleLearner
from .llm_api import BrainAPI
from .store import Store
from .store_location import unsafe_store_location
from .waiting import WaitingBrainAPI

STORE_ISSUE = "store_unavailable"
STORE_OFFLINE_ISSUE = "store_offline"
API_TAKEN_ISSUE = "memory_api_taken"


@dataclass(slots=True)
class Brain:
    """Memory as set up for the entry: its LLM API, its store and the jobs its features add."""

    api: BrainAPI | WaitingBrainAPI
    store: Store
    consolidator: Consolidator | None = None
    learner: RuleLearner | None = None


async def async_setup_brain(hass: HomeAssistant, entry: ConfigEntry) -> Brain | None:
    """Set up memory when the Memory subentry exists; None without it or when its store cannot be used."""
    memory = memory_subentry(entry)
    if memory is None:
        for issue_id in (STORE_ISSUE, STORE_OFFLINE_ISSUE, API_TAKEN_ISSUE):
            ir.async_delete_issue(hass, DOMAIN, issue_id)
        return None
    root = await hass.async_add_executor_job(_resolved, memory.data[CONF_STORE_LOCATION])
    store = Store(hass, root)
    if (unsafe := await hass.async_add_executor_job(unsafe_store_location, hass, str(root))) is not None:
        LOGGER.warning("memory is off: %s", unsafe)
        _raise_issue(hass, STORE_ISSUE, problem=unsafe)
        return None
    ir.async_delete_issue(hass, DOMAIN, STORE_ISSUE)
    if await _offline(memory, store):
        LOGGER.warning("memory is waiting: store not found at %s (network share offline?)", root)
        _raise_issue(hass, STORE_OFFLINE_ISSUE, path=str(root))
        waiting = WaitingBrainAPI(hass, store, entry)
        return Brain(waiting, store) if _register(hass, entry, waiting) else None
    ir.async_delete_issue(hass, DOMAIN, STORE_OFFLINE_ISSUE)

    await store.async_setup()
    _record_store_initialized(hass, entry, memory, store)

    api = BrainAPI(hass, store, entry)
    if not _register(hass, entry, api):
        return None
    @callback
    def _compile_ahead() -> None:
        entry.async_create_background_task(hass, api.async_compile_ahead(), "ai_suite_compile_rules")

    entry.async_on_unload(store.listen_rules_changed(_compile_ahead))
    brain = Brain(api, store, consolidator=_setup_consolidator(hass, entry, store))
    brain.learner = _setup_self_improvement(hass, entry, store)
    _check_setup_gaps(hass)
    return brain


def _resolved(path: str) -> Path:
    """The store's folder with links followed; this touches the file system, so it runs in the executor."""
    return Path(path).resolve()


async def _offline(memory: ConfigSubentry, store: Store) -> bool:
    """The store was set up here before and is gone now: a share offline, not an empty folder to seed.

    Moving the store to a new location is a deliberate act and is allowed to seed; "initialized_path" is where it was
    last set up.
    """
    return (
        bool(memory.data.get("initialized"))
        and str(store.root) == memory.data.get("initialized_path")
        and not await store.async_exists()
    )


def _register(hass: HomeAssistant, entry: ConfigEntry, api: llm.API) -> bool:
    """Serve memory's LLM API; False when another integration already serves its id."""
    try:
        entry.async_on_unload(llm.async_register_api(hass, api))
    except HomeAssistantError as err:
        LOGGER.error("memory is off: %s", err)
        _raise_issue(hass, API_TAKEN_ISSUE, api_id=api.id)
        return False
    ir.async_delete_issue(hass, DOMAIN, API_TAKEN_ISSUE)
    return True


def _raise_issue(hass: HomeAssistant, issue_id: str, **placeholders: str) -> None:
    ir.async_create_issue(
        hass, DOMAIN, issue_id, is_fixable=False, severity=ir.IssueSeverity.ERROR, translation_key=issue_id,
        translation_placeholders=placeholders,
    )


def _record_store_initialized(hass: HomeAssistant, entry: ConfigEntry, memory: ConfigSubentry, store: Store) -> None:
    if memory.data.get("initialized") and memory.data.get("initialized_path") == str(store.root):
        return
    hass.config_entries.async_update_subentry(
        entry, memory, data={**memory.data, "initialized": True, "initialized_path": str(store.root)}
    )


def _check_setup_gaps(hass: HomeAssistant) -> None:
    """Raise a repair issue while no conversation agent uses memory.

    Memory then loads and does nothing; it is one click from working and produces no error, so a repair issue says it.
    """
    # No conversation agent has ticked "Second Brain": the API is registered and never asked for a tool.
    selected = any(
        BRAIN_API_ID in (sub.data.get("llm_hass_api") or [])
        for other in hass.config_entries.async_entries()
        for sub in other.subentries.values()
    ) or any(BRAIN_API_ID in (other.options.get("llm_hass_api") or []) for other in hass.config_entries.async_entries())
    _set_issue(hass, "no_agent_selected", not selected)


def _set_issue(hass: HomeAssistant, issue_id: str, present: bool) -> None:
    if present:
        ir.async_create_issue(
            hass, DOMAIN, issue_id, is_fixable=False, severity=ir.IssueSeverity.WARNING, translation_key=issue_id
        )
    else:
        ir.async_delete_issue(hass, DOMAIN, issue_id)


def _hour_minute(value: str) -> tuple[int, int]:
    """Split a TimeSelector value ("HH:MM:SS") into (hour, minute)."""
    parts = value.split(":")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def _setup_self_improvement(hass: HomeAssistant, entry: ConfigEntry, store: Store) -> RuleLearner | None:
    """The per-turn review and the nightly pass that turns its notes into rules, if the feature is added.

    Its own subentry, not the librarian's: the librarian is one LLM call a night, the review one per turn, so they
    are worth adding and removing separately. The pass owns failures.md end to end, so it lives here too.
    """
    subentry = subentry_of(entry, SUBENTRY_SELF_IMPROVE)
    if subentry is None or (client := background_client(hass, entry)) is None:
        return None
    effort = subentry.data.get(CONF_SELF_IMPROVE_EFFORT, DEFAULT_SELF_IMPROVE_EFFORT)
    analyzer = TurnAnalyzer(hass, entry, store, client, effort)
    entry.async_on_unload(async_subscribe_chat_logs(hass, analyzer.on_chat_log))
    entry.async_on_unload(analyzer.async_shutdown)
    learner = RuleLearner(hass, store, client, effort)
    _schedule(hass, entry, "learner", learner.async_run, subentry.data.get(CONF_LEARN_TIME, DEFAULT_LEARN_TIME))
    return learner


def _setup_consolidator(hass: HomeAssistant, entry: ConfigEntry, store: Store) -> Consolidator | None:
    """The librarian, if the feature is added; its nightly run only when switched on (the action works either way)."""
    subentry = subentry_of(entry, SUBENTRY_LIBRARIAN)
    if subentry is None or (client := background_client(hass, entry)) is None:
        return None
    consolidator = Consolidator(
        hass, store, client, subentry.data.get(CONF_CONSOLIDATE_EFFORT, DEFAULT_SELF_IMPROVE_EFFORT)
    )
    if subentry.data.get(CONF_CONSOLIDATE_ENABLED, True):
        at = subentry.data.get(CONF_CONSOLIDATE_TIME, DEFAULT_CONSOLIDATE_TIME)
        _schedule(hass, entry, "librarian", consolidator.async_run, at)
    return consolidator


def _schedule(
    hass: HomeAssistant, entry: ConfigEntry, name: str, run: Callable[[], Coroutine[Any, Any, str]], at: str
) -> None:
    """Run the job daily at `at`, as a task of the entry, so an unload cancels a run still going."""
    hour, minute = _hour_minute(at)

    @callback
    def _start(_now: datetime) -> None:
        entry.async_create_background_task(hass, _scheduled_run(name, run), f"ai_suite_{name}")

    entry.async_on_unload(async_track_time_change(hass, _start, hour=hour, minute=minute, second=0))


async def _scheduled_run(name: str, run: Callable[[], Coroutine[Any, Any, str]]) -> None:
    try:
        summary = await run()
    except RunFailed as err:
        LOGGER.warning("%s changed nothing: %s", name, err)
        return
    LOGGER.info("%s: %s", name, summary)
