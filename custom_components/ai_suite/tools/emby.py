"""Emby: list the Emby (or Jellyfin) library and start playback on a player, waking an Android TV player if needed."""
import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
import ipaddress
import logging
from typing import Any

import aiohttp
import probatio

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, llm, selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.hass_dict import HassKey

from ..announce import async_tell
from ..languages import language_of_tag
from .spec import (
    READ_ONLY_REMOTE,
    RecordFailure,
    ToolArgs,
    ToolError,
    ToolSpec,
    spec_tools,
)

CONF_EMBY_URL = "emby_url"
CONF_EMBY_API_KEY = "emby_api_key"
CONF_WAKE_ENTITIES = "wake_entities"

_LOGGER = logging.getLogger(__name__)

_STATUS_FILTER = {
    "unwatched": "IsUnplayed",
    "watched": "IsPlayed",
    "in_progress": "IsResumable",
}
# (playstate filter, sort field) per Emby's "Browsing the Library" wiki; "watching" sorts like Kodi's resume row.
_RECENT = {
    "watching": ("IsResumable", "DatePlayed"),
    "played": ("IsPlayed", "DatePlayed"),
    "added": (None, "DateCreated"),
}

# What is resumable per item_type: a series is never started, its episodes are.
_RESUME_TYPE_MAP = {"movie": "Movie", "series": "Episode", "all": "Movie,Episode"}

# Items per list: enough for an overview without flooding the prompt.
SEARCH_LIMIT = 25
# Ceiling for full=true, so one request cannot blow the model's context on a huge library.
FULL_LIMIT = 1000

_TYPE_MAP = {"movie": "Movie", "series": "Series", "all": "Movie,Series"}

# How long a launched player gets to show its Emby session, and the pause between reads: Kodi's add-on takes seconds.
WAKE_TIMEOUT_S = 20
WAKE_POLL_S = 1

# The Emby client on Android TV: Kodi with the Emby add-on. play_emby starts no other app.
WAKE_DEFAULT_APP = "org.xbmc.kodi"
# The word the app's Emby add-on puts in its sessions' Client ("Emby for Kodi Next Gen").
WAKE_CLIENT = "kodi"
# KEYCODE_WAKEUP: a sleeping device keeps the app off Emby. Not the library's select_source: its focused-app read
# fails silently on Android versions it has no variant for (e.g. 15), and nothing starts.
WAKE_COMMAND = (
    "input keyevent KEYCODE_WAKEUP; "
    "monkey -p {app} -c android.intent.category.LEANBACK_LAUNCHER --pct-syskeys 0 1 >/dev/null 2>&1"
)


# The background jobs still running, by name: a new job under a name replaces the old one (Emby.run_in_background).
_BACKGROUND_JOBS: HassKey[dict[str, asyncio.Task]] = HassKey("ai_suite_emby_background_jobs")

# ValueError: a body that is no JSON, or holds no user (async_user_id).
EMBY_ERRORS = (aiohttp.ClientError, TimeoutError, ValueError)

_CONTROL = {"pause": "Pause", "resume": "Unpause", "stop": "Stop"}
_STATE_AFTER = {"pause": "paused", "resume": "playing", "stop": "stopped"}
# Reads of /Sessions and the seconds between them: a player gets 5 s to show a command's state.
CONTROL_READS = 10
CONTROL_POLL_S = 0.5


class EmbyClient:
    """Minimal Emby REST client: search the library, list sessions, start play."""

    def __init__(self, hass, url: str, api_key: str) -> None:
        """Talk to the Emby (or Jellyfin) server at `url` with `api_key`."""
        self._hass = hass
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._user_id: str | None = None

    @property
    def available(self) -> bool:
        """Whether both the URL and the API key are set."""
        return bool(self._url and self._api_key)

    async def async_user_id(self) -> str:
        """The Emby user id, for the per-user playstate calls (cached).

        Assumes a single Emby user: the first the server reports. Raises ValueError when it reports none.
        """
        if self._user_id:
            return self._user_id
        data = await self._get("/Users", {})
        for user in data if isinstance(data, list) else []:
            if isinstance(user, dict) and (user_id := user.get("Id")):
                self._user_id = user_id
                return user_id
        raise ValueError("Emby reported no users, so playstate is unavailable.")

    async def _user_items(self, params: dict) -> object:
        """GET the single user's /Items — the playstate-scoped library view."""
        user_id = await self.async_user_id()
        return await self._get(f"/Users/{user_id}/Items", params)

    def _headers(self) -> dict:
        # X-Emby-Token is the server-wide API-key auth for both Emby and Jellyfin.
        return {"X-Emby-Token": self._api_key, "Accept": "application/json"}

    async def _get(self, path: str, params: dict) -> object:
        session = async_get_clientsession(self._hass)
        async with session.get(
            f"{self._url}{path}",
            headers=self._headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _library_params(self, item_type: str, genre: str) -> dict:
        """The /Items filters shared by search and count."""
        params = {
            "IncludeItemTypes": _TYPE_MAP.get(item_type, _TYPE_MAP["all"]),
            "Recursive": "true",
        }
        if genre:
            params["Genres"] = genre  # Emby matches genre by name, pipe-separated
        return params

    async def async_count(self, item_type: str, genre: str) -> int:
        """Total items matching the type/genre — the answer to a 'how many'."""
        # One row is enough: Emby's TotalRecordCount counts the whole match regardless of Limit.
        params = {**self._library_params(item_type, genre), "Limit": 1}
        data = await self._get("/Items", params)
        if not isinstance(data, dict):
            return 0
        return data.get("TotalRecordCount", len(data.get("Items", [])))

    async def async_search(
        self, query: str, item_type: str, genre: str, limit: int, status: str = "all"
    ) -> tuple[list[dict], int]:
        """Return (items capped at `limit`, total matching in the library), each item with its watch state.

        An empty query lists all items of the type/genre; a status other than "all" filters by watch state.
        """
        params = {
            **self._library_params(item_type, genre),
            "Limit": limit,
            "Fields": "ProductionYear,UserData",
        }
        if query:
            params["SearchTerm"] = query
        if status != "all":
            params["Filters"] = _STATUS_FILTER[status]
        data = await self._user_items(params)
        if not isinstance(data, dict):
            return [], 0
        items = data.get("Items", [])
        return items, data.get("TotalRecordCount", len(items))

    async def async_recent(
        self, kind: str, item_type: str, limit: int
    ) -> list[dict]:
        """The user's items by recency, newest first: continue-watching, recently played, or added."""
        playstate_filter, sort_by = _RECENT[kind]
        params = {
            **self._library_params(item_type, ""),
            "Limit": limit,
            "Fields": "ProductionYear,UserData",
            "SortBy": sort_by,
            "SortOrder": "Descending",
        }
        if kind == "watching":
            params["IncludeItemTypes"] = _RESUME_TYPE_MAP.get(
                item_type, _RESUME_TYPE_MAP["all"]
            )
        if playstate_filter:
            params["Filters"] = playstate_filter
        data = await self._user_items(params)
        return data.get("Items", []) if isinstance(data, dict) else []

    async def async_now_playing(self) -> list[dict]:
        """Sessions currently playing something (each carries NowPlayingItem)."""
        return [s for s in await self.async_sessions() if s.get("NowPlayingItem")]

    async def _find_series_id(self, series: str) -> str | None:
        data = await self._get(
            "/Items",
            {
                "IncludeItemTypes": "Series",
                "Recursive": "true",
                "SearchTerm": series,
                "Limit": 1,
            },
        )
        items = data.get("Items", []) if isinstance(data, dict) else []
        return items[0].get("Id") if items else None

    async def async_next_up(self, series: str) -> list[dict]:
        """Next unwatched episodes — for one series if given, else across all."""
        # Emby 4.6 merged next-up into continue watching: without LegacyNextUp a query over all series is empty.
        # Jellyfin ignores the flag.
        params = {"UserId": await self.async_user_id(), "Limit": SEARCH_LIMIT, "LegacyNextUp": "true"}
        if series:
            series_id = await self._find_series_id(series)
            if not series_id:
                return []
            params["SeriesId"] = series_id
        data = await self._get("/Shows/NextUp", params)
        return data.get("Items", []) if isinstance(data, dict) else []

    async def async_sessions(self) -> list[dict]:
        """The server's connected sessions (players)."""
        data = await self._get("/Sessions", {})
        return data if isinstance(data, list) else []

    async def async_play(self, session_id: str, item_id: str) -> None:
        """Start the item on the session now."""
        await self._post(f"/Sessions/{session_id}/Playing", {"ItemIds": item_id, "PlayCommand": "PlayNow"})

    async def async_playstate(self, session_id: str, command: str) -> None:
        """Send the session a playstate command (Pause, Unpause, Stop); Emby accepts it whether the player obeys."""
        await self._post(f"/Sessions/{session_id}/Playing/{command}", {})

    async def _post(self, path: str, params: dict) -> None:
        session = async_get_clientsession(self._hass)
        async with session.post(
            f"{self._url}{path}",
            headers=self._headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()


_LABEL = {"movie": "movies", "series": "series", "all": "items"}


def _scope(item_type: str, genre: str = "", query: str = "") -> str:
    """Human phrase for what was asked."""
    label = _LABEL.get(item_type, "items")
    if genre:
        label = f"{genre} {label}"
    if query:
        label += f" matching '{query}'"
    return label


def _playstate(item: dict) -> str:
    """Watched / percent / in-progress / unwatched; empty when the item carries no UserData."""
    if "UserData" not in item:
        return ""  # a server-scoped call carries no playstate
    user_data = item["UserData"] or {}
    if user_data.get("Played"):
        return " — watched"
    percent = user_data.get("PlayedPercentage")
    if isinstance(percent, (int, float)) and percent > 0:
        return f" — {round(percent)}% watched"
    if user_data.get("PlaybackPositionTicks"):
        return " — in progress"
    return " — unwatched"


def _item_title(item: dict) -> str:
    """Readable title: episodes carry their series and SxxExx, movies their year."""
    name = item.get("Name", "(untitled)")
    series = item.get("SeriesName")
    season = item.get("ParentIndexNumber")
    episode = item.get("IndexNumber")
    if series and season is not None and episode is not None:
        return f"{series} S{season:02d}E{episode:02d} — {name}"
    year = item.get("ProductionYear")
    return f"{name} ({year})" if year else name


def _format_items(items: list[dict]) -> str:
    """One line per library item, ending in the id the play tool needs."""
    lines = []
    for it in items:
        kind = it.get("Type", "")
        lines.append(
            f"- {_item_title(it)} [{kind}]{_playstate(it)} — id: {it.get('Id', '')}"
        )
    return "\n".join(lines)


def _controllable(sessions: list[dict]) -> list[dict]:
    """Sessions that can actually be told to play (a player must be connected)."""
    return [s for s in sessions if s.get("SupportsRemoteControl")]


def _session_name(s: dict) -> str:
    parts = [s.get("DeviceName"), s.get("Client")]
    return " / ".join(p for p in parts if p) or s.get("Id", "?")


def _match_session(sessions: list[dict], player: str) -> dict | None:
    """First controllable session whose device name or client contains `player`, in any case."""
    needle = player.strip().lower()
    for s in _controllable(sessions):
        if needle in (s.get("DeviceName") or "").lower() or needle in (
            s.get("Client") or ""
        ).lower():
            return s
    return None


def _session_on_host(sessions: list[dict], host: str | None) -> dict | None:
    """The controllable session connecting from `host`, the device's ADB address: its player, whatever its name."""
    if (address := _address(host)) is None:
        return None
    for s in _controllable(sessions):
        if _address(s.get("RemoteEndPoint")) == address:
            return s
    return None


def _address(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The IP address in `text`, an IPv4-mapped IPv6 one as IPv4; None for a name or nothing."""
    try:
        address = ipaddress.ip_address(text or "")
    except ValueError:
        return None
    return getattr(address, "ipv4_mapped", None) or address


def _entity_host(hass, entity_id: str) -> str | None:
    """The ADB host of an androidtv entity's config entry, or None if unknown."""
    # An unknown host only disables the IP match, not the wake.
    entry = er.async_get(hass).async_get(entity_id)
    if entry is None or entry.config_entry_id is None:
        return None
    config_entry = hass.config_entries.async_get_entry(entry.config_entry_id)
    return config_entry.data.get("host") if config_entry else None


async def _launch(hass: HomeAssistant, wake: Wake) -> None:
    """Wake the device and start the app on it (WAKE_COMMAND) through the androidtv integration."""
    await hass.services.async_call(
        "androidtv",
        "adb_command",
        {"entity_id": wake.entity_id, "command": WAKE_COMMAND.format(app=wake.app)},
        blocking=True,
    )


async def _connected(client: EmbyClient, wake: Wake, before: set[str]) -> dict | None:
    """The launched player's session, read every WAKE_POLL_S for WAKE_TIMEOUT_S; None when it does not come."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAKE_TIMEOUT_S
    fresh: list[dict] = []
    while loop.time() < deadline:
        await asyncio.sleep(WAKE_POLL_S)
        try:
            sessions = await client.async_sessions()
        except EMBY_ERRORS:
            continue  # server hiccup mid-wait — keep polling until the deadline
        if (on_host := _session_on_host(sessions, wake.host)) is not None:
            return on_host
        fresh = [s for s in _controllable(sessions) if s.get("Id") not in before]
        if fresh and _address(wake.host) is None:
            return fresh[0]
    # Behind a proxy Emby sees the proxy's address: then the one new session of the launched app is the player.
    launched = [s for s in fresh if WAKE_CLIENT in (s.get("Client") or "").lower()]
    return launched[0] if len(launched) == 1 else None


async def _play_when_connected(
    hass: HomeAssistant, client: EmbyClient, wake: Wake, item_id: str, before: set[str], context: llm.LLMContext | None
) -> None:
    """Start the item once the launched player connects; tell the user when it does not."""
    try:
        if (session := await _connected(client, wake, before)) is not None:
            await client.async_play(session["Id"], item_id)
            return
        _LOGGER.warning("emby: %s did not connect to Emby within %s s", wake.entity_id, WAKE_TIMEOUT_S)
    except EMBY_ERRORS as err:
        _LOGGER.warning("emby: starting the item on %s failed: %s: %s", wake.entity_id, type(err).__name__, err)
    language = language_of_tag(context.language if context and context.language else "")
    await _tell(hass, context, language.replies.speech["playback_not_started"].format(player=wake.name))


async def _tell(hass: HomeAssistant, context: llm.LLMContext | None, message: str) -> None:
    """Say `message` on the satellite the request came from; a notification when it came from elsewhere."""
    await async_tell(hass, context.device_id if context else None, message, title="AI Suite")


def _filter_candidates(
    candidates: dict[str, str], allowed: list[str] | None
) -> dict[str, str]:
    """The candidate set the user's player picker allows.

    `allowed` None: every androidtv player. A list is exclusive, so no other device gets an app started on it.
    """
    if allowed is None:
        return candidates
    return {eid: name for eid, name in candidates.items() if eid in allowed}


def _wake_candidates(hass) -> dict[str, str]:
    """The androidtv media players known to Home Assistant: {entity_id: name}."""
    if hass is None:  # tests call the tools without a hass
        return {}
    registry = er.async_get(hass)
    candidates = {}
    for entry in registry.entities.values():
        if entry.domain != "media_player" or entry.platform != "androidtv":
            continue
        state = hass.states.get(entry.entity_id)
        candidates[entry.entity_id] = (
            state.attributes.get("friendly_name", "") if state else ""
        )
    return candidates


def _match_wake_entity(
    candidates: dict[str, str], player: str
) -> tuple[str | None, list[str]]:
    """The one androidtv player the request name fits, or which ones are ambiguous.

    Several fits return (None, hits): launching on a guess would start an app on the wrong TV.
    """
    needle = player.strip().lower()
    if not needle:
        return None, []
    request_words = set(needle.split())
    hits = []
    for entity_id, name in candidates.items():
        if set(name.lower().split()) & request_words or needle in entity_id.replace("_", " "):
            hits.append(entity_id)
    if len(hits) == 1:
        return hits[0], []
    return None, hits


@dataclass(frozen=True, slots=True)
class Emby:
    """What the Emby tools work with: the server's client, and the Android TV players play_emby may start."""

    client: EmbyClient
    wake_entities: list[str] | None  # None: every androidtv player
    # Runs a job past the tool call, owned by the config entry: the start of a launched player. A job under the name
    # of one still running replaces it, so a second request for the same player wins.
    run_in_background: Callable[[Coroutine[Any, Any, None], str], object]


@dataclass(frozen=True, slots=True)
class Wake:
    """An Android TV player to start: its androidtv entity and name, the app, and the device's address if known."""

    entity_id: str
    name: str
    app: str
    host: str | None


def _failed(tool: str, message: str) -> ToolError:
    """A failed call, logged: tool results vanish into the conversation, so the HA log is where debugging happens."""
    _LOGGER.warning("emby tool %s failed: %s", tool, message)
    return ToolError(message, learn=True)


async def count_emby(_hass: HomeAssistant, emby: Emby, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Count the items of the asked type and genre."""
    item_type = args.get("item_type", "all")
    genre = args.get("genre", "")
    try:
        count = await emby.client.async_count(item_type, genre)
    except EMBY_ERRORS as err:
        raise _failed("count_emby", f"Cannot reach Emby: {type(err).__name__}: {err}") from err
    return f"{count} {_scope(item_type, genre)} in the Emby library."


async def search_emby(_hass: HomeAssistant, emby: Emby, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Search the library."""
    query = args.get("query", "")
    item_type = args.get("item_type", "all")
    genre = args.get("genre", "")
    limit = FULL_LIMIT if args.get("full") else SEARCH_LIMIT
    try:
        items, total = await emby.client.async_search(query, item_type, genre, limit, args.get("status", "all"))
    except aiohttp.ClientResponseError as err:
        raise _failed("search_emby", f"Emby returned HTTP {err.status}") from err
    except EMBY_ERRORS as err:
        raise _failed("search_emby", f"Cannot reach Emby: {type(err).__name__}: {err}") from err
    scope = _scope(item_type, genre, query)
    if not items:
        return f"No {scope} in the Emby library."
    shown = len(items)
    if total > shown:
        header = f"{total} {scope} in Emby (showing first {shown} — set full=true for all):"
    else:
        header = f"{total} {scope} in Emby:"
    return f"{header}\n{_format_items(items)}"


async def recent_emby(_hass: HomeAssistant, emby: Emby, args: ToolArgs, _context: llm.LLMContext) -> str:
    """List the recent items."""
    kind = args["kind"]
    try:
        items = await emby.client.async_recent(kind, args.get("item_type", "all"), SEARCH_LIMIT)
    except aiohttp.ClientResponseError as err:
        raise _failed("recent_emby", f"Emby returned HTTP {err.status}") from err
    except EMBY_ERRORS as err:
        raise _failed("recent_emby", f"{type(err).__name__}: {err}") from err
    if not items:
        labels = {"watching": "in progress", "played": "recently played", "added": "recently added"}
        return f"Nothing {labels[kind]} in Emby."
    return f"Emby, {kind} (newest first):\n{_format_items(items)}"


async def now_playing_emby(_hass: HomeAssistant, emby: Emby, _args: ToolArgs, _context: llm.LLMContext) -> str:
    """Say what each session plays."""
    try:
        sessions = await emby.client.async_now_playing()
    except EMBY_ERRORS as err:
        raise _failed("now_playing_emby", f"Cannot reach Emby: {type(err).__name__}: {err}") from err
    if not sessions:
        return "Nothing is playing on Emby right now."
    lines = []
    for session in sessions:
        item = session.get("NowPlayingItem") or {}
        lines.append(f"- {_session_name(session)}: {_item_title(item)} [{item.get('Type', '')}] ({_play_state(session)})")
    return "Playing now:\n" + "\n".join(lines)


async def control_emby(_hass: HomeAssistant, emby: Emby, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Pause, resume or stop what a player plays, once the player shows it."""
    command = args["command"]
    try:
        target = _controlled_session(await emby.client.async_now_playing(), args.get("player", ""))
        await emby.client.async_playstate(target["Id"], _CONTROL[command])
        shown = await _shows(emby.client, target["Id"], _STATE_AFTER[command])
    except aiohttp.ClientResponseError as err:
        raise _failed("control_emby", f"Emby returned HTTP {err.status}") from err
    except EMBY_ERRORS as err:
        raise _failed("control_emby", f"Cannot reach Emby: {type(err).__name__}: {err}") from err
    name = _session_name(target)
    if not shown:
        raise _failed("control_emby", f"Emby took the {command} command, but {name} does not show it: the player "
                                      "may not obey remote control")
    return f"{name}: {_STATE_AFTER[command]}."


def _controlled_session(playing: list[dict], player: str) -> dict:
    """The playing session the request means: the one `player` names, or the only one; a ToolError otherwise."""
    controllable = _controllable(playing)
    names = ", ".join(_session_name(s) for s in controllable) or "nothing"
    if player:
        if (session := _match_session(controllable, player)) is not None:
            return session
        raise _failed("control_emby", f"No playing Emby player matches '{player}'. Playing: {names}.")
    if len(controllable) == 1:
        return controllable[0]
    if not controllable:
        raise _failed("control_emby", "Nothing plays on an Emby player that can be controlled.")
    raise _failed("control_emby", f"Several Emby players are playing: {names}. Name one.")


async def _shows(client: EmbyClient, session_id: str, state: str) -> bool:
    """Whether the session shows `state` within CONTROL_READS reads; a session gone has stopped."""
    for _ in range(CONTROL_READS):
        await asyncio.sleep(CONTROL_POLL_S)
        try:
            sessions = {s.get("Id"): s for s in await client.async_sessions()}
        except EMBY_ERRORS:
            continue  # the command was taken: a server hiccup is no reason to report it failed
        if _play_state(sessions.get(session_id, {})) == state:
            return True
    return False


def _play_state(session: dict) -> str:
    """What a session shows: playing, paused or stopped."""
    if not session.get("NowPlayingItem"):
        return "stopped"
    return "paused" if (session.get("PlayState") or {}).get("IsPaused") else "playing"


async def next_up_emby(_hass: HomeAssistant, emby: Emby, args: ToolArgs, _context: llm.LLMContext) -> str:
    """List the next episodes."""
    series = args.get("series", "")
    try:
        items = await emby.client.async_next_up(series)
    except aiohttp.ClientResponseError as err:
        raise _failed("next_up_emby", f"Emby returned HTTP {err.status}") from err
    except EMBY_ERRORS as err:
        raise _failed("next_up_emby", f"{type(err).__name__}: {err}") from err
    if not items:
        where = f" for '{series}'" if series else ""
        return f"No next-up episodes{where}."
    return f"Next up:\n{_format_items(items)}"


async def play_emby(hass: HomeAssistant, emby: Emby, args: ToolArgs, context: llm.LLMContext) -> str:
    """Start the item on the player; a player that is not connected is started, and gets the item in the background.

    Answers once the app is launched; a start that fails is told to the user later (_play_when_connected).
    """
    item_id = args["item_id"]
    player = args["player"]
    try:
        sessions = await emby.client.async_sessions()
    except EMBY_ERRORS as err:
        raise _failed("play_emby", f"Cannot reach Emby: {type(err).__name__}: {err}") from err

    candidates = _filter_candidates(_wake_candidates(hass), emby.wake_entities)
    target = _match_session(sessions, player)
    wake = None
    if target is None and (wake := _wake(hass, candidates, player, emby.wake_entities)) is not None:
        # Emby may already be online on that device under a name the request does not use: nothing to launch.
        target = _session_on_host(sessions, wake.host)
    if target is not None:
        await _play(emby.client, target, item_id)
        return f"Playing on {_session_name(target)}."
    if wake is None:
        raise _failed("play_emby", _no_player(sessions, candidates, player))
    try:
        await _launch(hass, wake)
    except HomeAssistantError as err:
        raise _failed("play_emby", f"Starting '{wake.name}' failed: {type(err).__name__}: {err}") from err
    before = {s["Id"] for s in _controllable(sessions)}
    emby.run_in_background(
        _play_when_connected(hass, emby.client, wake, item_id, before, context), f"ai_suite_emby_start {wake.entity_id}"
    )
    return (
        f"{wake.name} is being started: the item plays as soon as its Emby player connects (within "
        f"{WAKE_TIMEOUT_S} s). If it does not connect, the user is told then."
    )


async def _play(client: EmbyClient, session: dict, item_id: str) -> None:
    try:
        await client.async_play(session["Id"], item_id)
    except aiohttp.ClientResponseError as err:
        raise _failed("play_emby", f"Emby returned HTTP {err.status} starting playback") from err
    except EMBY_ERRORS as err:
        raise _failed("play_emby", f"Playback failed: {type(err).__name__}: {err}") from err


def _wake(hass: HomeAssistant, candidates: dict[str, str], player: str, wake_entities: list[str] | None) -> Wake | None:
    """The Android TV player the request names, to start; None when none fits, a ToolError when several do."""
    target, note = _wake_target(candidates, player, wake_entities)
    if target is None:
        if note:
            raise _failed("play_emby", note)
        return None
    entity_id, app = target
    return Wake(entity_id, candidates[entity_id] or entity_id, app, _entity_host(hass, entity_id))


def _no_player(sessions: list[dict], candidates: dict[str, str], player: str) -> str:
    """Why nothing plays: the connected players, and the Android TV players that can be started."""
    names = [_session_name(s) for s in _controllable(sessions)]
    available = ", ".join(names) if names else "none are connected right now"
    message = f"No connected Emby player matches '{player}'. Available: {available}."
    # Listing the startable players keeps the model from reaching for other media players.
    if candidates:
        startable = ", ".join(name or eid for eid, name in candidates.items())
        message += (
            " If the player is an Android TV, name it by its device or room name and it will be started "
            f"automatically. Players that can be started: {startable}. Other media players cannot run Emby."
        )
    return message


def _wake_target(
    candidates: dict[str, str], player: str, wake_entities: list[str] | None
) -> tuple[tuple[str, str] | None, str]:
    """What to launch for this player request: ((entity_id, app), note).

    (None, "") when nothing fits; (None, note) when several players fit and the user must pick one.
    """
    if not candidates:
        return None, ""
    # A single configured player is the target whatever the request calls it: the user picked it for this.
    if wake_entities and len(candidates) == 1:
        return (next(iter(candidates)), WAKE_DEFAULT_APP), ""
    hit, ambiguous = _match_wake_entity(candidates, player)
    if hit is not None:
        return (hit, WAKE_DEFAULT_APP), ""
    if ambiguous:
        listed = ", ".join(f"{eid} ({candidates[eid]})" for eid in ambiguous)
        return None, f"'{player}' matches several Android TV players: {listed}. Name one of them."
    return None, ""


SPECS: tuple[ToolSpec[Emby], ...] = (
    ToolSpec(
        name="count_emby",
        description=(
            "How many movies or series are in the library — returns just the number, "
            "no titles, so use it for every 'how many' question (faster than listing). "
            "item_type='movie', 'series' or 'all' (default all). Pass genre (e.g. "
            "'Action', 'Comedy') to count one genre — this is how 'how many action "
            "films' is answered. To see the titles instead, use search_emby."
        ),
        parameters=probatio.Schema(
            {probatio.Optional("item_type"): probatio.In(list(_TYPE_MAP)), probatio.Optional("genre"): str}
        ),
        run=count_emby,
        annotations=READ_ONLY_REMOTE,
    ),
    ToolSpec(
        name="search_emby",
        description=(
            "Find or list titles in the Emby library — returns each title with its "
            "year and an id for play_emby. Give query to search by title, or leave it "
            "out to list a whole type or genre. item_type='movie', 'series' or 'all' "
            "(default all); genre filters by genre (e.g. 'Action'). status filters by "
            "watch state: 'unwatched', 'watched' or 'in_progress' (default 'all'). "
            "Each title shows its watch state. Returns "
            "the first 25; set full=true only when the user explicitly asks for the "
            "entire list. For a plain 'how many' use count_emby; for 'what did I watch "
            "recently' or 'continue watching' use recent_emby. Reads only; starts nothing."
        ),
        parameters=probatio.Schema(
            {
                probatio.Optional("query"): str,
                probatio.Optional("item_type"): probatio.In(list(_TYPE_MAP)),
                probatio.Optional("genre"): str,
                probatio.Optional("status"): probatio.In(["all", *_STATUS_FILTER]),
                probatio.Optional("full"): bool,
            }
        ),
        run=search_emby,
        annotations=READ_ONLY_REMOTE,
    ),
    ToolSpec(
        name="recent_emby",
        description=(
            "Recently-active titles for the Emby user, newest first. kind='watching' "
            "= started but not finished, ordered by when it was last started — the "
            "first entry is what the user watched last. ALWAYS use 'watching' for "
            "resume / continue / 'weiter schauen' / 'zuletzt geschaut' / 'what was I "
            "watching'. 'played' lists only FINISHED titles, never use it to resume. "
            "'added' = newly added to the library. Optional item_type "
            "('movie', 'series', 'all'). Each title shows its watch state and an id "
            "for play_emby."
        ),
        parameters=probatio.Schema(
            {
                probatio.Required("kind"): probatio.In(list(_RECENT)),
                probatio.Optional("item_type"): probatio.In(list(_TYPE_MAP)),
            }
        ),
        run=recent_emby,
        annotations=READ_ONLY_REMOTE,
    ),
    ToolSpec(
        name="now_playing_emby",
        description=(
            "What is playing on each Emby player right now — the player, the title, "
            "and whether it is paused. No arguments. Use for 'what's playing' or "
            "'what's on in Room 1'."
        ),
        parameters=probatio.Schema({}),
        run=now_playing_emby,
        annotations=READ_ONLY_REMOTE,
    ),
    ToolSpec(
        name="next_up_emby",
        description=(
            "The next unwatched episode to watch. Give series to ask about one show "
            "('what's my next episode of X'), or leave it out for next-up across all "
            "your started series. Returns episodes with an id for play_emby."
        ),
        parameters=probatio.Schema({probatio.Optional("series"): str}),
        run=next_up_emby,
        annotations=READ_ONLY_REMOTE,
    ),
    ToolSpec(
        name="play_emby",
        description=(
            "Start playing an Emby item on a player. Pass item_id (from search_emby) "
            "and player — the name of the target player as shown in Emby, or the "
            "room/device name of its Android TV (e.g. 'Room 1', 'Player 1', "
            "'TV 1'); a partial name is enough. A player that is not connected yet "
            "is started by this tool: it wakes the Android TV and launches Kodi, then "
            "answers at once, and the item starts as soon as the player connects. Only Android TV (ADB) "
            "players can be started — never try to start Emby through other media "
            "players such as receivers. If the player cannot be reached, the reply "
            "says so and names the players that can be started."
        ),
        parameters=probatio.Schema({probatio.Required("item_id"): str, probatio.Required("player"): str}),
        run=play_emby,
    ),
    ToolSpec(
        name="control_emby",
        description=(
            "Pause, resume or stop what an Emby player plays: command='pause', 'resume' or 'stop'. player: the "
            "player's name as now_playing_emby shows it, a partial name is enough; leave it out when only one "
            "player plays. The result says what the player then shows."
        ),
        parameters=probatio.Schema(
            {probatio.Required("command"): probatio.In(list(_CONTROL)), probatio.Optional("player"): str}
        ),
        run=control_emby,
    ),
)


def build(
    hass: HomeAssistant, entry: ConfigEntry, data: Mapping[str, Any], record_failure: RecordFailure | None
) -> list[llm.Tool]:
    """The Emby tools, built from the set's settings; none while the server is not set."""
    url = (data.get(CONF_EMBY_URL) or "").strip()
    api_key = (data.get(CONF_EMBY_API_KEY) or "").strip()
    if not url or not api_key:
        return []
    # An empty or missing selection means no restriction, not no wake.
    raw_wake = data.get(CONF_WAKE_ENTITIES)
    wake_entities = [raw_wake] if isinstance(raw_wake, str) else (raw_wake or None)
    def run_in_background(job: Coroutine[Any, Any, None], name: str) -> object:
        running = hass.data.setdefault(_BACKGROUND_JOBS, {})
        if (earlier := running.pop(name, None)) is not None:
            earlier.cancel()
        task = entry.async_create_background_task(hass, job, name)
        if not task.done():
            running[name] = task
            task.add_done_callback(lambda done: running.pop(name) if running.get(name) is done else None)
        return task

    return spec_tools(SPECS, Emby(EmbyClient(hass, url, api_key), wake_entities, run_in_background), record_failure)


def subentry_schema(data: dict) -> dict:
    """The fields of the Emby subentry form."""
    # suggested_value, not default: the frontend omits a cleared field, and a default would put the old value back.
    return {
        probatio.Optional(
            CONF_EMBY_URL, description={"suggested_value": data.get(CONF_EMBY_URL, "")}
        ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
        probatio.Optional(
            CONF_EMBY_API_KEY,
            description={"suggested_value": data.get(CONF_EMBY_API_KEY, "")},
        ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
        probatio.Optional(
            CONF_WAKE_ENTITIES,
            description={"suggested_value": data.get(CONF_WAKE_ENTITIES)},
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                multiple=True,
                filter={"domain": "media_player", "integration": "androidtv"},
            )
        ),
    }


async def async_validate(hass, data: dict) -> str | None:
    """Validate the Emby subentry fields. Error string, or None if OK."""
    url = (data.get(CONF_EMBY_URL) or "").strip()
    api_key = (data.get(CONF_EMBY_API_KEY) or "").strip()
    if not url and not api_key:
        return None
    if not url or not api_key:
        return "Set both the Emby URL and API key, or leave both empty."

    session = async_get_clientsession(hass)
    try:
        async with session.get(
            f"{url.rstrip('/')}/System/Info",
            headers={"X-Emby-Token": api_key, "Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status in (401, 403):
                return "Authentication failed: check the API key"
            if resp.status != 200:
                return f"Emby returned HTTP {resp.status}"
    except aiohttp.ClientConnectorError:
        return f"Cannot connect to {url}"
    except TimeoutError:
        return f"Connection timed out connecting to {url}"
    except aiohttp.ClientError as e:
        return f"Validation error: {e}"
    return None
