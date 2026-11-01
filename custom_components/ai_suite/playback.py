"""What a media player plays after a play request: actions only send it, the player's state says whether it started.

Music Assistant's play_media and HA's media_player.play_media return once the request is sent, also when the name
matched another item, and HA skips an unavailable player without an error. A reply built on the action alone would
confirm what never played.
"""

import asyncio
from dataclasses import dataclass

from homeassistant.components.media_player import (
    ATTR_MEDIA_ALBUM_NAME,
    ATTR_MEDIA_ARTIST,
    ATTR_MEDIA_CONTENT_ID,
    ATTR_MEDIA_TITLE,
    MediaPlayerState,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.event import async_track_state_change_event

# How long a player gets to report that it plays. A stream that buffers longer reads as not started.
START_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class Playback:
    """What a player showed: its state and the item it had."""

    state: str | None
    item: tuple[str | None, str | None]  # media_content_id, media_title


def playback_of(state: State | None) -> Playback:
    """The player's playback as it is now (before a play request)."""
    if state is None:
        return Playback(None, (None, None))
    return Playback(state.state, (state.attributes.get(ATTR_MEDIA_CONTENT_ID), state.attributes.get(ATTR_MEDIA_TITLE)))


async def async_started(hass: HomeAssistant, entity_id: str, before: Playback) -> State | None:
    """The player's state once it plays an item other than `before`'s; None when it does not play in START_TIMEOUT_S.

    A player that plays at the timeout counts as started: the same item asked again keeps its id and title.
    `unchanged` tells that from a request that changed nothing.
    """
    changed = asyncio.Event()

    @callback
    def check(event: Event[EventStateChangedData]) -> None:
        if _started(event.data["new_state"], before):
            changed.set()

    unsubscribe = async_track_state_change_event(hass, entity_id, check)
    try:
        if not _started(hass.states.get(entity_id), before):
            async with asyncio.timeout(START_TIMEOUT_S):
                await changed.wait()
    except TimeoutError:
        pass
    finally:
        unsubscribe()
    state = hass.states.get(entity_id)
    return state if state is not None and state.state == MediaPlayerState.PLAYING else None


def unchanged(state: State, before: Playback) -> bool:
    """Whether the player plays the item it played before the request: the same item asked again, or no change."""
    return before.state == MediaPlayerState.PLAYING and playback_of(state).item == before.item


def _started(state: State | None, before: Playback) -> bool:
    if state is None or state.state != MediaPlayerState.PLAYING:
        return False
    return before.state != MediaPlayerState.PLAYING or playback_of(state).item != before.item


def now_playing(state: State) -> str:
    """The item a playing player shows, as "Title by Artist (Album)"; empty when it shows none."""
    title, artist, album = (state.attributes.get(key) for key in (ATTR_MEDIA_TITLE, ATTR_MEDIA_ARTIST,
                                                                 ATTR_MEDIA_ALBUM_NAME))
    text = " by ".join(part for part in (title, artist) if part)
    return f"{text} ({album})" if text and album else text or (album or "")


def names(state: State, name: str) -> bool:
    """Whether the playing item's title, artist or album holds `name` (case ignored)."""
    wanted = name.casefold()
    return any(wanted in str(state.attributes.get(key) or "").casefold()
               for key in (ATTR_MEDIA_TITLE, ATTR_MEDIA_ARTIST, ATTR_MEDIA_ALBUM_NAME))
