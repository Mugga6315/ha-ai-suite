"""What to play: the name and the artist from the request, the kind of media from the decider.

Code cuts the name and the artist from the request (every language's phrasings, languages/), then the decider says
what kind of media that name is. Music Assistant finds the item by name (music_assistant.play_media), as its own LLM
blueprint has it filled (music-assistant/voice-support, llm_voice_script.yaml). No HA imports.

Asking about the cut name beats asking about the whole request; a yes/no question per kind does worse, its answers
are not comparable with each other.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import re

from ..languages import alternatives, merged
from .turn import HandOff

# Music Assistant's media types that a voice request names (its play_media `media_type`).
MEDIA_TYPES = {
    "artist": "an artist or a band",
    "track": "a song",
    "album": "an album",
    "playlist": "a playlist",
    "radio": "a radio station",
}

_POLITE = re.compile(rf"\b{alternatives(lambda language: language.polite)}\b", re.IGNORECASE)
_PLAY = re.compile(rf"^{alternatives(lambda language: language.media.play)}\s+", re.IGNORECASE)
_PLAY_END = re.compile(rf"\s+{alternatives(lambda language: language.media.play_end)}$", re.IGNORECASE)
_KIND = re.compile(
    rf"\b{alternatives(lambda language: language.media.kind_articles)}?\s*"
    rf"{alternatives(lambda language: language.media.kinds)}\b",
    re.IGNORECASE,
)
_BY = re.compile(rf"\s+{alternatives(lambda language: language.media.by)}\s+", re.IGNORECASE)
_ROOM_WORDS = alternatives(lambda language: language.rooms.before_room)
# Words for music in general: before the word for "by", the artist is what to play.
_ANYTHING = merged(lambda language: language.media.anything)
# A place in the queue (next, previous, last), no name. Music Assistant finds a title with such a word in it, and
# some deciders send such skips down the play path.
_SKIP = re.compile(
    rf"^{alternatives(lambda language: language.media.skip_articles)}?\s*"
    rf"{alternatives(lambda language: language.media.queue)}"
    rf"(?:\s+{alternatives(lambda language: language.media.again)})?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Media:
    """What Music Assistant should find and play."""

    type: str | None  # None: the decider was unsure; `candidates` are its likeliest types, for a lookup by name
    name: str
    artist: str | None = None
    candidates: tuple[str, ...] = ()


# Music Assistant's search (music_assistant.search) answers per media type under these keys.
RESULT_KEY = {"artist": "artists", "album": "albums", "track": "tracks", "playlist": "playlists", "radio": "radio"}
# An unsure media type is looked up among the decider's likeliest types only: by name alone nearly every name is
# also a playlist, an album or a podcast somewhere (a station's name: a playlist and a radio station), among the
# likeliest it is one (among radio, track, artist: radio).
LOOKUP_TYPES = 3


def type_found(name: str, candidates: Iterable[str], found: dict[str, list[dict]]) -> str | None:
    """The one candidate type with an item of exactly that name (case aside) in a search's answer.

    None for none or several.
    """
    wanted = name.casefold()
    named = [kind for kind in candidates
             if any((item.get("name") or "").casefold() == wanted for item in found.get(RESULT_KEY[kind], []))]
    return named[0] if len(named) == 1 else None


def named_in(text: str, rooms: Iterable[str], room_named: bool) -> tuple[str, str | None]:
    """The name and the artist the request names, its room phrase left out; HandOff when no name is left.

    `room_named`: the decider heard a room, which the text may call by a word the home has no alias for (a room word
    in another language than the area's name).
    """
    rest = _POLITE.sub("", text.strip(" .!?"))
    rest = _PLAY_END.sub("", _PLAY.sub("", rest.strip()))
    rest = re.sub(r"\s+", " ", _KIND.sub("", _without_room(rest, rooms, room_named))).strip()
    parts = _BY.split(rest, maxsplit=1)
    name = parts[0]
    artist = re.sub(r"^the\s+", "", parts[1], flags=re.IGNORECASE) if len(parts) > 1 else None
    if artist and name.lower() in _ANYTHING:
        name, artist = artist, None
    if not name or name.lower() in _ANYTHING:  # music in general: no item to find
        raise HandOff(f"nothing to play named in {text!r}")
    if _SKIP.match(name):
        raise HandOff(f"a skip, not a name to play: {text!r}")
    return name, artist


def kind_question(name: str, artist: str | None) -> tuple[str, dict]:
    """What the decider is told the user wants to play, and the question what kind of media that is."""
    wanted = f'The user wants to play "{name}"' + (f" by {artist}" if artist else "")
    return wanted, {"type": "choice", "instructions": f'What is "{name}"?', "criteria": MEDIA_TYPES}


def _without_room(text: str, rooms: Iterable[str], room_named: bool) -> str:
    """The text without its room phrase.

    That is one of the home's room names, else (a room was heard) the closing "im/in der/in the <word>". A title ending
    like a room phrase stays when no room was heard.
    """
    if names := sorted({room for room in rooms if room}, key=len, reverse=True):
        known = "|".join(re.escape(name) for name in names)
        cut = re.sub(rf"\s+{_ROOM_WORDS}\s+(?:{known})\b", "", text, flags=re.IGNORECASE)
        if cut != text:
            return cut
    return re.sub(rf"\s+{_ROOM_WORDS}\s+\w+$", "", text, flags=re.IGNORECASE) if room_named else text
