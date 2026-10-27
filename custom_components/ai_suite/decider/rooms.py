"""Whether a request names a room, by its words.

The home's own areas and their aliases count, and every language's words for rooms and floors (languages/), also
those a home does not have, also inside compounds. No HA imports.
"""

from collections.abc import Mapping

from ..languages import pattern

# Matched anywhere in a word, so compounds count. A match where no room is meant ("Hallo" holds "hall") only keeps a
# hand-off; a room word missing here lets a doubted room pass as none.
_ROOM_WORDS = pattern(lambda language: language.rooms.words)


def names_a_room(text: str, areas: Mapping[str, list[str]]) -> bool:
    """Whether `text` holds a room or floor word, or one of the home's area names or aliases."""
    lowered = text.casefold()
    own_names = (name.casefold() for area, aliases in areas.items() for name in (area, *aliases))
    return _ROOM_WORDS.search(text) is not None or any(name in lowered for name in own_names)
