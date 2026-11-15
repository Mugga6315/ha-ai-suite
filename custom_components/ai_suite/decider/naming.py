"""Whether a request's words name a device."""

from collections import Counter
import re
from typing import NamedTuple

from ..languages import alternatives, merged, pattern
from .decision import Candidate, words
from .rooms import names_a_room

# No word boundary: a compound holds its light word anywhere.
LIGHT_WORDS = pattern(lambda language: language.lights.words)
# A name word counts inside a compound of the request from this length on; shorter ones match inside unrelated words.
_MIN_COMPOUND_PART = 4
# Light as such, no kind of light (names_a_light_kind), and the words that may stand before it.
_GENERIC_LIGHT_WORDS = alternatives(lambda language: language.lights.generic)
_GENERIC_LIGHTS = re.compile(rf"^{_GENERIC_LIGHT_WORDS}$", re.IGNORECASE)
_ENDS_IN_GENERIC_LIGHT = re.compile(rf"^(.+?)-?{_GENERIC_LIGHT_WORDS}$", re.IGNORECASE)
_NOT_A_KIND = merged(lambda language: language.lights.not_a_kind)


def names(device: Candidate, text: str) -> bool:
    """Whether the text names `device` by its name or an alias."""
    return any(_says(name, text) for name in (device.name, *device.aliases))


def _says(name: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) is not None


def named_only_by(device: Candidate, devices: dict[str, Candidate], text: str) -> bool:
    """Whether the text names `device` by its name or an alias, and names no other device that way."""
    return names(device, text) and not any(names(d, text) for d in devices.values() if d is not device)


def named_without_room(device: Candidate, devices: dict[str, Candidate], text: str) -> bool:
    """Whether the text names `device` by its name without room words, and no other device of its room that way."""
    rooms = {word for d in devices.values() if d.area for word in words(d.area)}
    others = [d for d in devices.values() if d.area == device.area and d.entity_id != device.entity_id]
    return _names_short(device, text, rooms) and not any(
        names(d, text) or _names_short(d, text, rooms) for d in others
    )


def _names_short(device: Candidate, text: str, rooms: set[str]) -> bool:
    for name in (device.name, *device.aliases):
        kept = [word for word in name.split() if not _a_room_word(word.casefold(), rooms)]
        if kept and len(kept) < len(name.split()) and re.search(rf"\b{re.escape(' '.join(kept))}\b", text, re.IGNORECASE):
            return True
    return False


def names_a_light_kind(text: str, devices: dict[str, Candidate]) -> bool:
    """Whether the text names a kind of light, not light as such."""
    rooms = {word for device in devices.values() if device.area for word in device.area.casefold().split()}
    words = re.findall(r"[\w-]+", text.casefold())
    for index, word in enumerate(words):
        if not LIGHT_WORDS.search(word):
            continue
        if not _GENERIC_LIGHTS.match(word):
            first_part = _first_part(word)
            if first_part is None or not _a_room_word(first_part, rooms):
                return True
            continue
        before = words[index - 1] if index else ""
        if before and before not in _NOT_A_KIND and not _a_room_word(before, rooms):
            return True
    return False


def _a_room_word(word: str, rooms: set[str]) -> bool:
    return word in rooms or names_a_room(word, {})


def named_better_than(device: Candidate, kind: list[Candidate], text: str) -> bool:
    """Whether the text shares more of the words that set a device apart with another device of its kind."""
    others = [d for d in kind if d.entity_id != device.entity_id]
    # Without its light word, a light's name may hold no word that sets it apart: its full name said decides.
    if not _full_name_said(device, kind, text) and any(_full_name_said(d, kind, text) for d in others):
        return True
    rooms = _area_words(kind)
    named = {d.entity_id: {word for word in _name_words(d) if not _a_room_word(word.text, rooms)} for d in kind}
    named.setdefault(device.entity_id, {word for word in _name_words(device) if not _a_room_word(word.text, rooms)})
    counts = Counter(word for names in named.values() for word in names)
    first_parts = _first_parts_said(kind, text)

    def distinct(entity_id: str) -> int:
        return _shared({word for word in named[entity_id] if counts[word] == 1}, text, first_parts)

    own = distinct(device.entity_id)
    return any(distinct(entity_id) > own for entity_id in named if entity_id != device.entity_id)


def named_best_by(device: Candidate, kind: list[Candidate], text: str) -> bool:
    """Whether the text shares more of the device's name and alias words than with any other device of its kind."""
    first_parts = _first_parts_said(kind, text)
    shared = {d.entity_id: _shared(_name_words(d), text, first_parts) for d in kind}
    best = max(shared.values())
    return best > 0 and shared[device.entity_id] == best and sum(n == best for n in shared.values()) == 1


def _full_name_said(device: Candidate, kind: list[Candidate], text: str) -> bool:
    """Whether the text says a light's full name, not only inside a longer one of another light of the kind."""
    if device.domain != "light":
        return False
    rooms = _area_words(kind)
    others = [d for d in kind if d.domain == "light" and d.entity_id != device.entity_id]
    longer = [name for d in others for name in _full_names_said(d, text, rooms)]
    return any(
        not any(len(other) > len(name) and _says(name, other) for other in longer)
        for name in _full_names_said(device, text, rooms)
    )


def _full_names_said(device: Candidate, text: str, rooms: set[str]) -> list[str]:
    """The device's full names the text says; none of light words only (aliases excepted) or room words only."""
    full_names = [*([] if _light_words_only(device.name) else [device.name]), *device.aliases]
    return [name for name in full_names if not words(name) <= rooms and _says(name, text)]


def _area_words(kind: list[Candidate]) -> set[str]:
    return {word for d in kind if d.area for word in words(d.area)}


def _light_words_only(name: str) -> bool:
    return all(_GENERIC_LIGHTS.match(word) for word in words(name))


class _NameWord(NamedTuple):
    """A word of a device's name.

    `first_part`: `text` is the part of a compound light word before the light word, said only as such a part.
    """

    text: str
    first_part: bool = False


def _name_words(device: Candidate) -> set[_NameWord]:
    """The words of the device's name and aliases; a light's without its generic light words."""
    named = words(" ".join((device.name, *device.aliases)))
    if device.domain != "light":
        return {_NameWord(word) for word in named}
    return {_light_name_word(word) for word in named if not _GENERIC_LIGHTS.match(word)}


def _light_name_word(word: str) -> _NameWord:
    first_part = _first_part(word)
    return _NameWord(first_part, first_part=True) if first_part else _NameWord(word)


def _first_part(word: str) -> str | None:
    """The part of a compound light word before its generic light word; None for any other word."""
    compound = _ENDS_IN_GENERIC_LIGHT.match(word)
    return compound.group(1) if compound else None


def _first_parts_said(kind: list[Candidate], text: str) -> set[str]:
    """The first parts of the kind's names that the text says, as the first part of a compound light word."""
    name_words = {word for d in kind for word in _name_words(d)}
    first_parts = {word.text for word in name_words if word.first_part}
    every_word = {word.text for word in name_words}
    said = set()
    for part in {part for word in words(text) if (part := _first_part(word))}:
        if part in every_word:
            said |= {part} & first_parts
        else:
            said |= {first for first in first_parts if len(first) >= _MIN_COMPOUND_PART and part.endswith(first)}
    return said


def _shared(name_words: set[_NameWord], text: str, first_parts: set[str]) -> int:
    """How many of the name words the text says; `first_parts`: the first parts it says (`_first_parts_said`)."""
    asked = words(text)
    return sum(_said(name_word, asked, first_parts) for name_word in name_words)


def _said(name_word: _NameWord, asked: set[str], first_parts: set[str]) -> bool:
    """Whether the request says the name word: a first part as one of `first_parts`, any other word as or in a word."""
    if name_word.first_part:
        return name_word.text in first_parts
    word = name_word.text
    return any(said == word or (len(word) >= _MIN_COMPOUND_PART and word in said) for said in asked)


def is_a_light(device: Candidate) -> bool:
    """Whether the device is a light, or is named as one."""
    return device.domain == "light" or any(LIGHT_WORDS.search(name) for name in (device.name, *device.aliases))
