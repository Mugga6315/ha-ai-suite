"""Whether a request's words name a device.

A request names one by its name or an alias, without its room words, or better than another device of its kind. No HA
imports.
"""

from collections import Counter
import re

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
    return any(re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE) for name in (device.name, *device.aliases))


def named_only_by(device: Candidate, devices: dict[str, Candidate], text: str) -> bool:
    """Whether the text names `device` by its name or an alias, and names no other device that way."""
    return names(device, text) and not any(names(d, text) for d in devices.values() if d is not device)


def named_without_room(device: Candidate, devices: dict[str, Candidate], text: str) -> bool:
    """Whether the text names `device` by its name without room words.

    Those are the words of its name that are no room word, and no other device of its room may be named by its name or
    so: else a request that names the device that way, with its room said apart, names no device.
    """
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
    """Whether the text names a kind of light, not light as such.

    That is a compound of a light word, or a light word after a word that is no room and no determiner. A room word may
    stand before it or in front of it in one word: the home's area words, or a room word (rooms.py) where the home's
    areas are named in another language. A request for one kind of light in a room must not switch the room's other
    lights off as one of all its lights.
    """
    rooms = {word for device in devices.values() if device.area for word in device.area.casefold().split()}
    words = re.findall(r"[\w-]+", text.casefold())
    for index, word in enumerate(words):
        if not LIGHT_WORDS.search(word):
            continue
        if not _GENERIC_LIGHTS.match(word):
            compound = _ENDS_IN_GENERIC_LIGHT.match(word)
            if compound is None or not _a_room_word(compound.group(1), rooms):
                return True
            continue
        before = words[index - 1] if index else ""
        if before and before not in _NOT_A_KIND and not _a_room_word(before, rooms):
            return True
    return False


def _a_room_word(word: str, rooms: set[str]) -> bool:
    return word in rooms or names_a_room(word, {})


def named_better_than(device: Candidate, kind: list[Candidate], text: str) -> bool:
    """Whether the text names another device of its kind better than `device`.

    Whether the text shares more of the words that set a device apart with another device of its kind than with
    `device`: words in only one name of the kind, no room words. A compound light word holds a room word that a switch
    named after that room shares, not the light the request means; a word in two other lights' names must not take a
    request from the light it names.
    """
    rooms = {word for d in kind if d.area for word in words(d.area)}
    named = {d.entity_id: {word for word in _name_words(d) if not _a_room_word(word, rooms)} for d in kind}
    named.setdefault(device.entity_id, {word for word in _name_words(device) if not _a_room_word(word, rooms)})
    counts = Counter(word for names in named.values() for word in names)

    def distinct(entity_id: str) -> int:
        return _shared({word for word in named[entity_id] if counts[word] == 1}, text)

    own = distinct(device.entity_id)
    return any(distinct(entity_id) > own for entity_id in named if entity_id != device.entity_id)


def named_best_by(device: Candidate, kind: list[Candidate], text: str) -> bool:
    """Whether the text shares more of the device's name and alias words than with any other device of its kind."""
    shared = {d.entity_id: _shared_name_words(d, text) for d in kind}
    best = max(shared.values())
    return best > 0 and shared[device.entity_id] == best and sum(n == best for n in shared.values()) == 1


def _shared_name_words(device: Candidate, text: str) -> int:
    """How many of the device's name and alias words the text holds.

    Words inside its compounds count too (a German question's compound holding a word of an English name).
    """
    return _shared(_name_words(device), text)


def _name_words(device: Candidate) -> set[str]:
    return words(" ".join((device.name, *device.aliases)))


def _shared(name_words: set[str], text: str) -> int:
    asked = words(text)
    return sum(
        any(said == word or (len(word) >= _MIN_COMPOUND_PART and word in said) for said in asked) for word in name_words
    )


def is_a_light(device: Candidate) -> bool:
    """Whether the device is a light, or is named as one."""
    return device.domain == "light" or any(LIGHT_WORDS.search(name) for name in (device.name, *device.aliases))
