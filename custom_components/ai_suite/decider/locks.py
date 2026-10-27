"""Whether a text says to lock or to unlock, by its words.

Locking is a closed set of words that carry their direction, and a wrong answer or action is a security fact: a
decider may take a word for locked as closed with high confidence, or lock the door for an unlock command whose verb
is split, whatever its options say. Only the languages with a module (languages/). No HA imports.
"""

import re

from ..languages import LANGUAGES, pattern

LOCKED, UNLOCKED = "locked", "unlocked"
# The form's action for each direction (form.ACTIONS).
_ACTIONS = {LOCKED: "lock", UNLOCKED: "unlock"}
LOCK_ACTIONS = frozenset(_ACTIONS.values())

_UNLOCKED = pattern(lambda language: language.locks.unlocked)
_LOCKED = pattern(lambda language: language.locks.locked)
# A command whose verb is split, by each language that splits its verbs, with the particles that unlock.
_SPLITS = [
    (re.compile(language.locks.split, re.IGNORECASE), language.locks.unlock_particles)
    for language in LANGUAGES.values()
    if language.locks.split
]


def lock_direction(text: str) -> str | None:
    """LOCKED or UNLOCKED when the text says one of them, else None; both: None."""
    unlocked = _UNLOCKED.search(text) is not None
    locked = _LOCKED.search(text) is not None
    for split, unlocking in _SPLITS:
        if found := split.search(text):
            if found["particle"].lower() in unlocking:
                unlocked = True
            else:
                locked = True
    if unlocked == locked:
        return None
    return UNLOCKED if unlocked else LOCKED


def lock_action(text: str) -> str | None:
    """The form's lock action ("lock" or "unlock") the text's words say, else None."""
    direction = lock_direction(text)
    return _ACTIONS[direction] if direction else None
