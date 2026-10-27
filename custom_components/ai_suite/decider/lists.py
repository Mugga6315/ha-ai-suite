"""Lists (shopping, to-do): the decider says what to do and on which list.

For an item already on the list it picks one of the list's own entries, read from HA at request time, so a request in
one language finds the entry written in another without any word map. A new item's text is cut from the request by
rules (every language's phrasings, languages/). HA's own list intents do the change. No HA imports.
"""

from dataclasses import dataclass
import re

from ..languages import alternatives, merged, pattern
from .turn import HandOff


@dataclass(frozen=True, slots=True)
class ListAction:
    """A list action: the HA intent and how its note says what was done."""
    description: str
    intent: str | None  # None: a read, nothing changes
    # What happened to the entry, for the answer step. The action's id ("list_done") alone is worded as a removal for
    # a tick-off.
    done: str = ""


LIST_ACTIONS: dict[str, ListAction] = {
    "list_add": ListAction("put something on a shopping or to-do list", "HassListAddItem", "added to"),
    "list_done": ListAction("tick something off a list, or report that a task on it is done", "HassListCompleteItem",
                            "ticked off as done on"),
    "list_remove": ListAction("remove or delete something from a list", "HassListRemoveItem", "removed from"),
    "list_read": ListAction("ask what is on a list, or whether something is on it", None),
}
NO_ITEM = "none"
# The list route's question (form.py): only the list actions, with narrower wording, which a small decider gets right
# more often than the form's full menu.
LIST_STEP = {"type": "choice", "instructions": "What should happen on the list?",
             "criteria": {"list_add": "put a new entry on the list",
                          "list_done": "tick an entry that is on the list off as done",
                          "list_remove": "delete an entry from the list",
                          "list_read": "tell what is on the list or whether something is"}}
# Ticking off is not removing: deciders take a request to tick an entry off for remove with confidence even after
# the step, and text models remove the entry too. A request in these words never deletes or adds: a remove becomes
# the tick-off, unless the text also says delete (deleting the entries already ticked off).
_TICK_OFF = pattern(lambda language: language.lists.tick_off)
_DELETE = pattern(lambda language: language.lists.delete)

_POLITE = alternatives(lambda language: language.polite)
_ADD = re.compile(rf"^(?:{_POLITE}\s+)?{alternatives(lambda language: language.lists.add)}\s+", re.IGNORECASE)
_NEED = re.compile(
    rf"^{alternatives(lambda language: language.lists.need)}\s+(?P<item>.+?)(?:\s*[—–,-]\s*.*)?$", re.IGNORECASE
)
# The closing list phrase: a preposition, an article or possessive, at most two words ending in the word for a list,
# and whatever follows it.
_ON_LIST = re.compile(
    rf"\s+{alternatives(lambda language: language.lists.to_list)}\s+"
    rf"(?:{alternatives(lambda language: language.lists.list_owners)}\s+)?"
    rf"[\w-]*(?:\s+[\w-]+)?\s*{alternatives(lambda language: language.lists.list)}\b.*$",
    re.IGNORECASE,
)
_ARTICLE = re.compile(rf"^{alternatives(lambda language: language.lists.item_articles)}\s+", re.IGNORECASE)
# A question whether one item is on a list, once its closing list phrase is cut: the question words, then the item.
_IS_ON = re.compile(
    rf"^{alternatives(lambda language: language.lists.is_on)}\s+"
    rf"(?:{alternatives(lambda language: language.lists.still)}\s+)?"
    rf"{alternatives(lambda language: language.lists.asked_articles)}?\s*(?P<item>.+)$",
    re.IGNORECASE,
)
_PRONOUNS = merged(lambda language: language.lists.pronouns)


def checked_action(action: str, text: str) -> str:
    """The list action.

    When the text asks to tick off: a remove is the tick-off, an add or a text that also says delete hands off.
    """
    if action not in ("list_remove", "list_add") or not _TICK_OFF.search(text):
        return action
    if action == "list_remove" and not _DELETE.search(text):
        return "list_done"
    raise HandOff(f"the user ticks off, the decider says {action}")


def item_to_add(text: str) -> str:
    """The new item the request names.

    HandOff when none is found or the request names no list: writing something onto anything but a list is no list
    request.
    """
    said = text.strip(" .!?")
    if needed := _NEED.match(said):
        item = needed["item"]
    elif _ON_LIST.search(said):
        item = _ON_LIST.sub("", _ADD.sub("", said))
    else:  # something written onto something that is no list: which list, if any, is not said
        raise HandOff(f"no list named in {text!r}")
    item = _ARTICLE.sub("", re.sub(r"\s+", " ", item).strip(" .!?,"))
    if not item or item.lower() in _PRONOUNS:
        raise HandOff(f"no item named in {text!r}")
    return item


def item_asked(text: str) -> str | None:
    """The item a question asks about; None for a question about the whole list."""
    said = text.strip(" .!?")
    if not _ON_LIST.search(said) or not (asked := _IS_ON.match(_ON_LIST.sub("", said))):
        return None
    return asked["item"].strip(" ,") or None


def list_question(names: list[str]) -> dict:
    """The question which list the user means."""
    return {"type": "choice", "instructions": "Which list does the user mean?",
            "criteria": dict.fromkeys(names, "")}


def item_question(entries: list[str]) -> dict:
    """Which of the list's entries the user means; the entries are what the list holds now."""
    return {"type": "choice", "instructions": "Which entry of the list does the user mean?",
            "criteria": {**dict.fromkeys(entries, ""), NO_ITEM: "none of these entries"}}
