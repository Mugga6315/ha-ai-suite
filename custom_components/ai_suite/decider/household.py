"""The household's rules as Second Brain compiles them for a decider (its rule_compiler).

The decider cannot read rules, so each rule comes as fields code uses. A request rule hands the request to the text
model, which has the rule in its prompt: by words in the request ("triggers") or by the kind of request the form names
("scopes"). Names are aliases; answer rules go to the model that words decided replies. Given as text to every decider
question, rules send requests astray: an alias rule pulls a request for another room to the room it names. A request
rule with an effect in the decider's terms (rule_compiler.py's fields) is followed by code instead: `follow`. No HA
imports.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import re
from typing import Any

from ..languages import LANGUAGES, pattern
from .form import ACTIONS, PLAY, Form
from .locks import LOCK_ACTIONS, lock_action
from .media import MEDIA_TYPES
from .turn import HandOff

ANSWER, NAME, REQUEST, PENDING = "answer", "name", "request", "pending"
ANY_DEVICE = "any"
_EFFECT_FIELDS = frozenset({"media_type", "name", "action", "limit"})
_LIMIT_FIELDS = frozenset({"slot", "min", "max"})
_VALUE_SLOTS = frozenset(action.value_slot for action in ACTIONS.values() if action.value_slot)
# A negation turns any command around, and trigger words cannot exclude it: an action a rule sets hands off where the
# text negates, or where its language has no word list (closed word sets for what is security-relevant, in the
# languages with a module: languages/). A wrong hit only hands off.
# Which way an action turns a device. A rule's action replaces only one of its own way ("close" → "lock"), never the
# other ("unlock" → "lock": a rule triggered by a door's name would lock the door for a request to unlock it).
_DIRECTION = {
    **dict.fromkeys(("turn_on", "open", "unlock", "resume", "unmute", "volume_up"), "on"),
    **dict.fromkeys(("turn_off", "close", "lock", "pause", "stop", "mute", "volume_down"), "off"),
}
_NEGATION = pattern(lambda language: language.requests.negation)


@dataclass(frozen=True)
class Scope:
    """The kind of request a rule changes, whatever its words."""
    request: str  # an action or route of the form (Second Brain's rule_compiler.REQUEST_KINDS)
    device_kind: str  # a device domain, or ANY_DEVICE
    only_without_room: bool


@dataclass(frozen=True)
class Limit:
    """A limit a rule sets on a value."""
    slot: str  # the HA slot an action's value fills (form.ACTIONS value_slot): temperature, brightness, ...
    min: int | None
    max: int | None


@dataclass(frozen=True)
class Effect:
    """What a request rule changes, in the decider's own answers."""

    media_type: str | None = None  # media.MEDIA_TYPES
    name: str | None = None  # with media_type: the name to play the rule is about
    action: str | None = None  # form.ACTIONS
    limit: Limit | None = None


@dataclass(frozen=True)
class Rule:
    """One compiled household rule."""
    text: str
    kind: str
    names: tuple[tuple[str, str], ...] = ()  # (the household's word, the room or device it means)
    triggers: tuple[str, ...] = ()
    scopes: tuple[Scope, ...] = ()
    effect: Effect | None = None  # None: the rule hands its requests off


@dataclass(frozen=True)
class Household:
    """The household's rules, compiled."""
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, compiled: list[dict[str, Any]]) -> Household:
        """Second Brain's compiled rules; a rule it sent in an unknown shape counts as not compiled (PENDING)."""
        return cls(tuple(_rule(raw) for raw in compiled))

    @property
    def pending(self) -> tuple[Rule, ...]:
        """Rules Second Brain could not compile: what they change is unknown."""
        return tuple(rule for rule in self.rules if rule.kind == PENDING)

    @property
    def text(self) -> str:
        """Every rule's text, as bullets (the media question reads them: decision.with_rules)."""
        return "\n".join(f"- {rule.text}" for rule in self.rules)

    @property
    def answer_rules(self) -> tuple[str, ...]:
        """The rules for how replies are worded."""
        return tuple(rule.text for rule in self.rules if rule.kind == ANSWER)

    def aliases(self) -> dict[str, tuple[str, ...]]:
        """The household's words by the lowercased room or device name they mean."""
        found: dict[str, tuple[str, ...]] = {}
        for rule in self.rules:
            for word, means in rule.names:
                found[means.lower()] = (*found.get(means.lower(), ()), word)
        return found

    def triggered(self, text: str) -> Rule | None:
        """The request rule whose trigger words are all in the text."""
        return next((rule for rule in self._requests() if trigger_fires(rule.triggers, text)), None)

    def in_scope(self, request: str, device_kind: str | None, room_named: bool) -> Rule | None:
        """The request rule about this kind of request, whatever its words."""
        for rule in self._requests():
            for scope in rule.scopes:
                if scope.request == request and scope.device_kind in (ANY_DEVICE, device_kind) and \
                        not (scope.only_without_room and room_named):
                    return rule
        return None

    def _requests(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.kind == REQUEST]


def trigger_fires(triggers: Iterable[str], text: str) -> bool:
    """Whether every word of one of the triggers is in the text: how a request rule is matched."""
    have = _words(text)
    return any(_words(trigger) <= have for trigger in triggers if trigger)


def _rule(raw: dict[str, Any]) -> Rule:
    try:
        return Rule(
            str(raw["text"]),
            str(raw["kind"]),
            tuple((str(n["word"]), str(n["means"])) for n in raw.get("names") or ()),
            tuple(str(t) for t in raw.get("triggers") or ()),
            tuple(Scope(str(s["request"]), str(s["device_kind"]), bool(s["only_without_room"]))
                  for s in raw.get("scopes") or ()),
            _effect(raw.get("effect")),
        )
    except (KeyError, TypeError, AttributeError):
        return Rule(str(raw.get("text", raw)) if isinstance(raw, dict) else str(raw), PENDING)


def follow(rule: Rule, text: str, language: str | None, filled: Form) -> Form:
    """The form with the rule's effect in place of the decider's answers.

    HandOff when the rule has no effect, or its effect does not fit the request: the text model, which has the rule,
    answers then. An action the rule sets counts as the text's lock words: a request to close a door names no
    locking, the household means it. `language`: the reply language the decider named (None: unknown).
    """
    effect = rule.effect
    if effect is None:
        raise HandOff(f"a household rule covers it: {rule.text}")
    if effect.media_type is not None:
        if filled.action != PLAY or filled.media is None:
            raise HandOff(f"a household rule's media type, the request plays nothing: {rule.text}")
        # Only for the name the rule is about: its trigger also fires on a title by the artist the rule names, and
        # the name cut from the text can be wrong (an artist named in the possessive); then the text model decides.
        if filled.media.name.casefold() != (effect.name or "").casefold():
            raise HandOff(f"a household rule's media name, the request plays something else: {rule.text}")
        filled = replace(filled, media=replace(filled.media, type=effect.media_type, candidates=()))
    if effect.action is not None:
        if language not in LANGUAGES or _NEGATION.search(text):
            raise HandOff(f"a household rule's action, the request may say not to: {rule.text}")
        if filled.action not in ACTIONS:
            raise HandOff(f"a household rule's action, the request is no command: {rule.text}")
        if not _same_way(effect.action, filled.action, text):
            raise HandOff(f"a household rule's action against the request's: {rule.text}")
        filled = replace(filled, action=effect.action, action_from_rule=True)
    if effect.limit is not None and not _within(effect.limit, filled):
        raise HandOff(f"outside a household rule's limit: {rule.text}")
    return filled


def _same_way(ruled: str, decided: str, text: str) -> bool:
    """Whether the rule's action goes the way the request does.

    The request's way is the decider's action, or the text's own lock words, which win over it (a decider may lock the
    door for a German unlock command whose verb is split).
    """
    said = lock_action(text)
    if said is not None and ruled in LOCK_ACTIONS and said != ruled:
        return False
    if ruled == decided:
        return True
    return _DIRECTION.get(ruled) is not None and _DIRECTION.get(ruled) == _DIRECTION.get(decided)


def _within(limit: Limit, filled: Form) -> bool:
    slot = ACTIONS[filled.action].value_slot if filled.action in ACTIONS else None
    value = filled.value
    return (
        slot == limit.slot and value is not None
        and (limit.min is None or value >= limit.min) and (limit.max is None or value <= limit.max)
    )


def _effect(raw: object) -> Effect | None:
    """The compiled rule's effect; None, so the rule hands off, for no effect or one outside the compiler's lists."""
    if not isinstance(raw, dict) or set(raw) - _EFFECT_FIELDS:
        return None
    media_type, name, action, limit = raw.get("media_type"), raw.get("name"), raw.get("action"), _limit(raw.get("limit"))
    if media_type is not None and (media_type not in MEDIA_TYPES or not isinstance(name, str) or not name.strip()):
        return None
    if action is not None and action not in ACTIONS:
        return None
    if raw.get("limit") is not None and limit is None:
        return None
    if media_type is None and action is None and limit is None:
        return None
    return Effect(media_type, name if media_type is not None else None, action, limit)


def _limit(raw: object) -> Limit | None:
    if not isinstance(raw, dict) or set(raw) - _LIMIT_FIELDS or raw.get("slot") not in _VALUE_SLOTS:
        return None
    low, high = raw.get("min"), raw.get("max")
    if not all(bound is None or (isinstance(bound, int) and not isinstance(bound, bool)) for bound in (low, high)):
        return None
    if low is None and high is None:
        return None
    return Limit(raw["slot"], low, high)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))
