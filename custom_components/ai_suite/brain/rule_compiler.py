"""Household rules compiled for small decision models.

A rule in rules.md is written for a human and a text model. A small decision
model (a decision agent's decider) cannot read rules: given them as text, an
alias rule (another word for a room) pulls other rooms' requests to the room
it names. So the text model translates each rule once into fields
code can use:

- kind "answer": how replies are worded; for the model that words replies.
- kind "name": the household's word for a room, device or list (an alias).
- kind "request": which requests the rule changes: by words in the request
  ("triggers") or by the kind of request whatever its words ("scopes"). A
  decision agent hands such requests to the text model, which has the rule.
- kind "text_model": about the text model's tools and notes; nothing else.

REQUEST_KINDS and DEVICE_KINDS are the contract with decision agents: the
request and device names their forms produce.
"""
import hashlib
import json

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar

from ..decider.decision import NO_AREA_LABEL
from ..decider.domains import DOMAINS as DECIDER_DOMAINS
from ..decider.form import ACTIONS as DECIDER_ACTIONS
from ..decider.household import trigger_fires
from ..decider.media import MEDIA_TYPES as DECIDER_MEDIA_TYPES
from ..home import area_name, exposed_states
from ..languages import LANGUAGE_NAMES
from ..llm_client import ChatClient, LLMError

COMPILE_TIMEOUT_SECONDS = 60
COMPILE_MAX_TOKENS = 1500  # the examples need room

# The routes in words written for the compiler; the actions in the decider's, so the two cannot drift.
REQUEST_KINDS = {
    "read": "ask for the current state or value of one device or sensor",
    "play": "play a song, artist, album, playlist or radio station",
    "list": "add to, tick off, remove from or read a shopping or to-do list",
    "forecast": "ask about the weather of a coming day or days",
    "calendar": "ask what is in the calendar, or when an appointment or event is",
    "past": "the highest, lowest, average or total value of a sensor on an earlier day",
    "changed": "when a device last changed (a door last opened, a light last switched on)",
    "remember": "remember something the user tells",
    "chat": "only thanks, a greeting or a goodbye",
} | {name: action.description for name, action in DECIDER_ACTIONS.items()}
DEVICE_KINDS = (*DECIDER_DOMAINS, "any")
KINDS = ("answer", "name", "request", "text_model")

# The decision agent's form answers a rule's effect may name, read from the decider itself.
MEDIA_TYPES = tuple(DECIDER_MEDIA_TYPES)
ACTIONS = tuple(DECIDER_ACTIONS)
LIMIT_SLOTS = {action.value_slot: name for name, action in DECIDER_ACTIONS.items() if action.value_slot}
EXAMPLES_PER_SIDE = 5
# Triggers are given in every language the decider reads (languages/).
_NAMES = list(LANGUAGE_NAMES.values())
_LANGUAGES_IN_WORDS = f"{', '.join(_NAMES[:-1])} and {_NAMES[-1]}" if len(_NAMES) > 1 else _NAMES[0]

SYSTEM = """You translate one household rule of a smart home voice assistant into a form a small decision model can use.
The small model cannot read rules; it only picks from menus. Pick the rule's kind:
- "answer": how replies are worded (language, length, units, tone, form of address). It never changes what is done or what a reply contains.
- "name": the household's own word for one of the home's rooms, devices or lists. Give the word and the exact room, device or list name it means.
- "request": it changes what certain requests mean, what is done for them or what their answer contains (a phrase meaning an action or a scene, a limit, a condition, a default room, a meaning of a word).
- "text_model": about how the assistant uses its tools, notes or context. It does not change what any request means.
For "request", say which requests the rule touches, in one or both ways:
- "triggers": when the rule is about certain words. Core words, as short as possible, in __LANGUAGES__: the words that must be in a request for the rule to matter, no verbs unless the verb carries the meaning. A request containing all words of one trigger is covered. Give at least one trigger per language, so every "changed" example contains all words of one trigger.
- "scopes": when the rule is about a kind of request whatever its words (a limit, a time condition, a default when no room is named, what an answer must contain). Pick each kind of request from the menu, the device kind it acts on ("any" for all), and whether the rule only matters when the request names no room. Leave scopes empty for a rule about certain words.
For "request", also give:
- "effect": what the rule changes, only when its whole change is one of these: the media type a play request means ("media_type") with the name to play it is about ("name": the band's name for "Spiel Artist 1 means the band"), the action a request means instead of the one its words say ("action": a phrase meaning lock is "lock"), or a limit on a value ("limit": the value slot with "min" and/or "max", whole numbers). Fill the fields that apply, null for the others. null for the whole effect when the rule does more than one thing, depends on a time or a person, or changes anything else.
- "examples": "changed": __N__ requests the rule changes, using the rule's own words or their translation, never a verb or word swapped for another with the same meaning; "unchanged": __N__ requests that share words or devices with the rule but that it does not change. Short voice requests, __LANGUAGES__ mixed. Sort each request by the rule text alone: a request the rule covers is "changed" whatever else it names, such as a room or a time. When the rule is about a device, "unchanged" must hold the opposite action on the same device (the rule means lock: "unlock Door 1") and a question about the device alone ("is Door 1 open?").
Leave fields empty or null that do not belong to the kind.""".replace("__N__", str(EXAMPLES_PER_SIDE)).replace(
    "__LANGUAGES__", _LANGUAGES_IN_WORDS
)

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"enum": list(KINDS)},
        "names": {"type": "array", "items": {
            "type": "object",
            "properties": {"word": {"type": "string"}, "means": {"type": "string"}},
            "required": ["word", "means"], "additionalProperties": False}},
        "triggers": {"type": "array", "items": {"type": "string"}},
        "scopes": {"type": "array", "items": {
            "type": "object",
            "properties": {"request": {"enum": list(REQUEST_KINDS)}, "device_kind": {"enum": list(DEVICE_KINDS)},
                           "only_without_room": {"type": "boolean"}},
            "required": ["request", "device_kind", "only_without_room"], "additionalProperties": False}},
        "effect": {"anyOf": [{"type": "null"}, {
            "type": "object",
            "properties": {
                "media_type": {"enum": [*MEDIA_TYPES, None]},
                "name": {"type": ["string", "null"]},
                "action": {"enum": [*ACTIONS, None]},
                "limit": {"anyOf": [{"type": "null"}, {
                    "type": "object",
                    "properties": {"slot": {"enum": list(LIMIT_SLOTS)}, "min": {"type": ["integer", "null"]},
                                   "max": {"type": ["integer", "null"]}},
                    "required": ["slot", "min", "max"], "additionalProperties": False}]},
            },
            "required": ["media_type", "name", "action", "limit"], "additionalProperties": False}]},
        "examples": {
            "type": "object",
            "properties": {"changed": {"type": "array", "items": {"type": "string"}},
                           "unchanged": {"type": "array", "items": {"type": "string"}}},
            "required": ["changed", "unchanged"], "additionalProperties": False},
    },
    "required": ["kind", "names", "triggers", "scopes", "effect", "examples"],
    "additionalProperties": False,
}


# Stored with the compiled rules: a change to the contract, the prompt or the schema
# changes it, and rules compiled under another version are compiled again.
COMPILER_VERSION = hashlib.sha256(
    json.dumps(
        [REQUEST_KINDS, DEVICE_KINDS, KINDS, MEDIA_TYPES, ACTIONS, LIMIT_SLOTS, SYSTEM, SCHEMA], sort_keys=True
    ).encode()
).hexdigest()[:12]


class CompileError(Exception):
    """The text model gave no usable translation of a rule."""


async def async_compile_rule(client: ChatClient, home: str, rule: str) -> dict:
    """The rule's compiled fields plus its text, by the background model with thinking off."""
    menu = "\n".join(f"- {name}: {description}" for name, description in REQUEST_KINDS.items())
    try:
        compiled = await client.complete_json(
            f"{SYSTEM}\n\nKinds of request:\n{menu}\n\nThe home:\n{home}", rule,
            timeout=COMPILE_TIMEOUT_SECONDS, temperature=0, max_tokens=COMPILE_MAX_TOKENS, thinking=False,
            response_format={"type": "json_schema", "json_schema": {"name": "rule", "schema": SCHEMA}},
        )
    except LLMError as err:
        raise CompileError(f"{rule!r}: {err}") from err
    if compiled.get("kind") not in KINDS:
        raise CompileError(f"{rule!r}: no kind in {compiled!r}")
    return checked({"text": rule, **compiled})


def checked(compiled: dict) -> dict:
    """The record with its effect checked against the contract and its examples.

    The record with its effect kept only when the effect is a contract value and the triggers fit the rule's
    examples. Otherwise the effect is null (the decision agent hands off) and "check" says why, for diagnostics.

    A rule about words that passes loses its scopes: a model may give a rule about one name a scope covering every
    request of that kind as well. One that fails keeps them, so its scopes hand off what a too narrow trigger misses.
    """
    compiled = {**compiled, "effect": _normalized(compiled.get("effect"))}
    problem = _effect_problem(compiled) or _trigger_problem(compiled)
    if problem:
        return {**compiled, "effect": None, "check": problem}
    if compiled.get("triggers"):
        return {**compiled, "scopes": []}
    return compiled


def _normalized(effect: dict | None) -> dict | None:
    """An effect naming nothing is no effect."""
    if not effect or not any(effect.get(field) for field in ("media_type", "action", "limit")):
        return None
    return effect


def _effect_problem(compiled: dict) -> str | None:
    effect = compiled["effect"]
    if effect is None:
        return None
    if compiled["kind"] != "request":
        return f"an effect on a {compiled['kind']} rule"
    if effect.get("media_type") not in (*MEDIA_TYPES, None):
        return f"unknown media type {effect['media_type']!r}"
    if effect.get("media_type") and not (effect.get("name") or "").strip():
        return "a media type without the name it is about"
    if effect.get("action") not in (*ACTIONS, None):
        return f"unknown action {effect['action']!r}"
    limit = effect.get("limit")
    if limit is not None:
        if limit.get("slot") not in LIMIT_SLOTS:
            return f"unknown limit slot {limit.get('slot')!r}"
        if limit.get("min") is None and limit.get("max") is None:
            return "a limit without min or max"
    return None


def _trigger_problem(compiled: dict) -> str | None:
    """Only triggers can be checked here; whether an example falls in a scope needs the decider.

    A look-alike that triggers a media type effect is no problem: the decision agent applies it only when the name
    it cuts from the request is the effect's name, and hands off otherwise.
    """
    triggers = compiled.get("triggers") or []
    if not triggers:
        return None
    examples = compiled.get("examples") or {}
    changed, unchanged = examples.get("changed") or [], examples.get("unchanged") or []
    if not changed:
        return "no examples to check the triggers with"
    if missed := [text for text in changed if not trigger_fires(triggers, text)]:
        return f"triggers {triggers} miss requests the rule changes: {missed}"
    if (compiled["effect"] or {}).get("media_type"):
        return None
    if fired := [text for text in unchanged if trigger_fires(triggers, text)]:
        return f"triggers {triggers} also fire on requests the rule does not change: {fired}"
    return None


def home_context(hass: HomeAssistant) -> str:
    """The home's rooms and the devices exposed to voice, for resolving names."""
    lines = [f"{state.name} ({area_name(hass, state.entity_id) or NO_AREA_LABEL})" for state in exposed_states(hass)]
    rooms = ", ".join(sorted(area.name for area in ar.async_get(hass).async_list_areas()))
    return f"Rooms: {rooms}.\nDevices: {'; '.join(sorted(lines))}."
