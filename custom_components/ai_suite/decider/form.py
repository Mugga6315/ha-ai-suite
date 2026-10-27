"""The form flow: a model fills a fixed form (action, room, device, value), code checks it and builds the HA intent.

The model only reads the request; every decision a rule can make is made here. No HA imports.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..languages import LANGUAGE_NAMES, LANGUAGES, pattern
from ..monitor import recorder
from .api import DecisionClient, DecisionError
from .calendar import CALENDAR, CALENDAR_DESCRIPTION
from .changes import (
    CHANGE_QUESTION,
    CHANGED,
    CHANGED_DESCRIPTION,
    LOCK_CHECK,
    OPENING_CHANGES,
    can_make,
)
from .decision import (
    Candidate,
    labelled,
    lights_in,
    narrow,
    only_of_its_kind,
    parse_brightness,
    parse_temperature,
    spelled_out,
    state_text,
    with_rules,
)
from .domains import CHANGE_DOMAINS, READABLE, SWITCHED
from .forecast import FORECAST, FORECAST_DESCRIPTION
from .lists import LIST_STEP
from .locks import lock_direction
from .media import LOOKUP_TYPES, Media, kind_question, named_in
from .memory import REMEMBER, REMEMBER_DESCRIPTION
from .naming import LIGHT_WORDS, is_a_light, named_only_by, names
from .past import PAST, PAST_DESCRIPTION, STATISTIC_QUESTION
from .rooms import names_a_room
from .turn import HandOff


@dataclass(frozen=True, slots=True)
class Action:
    """An action of the form: what it does, which domains can do it, the HA intent and its slots."""
    description: str
    domains: frozenset[str]
    intent: str
    value_slot: str | None = None  # the HA slot the form's value fills
    fixed_slots: dict[str, str] = field(default_factory=dict)


_COVER = frozenset({"cover"})
MEDIA_DOMAINS = frozenset({"media_player"})
SENSOR_DOMAINS = frozenset({"sensor"})  # long-term statistics are a sensor's

ACTIONS: dict[str, Action] = {
    "turn_on": Action("switch a light, switch, fan, speaker or thermostat on", SWITCHED, "HassTurnOn"),
    "turn_off": Action("switch a light, switch, fan, speaker or thermostat off", SWITCHED, "HassTurnOff"),
    "open": Action("open a cover (screen, blind, shade)", _COVER, "HassTurnOn"),
    "close": Action("close a cover", _COVER, "HassTurnOff"),
    "set_position": Action("move a cover to a position in percent", _COVER, "HassSetPosition", "position"),
    "stop": Action("stop a moving cover, or stop the music", _COVER | MEDIA_DOMAINS, "HassStopMoving"),
    "set_brightness": Action("set a light's brightness in percent", frozenset({"light"}), "HassLightSet", "brightness"),
    "set_temperature": Action(
        "set a thermostat to a temperature", frozenset({"climate"}), "HassClimateSetTemperature", "temperature"
    ),
    # HA's on/off intents lock and unlock a lock (components/intent/__init__.py).
    "lock": Action("lock a door", frozenset({"lock"}), "HassTurnOn"),
    "unlock": Action("unlock a door", frozenset({"lock"}), "HassTurnOff"),
    "pause": Action("pause the music", MEDIA_DOMAINS, "HassMediaPause"),
    "resume": Action("continue paused music", MEDIA_DOMAINS, "HassMediaUnpause"),
    "next_track": Action("skip to the next song", MEDIA_DOMAINS, "HassMediaNext"),
    "previous_track": Action("go back to the previous song", MEDIA_DOMAINS, "HassMediaPrevious"),
    "set_volume": Action("set the volume in percent", MEDIA_DOMAINS, "HassSetVolume", "volume_level"),
    "volume_up": Action("make it louder", MEDIA_DOMAINS, "HassSetVolumeRelative", fixed_slots={"volume_step": "up"}),
    "volume_down": Action("make it quieter", MEDIA_DOMAINS, "HassSetVolumeRelative", fixed_slots={"volume_step": "down"}),
    "mute": Action("mute a speaker", MEDIA_DOMAINS, "HassMediaPlayerMute"),
    "unmute": Action("unmute a speaker (sound back on)", MEDIA_DOMAINS, "HassMediaPlayerUnmute"),
}

# Remembering is the text model's: it sees what is stored, files by topic and tells a rule from a fact; the decider's
# own path filed rules as inbox facts and added corrections beside the old fact. The decider keeps the answer, so its
# route question stays as it is.
REMEMBER_HANDOFF = "remember: the text model files memories"

# A question about one device's current state: code reads it, nothing changes.
READ = "read"
READ_DESCRIPTION = ("ask for the current state or value of one device or sensor (temperature, is it on, what plays, "
                    "a total so far today)")
# Playing names its media: the decider answers its kind, code cuts its name from the text (media.py). The LLM form
# has no media field, so there it is recognised only to hand off.
PLAY = "play"
# Thanks, greetings, goodbyes: nothing to do, the answer step replies without tools.
CHAT = "chat"
CHAT_DESCRIPTION = "only thanks, a greeting or a goodbye: nothing to do, remember, forget or find"
# The language the reply should be in: the answer step and the fallback agent sometimes reply in another one (an
# English request answered in German).
LANGUAGE_CHECK: dict[str, Any] = {"type": "choice", "instructions": "Which language is the user's message written in?",
                  "criteria": {**LANGUAGE_NAMES, "other": "another language"}}
# A decider takes some requests for chat (a request to forget something, a report that a task is done: the answer
# then claims an action that did not happen). A second, categorising question confirms it.
CHAT_CHECK = {"type": "choice", "instructions": "What is the user's message?",
              "criteria": {CHAT: "only thanks, a greeting or a goodbye, with nothing else in it",
                           "request": "asks or tells the assistant to do, remember, forget, find, tick off or tell "
                                      "something, or reports that a task is done"}}
# How many requests the latest message holds: one, several (split and decided part by part), or a group with some
# devices left out (hands off; no rule acts on "all but X", and word lists miss most ways to say it: a missed
# exclusion would switch every light of the room off). One categorising choice instead of two yes / no questions,
# which the decider does not weigh against each other: "leaves out" also fires on real two-request sentences.
# Asked about the latest message alone: with the previous turn in the state, the decider takes single follow-ups
# for several requests.
ONE, SEVERAL, ALL_BUT = "one", "several", "all_but"
REQUESTS_QUESTION = {"type": "choice", "instructions": "How many separate things does the user ask for?", "criteria": {
    ONE: "one thing",
    SEVERAL: "two or more things, each a device to change or a question to answer",
    ALL_BUT: "one change for a group of devices, except one or more of them",
}}
NO_ACTION = "none"
NO_DEVICE = "none"
NO_ROOM = "not mentioned"
OTHER_ROOM = "another room"

# A light request for a room that names no light means every light there, as HA's own sentences have it; the form's
# one device would leave the other lights on while the reply claims the room.
# In a sure room with several lights the decider says which is meant; code acts on the room only for these.
ROOM_ACTIONS = frozenset({"turn_on", "turn_off", "set_brightness"})
ALL_LIGHTS = "all"
SCOPE_QUESTION = {"type": "choice", "instructions": "Does the user mean one particular light, or all lights in the room?",
                  "criteria": {"one": "one particular light", ALL_LIGHTS: "all lights in the room"}}
CONTROLLABLE = frozenset(domain for action in ACTIONS.values() for domain in action.domains)
# The first step: what kind of request this is. Each route then asks only its own questions.
CONTROL = "control"
LIST = "list"
# The routes whose request names a device, so the room picks among devices.
_DEVICE_ROUTES = frozenset({CONTROL, READ, PLAY, PAST, CHANGED})
ROUTES = {
    CONTROL: "change a device: switch it on or off, dim it, open, close or stop a cover, set a temperature, lock or "
             "unlock a door, or control the music that plays (pause, resume, skip, volume, mute)",
    READ: READ_DESCRIPTION,
    PLAY: "play a song, artist, album, playlist or radio station",
    LIST: "a shopping or to-do list: add, tick off or remove an entry, say that a task on it is done, or ask what "
          "is on it",
    FORECAST: FORECAST_DESCRIPTION,
    CALENDAR: CALENDAR_DESCRIPTION,
    PAST: PAST_DESCRIPTION,
    CHANGED: CHANGED_DESCRIPTION,
    REMEMBER: REMEMBER_DESCRIPTION,
    CHAT: CHAT_DESCRIPTION,
    NO_ACTION: "anything else: a timer, a general question",
}

# Words for a half in a percent slot.
_HALF = pattern(lambda language: language.requests.half)
# A request to stop is a stop: a decider may answer "close" for it with high confidence, which
# no threshold catches.
STOP_WORDS = pattern(lambda language: language.requests.stop)


@dataclass(frozen=True, slots=True)
class Form:
    """The form as the decider answered it, before code checks it (command.py)."""
    several_requests: bool
    action: str
    room: str
    device: str
    value: int | None
    media: Media | None = None  # what to play, for the play action
    statistic: str | None = None  # which value of an earlier day, for the past action (past.py)
    room_wide: bool = False  # every light in the room, not one device
    change: str | None = None  # which change of a device is asked, for the changed action (changes.py)
    action_from_rule: bool = False  # a household rule set the action (household.follow): it counts as the lock words


def devices_for_form(entities: list[Candidate]) -> dict[str, Candidate]:
    """The devices the form may name, by their unique label "Name (domain, measures, area)", in entity id order.

    HA's own order changes with every start, and the decider's answers change with the order.
    """
    return labelled(sorted((e for e in entities if e.domain in CONTROLLABLE | READABLE), key=lambda e: e.entity_id))


def reply_language(code: str, p: float, threshold: float, pipeline_language: str) -> str | None:
    """The code of the language the reply should be in (languages.LANGUAGE_NAMES).

    The decider's when it is sure, else the pipeline's. Without one the answer LLM picks, and a German request with an
    English title gets an English reply. None for another language: the answer LLM replies in the user's.
    """
    chosen = code if _passes("q", p, threshold) else pipeline_language.split("-", maxsplit=1)[0].lower()
    return chosen if chosen in LANGUAGE_NAMES else None


async def decider_form(
    client: DecisionClient,
    state: str,
    text: str,
    devices: dict[str, Candidate],
    areas: dict[str, list[str]],
    threshold: float,
    language: asyncio.Task[str | None],
    rules: str | None = None,
) -> Form:
    """The form answered by a decision model in steps of narrow choice questions.

    Each question asks for one judgment, over input narrowed to what it needs: first the route, the room and whether the
    request asks for several things; then only the route's own questions, the device among the route's devices (in the
    room, when the room is sure). It cannot write numbers, so the value is parsed from the text; an answer below
    `threshold` hands off. An unsure room is no reason to hand off when the device stands without it
    (`_stands_without_room`). An unsure device leaves the device to code (command), which acts only where the room
    leaves one choice. `language`: the reply language the decider names (form.reply_language), awaited only for an
    unsure room. `rules`: the household's rules from Second Brain, shown only to the media type question
    (decision.with_rules).
    """
    first = {
        "route": {"type": "choice", "instructions": "What does the user want?", "criteria": ROUTES},
        "room": {"type": "choice", "instructions": "Which room does the user name?", "criteria": _room_menu(areas)},
    }
    latest = state_text(text)
    about_the_latest = {"requests": REQUESTS_QUESTION}
    if latest == state:
        answers = await _ask(client, state, about_the_latest | first)
    else:
        asked, on_latest = await asyncio.gather(_ask(client, state, first), _ask(client, latest, about_the_latest))
        answers = asked | on_latest
    requests, p_requests = _answer(answers, "requests")
    if requests == ALL_BUT:
        raise HandOff("the request leaves a device out")
    # Unsure, the single path decides as with any request: handing off would lose requests a decider is only a bit
    # unsure of (a play of an album); a split needs a sure "several".
    if requests == SEVERAL and _passes("requests", p_requests, threshold):
        # Decided part by part once the message is split (the agent's several-request path); the whole message's
        # route answers would fit none of them.
        return Form(True, NO_ACTION, NO_ROOM, NO_DEVICE, None)
    route = await _route(client, state, answers, threshold)
    room, p = _answer(answers, "room")
    sure_room = room if _passes("room", p, threshold) and room not in (NO_ROOM, OTHER_ROOM) else None
    # The change first: its devices make the menu (a lock word means the lock, not the door sensor beside it).
    change = await _change_asked(client, state, text, language, threshold) if route == CHANGED else None
    able = devices if change is None else {label: d for label, d in devices.items() if can_make(change, d)}
    action, device = await _route_answers(client, state, text, route, able, areas, sure_room, threshold)
    if _below("room", p, threshold):
        # An unsure room matters only where it picks the device: a question about an appointment is sure as a
        # calendar question, and its room does not matter.
        if route in _DEVICE_ROUTES and not await _stands_without_room(room, device, devices, text, state, areas, language):
            raise HandOff(f"room {room} p={p:.2f}")
        room = NO_ROOM
    if action == CHAT:
        await _confirm_chat(client, state, threshold)
    if action == REMEMBER:
        raise HandOff(REMEMBER_HANDOFF)
    media = await _media_answer(client, with_rules(state, rules), text, areas, room != NO_ROOM, threshold) \
        if action == PLAY else None
    statistic = _sure(await _ask(client, state, {"statistic": STATISTIC_QUESTION}), "statistic", threshold) \
        if action == PAST else None
    room_wide = action in ROOM_ACTIONS and sure_room is not None and lights_in(devices, sure_room) > 1 and \
        _sure(await _ask(client, state, {"scope": SCOPE_QUESTION}), "scope", threshold) == ALL_LIGHTS
    return Form(False, action, room, device, _value_in(text, action), media, statistic, room_wide, change)


async def _stands_without_room(
    room: str,
    device: str,
    devices: dict[str, Candidate],
    text: str,
    state: str,
    areas: dict[str, list[str]],
    language: asyncio.Task[str | None],
) -> bool:
    """Whether the device the decider picked holds although it doubted the room.

    It holds when the text names the device by a name only it has (and the doubt is not about a room missing from the
    home), or the request holds no room word and the device is the home's only one of its kind, so its own area places
    it. A decider may doubt whether a device's name is a place while it is sure of the device. Room words are known
    only in the languages with a module (rooms.py); `state` holds the previous exchange, whose room a follow-up keeps.
    """
    picked = devices.get(device)
    if picked is None:
        return False
    if room != OTHER_ROOM and named_only_by(picked, devices, text):
        return True
    if not only_of_its_kind(picked, devices):
        return False
    return await language in LANGUAGES and not names_a_room(state, areas)


async def _change_asked(
    client: DecisionClient, state: str, text: str, language: asyncio.Task[str | None], threshold: float
) -> str:
    """Which change of a device the user asks about.

    Locked or unlocked by the text's words (locks.py), else the decider's answer. Without those words, an opening or
    closing in a language they do not cover hands off when the decider thinks the question is about a lock.
    """
    if locking := lock_direction(text):
        return locking
    change = _sure(await _ask(client, state, {"change": CHANGE_QUESTION}), "change", threshold)
    if change in OPENING_CHANGES and await language not in LANGUAGES:
        if _passes("lock", float((await _ask(client, state, {"lock": LOCK_CHECK}))["lock"]["noul"]), threshold):
            raise HandOff(f"changed: {change}, or a lock")
    return change


async def _route(client: DecisionClient, state: str, answers: dict[str, dict], threshold: float) -> str:
    """The decider's sure route, or the winner of a run-off between its two likeliest.

    The decider's sure route; unsure, a run-off between its two likeliest routes: a decider may split a question
    about a past appointment between two routes and be sure of one when only those two are offered. Of two
    options one always has p >= 0.5, so the run-off must clear the bar halfway between chance and certainty:
    (1 + threshold) / 2, 0.75 at the default 0.5 (chosen, not calibrated).
    """
    choice, p = _answer(answers, "route")
    if _passes("route", p, threshold):
        return choice
    likeliest = sorted(answers["route"].get("probabilities") or {}, key=answers["route"]["probabilities"].get)[-2:]
    if len(likeliest) < 2:
        raise HandOff(f"route {choice} p={p:.2f}")
    question = {"type": "choice", "instructions": "What does the user want?",
                "criteria": {route: ROUTES[route] for route in reversed(likeliest)}}
    return _sure(await _ask(client, state, {"route": question}), "route", (1 + threshold) / 2)


async def _route_answers(
    client: DecisionClient,
    state: str,
    text: str,
    route: str,
    devices: dict[str, Candidate],
    areas: dict[str, list[str]],
    room: str | None,
    threshold: float,
) -> tuple[str, str]:
    """The action and the device label for the route, from a second request of the route's own questions."""
    if route == CONTROL:
        questions = {"action": {"type": "choice", "instructions": "What does the user want done?",
                                "criteria": {name: action.description for name, action in ACTIONS.items()}},
                     "device": _device_question(_lights_if_asked(devices, text), text, areas, CONTROLLABLE, room)}
        answers = await _ask(client, state, questions)
        device = _device_answer(answers, threshold)
        return _action_answer(answers, text, devices.get(device), threshold), device
    if route in (READ, PLAY, PAST, CHANGED):
        domains = {READ: READABLE, PLAY: MEDIA_DOMAINS, PAST: SENSOR_DOMAINS, CHANGED: CHANGE_DOMAINS}[route]
        answers = await _ask(client, state, {"device": _device_question(devices, text, areas, domains, room)})
        return route, _device_answer(answers, threshold)
    if route == LIST:
        return _sure(await _ask(client, state, {"list": LIST_STEP}), "list", threshold), NO_DEVICE
    return route, NO_DEVICE


def _device_question(
    devices: dict[str, Candidate], text: str, areas: dict[str, list[str]], domains: frozenset[str], room: str | None
) -> dict:
    """The question which of the route's devices the user means.

    Which of the route's devices the user means: those of the route's domains in the sure room, with those without
    a room that the text names, when there are any; else all of them narrowed by the words they share with the
    request.
    """
    fitting = {label: d for label, d in devices.items() if d.domain in domains}
    in_room = {label: d for label, d in fitting.items()
               if room is not None and (d.area == room or (d.area is None and names(d, text)))}
    # The room already narrows: its words would drop a device named in another language than the request.
    shown = in_room or _narrowed(fitting, text, areas)
    return {"type": "choice", "instructions": "Which device or sensor does the user mean?",
            "criteria": {**{label: _described(d) for label, d in shown.items()}, NO_DEVICE: None}}


def _described(device: Candidate) -> str | None:
    """The names the device answers to, its own first.

    Aliases alone in another language pull a request in the name's language away from it. An alias that repeats the
    name is left out: a name said twice lowers the decider's confidence for a
    request naming it.
    """
    aliases = [alias for alias in device.aliases if alias.casefold() != device.name.casefold()]
    return ", ".join([device.name, *aliases]) if aliases else None


async def _ask(client: DecisionClient, state: str, questions: dict[str, dict]) -> dict[str, dict]:
    try:
        return await client.ask(state, questions)
    except DecisionError as err:
        raise HandOff(f"decision server: {err}") from err


def _action_answer(answers: dict[str, dict], text: str, device: Candidate | None, threshold: float) -> str:
    """The decider's sure action.

    Unsure, a stop the text says for a sure device that can stop: a decider may be sure of the blind but not of the
    action for a request to stop it.
    """
    _, p = _answer(answers, "action")
    unsure = _below("action", p, threshold)
    if unsure and device is not None and STOP_WORDS.search(text) and device.domain in ACTIONS["stop"].domains:
        return "stop"
    return _sure(answers, "action", threshold)


def _device_answer(answers: dict[str, dict], threshold: float) -> str:
    device, p = _answer(answers, "device")
    return device if _passes("device", p, threshold) else NO_DEVICE


async def ask_sure(client: DecisionClient, state: str, question: dict, threshold: float) -> str:
    """The decider's sure answer to one choice question; HandOff when it is unsure or the server fails."""
    choice, p = _answer(await _ask(client, state, {"q": question}), "q")
    if _below("q", p, threshold):
        raise HandOff(f"{question['instructions']} {choice} p={p:.2f}")
    return choice


async def _confirm_chat(client: DecisionClient, state: str, threshold: float) -> None:
    """HandOff unless the decider is sure the message is only chat."""
    # Asked as "kind", not ask_sure's "q": the decider reads the question's name, and CHAT_CHECK was checked so.
    if _sure(await _ask(client, state, {"kind": CHAT_CHECK}), "kind", threshold) != CHAT:
        raise HandOff("chat: the message asks for something")


async def _media_answer(
    client: DecisionClient, state: str, text: str, areas: dict[str, list[str]], room_named: bool, threshold: float
) -> Media:
    """What to play: the name from the text, its kind from the decider.

    Code cuts what to play from the text, the decider says what kind of media it is (media.py); unsure, its
    likeliest kinds go with the name, for a lookup in Music Assistant before playing.
    """
    rooms = [name for area, aliases in areas.items() for name in (area, *aliases)]
    name, artist = named_in(text, rooms, room_named)
    wanted, question = kind_question(name, artist)
    answers = await _ask(client, f"{state} {wanted}", {"media": question})
    media_type, p = _answer(answers, "media")
    if _passes("media", p, threshold):
        return Media(media_type, name, artist)
    probabilities = answers["media"].get("probabilities") or {}
    likeliest = sorted(probabilities, key=lambda kind: probabilities[kind], reverse=True)
    return Media(None, name, artist, tuple(likeliest[:LOOKUP_TYPES]))


def _lights_if_asked(devices: dict[str, Candidate], text: str) -> dict[str, Candidate]:
    """For a request for a light, the control menu of lights.

    For a request for a light, the control menu holds only lights and devices named as one, when there are any:
    `_light_if_asked` hands off any other pick, and the others only draw the decider's probability away (switches
    named after the room a compound light word names).
    """
    if not LIGHT_WORDS.search(text):
        return devices
    return {label: d for label, d in devices.items() if is_a_light(d)} or devices


def _narrowed(devices: dict[str, Candidate], text: str, areas: dict[str, list[str]]) -> dict[str, Candidate]:
    """The devices sharing the most words with the request (names, rooms, aliases).

    They keep their full-list labels, so the answer can be looked up in `devices`.
    """
    kept = narrow(list(devices.values()), text, areas)
    return {label: device for label, device in devices.items() if device in kept}


def _room_menu(areas: dict[str, list[str]]) -> dict[str, str | None]:
    """The rooms; an area with aliases is described by its name and them.

    The aliases alone read as the room's only names: with an area described by its alias alone, a decider takes
    another word for the area for another room; "<area>, also called <aliases>" keeps it the area.
    """
    menu: dict[str, str | None] = {
        area: f"{area}, also called {', '.join(aliases)}" if aliases else None for area, aliases in areas.items()
    }
    menu[NO_ROOM] = "the user names no room"
    menu[OTHER_ROOM] = "the user names a room that is not in this list"
    return menu


def _sure(answers: dict[str, dict], question: str, threshold: float) -> str:
    choice, p = _answer(answers, question)
    if _below(question, p, threshold):
        raise HandOff(f"{question} {choice} p={p:.2f}")
    return choice


def _passes(qid: str, p: float, bar: float) -> bool:
    """Whether the answer to `qid` clears the bar (p >= bar); the bar is recorded for the monitor."""
    recorder.bar(qid, bar)
    return p >= bar


def _below(qid: str, p: float, bar: float) -> bool:
    """Whether the answer to `qid` falls below the bar (p < bar); the bar is recorded for the monitor.

    Not `not _passes(...)`: for a NaN probability both comparisons are False.
    """
    recorder.bar(qid, bar)
    return p < bar


def _answer(answers: dict[str, dict], question: str) -> tuple[str, float]:
    """The model's choice and its probability."""
    choice = answers[question]["choice"]
    return choice, float((answers[question].get("probabilities") or {}).get(choice, 0.0))


def _value_in(text: str, action_name: str) -> int | None:
    """The number an action needs, from the text (digits, number words or a half).

    None when the action needs none or the text has no number.
    """
    slot = ACTIONS[action_name].value_slot if action_name in ACTIONS else None
    text = spelled_out(text)
    if slot == "temperature":
        temperature = parse_temperature(text)
        return None if temperature is None else round(temperature)
    return parse_brightness(_HALF.sub("50 %", text)) if slot else None


