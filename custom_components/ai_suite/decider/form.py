"""The form flow: a model fills a fixed form (action, room, device, value), code checks it and builds the HA intent."""

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
    previous_user_text,
    spelled_out,
    state_text,
    with_rules,
)
from .domains import CHANGE_DOMAINS, READABLE, SWITCHED
from .forecast import FORECAST, FORECAST_DESCRIPTION
from .lists import LIST_ACTIONS, LIST_STEP
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

# Remembering is the text model's: it sees what is stored and tells a rule from a fact. The route only recognises it.
REMEMBER_HANDOFF = "remember: the text model files memories"

# A question about one device's current state: code reads it, nothing changes.
READ = "read"
READ_DESCRIPTION = ("ask for the current state or value of one device or sensor (temperature, is it on, what plays, "
                    "a total so far today)")
# The decider answers the media's kind, code cuts its name from the text (media.py).
PLAY = "play"
# Thanks, greetings, goodbyes: nothing to do, the answer step replies without tools.
CHAT = "chat"
CHAT_DESCRIPTION = "only thanks, a greeting or a goodbye: nothing to do, remember, forget or find"
# The language the reply should be in: the answer step and the fallback agent sometimes reply in another one.
LANGUAGE_CHECK: dict[str, Any] = {"type": "choice", "instructions": "Which language is the user's message written in?",
                  "criteria": {**LANGUAGE_NAMES, "other": "another language"}}
# A decider takes some requests for chat, and the answer then claims an action that did not happen. A second,
# categorising question confirms it.
CHAT_CHECK = {"type": "choice", "instructions": "What is the user's message?",
              "criteria": {CHAT: "only thanks, a greeting or a goodbye, with nothing else in it",
                           "request": "asks or tells the assistant to do, remember, forget, find, tick off or tell "
                                      "something, or reports that a task is done"}}
# A decider takes some calendar changes for questions, and nothing changes. A second, categorising question confirms
# the request only asks; changes are the text model's, which has the tool that adds an entry.
CALENDAR_ASK, CALENDAR_CHANGE = "ask", "change"
CALENDAR_CHECK = {"type": "choice", "instructions": "What does the user want with the calendar?",
                  "criteria": {CALENDAR_ASK: "asks what is in the calendar, or when an appointment, birthday or event "
                                             "is or was",
                               CALENDAR_CHANGE: "wants an entry added, entered, moved, changed or deleted"}}
# "All but X" hands off: no rule acts on it, and a missed exclusion would switch every light of the room off. Asked
# about the latest message alone: with the turn before in the state, the decider takes follow-ups for several requests.
ONE, SEVERAL, ALL_BUT = "one", "several", "all_but"
REQUESTS_QUESTION = {"type": "choice", "instructions": "How many separate things does the user ask for?", "criteria": {
    ONE: "one thing",
    SEVERAL: "two or more things, each a device to change or a question to answer",
    ALL_BUT: "one change for a group of devices, except one or more of them",
}}
# A command for later is the text model's, which has the schedule's tools; the decision path would act now. This
# question and the route's "none" each miss some, together far fewer.
NOW, FOR_LATER = "no", "yes"
LATER_QUESTION = {
    "type": "choice",
    "instructions": "Does the user's message set a time, delay or duration for the action, or refer to an action "
                    "planned for later?",
    "criteria": {
        NOW: "no: the action is meant now",
        FOR_LATER: "yes: in 10 minutes, at 15:30, tonight, later, for an hour, or it cancels or lists something "
                   "scheduled",
    },
}
NO_ACTION = "none"
NO_DEVICE = "none"
NO_ROOM = "not mentioned"
OTHER_ROOM = "another room"

# A light request for a room that names no light means every light there, as in HA's own sentences; the form's one
# device would leave the other lights on. Code acts on the room only for these actions.
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
# The routes that act: a command for later among them hands off. A list acts only where its entry changes (below).
_ACTING_ROUTES = frozenset({CONTROL, PLAY})
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
    NO_ACTION: "anything else: a timer, a general question, or do something at a later time, after a delay or only "
               "for a while (in 10 minutes, at 15:30, tonight, for an hour), or cancel or list something planned for "
               "later",
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
    room_named: bool = False  # a room word in the request or the user's turn before; else the room is a guess
    room_clear: bool = False  # the decider's room answer clears the run-off bar

    @property
    def names_room(self) -> bool:
        """Whether the user named the form's room: a household rule only for requests without one does not apply."""
        return self.room != NO_ROOM and self.room_named


def devices_for_form(entities: list[Candidate]) -> dict[str, Candidate]:
    """The devices the form may name, by their unique label "Name (domain, measures, area)", in entity id order."""
    # HA's own order changes with every start, and the decider's answers change with the order.
    return labelled(sorted((e for e in entities if e.domain in CONTROLLABLE | READABLE), key=lambda e: e.entity_id))


def reply_language(code: str, p: float, threshold: float, pipeline_language: str) -> str | None:
    """The reply's language code (languages.LANGUAGE_NAMES): the decider's when it is sure, else the pipeline's.

    None for another language: the answer LLM replies in the user's.
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

    HandOff for an answer below `threshold`. `language`: the reply language the decider names, awaited only for an
    unsure room. `rules`: the household's rules, shown only to the media type question.
    """
    first = {
        "route": {"type": "choice", "instructions": "What does the user want?", "criteria": ROUTES},
        "room": {"type": "choice", "instructions": "Which room does the user name?", "criteria": _room_menu(areas)},
    }
    latest = state_text(text)
    about_the_latest = {"requests": REQUESTS_QUESTION, "later": LATER_QUESTION}
    if latest == state:
        answers = await _ask(client, state, about_the_latest | first)
        later = [answers]
    else:
        # A follow-up's time may be in the turn before.
        asked, on_latest = await asyncio.gather(
            _ask(client, state, first | {"later": LATER_QUESTION}), _ask(client, latest, about_the_latest)
        )
        answers = asked | on_latest
        later = [on_latest, asked]
    requests, p_requests = _answer(answers, "requests")
    if requests == ALL_BUT:
        raise HandOff("the request leaves a device out")
    # Unsure, the single path decides: handing off would lose requests a decider is only a bit unsure of.
    if requests == SEVERAL and _passes("requests", p_requests, threshold):
        # The whole message's route answers fit none of its parts. A time for later holds for the whole message: a
        # split may drop it.
        _check_now(later, threshold)
        return Form(True, NO_ACTION, NO_ROOM, NO_DEVICE, None)
    route = await _route(client, state, answers, threshold)
    if route in _ACTING_ROUTES:
        _check_now(later, threshold)
    room, p = _answer(answers, "room")
    if room == OTHER_ROOM and route in _DEVICE_ROUTES and names_a_room(text, areas):
        # A room the menu lacks may be one of the home's, misheard or in another language: the text model can tell.
        raise HandOff("another room: a room word the menu lacks")
    sure_room = room if _passes("room", p, threshold) and room not in (NO_ROOM, OTHER_ROOM) else None
    # The change first: its devices make the menu (a lock word means the lock, not the door sensor beside it).
    change = await _change_asked(client, state, text, language, threshold) if route == CHANGED else None
    able = devices if change is None else {label: d for label, d in devices.items() if can_make(change, d)}
    action, device = await _route_answers(client, state, text, route, able, areas, sure_room, threshold)
    if route == LIST and LIST_ACTIONS[action].intent is not None:  # a question about the list changes nothing
        _check_now(later, threshold)
    if _below("room", p, threshold):
        # An unsure room matters only where it picks the device.
        if route in _DEVICE_ROUTES and not await _stands_without_room(room, device, devices, text, state, areas, language):
            raise HandOff(_unsure(f"room {room}", p, threshold))
        room = NO_ROOM
    if action == CHAT:
        await _confirm_chat(client, state, threshold)
    if action == CALENDAR:
        await _confirm_calendar_question(client, state, threshold)
    if action == REMEMBER:
        raise HandOff(REMEMBER_HANDOFF)
    media = await _media_answer(client, with_rules(state, rules), text, areas, room != NO_ROOM, threshold) \
        if action == PLAY else None
    statistic = _sure(await _ask(client, state, {"statistic": STATISTIC_QUESTION}), "statistic", threshold) \
        if action == PAST else None
    before = previous_user_text(state)
    room_named = names_a_room(text, areas) or names_a_room(before, areas)
    # A room a request names is sure well above the bar; the decider gave rooms no word names well below it.
    room_clear = _passes("room_clear", p, _run_off(threshold))
    room_wide = action in ROOM_ACTIONS and sure_room is not None and room_named and room_clear and \
        lights_in(devices, sure_room) > 1 and asks_for_lights(action, text) and \
        _sure(await _ask(client, state, {"scope": SCOPE_QUESTION}), "scope", threshold) == ALL_LIGHTS
    return Form(False, action, room, device, _value_in(text, action), media, statistic, room_wide, change,
                room_named=room_named, room_clear=room_clear)


async def _stands_without_room(
    room: str,
    device: str,
    devices: dict[str, Candidate],
    text: str,
    state: str,
    areas: dict[str, list[str]],
    language: asyncio.Task[str | None],
) -> bool:
    """Whether the device the decider picked holds although it doubted the room."""
    picked = devices.get(device)
    if picked is None:
        return False
    if room != OTHER_ROOM and named_only_by(picked, devices, text):
        return True
    if not only_of_its_kind(picked, devices):
        return False
    # Room words are known only in the languages with a module (rooms.py); a follow-up keeps the room of `state`.
    return await language in LANGUAGES and not names_a_room(state, areas)


async def _change_asked(
    client: DecisionClient, state: str, text: str, language: asyncio.Task[str | None], threshold: float
) -> str:
    """Which change of a device the user asks about; HandOff for an opening that may be a lock's."""
    if locking := lock_direction(text):
        return locking
    change = _sure(await _ask(client, state, {"change": CHANGE_QUESTION}), "change", threshold)
    if change in OPENING_CHANGES and await language not in LANGUAGES:
        if _passes("lock", float((await _ask(client, state, {"lock": LOCK_CHECK}))["lock"]["noul"]), threshold):
            raise HandOff(f"changed: {change}, or a lock")
    return change


async def _route(client: DecisionClient, state: str, answers: dict[str, dict], threshold: float) -> str:
    """The decider's sure route, or the winner of a run-off between its two likeliest."""
    choice, p = _answer(answers, "route")
    if _passes("route", p, threshold):
        return choice
    likeliest = sorted(answers["route"].get("probabilities") or {}, key=answers["route"]["probabilities"].get)[-2:]
    if len(likeliest) < 2:
        raise HandOff(_unsure(f"route {choice}", p, threshold))
    question = {"type": "choice", "instructions": "What does the user want?",
                "criteria": {route: ROUTES[route] for route in reversed(likeliest)}}
    # Of two options one always has p >= 0.5: the run-off needs the higher bar.
    return _sure(await _ask(client, state, {"route": question}), "route", _run_off(threshold))


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
    """The question which of the route's devices the user means: those in the sure room, else those the words fit."""
    fitting = {label: d for label, d in devices.items() if d.domain in domains}
    in_room = {label: d for label, d in fitting.items()
               if room is not None and (d.area == room or (d.area is None and names(d, text)))}
    # The room already narrows: its words would drop a device named in another language than the request.
    shown = in_room or _narrowed(fitting, text, areas)
    return {"type": "choice", "instructions": "Which device or sensor does the user mean?",
            "criteria": {**{label: _described(d) for label, d in shown.items()}, NO_DEVICE: None}}


def _described(device: Candidate) -> str | None:
    """The names the device answers to, its own first."""
    # A name said twice lowers the decider's confidence for a request naming it.
    aliases = [alias for alias in device.aliases if alias.casefold() != device.name.casefold()]
    return ", ".join([device.name, *aliases]) if aliases else None


async def _ask(client: DecisionClient, state: str, questions: dict[str, dict]) -> dict[str, dict]:
    try:
        return await client.ask(state, questions)
    except DecisionError as err:
        raise HandOff(f"decision server: {err}") from err


def _action_answer(answers: dict[str, dict], text: str, device: Candidate | None, threshold: float) -> str:
    """The decider's sure action; unsure, a stop the text says for a sure device that can stop."""
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
        raise HandOff(_unsure(f"{question['instructions']} {choice}", p, threshold))
    return choice


async def _confirm_chat(client: DecisionClient, state: str, threshold: float) -> None:
    """HandOff unless the decider is sure the message is only chat."""
    # Asked as "kind", not ask_sure's "q": the decider reads the question's name, and CHAT_CHECK was checked so.
    if _sure(await _ask(client, state, {"kind": CHAT_CHECK}), "kind", threshold) != CHAT:
        raise HandOff("chat: the message asks for something")


async def _confirm_calendar_question(client: DecisionClient, state: str, threshold: float) -> None:
    """HandOff unless the decider is sure the request only asks about the calendar."""
    # Asked as "kind", like CHAT_CHECK: the decider reads the question's name.
    bar = _run_off(threshold)  # a wrong "ask" reads entries out while the entry asked for is not added
    choice, p = _answer(await _ask(client, state, {"kind": CALENDAR_CHECK}), "kind")
    unsure = _below("kind", p, bar)  # before the answer is read: the monitor shows the bar for either hand-off
    if choice != CALENDAR_ASK:
        raise HandOff(f"calendar: the request changes the calendar, kind {choice} p={p:.2f}")
    if unsure:
        raise HandOff(f"calendar: the request may change the calendar, {_unsure(f'kind {choice}', p, bar)}")


async def _media_answer(
    client: DecisionClient, state: str, text: str, areas: dict[str, list[str]], room_named: bool, threshold: float
) -> Media:
    """What to play: the name from the text, its kind from the decider.

    Unsure, its likeliest kinds go with the name, for a lookup in Music Assistant before playing.
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
    """For a request for a light, the control menu of lights and devices named as one, when there are any."""
    if not LIGHT_WORDS.search(text):
        return devices
    return {label: d for label, d in devices.items() if is_a_light(d)} or devices


def _narrowed(devices: dict[str, Candidate], text: str, areas: dict[str, list[str]]) -> dict[str, Candidate]:
    """The devices sharing the most words with the request (names, rooms, aliases), by their full-list labels."""
    kept = narrow(list(devices.values()), text, areas)
    return {label: device for label, device in devices.items() if device in kept}


def _room_menu(areas: dict[str, list[str]]) -> dict[str, str | None]:
    """The rooms; an area with aliases is described by its name and them."""
    menu: dict[str, str | None] = {
        area: f"{area}, also called {', '.join(aliases)}" if aliases else None for area, aliases in areas.items()
    }
    menu[NO_ROOM] = "the user names no room"
    menu[OTHER_ROOM] = "the user names a room that is not in this list"
    return menu


def _check_now(later: list[dict[str, dict]], threshold: float) -> None:
    """HandOff unless every answer to the later question is a sure "now"."""
    bar = _run_off(threshold)  # a wrong "now" acts at once
    for answers in later:
        choice, p = _answer(answers, "later")
        if choice != NOW:
            raise HandOff(f"a command for later: later {choice} p={p:.2f}")
        if _below("later", p, bar):
            raise HandOff(f"a command for later? {_unsure(f'later {choice}', p, bar)}")


def _sure(answers: dict[str, dict], question: str, threshold: float) -> str:
    choice, p = _answer(answers, question)
    if _below(question, p, threshold):
        raise HandOff(_unsure(f"{question} {choice}", p, threshold))
    return choice


def asks_for_lights(action: str, text: str) -> bool:
    """Whether a request may be for all of a room's lights: a light word in it, or dimming, which only lights do."""
    return action == "set_brightness" or LIGHT_WORDS.search(text) is not None


def _run_off(threshold: float) -> float:
    """The bar halfway between the threshold and certainty: (1 + threshold) / 2, 0.75 at the default 0.5."""
    return (1 + threshold) / 2


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


def _unsure(answer: str, p: float, bar: float) -> str:
    """The hand-off reason for an answer below its bar, with the bar: the run-off's is not the threshold."""
    return f"{answer} p={p:.2f} < {bar:.2f}"


def _answer(answers: dict[str, dict], question: str) -> tuple[str, float]:
    """The model's choice and its probability."""
    choice = answers[question]["choice"]
    return choice, float((answers[question].get("probabilities") or {}).get(choice, 0.0))


def _value_in(text: str, action_name: str) -> int | None:
    """The number an action needs, from the text (digits, number words or a half); None without one."""
    slot = ACTIONS[action_name].value_slot if action_name in ACTIONS else None
    text = spelled_out(text)
    if slot == "temperature":
        temperature = parse_temperature(text)
        return None if temperature is None else round(temperature)
    return parse_brightness(_HALF.sub("50 %", text)) if slot else None


