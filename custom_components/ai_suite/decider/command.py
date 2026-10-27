"""A filled form checked and turned into what code does.

The result is an HA intent on a device or a room, a read, a play, a list, forecast, calendar, past-value or last-change
request, or what the home lacks. Every check a rule can make is made here; one that fails hands off. No HA imports.
"""

from dataclasses import dataclass, replace
import re
from typing import Any

from ..languages import alternatives, pattern
from .calendar import CALENDAR
from .changes import CHANGED, can_make
from .decision import Candidate, RoomGroup, Target, lights_in, only_of_its_kind
from .domains import READABLE, TEMPERATURE
from .forecast import FORECAST
from .form import (
    ACTIONS,
    CHAT,
    MEDIA_DOMAINS,
    NO_ACTION,
    NO_DEVICE,
    NO_ROOM,
    OTHER_ROOM,
    PLAY,
    READ,
    REMEMBER_HANDOFF,
    ROOM_ACTIONS,
    SENSOR_DOMAINS,
    STOP_WORDS,
    Action,
    Form,
)
from .lists import LIST_ACTIONS
from .locks import lock_action, lock_direction
from .media import Media
from .memory import REMEMBER
from .naming import (
    LIGHT_WORDS,
    is_a_light,
    named_best_by,
    named_better_than,
    named_only_by,
    named_without_room,
    names,
    names_a_light_kind,
)
from .past import DAY, NO_STATISTIC, PAST
from .turn import HandOff, target_slots

# A question naming one of these quantities is answered only from a device that measures it: asked for the
# temperature of a room without a sensor, any other device there would give a made-up answer.
QUANTITIES = {
    "temperature": re.compile(
        rf"{alternatives(lambda language: language.reads.temperature)}"
        rf"|\b{alternatives(lambda language: language.reads.degrees)}\b",
        re.IGNORECASE,
    ),
    "humidity": pattern(lambda language: language.reads.humidity),
}
# Quantities a home's devices measure wherever they are, not per room (such a device usually has no area): a read
# naming one that no device of the home measures is answered as missing; handed off, a text model may make a value
# up (a battery level in a home without a battery).
HOME_QUANTITIES = {
    "battery": pattern(lambda language: language.reads.battery),
}
# A read is one device's current state. It cannot answer for many devices or about the past.
_MANY = pattern(lambda language: language.reads.many)
_STOPS = frozenset({"stop", "pause"})
# A question about outdoors is answered by the weather or an outdoor device, not by a room's sensor.
_OUTDOOR = pattern(lambda language: language.reads.outdoors)
# A follow-up without its own room means the previous turn's room, which the decider does not carry over.
_FOLLOW_UP = re.compile(
    rf"^\s*(?:{alternatives(lambda language: language.reads.follow_up_opener)},?\s*)?"
    rf"{alternatives(lambda language: language.reads.follow_up_and)}"
    rf"|{alternatives(lambda language: language.reads.there)}",
    re.IGNORECASE,
)
# An amount so far is read from a total (energy, kWh), a rate now from a power device (W): a home with only one of
# them must not answer the other question with it.
_SO_FAR = pattern(lambda language: language.reads.so_far)
_NOW = pattern(lambda language: language.reads.now)
# A question for the time something happened: statistics hold values, not times.
_WHEN = pattern(lambda language: f"{language.time.when}|{language.time.time_question}")
_PAST = pattern(lambda language: f"{language.time.when}|{language.time.past}")


@dataclass(frozen=True, slots=True)
class Command:
    """What HA runs: an intent with its slots on one device."""

    action: str
    intent: str
    target: Target
    slots: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Reading:
    """What code reads: the current state of one device."""

    target: Candidate


@dataclass(frozen=True, slots=True)
class Chat:
    """Thanks, a greeting, a goodbye: nothing to do."""


@dataclass(frozen=True, slots=True)
class ListRequest:
    """A change to or a question about a list: which list and which entry, the decider answers next (lists.py)."""

    action: str


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """A question about the weather of coming days (forecast.py)."""


@dataclass(frozen=True, slots=True)
class CalendarRequest:
    """A question about the calendar (calendar.py)."""


@dataclass(frozen=True, slots=True)
class PastRequest:
    """A sensor's value over an earlier day (past.py); the day comes from the text."""

    target: Candidate
    statistic: str


@dataclass(frozen=True, slots=True)
class ChangeRequest:
    """When a device last made a change (changes.py), from its recorded state history."""

    target: Candidate
    change: str


@dataclass(frozen=True, slots=True)
class Missing:
    """What the home lacks for the request: no such room, or no device there that could do it.

    Said, not handed off: the text model, asked the same, may give a value from another room.
    """

    fact: str


@dataclass(frozen=True, slots=True)
class Playing:
    """What a player should find and play."""

    target: Candidate
    media: Media


Result = (
    Command | Reading | Playing | Missing | Chat | ListRequest | ForecastRequest | CalendarRequest | PastRequest
    | ChangeRequest
)


def command(
    form: Form, devices: dict[str, Candidate], text: str, here: str | None = None, previous: str | None = None
) -> Result:
    """What code does for the form, or HandOff when a rule says the decision path must not act.

    The HA intent, the read, the playing or what is missing for what the form asks, or HandOff when a rule says
    the decision path must not act. `here` is the area of the voice satellite the request came from, if any;
    `previous`: the conversation's previous exchange (decision.previous_turn), None on its first turn.
    """
    if (request := _request_without_device(form)) is not None:
        return request
    if form.room == OTHER_ROOM:
        return Missing("the room the user names is not in this home")
    if form.action == PAST:
        return _past_request(form, devices, text, here)
    if form.action == CHANGED:
        return _change_request(form, devices, text, here)
    if form.action == PLAY:
        return _play_request(form, devices, text, here)
    if form.action == READ and (reading := _quantity_read(form, devices, text, here, previous)) is not None:
        return reading
    if (room_lights := _room_lights(form, devices, text)) is not None:
        return room_lights
    _one_light_named(form, devices, text)
    if form.device == NO_DEVICE:
        if form.action == READ:
            raise HandOff("read: no device")
        return _missing_capable(form, devices, text)
    picked = _at_satellite(devices[form.device], devices, text, here, form.room)
    if isinstance(picked, Missing):
        return picked
    _placed(picked, devices, text, form.room, here)
    if form.action == READ:
        return _device_read(form, picked, devices, text, previous)
    return _device_command(form, picked, devices, text)


def _request_without_device(form: Form) -> Result | None:
    """The request when it names no device: chat, a list, the forecast, the calendar.

    HandOff for those the decision path leaves to the text model; None for a request about a device.
    """
    if form.several_requests:
        raise HandOff("several requests")
    if form.action == CHAT:
        return Chat()
    if form.action == REMEMBER:
        raise HandOff(REMEMBER_HANDOFF)
    if form.action in LIST_ACTIONS:
        return ListRequest(form.action)
    if form.action == FORECAST:
        return ForecastRequest()
    if form.action == CALENDAR:
        return CalendarRequest()
    if form.action == NO_ACTION:
        raise HandOff("form: no action")
    return None


def _play_request(form: Form, devices: dict[str, Candidate], text: str, here: str | None) -> Playing | Missing:
    if form.media is None:
        raise HandOff("play: the form names no media")
    player = _player(form, devices, text, here)
    return player if isinstance(player, Missing) else Playing(player, form.media)


def _quantity_read(
    form: Form, devices: dict[str, Candidate], text: str, here: str | None, previous: str | None
) -> Reading | Missing | None:
    """A read of a quantity in a place.

    The place is the named room, a follow-up's room or the satellite's area; Missing when no device of the home
    measures the quantity. None for a read of one device.
    """
    if form.room != NO_ROOM and (quantity := _quantity_in(text) or _followed(text, previous)):
        return _read_in_room(form, quantity, devices, text)
    if here is not None and (quantity := _here_asked(form, devices, text, here)):
        return _read_in_room(replace(form, room=here), quantity, devices, text)
    return _unmeasured_in_home(devices, text)


def _room_lights(form: Form, devices: dict[str, Candidate], text: str) -> Command | None:
    """Every light of the room, when the form or the text means them all; None otherwise."""
    if form.room_wide and not (form.device in devices and named_only_by(devices[form.device], devices, text)):
        if any(names(d, text) for d in devices.values() if d.domain == "light" and d.area == form.room):
            raise HandOff(f"all lights in {form.room}, but the text names some")
        if names_a_light_kind(text, devices):
            raise HandOff(f"all lights in {form.room}, but the text names a kind of light")
        return _room_command(form)
    if _means_the_rooms_lights(form, devices, text):
        return _room_command(form)
    return None


def _device_read(
    form: Form, picked: Candidate, devices: dict[str, Candidate], text: str, previous: str | None
) -> Reading:
    target = _in_room(_readable_target(picked, text), form.room, devices, text)
    if previous and form.room == NO_ROOM and target.area and _FOLLOW_UP.search(text):
        raise HandOff("a follow-up's room is the previous turn's")
    _lock_read_from_lock(target, text)
    return Reading(target)


def _device_command(form: Form, picked: Candidate, devices: dict[str, Candidate], text: str) -> Command | Missing:
    if STOP_WORDS.search(text) and form.action not in _STOPS:
        raise HandOff(f"the user asks to stop, the form says {form.action}")
    action = ACTIONS[form.action]
    try:
        capable = _capable_target(form.action, picked, devices)
    except HandOff:
        if form.room == NO_ROOM:
            raise
        return _missing_capable(form, devices, text)  # the decider picked a device, the room has none that can do it
    target = _light_if_asked(_in_room(capable, form.room, devices, text), text)
    if not form.action_from_rule:
        _lock_as_said(form.action, target, text)
    return Command(form.action, _intent(form.action, target), target, _slots(action, target, form.value))


def parts_overlap(results: list[object]) -> bool:
    """Whether two parts of one message act on the same device.

    That is the same entity twice, or a room's lights beside one of its lights. Parts that overlap may undo each other,
    or be an exclusion the decider took for two requests (all lights off, one of them left on).
    """
    targets = [target for result in results if (target := getattr(result, "target", None)) is not None]
    entities = [t.entity_id for t in targets if isinstance(t, Candidate)]
    groups = [t for t in targets if isinstance(t, RoomGroup)]
    if len(entities) != len(set(entities)) or len(groups) != len(set(groups)):
        return True
    return any(t.domain == g.domain and t.area == g.area for g in groups for t in targets if isinstance(t, Candidate))


def _past_request(
    form: Form, devices: dict[str, Candidate], text: str, here: str | None
) -> PastRequest | Missing:
    """The sensor whose earlier value is asked.

    It is one sensor, in the named room (else the satellite's), that measures what the question names.
    """
    if unmeasured := _unmeasured(form, devices, text, here):
        return unmeasured
    statistic = _statistic_asked(form, text)
    if form.device == NO_DEVICE:
        raise HandOff("past: no device")
    picked = _at_satellite(devices[form.device], devices, text, here, form.room)
    if isinstance(picked, Missing):
        return picked
    _placed(picked, devices, text, form.room, here)
    target = _in_room(picked, form.room, devices, text)
    if target.domain not in SENSOR_DOMAINS:
        raise HandOff(f"{target.entity_id} has no statistics")
    if _MANY.search(text):
        raise HandOff("past: the question names many devices")
    for quantity, named in QUANTITIES.items():
        if named.search(text) and not _measures(target, quantity):
            raise HandOff(f"{target.entity_id} does not measure {quantity}")
    return PastRequest(target, statistic)


def _statistic_asked(form: Form, text: str) -> str:
    """The earlier value the question asks for.

    The earlier value the decider names; with none, a day's average, lowest and highest (past.DAY) when the
    question asks how a quantity was, not when. Otherwise HandOff.
    """
    if form.statistic not in (None, NO_STATISTIC):
        return form.statistic
    if _quantity_in(text) and not _WHEN.search(text):
        return DAY
    raise HandOff("past: no value asked that statistics hold")


def _lock_as_said(action: str, target: Candidate, text: str) -> None:
    """HandOff unless an action on a lock is the one the text's lock words say (locks.py).

    A decider may lock the door for a German unlock command whose verb is split; closing a door means locking it to
    some people and not to others, so the request decides, not the decider.
    """
    if target.domain != "lock":
        return
    said = lock_action(text)
    if said != action:
        raise HandOff(f"{target.entity_id}: the decider says {action}, the words say {said or 'neither'}")


def _lock_read_from_lock(target: Candidate, text: str) -> None:
    """Hand off a question in lock words that a device other than a lock would answer.

    HandOff when a question in lock words (locks.py) would be answered by a device that is no lock: a decider may
    pick another device's sensor for a question whether a door is locked, and that sensor's "off" is worded as not
    locked. Handed off, not declined: a home may hold its lock as another kind of entity, which the text model
    can still find.
    """
    if target.domain != "lock" and lock_direction(text) is not None:
        raise HandOff(f"{target.entity_id} is no lock, the question asks whether one is locked")


def _change_request(
    form: Form, devices: dict[str, Candidate], text: str, here: str | None
) -> ChangeRequest | Missing:
    """The device whose last change is asked.

    It is the decider's sure pick, in the named room (else the satellite's), able to make the change.
    """
    if form.change is None or form.device == NO_DEVICE:
        raise HandOff("changed: no device")
    picked = _at_satellite(devices[form.device], devices, text, here, form.room)
    if isinstance(picked, Missing):
        return picked
    _placed(picked, devices, text, form.room, here)
    target = _in_room(picked, form.room, devices, text)
    if not can_make(form.change, target):
        raise HandOff(f"{target.entity_id} cannot be {form.change}")
    return ChangeRequest(target, form.change)


def _room_command(form: Form) -> Command:
    """Every light in the form's room: HA's intent with area and domain."""
    if form.action not in ROOM_ACTIONS or form.room in (NO_ROOM, OTHER_ROOM):
        raise HandOff(f"{form.action} on a whole room")
    group = RoomGroup("light", form.room)
    action = ACTIONS[form.action]
    return Command(form.action, action.intent, group, _slots(action, group, form.value))


def _means_the_rooms_lights(form: Form, devices: dict[str, Candidate], text: str) -> bool:
    """Whether a light request means every light of the room it names.

    It does when the room has several, and the text names no light and no kind of light, as HA's own sentences have it.
    The room's lights are those HA's area intent picks, by the registry's areas: a light without a room, or with the
    room only in its name, is not one of them.
    """
    return (form.action in ROOM_ACTIONS and form.room not in (NO_ROOM, OTHER_ROOM)
            and lights_in(devices, form.room) > 1 and LIGHT_WORDS.search(text) is not None
            and not any(names(d, text) for d in devices.values() if d.domain == "light")
            and not names_a_light_kind(text, devices))


def _one_light_named(form: Form, devices: dict[str, Candidate], text: str) -> None:
    """Hand off when one of several lights would be switched without the text naming it.

    HandOff when one light of several in the named room would be switched without the text naming it: the
    decider's guess could be the wrong lamp, and a wrong action weighs more than a question back.
    """
    picked = devices.get(form.device)
    if (form.action in ROOM_ACTIONS and picked is not None and picked.domain == "light"
            and form.room not in (NO_ROOM, OTHER_ROOM) and lights_in(devices, form.room) > 1
            and not named_only_by(picked, devices, text) and not named_without_room(picked, devices, text)):
        raise HandOff(f"one of the lights in {form.room}, none named")


def _quantity_in(text: str) -> str | None:
    """The one quantity the question names (temperature, humidity); None for none or several."""
    named = [quantity for quantity, words in QUANTITIES.items() if words.search(text)]
    return named[0] if len(named) == 1 else None


def _followed(text: str, previous: str | None) -> str | None:
    """The quantity of the previous exchange, for a follow-up that names only a room.

    Such a follow-up asks for another room after one room's temperature. The quantity is the previous question's, or the
    reply's when that question was a follow-up too. None for none or several, and for a text that is no follow-up.
    """
    if previous is None or not _FOLLOW_UP.search(text):
        return None
    return _quantity_in(previous)


def _here_asked(form: Form, devices: dict[str, Candidate], text: str, here: str | None) -> str | None:
    """The quantity a satellite request asks about where it is.

    The quantity of a place a satellite request asks about where it is: no room, no device named, not outdoors,
    no follow-up. Else a question about the satellite's own room may read another room's sensor.
    """
    picked = devices.get(form.device)
    if here is None or form.room != NO_ROOM or _OUTDOOR.search(text) or _FOLLOW_UP.search(text):
        return None
    if picked is not None and named_only_by(picked, devices, text):
        return None
    return _quantity_in(text)


def _unmeasured(form: Form, devices: dict[str, Candidate], text: str, here: str | None) -> Missing | None:
    """Missing when the place a quantity is asked of has no device measuring it.

    Missing when the place a quantity is asked of (the named room, else the satellite's, as for a current read)
    has no device measuring it. Checked before what the past question asks: a question how a quantity was in the
    satellite's room, asked where no device measures it, names no statistic, and the text model, handed it, gives
    another room's values as this room's.
    """
    place: str | None
    if form.room != NO_ROOM and (quantity := _quantity_in(text)):
        place = form.room
    elif quantity := _here_asked(form, devices, text, here):
        place = here
    else:
        return None
    if any(d.area == place and _measures(d, quantity) for d in devices.values()):
        return None
    return Missing(f"there is no device in {place} that measures {quantity}")


def _unmeasured_in_home(devices: dict[str, Candidate], text: str) -> Missing | None:
    """Missing when the read names a quantity of HOME_QUANTITIES that no device of the home measures."""
    for quantity, words in HOME_QUANTITIES.items():
        if words.search(text) and not any(_measures(d, quantity) for d in devices.values()):
            return Missing(f"there is no device in this home that measures {quantity}")
    return None


def _read_in_room(form: Form, quantity: str, devices: dict[str, Candidate], text: str) -> Reading | Missing:
    """A quantity in a named room, read from the room's device that measures it.

    That is the decider's pick when it is one of them, else the only one. None there: Missing; several and no pick among
    them: HandOff.
    """
    _one_device_now(text)
    fitting = [d for d in devices.values() if d.area == form.room and _measures(d, quantity)]
    if not fitting:
        return Missing(f"there is no device in {form.room} that measures {quantity}")
    picked = devices.get(form.device)
    if picked not in fitting and len(fitting) > 1:
        raise HandOff(f"{len(fitting)} devices in {form.room} measure {quantity}")
    return Reading(_readable_target(picked if picked in fitting else fitting[0], text))


def _missing_capable(form: Form, devices: dict[str, Candidate], text: str) -> Missing:
    """Missing when the named room has no device that could do the action.

    No device picked for an action in a named room: Missing when the room has none that could do it, and the text
    names no device without a room that could: such a device may be anywhere (a request naming it with a room must
    not be answered as no device in that room).
    """
    action = ACTIONS[form.action]
    if form.room == NO_ROOM or any(d.area == form.room and d.domain in action.domains for d in devices.values()):
        raise HandOff(f"form {form.action}: no device")
    if any(d.area is None and d.domain in action.domains and names(d, text) for d in devices.values()):
        raise HandOff(f"form {form.action}: the text names a device without a room")
    return Missing(f"there is no device in {form.room} that can {action.description}")


def _player(form: Form, devices: dict[str, Candidate], text: str, here: str | None) -> Candidate | Missing:
    """The player to play on.

    That is the one the request names, else the only player in the named room or, without one, in the satellite's area
    (as HA's own search-and-play does). Several or none there: HandOff, or Missing when the decider's pick is in another
    room than the satellite's.
    """
    if form.device != NO_DEVICE:
        picked = devices[form.device]
        unplaced = form.room == NO_ROOM and here is None and not named_only_by(picked, devices, text)
        if unplaced and sum(d.domain in MEDIA_DOMAINS for d in devices.values()) > 1:
            # A typed play request with several players: the decider's pick is a guess. No room, no satellite and
            # no player named leaves the player open.
            raise HandOff("several players, the request places none")
        local = _at_satellite(picked, devices, text, here, form.room)
        if isinstance(local, Missing):
            return local
        target = _in_room(local, form.room, devices, text)
        if target.domain not in MEDIA_DOMAINS:
            raise HandOff(f"{target.entity_id} cannot play")
        return target
    area = here if form.room == NO_ROOM else form.room
    players = [d for d in devices.values() if d.domain in MEDIA_DOMAINS and area is not None and d.area == area]
    if len(players) != 1:
        raise HandOff(f"{len(players)} players in {area}")
    return players[0]


def _at_satellite(
    picked: Candidate, devices: dict[str, Candidate], text: str, here: str | None, room: str
) -> Candidate | Missing:
    """The device of the picked kind in the satellite's area.

    Spoken to a satellite, naming neither a room nor the device: the device of the picked kind in the satellite's
    area, as HA's own prompt says ("You are in area ..."). Several of that kind there: HandOff. None there: Missing
    when the pick is in another room, never the pick as the one asked: a question about the satellite's own room must
    not read another room's sensor. A pick without an area stands: it may be anywhere.
    """
    if here is None or room != NO_ROOM or picked.area == here or named_only_by(picked, devices, text):
        return picked
    kind = (picked.domain, picked.measures)
    local = [d for d in devices.values() if (d.domain, d.measures) == kind and d.area == here]
    if len(local) > 1:
        raise HandOff(f"several {picked.domain} devices in {here}, the satellite's area")
    if local:
        return local[0]
    if picked.area is None:
        return picked
    return Missing(f"there is no device like {picked.name} in {here}, where the user is; {picked.name} is in "
                   f"{picked.area}")


def _light_if_asked(target: Candidate, text: str) -> Candidate:
    """A request for a light acts on a light, or on a switch named as one.

    Else a decider may switch on another switch named after the room a compound light word names.
    """
    if LIGHT_WORDS.search(text) and not is_a_light(target):
        raise HandOff(f"the user asks for a light, {target.entity_id} is none")
    return target


def _in_room(target: Candidate, room: str, devices: dict[str, Candidate], text: str) -> Candidate:
    """The target, when the room the request names allows it.

    The target when the request names no room or the target is in it, or when the target has no room and the
    text names it by a name only it has: an area is not mandatory in HA, and a room word cannot place such a device.
    """
    if room in (NO_ROOM, target.area):
        return target
    if target.area is None and named_only_by(target, devices, text):
        return target
    raise HandOff(f"{target.entity_id} is not in {room}")


def _readable_target(picked: Candidate, text: str) -> Candidate:
    """The picked device, when it can answer the question.

    The picked device, when it can be read, the question asks for one device's current state, and the device
    measures every quantity the question names. Otherwise HandOff.
    """
    if picked.domain not in READABLE:
        raise HandOff(f"{picked.entity_id} cannot be read")
    _one_device_now(text)
    outdoor_named = any(_OUTDOOR.search(name) for name in (picked.name, *picked.aliases, picked.area or ""))
    if _OUTDOOR.search(text) and picked.domain != "weather" and not outdoor_named:
        raise HandOff(f"the question is about outdoors, {picked.entity_id} is not")
    for quantity, named in QUANTITIES.items():
        if named.search(text) and not _measures(picked, quantity):
            raise HandOff(f"{picked.entity_id} does not measure {quantity}")
    kind = (picked.measures or "").split(" ")[0]
    if (kind == "power" and _SO_FAR.search(text)) or (kind == "energy" and _NOW.search(text)):
        raise HandOff(f"{picked.entity_id} measures {kind}, the question asks for the other")
    return picked


def _one_device_now(text: str) -> None:
    """HandOff unless the question asks for one device's current state."""
    if _MANY.search(text):
        raise HandOff("a read answers for one device, the question names many")
    if _PAST.search(text):
        raise HandOff("a read gives the current state, the question asks about the past")


def _measures(device: Candidate, quantity: str) -> bool:
    if quantity == "temperature" and device.domain in TEMPERATURE:
        return True
    return quantity in (device.measures or "")


def _capable_target(action_name: str, picked: Candidate, devices: dict[str, Candidate]) -> Candidate:
    """The picked device, or the one device in its room that can do the action.

    A mute request for which the decider picked the satellite's microphone-mute switch means the room's media player.
    Otherwise HandOff.
    """
    domains = ACTIONS[action_name].domains
    if picked.domain in domains:
        return picked
    capable = [d for d in devices.values() if d.domain in domains and picked.area and d.area == picked.area]
    if len(capable) != 1:
        raise HandOff(f"{action_name} is not possible on {picked.entity_id}")
    return capable[0]


def _intent(action_name: str, target: Candidate) -> str:
    # HA stops covers and valves only: stopping a media player is a pause.
    if action_name == "stop" and target.domain == "media_player":
        return "HassMediaPause"
    return ACTIONS[action_name].intent


def _slots(action: Action, target: Target, value: int | None) -> dict[str, Any]:
    slots: dict[str, Any] = {**target_slots(target), **action.fixed_slots}
    if action.value_slot:
        if value is None:
            raise HandOff(f"{action.intent} without a value")
        slots[action.value_slot] = value
    return slots


def _placed(picked: Candidate, devices: dict[str, Candidate], text: str, room: str, here: str | None) -> None:
    """Hand off when nothing places the decider's pick among the devices of its kind.

    HandOff when nothing places the decider's pick among the devices of its kind: no room named, not in the
    satellite's area, not the home's only one of its kind, and the text's words name it no better than another of
    its kind, and it is not the only one of its kind that plays. Otherwise the decider's pick is a guess: one of two
    lamps that share their name's words, one of two thermostats; for a question what plays with two players, the
    playing one is the one meant. Also with a room: HandOff when the text names another device of its kind better (a
    room taken from the previous answer must not turn a follow-up that names a device into a read of another device
    of its kind).
    """
    kind = [d for d in devices.values() if (d.domain, d.measures) == (picked.domain, picked.measures)]
    if named_better_than(picked, kind, text):
        raise HandOff(f"{picked.entity_id}: the text names another device of its kind better")
    if room != NO_ROOM or (here is not None and picked.area == here) or only_of_its_kind(picked, devices):
        return
    if named_only_by(picked, devices, text) or named_best_by(picked, kind, text):
        return
    if picked.playing and sum(d.playing for d in kind) == 1:
        return
    raise HandOff(f"{picked.entity_id}: neither a room nor its name sets it apart from the others of its kind")
