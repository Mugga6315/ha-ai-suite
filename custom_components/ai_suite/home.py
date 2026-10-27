"""The home as HA holds it, read for the decider, the rule compiler and the tools.

It reads the exposed devices with their rooms and names, the entity a name means, the rooms' names, the room a request
came from, the calendars' entries, and HA's own words for states.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Any, cast

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util, slugify

from .decider.calendar import Event
from .decider.decision import Candidate, RoomGroup, Target, measures
from .decider.household import Household

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CalendarRead:
    """The entries each calendar answered with, and the error of each calendar that failed."""

    events: dict[str, list[Event]]  # by calendar entity id
    failed: dict[str, str]


def exposed_candidates(hass: HomeAssistant, household: Household) -> list[Candidate]:
    """The exposed devices, their aliases in HA with the household's words for them from Second Brain's rules."""
    words = household.aliases()
    return [
        Candidate(
            state.entity_id,
            state.name,
            state.domain,
            area_name(hass, state.entity_id),
            measures(dict(state.attributes)),
            entity_aliases(hass, state.entity_id) + words.get(state.name.lower(), ()),
            playing=state.domain == "media_player" and state.state == "playing",
        )
        for state in exposed_states(hass)
    ]


def area_aliases(hass: HomeAssistant, household: Household) -> dict[str, list[str]]:
    """Each area's aliases in HA, with the household's words for it from Second Brain's rules."""
    words = household.aliases()
    return {
        area.name: sorted(area.aliases) + list(words.get(area.name.lower(), ()))
        for area in ar.async_get(hass).async_list_areas()
    }


def exposed_states(hass: HomeAssistant, domain: str | None = None) -> list[State]:
    """The states of the entities exposed to voice assistants; only the domain's, when given."""
    return [
        state
        for state in hass.states.async_all(domain)
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]


def entity_named(hass: HomeAssistant, given: str, *, domain: str | None = None, exposed_only: bool = True) -> str | None:
    """The entity `given` names: its id, its name or one of its aliases, in any case; None when no entity fits.

    Exact matches only, so no wrong entity is picked. A model often writes an exposed name slugified into an id that
    does not exist: such an id finds the entity of its domain whose name slugifies to it. `domain`: only its entities.
    `exposed_only`: only the entities exposed to voice assistants.
    """

    def allowed(entity_id: str) -> bool:
        return (domain is None or entity_id.startswith(f"{domain}.")) and (
            not exposed_only or async_should_expose(hass, conversation.DOMAIN, entity_id)
        )

    needle = given.strip().lower()
    if not needle:
        return None
    if hass.states.get(needle) is not None:
        return needle if allowed(needle) else None
    guessed_domain, _, guessed_object = needle.partition(".")

    def fits(entity_id: str, names: Iterable[Any]) -> bool:
        names = [name for name in names if isinstance(name, str)]  # the registry's marker for the computed name is none
        if any(name.strip().lower() == needle for name in names):
            return True
        return (
            bool(guessed_object)
            and entity_id.startswith(f"{guessed_domain}.")
            and any(slugify(name) == guessed_object for name in names)
        )

    for state in hass.states.async_all():
        if allowed(state.entity_id) and fits(state.entity_id, (state.attributes.get("friendly_name"),)):
            return state.entity_id
    for entry in er.async_get(hass).entities.values():
        if allowed(entry.entity_id) and fits(entry.entity_id, (entry.name, entry.original_name, *(entry.aliases or ()))):
            return entry.entity_id
    return None


def entity_aliases(hass: HomeAssistant, entity_id: str) -> tuple[str, ...]:
    """The user-set aliases; the registry may also hold a marker for the computed name (not a string)."""
    entity = er.async_get(hass).async_get(entity_id)
    return tuple(alias for alias in entity.aliases if isinstance(alias, str)) if entity else ()


def area_name(hass: HomeAssistant, entity_id: str) -> str | None:
    """Entity's own area, else its device's area (HA's resolution order)."""
    entity = er.async_get(hass).async_get(entity_id)
    if entity is None:
        return None
    area_id = entity.area_id
    if not area_id and entity.device_id and (device := dr.async_get(hass).async_get(entity.device_id)):
        area_id = device.area_id
    area = ar.async_get(hass).async_get_area(area_id) if area_id else None
    return area.name if area else None


def satellite_area(hass: HomeAssistant, device_id: str | None) -> str | None:
    """The area of the device a request came from (a voice satellite), as HA's LLM API finds it."""
    device = dr.async_get(hass).async_get(device_id) if device_id else None
    area_id = dr.async_get_effective_area_id(hass, device) if device else None
    area = ar.async_get(hass).async_get_area(area_id) if area_id else None
    return area.name if area else None


def unavailable_reason(hass: HomeAssistant, target: Target) -> str | None:
    """Why the device cannot act: HA has it as unavailable.

    HA's actions only warn about such an entity, so the reply would claim what never happened (a play on an
    unavailable player). None for a room's lights (HA's area intent picks the available ones) and an available
    device.
    """
    if isinstance(target, RoomGroup):
        return None
    state = hass.states.get(target.entity_id)
    if state is None or state.state == STATE_UNAVAILABLE:
        return f"{target.name} is not available right now"
    return None


async def read_calendars(hass: HomeAssistant, calendars: list[str], start: datetime, end: datetime) -> CalendarRead:
    """The calendars' entries from `start` to `end`, each calendar asked on its own so that a broken one hides none.

    Read without the user's context, as a read changes nothing. An entry that is no calendar entry is logged and left
    out.
    """
    events: dict[str, list[Event]] = {}
    failed: dict[str, str] = {}
    for entity_id in calendars:
        try:
            response = await hass.services.async_call(
                "calendar",
                "get_events",
                {"entity_id": entity_id, "start_date_time": start.isoformat(), "end_date_time": end.isoformat()},
                blocking=True,
                return_response=True,
            )
        except HomeAssistantError as err:
            failed[entity_id] = str(err)
            continue
        found = cast(dict[str, Any], (response or {}).get(entity_id) or {})
        events[entity_id] = []
        for raw in found.get("events", []):
            try:
                events[entity_id].append(calendar_event(raw))
            except (KeyError, ValueError) as err:
                _LOGGER.warning("%s: an entry without a start, left out: %r (%s)", entity_id, raw, err)
    return CalendarRead(events, failed)


def calendar_event(raw: dict[str, Any]) -> Event:
    """One of HA's calendar entries in the home's local time; KeyError or ValueError when its start is no date.

    HA passes on each calendar's own time zone (calendar/helper.py `event_dict_factory`), so an entry kept in UTC
    would otherwise be read at the wrong hour, and near midnight on the wrong day.
    """
    end = raw.get("end")
    return Event(raw.get("summary") or "", _local(raw["start"]), raw.get("location") or None, _local(end) if end else None)


def _local(value: str) -> datetime | date:
    """A date-only value is a date (all day); a time is in the home's local time zone."""
    if "T" not in value:
        return date.fromisoformat(value)
    if (parsed := dt_util.parse_datetime(value)) is None:
        raise ValueError(f"no date and time: {value!r}")
    return dt_util.as_local(parsed)


async def state_words(hass: HomeAssistant, domain: str, language: str) -> dict[str, str]:
    """HA's own words for a domain's states in `language`, by state."""
    prefix = f"component.{domain}.entity_component._.state."
    translations = await async_get_translations(hass, language, "entity_component", [domain])
    return {key.removeprefix(prefix): word for key, word in translations.items() if key.startswith(prefix)}
