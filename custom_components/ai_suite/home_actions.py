"""A decided command run in Home Assistant.

Devices go through HA's intents, music through Music Assistant; lists, forecasts, calendars, earlier values and last
changes are read through HA's actions and the recorder. Each run gives the note the answer model words (api.DONE,
NOT_POSSIBLE, NOTHING_TO_DO, or the facts read); what cannot be done as decided hands off.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.recorder import get_instance, history
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, intent
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .decider import command as decided
from .decider.api import DONE, NOT_POSSIBLE, NOTHING_TO_DO
from .decider.calendar import (
    AHEAD_DAYS,
    BACK_DAYS,
    NO_EVENT,
    Event,
    day_facts,
    event_fact,
    event_question,
    looks_back,
)
from .decider.changes import change_fact, last_change
from .decider.days import day_named, days_asked, past_day_asked
from .decider.decision import Candidate, RoomGroup, located, read_answer
from .decider.forecast import forecast_days, forecast_facts
from .decider.lists import (
    LIST_ACTIONS,
    NO_ITEM,
    checked_action,
    item_asked,
    item_question,
    item_to_add,
    list_question,
)
from .decider.media import Media, type_found
from .decider.past import STATISTIC_TYPES, past_note
from .decider.turn import HandOff, Timings, describe
from .home import read_calendars, state_words, unavailable_reason
from .monitor import recorder as monitor

# Player states whose media attributes are what plays now; other states keep the last title.
MEDIA_CURRENT = frozenset({"playing", "paused", "buffering"})
# The integration whose players play by name (music_assistant.play_media).
MUSIC_ASSISTANT = "music_assistant"
TODO = "todo"
WEATHER = "weather"
CALENDAR = "calendar"
RECORDER = "recorder"
DAY = timedelta(days=1)
# State changes read for a last-change question, newest first: enough for a busy sensor's days the recorder keeps.
CHANGES_READ = 1000
CALENDAR_AHEAD = timedelta(days=AHEAD_DAYS)
CALENDAR_BACK = timedelta(days=BACK_DAYS)

# One choice question over the turn's state; the decider's sure answer, HandOff when it is unsure (form.ask_sure).
Ask = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(slots=True)
class HomeActions:
    """Runs one decided command for one request (`user_input`, in the reply language)."""

    hass: HomeAssistant
    user_input: conversation.ConversationInput
    timings: Timings
    entities: list[Candidate]  # the exposed devices
    ask: Ask

    async def async_run(self, command: decided.Result) -> intent.IntentResponse:
        """The note for the answer model; HandOff when the command cannot be done as decided."""
        match command:
            case decided.Chat():
                self.timings.mark("chat")
                return self._note(f"{NOTHING_TO_DO}: the user thanks, greets or ends the conversation")
            case decided.ListRequest():
                return await self._list(command)
            case decided.ForecastRequest():
                return await self._forecast()
            case decided.CalendarRequest():
                return await self._calendar()
            case decided.PastRequest():
                return await self._past(command)
            case decided.ChangeRequest():
                return await self._change(command)
            case decided.Missing():
                self.timings.mark("missing")
                return self._note(f"{NOT_POSSIBLE}: {command.fact}")
            case decided.Reading():
                self.timings.mark(f"target:{describe(command.target)}")
                return await self._read(command.target)
        self.timings.mark(f"target:{describe(command.target)}")
        if unavailable := unavailable_reason(self.hass, command.target):
            return self._note(f"{NOT_POSSIBLE}: {unavailable}")
        if isinstance(command, decided.Playing):
            return await self._play(command)
        return await self._act(command)

    async def _act(self, command: decided.Command) -> intent.IntentResponse:
        await self._handle_intent(command.intent, command.slots, describe(command.target))
        value = next((v for k, v in command.slots.items() if k not in ("name", "domain", "area")), None)
        target = command.target.label if isinstance(command.target, RoomGroup) else command.target.name
        return self._note(f"{DONE}: {command.action} {target}" + (f" {value}" if value else ""))

    async def _list(self, request: decided.ListRequest) -> intent.IntentResponse:
        """Change or read a list.

        The decider picks the list and, for an entry already on it, one of the list's current entries (lists.py);
        HA's list intents do the change.
        """
        lists = {c.name: c for c in self.entities if c.domain == TODO}
        if not lists:
            raise HandOff("no list exposed")
        text = self.user_input.text
        name = checked_action(request.action, text)
        chosen = lists[await self.ask(list_question(list(lists)))] if len(lists) > 1 else next(iter(lists.values()))
        self.timings.mark(f"list:{chosen.entity_id}")
        action = LIST_ACTIONS[name]
        if action.intent == "HassListAddItem":
            item = item_to_add(text)
        else:
            entries = await self._list_entries(chosen.entity_id)
            if action.intent is None:
                held = ", ".join(entries) if entries else "nothing"
                asked = await self._asked_entry(entries)
                return self._note(f"{chosen.name} holds: {held}" + (f". {asked}" if asked else ""))
            item = await self.ask(item_question(entries)) if entries else NO_ITEM
            if item == NO_ITEM:
                raise HandOff(f"no entry of {chosen.name} fits")
        await self._handle_intent(action.intent, {"name": chosen.name, "item": item}, chosen.entity_id)
        return self._note(f"{DONE}: {item} {action.done} {chosen.name}")

    async def _asked_entry(self, entries: list[str]) -> str | None:
        """The answer to whether one item is on the list.

        For a question whether one item is on the list, the answer as a fact (the item, then whether it is on it);
        the entry is the decider's pick among the list's entries. None when the question names no item. Given the
        entries alone, small answer models confirm an item missing from them.
        """
        if (item := item_asked(self.user_input.text)) is None:
            return None
        entry = await self.ask(item_question(entries)) if entries else NO_ITEM
        return f"{item}: not on it" if entry == NO_ITEM else f"{entry}: on it"

    async def _list_entries(self, entity_id: str) -> list[str]:
        """The list's open entries, as HA holds them now."""
        response = await self._call(TODO, "get_items", {"entity_id": entity_id, "status": ["needs_action"]})
        return [item["summary"] for item in response[entity_id]["items"]]

    async def _forecast(self) -> intent.IntentResponse:
        """The daily forecast for the days the text names (forecast.py), as facts for the answer step."""
        text, today = self.user_input.text, dt_util.now().date()
        days_asked(text, today)  # HandOff before any request when the text names no day
        weathers = {c.name: c for c in self.entities if c.domain == WEATHER}
        if not weathers:
            raise HandOff("no weather exposed")
        if len(weathers) > 1:
            question = {"type": "choice", "instructions": "Which weather service does the user mean?",
                        "criteria": dict.fromkeys(weathers, "")}
            chosen = weathers[await self.ask(question)]
        else:
            chosen = next(iter(weathers.values()))
        self.timings.mark(f"forecast:{chosen.entity_id}")
        response = await self._call(WEATHER, "get_forecasts", {"entity_id": chosen.entity_id, "type": "daily"})
        # An entry whose time HA does not give as a datetime cannot be placed on a day.
        entries = [
            (dt_util.as_local(when), entry)
            for entry in response[chosen.entity_id]["forecast"]
            if (when := dt_util.parse_datetime(entry["datetime"])) is not None
        ]
        facts = forecast_facts(entries, forecast_days(text, entries, today), today,
                               self.hass.config.units.temperature_unit,
                               await state_words(self.hass, WEATHER, self.user_input.language))
        return self._note(f"Forecast of {chosen.name}: {facts}")

    async def _calendar(self) -> intent.IntentResponse:
        """The calendars' entries of the days the text names, or the one entry the decider picks (decider/calendar.py)."""
        calendars = [c.entity_id for c in self.entities if c.domain == CALENDAR]
        if not calendars:
            raise HandOff("no calendar exposed")
        text, now = self.user_input.text, dt_util.now()
        try:
            days = days_asked(text, now.date())
        except HandOff:
            days = None
        if days:
            first = dt_util.start_of_local_day(days[0])
            events = await self._calendar_events(calendars, first, dt_util.start_of_local_day(days[-1]) + DAY)
            facts = day_facts(events, days, now.date())
        else:
            back = looks_back(text)
            start, end = (now - CALENDAR_BACK, now) if back else (now, now + CALENDAR_AHEAD)
            events = sorted(await self._calendar_events(calendars, start, end), key=_event_time, reverse=back)
            question, by_summary = event_question(events, text)
            choice = await self.ask(question)
            if choice == NO_EVENT:
                raise HandOff("no calendar entry fits")
            facts = event_fact(by_summary[choice], now.date())
        self.timings.mark("calendar")
        # Each fact carries its own date. Today's date beside it would be taken for the answer to a question when an
        # entry is: today instead of the entry's day.
        return self._note(f"Calendar: {facts}")

    async def _calendar_events(self, calendars: list[str], start: datetime, end: datetime) -> list[Event]:
        """Every calendar's entries; HandOff when one fails, as an answer without its entries could be wrong."""
        step = monitor.service_opened(
            CALENDAR, "get_events",
            {"entity_id": calendars, "start_date_time": start.isoformat(), "end_date_time": end.isoformat()},
        )
        read = await read_calendars(self.hass, calendars, start, end)
        if read.failed:
            err = HomeAssistantError("; ".join(f"{calendar}: {error}" for calendar, error in read.failed.items()))
            monitor.step_failed(step, err)
            raise HandOff(f"{CALENDAR}.get_events: {err}")
        monitor.action_returned(step, {calendar: {"events": events} for calendar, events in read.events.items()})
        return [event for events in read.events.values() for event in events]

    async def _past(self, request: decided.PastRequest) -> intent.IntentResponse:
        """A sensor's value over the earlier day the text names, from HA's long-term statistics (past.py).

        Called without the user's context: HA allows the action to admins only, and reading statistics changes nothing.
        """
        today = dt_util.now().date()
        day = past_day_asked(self.user_input.text, today)
        start = dt_util.start_of_local_day(day)
        target = request.target.entity_id
        response = await self._call(
            RECORDER, "get_statistics",
            {"statistic_ids": [target], "start_time": start, "end_time": start + DAY, "period": "hour",
             "types": STATISTIC_TYPES},
        )
        state = self.hass.states.get(target)
        unit = state.attributes.get("unit_of_measurement", "") if state else ""
        note = past_note(request.statistic, response["statistics"].get(target, []), unit)
        self.timings.mark(f"past:{target}")
        return self._note(f"{located(request.target)} on {day_named(day, today)}: {note}", answer=True)

    async def _change(self, request: decided.ChangeRequest) -> intent.IntentResponse:
        """When the device last made the asked change, from HA's recorded state history (changes.py).

        The history covers all the days the recorder keeps.
        """
        if RECORDER not in self.hass.config.components:
            raise HandOff("changed: HA records no history")
        recorder = get_instance(self.hass)
        now = dt_util.utcnow()
        start = now - timedelta(days=recorder.keep_days)
        target = request.target.entity_id
        states = await recorder.async_add_executor_job(
            lambda: history.state_changes_during_period(
                self.hass, start, now, target, no_attributes=True, descending=True, limit=CHANGES_READ
            )
        )
        # The state the history begins with is dated to its start: it was so already, no change then (taken as a
        # change, a state held through the whole window would read as made at its start).
        found = last_change(
            [(state.state, dt_util.as_local(state.last_changed) if state.last_changed > start else None)
             for state in states.get(target, [])],
            request.change, request.target.domain,
        )
        self.timings.mark(f"changed:{target}")
        fact = change_fact(located(request.target), request.change, found, dt_util.as_local(start))
        return self._note(fact, answer=True)

    async def _play(self, playing: decided.Playing) -> intent.IntentResponse:
        """Music Assistant finds the media by name and plays it (music_assistant.play_media).

        Its action ignores other players without an error, so a player that is not Music Assistant's hands off first.
        Not built: players outside Music Assistant, through HA's own HassMediaSearchAndPlay.
        """
        player, media = playing.target, playing.media
        entry = er.async_get(self.hass).async_get(player.entity_id)
        if entry is None or entry.platform != MUSIC_ASSISTANT:
            raise HandOff(f"{player.entity_id} is not a Music Assistant player")
        if media.type is None:
            media = replace(media, type=await self._media_type_found(media, entry.config_entry_id))
        data = {"entity_id": player.entity_id, "media_id": media.name, "media_type": media.type}
        # Nothing plays on an error: the fallback agent can still try.
        await self._call(MUSIC_ASSISTANT, "play_media", data | ({"artist": media.artist} if media.artist else {}),
                         as_user=True)
        by = f" by {media.artist}" if media.artist else ""
        return self._note(f"{DONE}: play {media.type} {media.name}{by} on {player.name}")

    async def _media_type_found(self, media: Media, music_assistant: str | None) -> str:
        """The media type the decider was unsure of, from Music Assistant's search.

        It is the one of its likeliest types with an item of exactly that name. None or several: HandOff.
        """
        search = {"config_entry_id": music_assistant, "name": media.name, "media_type": list(media.candidates)}
        found = await self._call(MUSIC_ASSISTANT, "search", search | ({"artist": media.artist} if media.artist else {}))
        if (kind := type_found(media.name, media.candidates, found or {})) is None:
            raise HandOff(f"media type of {media.name!r}: none or several of {media.candidates} by that name")
        return kind

    async def _read(self, target: Candidate) -> intent.IntentResponse:
        state = self.hass.states.get(target.entity_id)
        if state is None:
            raise HandOff(f"no state for {target.entity_id}")
        language = self.user_input.language
        # HA's own state wording in the turn's language; numbers and unknown states stay raw.
        shown = (await state_words(self.hass, target.domain, language)).get(state.state, state.state)
        answer = read_answer(target, shown, dict(state.attributes), language, self.hass.config.units.temperature_unit,
                             media_current=state.state in MEDIA_CURRENT)
        return self._note(answer, answer=True)

    async def _handle_intent(self, intent_type: str, slots: dict[str, Any], target: str) -> None:
        """Run the HA intent; an intent HA refuses for the target hands off."""
        user_input = self.user_input
        step = monitor.intent_opened(intent_type, slots, target)
        try:
            response = await intent.async_handle(
                self.hass, DOMAIN, intent_type, {key: {"value": value} for key, value in slots.items()},
                user_input.text, user_input.context, user_input.language, assistant=conversation.DOMAIN,
                device_id=user_input.device_id, satellite_id=user_input.satellite_id,
            )
        except intent.IntentError as err:
            monitor.step_failed(step, err)
            raise HandOff(f"{intent_type} {target}: {err}") from err
        monitor.action_returned(step, response)

    async def _call(self, domain: str, service: str, data: dict[str, Any], *, as_user: bool = False) -> Any:
        """An HA action with its response; HandOff when it fails.

        Reads run without the user's context (some are for admins only); a change (`as_user`) runs with it.
        """
        context = self.user_input.context if as_user else None
        step = monitor.service_opened(domain, service, data)
        try:
            response = await self.hass.services.async_call(
                domain, service, data, blocking=True, context=context, return_response=not as_user,
            )
        except HomeAssistantError as err:
            monitor.step_failed(step, err)
            raise HandOff(f"{domain}.{service}: {err}") from err
        monitor.action_returned(step, response)
        return response

    def _note(self, speech: str, *, answer: bool = False) -> intent.IntentResponse:
        """The result note for the answer model; `answer`: it answers a question (no action ran)."""
        response = intent.IntentResponse(language=self.user_input.language)
        if answer:
            response.response_type = intent.IntentResponseType.QUERY_ANSWER
        response.async_set_speech(speech)
        return response


def _event_time(event: Event) -> datetime:
    """When an entry starts, an all-day one at the local midnight of its day."""
    return event.start if isinstance(event.start, datetime) else dt_util.start_of_local_day(event.start)
