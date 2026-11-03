"""Commands for later and timers: what runs at a set time without the model.

For "turn off the TV in 30 minutes" the model calls schedule_command with the tool it would call now, that tool's
arguments and the time. The call is checked against the tool at once and stored with the LLM API that provides the
tool; at the time the same tool of that API runs with the same arguments, the request's language and its device. The
APIs are Home Assistant's, the home's: no agent is asked. A timer ("Stell einen Timer auf 5 Minuten") is an entry that
rings where it was set (ring.py). A command that fails at its time is told the same way: on the satellite it was asked
on, else as a notification. A command runs as the user who asked for it. Stored entries survive a restart. One whose
time passed while Home Assistant was down is dropped: nobody asked for it late. Listeners (the Timers sensor) hear of
every change.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
import json
import logging
import re
from typing import Any
import uuid

import probatio

from homeassistant.components import conversation
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CALLBACK_TYPE, Context, Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey

from ..announce import async_tell
from ..const import AGENT_PLATFORM, DOMAIN
from ..home import satellite_area
from ..languages import language_of_tag
from ..tools.spec import READ_ONLY, SpecTool, ToolArgs, ToolError, ToolSpec
from ..tools.when import WEEKDAYS
from .ring import async_ring
from .tools import WrappedTool, visible_tools

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = f"{DOMAIN}.schedule"
STORAGE_VERSION = 1
SCHEDULE_TOOL = "schedule_command"
TIMER_TOOL = "start_timer"
LIST_TOOL = "list_timers_and_commands"
CANCEL_TOOL = "cancel_timer_or_command"
# What an entry does when its time comes: call a tool, or ring.
COMMAND, TIMER = "command", "timer"

_CLOCK_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
# A call the model repeats (in the same round or the next) is no second command: the same tool, arguments and time as
# asked, for about the same time. Rounds are seconds apart, so their in_seconds times differ by that much. "Off in 10 s,
# on in 20 s, off in 30 s" asks two different times for the same call: two commands.
SAME_TIME = timedelta(seconds=30)
# The longest delay a timer or command may have, in seconds or as a time.
MAX_DELAY = timedelta(days=366)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduledEntry:
    """One entry of the schedule: a tool call for later (kind COMMAND), or a timer that rings (kind TIMER).

    `asked`: the time as the model gave it ("in_seconds=600", "at=15:30"). A command's `tool`: its name as the model saw
    it (for the list); `api_id`, `api_tool`: the LLM API that provides it and its name there, by which it is called. A
    timer has its `name` (may be empty) and no tool. `user_id`: who asked; a command runs as them.
    """

    id: str
    kind: str
    created: datetime  # UTC
    run_at: datetime  # UTC
    asked: str
    name: str = ""
    tool: str = ""
    api_id: str = ""
    api_tool: str = ""
    arguments: Mapping[str, Any]
    language: str | None
    device_id: str | None
    user_id: str | None

    def as_stored(self) -> dict[str, Any]:
        """The entry as the storage file holds it."""
        return {
            "id": self.id, "kind": self.kind, "created": self.created.isoformat(), "run_at": self.run_at.isoformat(),
            "asked": self.asked, "name": self.name, "tool": self.tool, "api_id": self.api_id, "api_tool": self.api_tool,
            "arguments": dict(self.arguments), "language": self.language, "device_id": self.device_id,
            "user_id": self.user_id,
        }

    @classmethod
    def from_stored(cls, data: Mapping[str, Any]) -> ScheduledEntry:
        """The entry from the storage file; KeyError, TypeError or ValueError for an entry that cannot be read."""
        run_at, created = _aware(data["run_at"]), _aware(data["created"])
        if data["kind"] not in (COMMAND, TIMER):
            raise ValueError(f"kind {data['kind']} is unknown")
        if not isinstance(data["arguments"], Mapping):
            raise TypeError("arguments are not a mapping")
        return cls(
            id=data["id"], kind=data["kind"], created=created, run_at=run_at, asked=data["asked"], name=data["name"],
            tool=data["tool"], api_id=data["api_id"], api_tool=data["api_tool"], arguments=data["arguments"],
            language=data["language"], device_id=data["device_id"], user_id=data["user_id"],
        )


def _aware(text: str) -> datetime:
    """A stored time; ValueError for one without a time zone."""
    when = dt_util.parse_datetime(text, raise_on_error=True)
    if when.tzinfo is None:
        raise ValueError(f"{text} has no time zone")
    return when


class Schedule:
    """The home's timers and commands for later: stored, each armed for its time, run when it comes.

    It belongs to the integration, not to its entry: an entry whose setup waits for its LLM server must not keep a
    timer from ringing or a command from running.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Nothing is armed before async_load."""
        self._hass = hass
        self._store: Store[dict[str, list[dict[str, Any]]]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._entries: dict[str, ScheduledEntry] = {}
        self._disarm: dict[str, CALLBACK_TYPE] = {}
        self._listeners: list[Callable[[], None]] = []

    async def async_load(self) -> None:
        """Arm the stored entries still ahead; drop those whose time passed while Home Assistant was down.

        An entry that cannot be read (a hand edit) is dropped with a warning: it must not keep the entry from loading.
        """
        stored = (await self._store.async_load() or {}).get("entries", [])
        now = dt_util.utcnow()
        for data in stored:
            try:
                entry = ScheduledEntry.from_stored(data)
            except (KeyError, TypeError, ValueError) as err:
                _LOGGER.warning("Dropped a schedule entry that cannot be read: %r (%s)", data, err)
                continue
            if entry.run_at > now:
                self._arm(entry)
        if len(self._entries) < len(stored):
            await self._async_save()

    @callback
    def async_listen(self, changed: Callable[[], None]) -> CALLBACK_TYPE:
        """Call `changed` after every change of the entries; returns the function that stops it."""
        self._listeners.append(changed)
        return lambda: self._listeners.remove(changed)

    @callback
    def async_unload(self) -> None:
        """Disarm every entry and keep it; the stored entries are armed again at the next load.

        Kept, so that one added or cancelled afterwards (a turn still running while Home Assistant stops) is saved with
        the rest.
        """
        for disarm in self._disarm.values():
            disarm()
        self._disarm.clear()

    async def async_clear(self) -> None:
        """Disarm and forget every entry, and delete the store: the integration's entry is gone."""
        self.async_unload()
        self._entries.clear()
        self._changed()
        await self._store.async_remove()

    async def async_add(self, entry: ScheduledEntry) -> None:
        """Store `entry` and arm it."""
        self._arm(entry)
        self._changed()
        await self._async_save()

    async def async_cancel(self, entry_id: str) -> None:
        """Disarm and forget the entry with `entry_id`; KeyError for one that is not scheduled."""
        self._disarm.pop(entry_id)()
        del self._entries[entry_id]
        self._changed()
        await self._async_save()

    def pending(self) -> list[ScheduledEntry]:
        """The timers and commands for later, the next one first."""
        return sorted(self._entries.values(), key=lambda entry: entry.run_at)

    def scheduled_like(self, entry: ScheduledEntry) -> ScheduledEntry | None:
        """An entry already scheduled with the same call and asked time, from the same device, for about the same time.

        The same timer asked on two satellites is two timers: each rings in its own room.
        """
        return next((pending for pending in self._entries.values()
                     if _same_call(pending) == _same_call(entry)
                     and abs(pending.run_at - entry.run_at) <= SAME_TIME), None)

    def tools(self, offered: list[WrappedTool]) -> list[llm.Tool]:
        """The schedule's tools for a model that is `offered` these tools to run later."""
        scheduling = Scheduling(self, {tool.name: tool for tool in offered if tool.api_id is not None})
        return [WrappedTool(SpecTool(spec, scheduling), spec.name) for spec in SPECS]

    def _arm(self, entry: ScheduledEntry) -> None:
        self._entries[entry.id] = entry
        self._disarm[entry.id] = async_track_point_in_utc_time(self._hass, partial(self._due, entry.id), entry.run_at)

    @callback
    def _due(self, entry_id: str, _now: datetime) -> None:
        del self._disarm[entry_id]
        entry = self._entries.pop(entry_id)
        self._changed()
        self._hass.async_create_background_task(self._async_run(entry), f"{DOMAIN} scheduled {entry.kind} {entry.id}")

    async def _async_run(self, entry: ScheduledEntry) -> None:
        """Ring the timer or run the command, already removed from the store: one that fails is not run again."""
        await self._async_save()
        try:
            if entry.kind == TIMER:
                await async_ring(self._hass, entry.id, entry.name, entry.language, entry.device_id)
            else:
                await _async_call(self._hass, entry)
        # Invalid: the tool's arguments changed since the command was stored (a Home Assistant update).
        except (HomeAssistantError, probatio.Invalid) as err:
            _LOGGER.warning("Scheduled %s failed: %s", _described(entry), err)
        except Exception:  # a bug in a tool: logged with the entry, as no caller is left to raise to
            _LOGGER.exception("Scheduled %s failed", _described(entry))
        else:
            return
        if entry.kind == COMMAND:
            await _async_tell_failed(self._hass, entry)

    @callback
    def _changed(self) -> None:
        for changed in list(self._listeners):
            changed()

    async def _async_save(self) -> None:
        await self._store.async_save({"entries": [entry.as_stored() for entry in self.pending()]})


DATA_SCHEDULE: HassKey[Schedule] = HassKey(f"{DOMAIN}_schedule")


async def async_setup_schedule(hass: HomeAssistant) -> None:
    """Load the home's schedule when Home Assistant sets up the integration; disarm it when Home Assistant stops."""
    schedule = Schedule(hass)
    await schedule.async_load()
    hass.data[DATA_SCHEDULE] = schedule

    @callback
    def stop(_event: Event) -> None:
        schedule.async_unload()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, stop)


async def async_remove_schedule(hass: HomeAssistant) -> None:
    """Forget the home's timers and commands and delete their store: the integration's entry is gone.

    Home Assistant removes a disabled entry without setting the integration up first: then only the store is there.
    """
    if (schedule := hass.data.get(DATA_SCHEDULE)) is not None:
        await schedule.async_clear()
    else:
        await Store(hass, STORAGE_VERSION, STORAGE_KEY).async_remove()


@dataclass(frozen=True, slots=True)
class Scheduling:
    """What the schedule's tools work with: the schedule, and the tools a command may call, by the model's names."""

    schedule: Schedule
    tools: Mapping[str, WrappedTool]


async def _async_call(hass: HomeAssistant, command: ScheduledEntry) -> None:
    """Call the command's tool of its LLM API as a model gets it; a HomeAssistantError says why it failed."""
    context = llm.LLMContext(
        platform=AGENT_PLATFORM, context=Context(user_id=command.user_id), language=command.language,
        assistant=conversation.DOMAIN,
        device_id=command.device_id,
    )
    instance = await llm.async_get_api(hass, command.api_id, context)
    tool = next((tool for tool in visible_tools([instance]) if tool.tool.name == command.api_tool), None)
    if tool is None:
        raise HomeAssistantError(f"{command.api_id} no longer has this tool")
    tool_input = llm.ToolInput(tool_name=tool.name, tool_args=dict(command.arguments))
    result = await tool.async_call(hass, tool_input, context)
    if isinstance(result, llm.ToolResult) and result.error:
        raise HomeAssistantError(str(result.data))
    _raise_if_failed(result.data if isinstance(result, llm.ToolResult) else result)


def _raise_if_failed(data: Mapping[str, Any]) -> None:
    """HomeAssistantError for an intent's result that failed, wholly or for some of its targets."""
    details = data.get("data")
    failed = (details.get("failed") or []) if isinstance(details, Mapping) else []
    if data.get("response_type") == "error" or failed:
        names = ", ".join(str(target.get("name") or target.get("id")) for target in failed)
        raise HomeAssistantError(f"failed for {names}" if names else str(data))


async def _async_tell_failed(hass: HomeAssistant, command: ScheduledEntry) -> None:
    """Tell the user that the command did not work, where it was asked: nobody else would notice."""
    speech = language_of_tag(command.language or hass.config.language).replies.speech
    time = f"{dt_util.as_local(command.run_at):%H:%M}"
    if target := command.arguments.get("name") or command.arguments.get("area"):
        message = speech["command_failed"].format(target=target, time=time)
    else:
        message = speech["command_failed_untargeted"].format(time=time)
    await async_tell(hass, command.device_id, message, title=speech["command_failed_title"],
                     notification_id=f"{DOMAIN}_command_{command.id}")


async def schedule_command(hass: HomeAssistant, scheduling: Scheduling, args: ToolArgs, context: llm.LLMContext) -> str:
    """Check the call against its tool and schedule it."""
    tool = scheduling.tools.get(args["tool"])
    if tool is None:
        raise ToolError(f"There is no tool '{args['tool']}'. Give the name of one of your tools.")
    arguments = args.get("arguments") or {}
    try:
        tool.parameters(arguments)
    except probatio.Invalid as err:
        raise ToolError(f"The arguments do not fit {tool.name}: {err}") from err
    assert tool.api_id is not None  # Scheduling.tools holds only tools of an API
    command = ScheduledEntry(
        id=uuid.uuid4().hex[:6], kind=COMMAND, created=dt_util.utcnow(), run_at=_run_at(args), asked=_asked(args),
        tool=tool.name, api_id=tool.api_id, api_tool=tool.tool.name, arguments=arguments, language=context.language,
        device_id=context.device_id, user_id=_user_id(context),
    )
    if (same := scheduling.schedule.scheduled_like(command)) is not None:
        return f"Already scheduled for {_when(same)}, id {same.id}. Nothing was done now."
    await scheduling.schedule.async_add(command)
    return f"Scheduled for {_when(command)}, id {command.id}. Nothing was done now."


async def start_timer(_hass: HomeAssistant, scheduling: Scheduling, args: ToolArgs, context: llm.LLMContext) -> str:
    """Start a timer that rings where it was set."""
    timer = ScheduledEntry(
        id=uuid.uuid4().hex[:6], kind=TIMER, created=dt_util.utcnow(), run_at=_run_at(args), asked=_asked(args),
        name=(args.get("name") or "").strip(), arguments={}, language=context.language, device_id=context.device_id,
        user_id=_user_id(context),
    )
    if (same := scheduling.schedule.scheduled_like(timer)) is not None:
        return f"Already running: rings {_when(same)} ({_left(same)} left), id {same.id}."
    await scheduling.schedule.async_add(timer)
    return f"Timer started: rings {_when(timer)} ({_left(timer)} left), id {timer.id}."


async def list_timers_and_commands(
    hass: HomeAssistant, scheduling: Scheduling, _args: ToolArgs, context: llm.LLMContext
) -> str:
    """Every timer and command for later, the next one first, with the time left and where it was set.

    The schedule holds the whole home's timers: "cancel the timer" in the kitchen means the kitchen's, so each line
    says whether it was set on the device asking ("here") or in which room.
    """
    pending = scheduling.schedule.pending()
    if not pending:
        return "No timers are running and no commands are scheduled."
    return "\n".join(f"{_listed(entry)}, {_where(hass, entry, context.device_id)}" for entry in pending)


async def cancel_timer_or_command(
    _hass: HomeAssistant, scheduling: Scheduling, args: ToolArgs, _context: llm.LLMContext
) -> str:
    """Cancel the timer or command with the given id."""
    try:
        await scheduling.schedule.async_cancel(args["id"])
    except KeyError as err:
        raise ToolError(f"Nothing with id '{args['id']}' is scheduled. Call {LIST_TOOL} for the ids.") from err
    return f"Cancelled {args['id']}."


def _user_id(context: llm.LLMContext) -> str | None:
    return context.context.user_id if context.context else None


def _same_call(entry: ScheduledEntry) -> tuple:
    """What two entries share when one repeats the other."""
    return (entry.kind, entry.name, entry.api_id, entry.api_tool, entry.arguments, entry.asked, entry.device_id)


def _described(entry: ScheduledEntry) -> str:
    """The entry for the log: a timer by its id and name, a command by its call."""
    if entry.kind == TIMER:
        return f"timer {entry.id} {entry.name!r}"
    return f"{entry.tool} {_arguments(entry)}"


def _listed(command: ScheduledEntry) -> str:
    if command.kind == TIMER:
        named = f" '{command.name}'" if command.name else ""
        return f"id {command.id}: timer{named}, rings {_when(command)} ({_left(command)} left)"
    return f"id {command.id}: {_when(command)} ({_left(command)} left), {command.tool} {_arguments(command)}"


def _where(hass: HomeAssistant, entry: ScheduledEntry, asking_device: str | None) -> str:
    """Where the entry was set, as the asking device sees it.

    "set here (Location 1)", "set in Location 2", "set in a chat".
    """
    if entry.device_id is None:
        return "set in a chat"
    area = satellite_area(hass, entry.device_id)
    if entry.device_id == asking_device:
        return f"set here ({area})" if area else "set here"
    return f"set in {area}" if area else "set on another device"


def _left(command: ScheduledEntry) -> str:
    """The time until the entry runs: "1 h 5 min 3 s", "45 s"."""
    seconds = max(0, round((command.run_at - dt_util.utcnow()).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [f"{hours} h" if hours else "", f"{minutes} min" if minutes else "", f"{seconds} s" if seconds else ""]
    return " ".join(part for part in parts if part) or "0 s"


def _run_at(args: ToolArgs) -> datetime:
    """The UTC time the command runs, from exactly one of in_seconds and at."""
    if ("in_seconds" in args) == ("at" in args):
        raise ToolError("Give exactly one of in_seconds and at.")
    if "in_seconds" in args:
        return dt_util.utcnow() + timedelta(seconds=args["in_seconds"])
    run_at = dt_util.as_utc(_clock_time(args["at"]))
    if run_at - dt_util.utcnow() > MAX_DELAY:
        raise ToolError(f"'{args['at']}' is more than {MAX_DELAY.days} days ahead.")
    return run_at


def _asked(args: ToolArgs) -> str:
    """The time as the model gave it."""
    return f"in_seconds={args['in_seconds']}" if "in_seconds" in args else f"at={args['at']}"


def _clock_time(text: str) -> datetime:
    """HH:MM the next time it comes, today or tomorrow; a date and time as given, if still ahead."""
    now = dt_util.now()
    if match := _CLOCK_TIME.match(text.strip()):
        hour, minute = int(match[1]), int(match[2])
        if hour > 23 or minute > 59:
            raise ToolError(f"'{text}' is not a time of day.")
        today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return today if today > now else today + timedelta(days=1)
    try:
        parsed = dt_util.parse_datetime(text.strip())
    except ValueError:  # well formed, but no such day: 2030-02-30
        parsed = None
    if parsed is None:
        raise ToolError(f"Could not read the time '{text}'. Use HH:MM or YYYY-MM-DDTHH:MM.")
    when = dt_util.as_local(parsed) if parsed.tzinfo is None else parsed
    if when <= now:
        raise ToolError(f"{when:%Y-%m-%d %H:%M} has already passed.")
    return when


def _when(command: ScheduledEntry) -> str:
    local = dt_util.as_local(command.run_at)
    return f"{WEEKDAYS[local.weekday()]} {local:%Y-%m-%d %H:%M:%S}"


def _arguments(command: ScheduledEntry) -> str:
    return json.dumps(dict(command.arguments), ensure_ascii=False, sort_keys=True)


# The time of an entry, as schedule_command and start_timer take it.
_IN_SECONDS = probatio.All(probatio.Coerce(int), probatio.Range(min=1, max=int(MAX_DELAY.total_seconds())))
_AT_DESCRIPTION = "Local time HH:MM, the next time it comes; YYYY-MM-DDTHH:MM for another day"

SPECS: tuple[ToolSpec[Scheduling], ...] = (
    ToolSpec(
        name=SCHEDULE_TOOL,
        description=(
            "Run a command later instead of now: after a delay or at a clock time ('turn off the TV in 30 minutes', "
            "'turn on the fan at 15:30', a sleep timer). Give the tool you would call now, its arguments, and exactly "
            "one of in_seconds or at. One call per command. in_seconds counts from now, but a time the user gives "
            "after an earlier command counts from that command, so add the two: 'on in 10 seconds and off again "
            "after 30 seconds' is in_seconds 10 and 40, and the light stays on for 30 seconds. 'Turn on the fan for "
            "10 minutes' is the fan's tool now and its off command in 600 seconds. Nothing runs now: tell the user "
            "when it will run."
        ),
        parameters=probatio.Schema({
            probatio.Required("tool", description="The tool to call later, by its name"): str,
            probatio.Optional("arguments", description="The tool's arguments, as you would give them now"): dict,
            probatio.Optional("in_seconds", description=(
                "Seconds from now until this command runs, the user's times added up: 'on in 10 seconds and off "
                "again after 2 minutes' is 10 for on and 130 for off. 10 minutes is 600"
            )): _IN_SECONDS,
            probatio.Optional("at", description=_AT_DESCRIPTION): str,
        }),
        run=schedule_command,
    ),
    ToolSpec(
        name=TIMER_TOOL,
        description=(
            "Start a timer: when it ends, it rings where the user asked ('Stell einen Timer auf 5 Minuten', 'a pasta "
            "timer for 8 minutes'). Give exactly one of in_seconds or at, and the timer's name if the user gives one. "
            f"A device to switch later is {SCHEDULE_TOOL}, not a timer."
        ),
        parameters=probatio.Schema({
            probatio.Optional("in_seconds", description="Seconds from now: 5 minutes is 300"): _IN_SECONDS,
            probatio.Optional("at", description=_AT_DESCRIPTION): str,
            probatio.Optional("name", description="The timer's name, if the user gives one ('Pasta')"): str,
        }),
        run=start_timer,
    ),
    ToolSpec(
        name=LIST_TOOL,
        description=(
            "The running timers and the commands scheduled for later, the next one first, each with its id, when it "
            "ends and the time left: for 'how long is left on the timer' too. When the user asks what is planned or "
            "scheduled, call this as well as the calendar: they may mean either. Each entry says where it was set, "
            "'here' being the device the user speaks to: for 'the timer' without more, take the one set here."
        ),
        parameters=probatio.Schema({}),
        run=list_timers_and_commands,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name=CANCEL_TOOL,
        description=f"Cancel a timer or a command scheduled for later, by the id {LIST_TOOL} gives.",
        parameters=probatio.Schema({probatio.Required("id"): str}),
        run=cancel_timer_or_command,
    ),
)
