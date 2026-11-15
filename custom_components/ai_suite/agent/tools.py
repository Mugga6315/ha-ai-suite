"""An agent's tools: the LLM APIs it selects, each tool under a plain name, some left out.

A plain name drops the prefixes HA puts in front (`<api>__<tool>`): small models read and call it more reliably.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import probatio

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.util import dt as dt_util, slugify
from homeassistant.util.json import JsonObjectType

from ..const import DOMAIN
from ..monitor import recorder
from ..tools.when import WEEKDAYS
from .media import checked

if TYPE_CHECKING:  # schedule.py imports this module
    from .schedule import Schedule

_LOGGER = logging.getLogger(__name__)

# What a plain name holds: a model's tool-call parser may read anything else as an operator.
_NOT_IN_A_NAME = re.compile(r"[^A-Za-z0-9_]")

# Where one of AI Suite's tools does a Home Assistant tool's job better, the agent has only ours. Only HA's own tools
# are replaced: which tools of other integrations an agent uses is the user's choice (its APIs and "Tools left out").
REPLACES = {
    # Every entry with its weekday; HA's gives ISO dates only, and a model that uses it alone names weekdays wrong.
    "get_calendar_events": {"calendar__get_events"},
}
# With the schedule, its timers replace HA's, which exist only on satellites that can ring and show nowhere else (no
# entity, no event). So the agent gets none of HA's timer tools, nor the line HA's prompts add on a device without them.
HA_TIMER_TOOLS = frozenset({
    intent.INTENT_START_TIMER, intent.INTENT_CANCEL_TIMER, intent.INTENT_CANCEL_ALL_TIMERS, intent.INTENT_TIMER_STATUS,
    intent.INTENT_INCREASE_TIMER, intent.INTENT_DECREASE_TIMER, intent.INTENT_PAUSE_TIMER, intent.INTENT_UNPAUSE_TIMER,
})
NO_TIMERS_LINE = "This device is not able to start timers."


class AgentAPI(llm.API):
    """The tools of one agent. Not registered: an agent hands it to its chat log itself."""

    def __init__(
        self,
        hass: HomeAssistant,
        agent_id: str,
        sources: Iterable[str],
        hidden: Iterable[str] = (),
        schedule: Schedule | None = None,
    ) -> None:
        """Merge the tools of `sources`, leaving out the `hidden` ones; with `schedule`, add its tools."""
        super().__init__(hass=hass, id=f"{DOMAIN}.{agent_id}", name="AI Suite")
        self._sources = list(sources)
        self._hidden = set(hidden)
        self._schedule = schedule

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """The merged tools and prompts for this request."""
        instances = [await llm.async_get_api(self.hass, api_id, llm_context) for api_id in self._sources]
        visible = [tool for tool in visible_tools(instances) if tool.name not in self._hidden]
        prompt = "\n\n".join(instance.api_prompt for instance in instances if instance.api_prompt)
        tools: list[llm.Tool] = list(visible)
        if self._schedule is not None:
            visible = [tool for tool in visible if plain_name(tool.tool.name) not in HA_TIMER_TOOLS]
            tools = [*visible, *self._schedule.tools(visible)]
            prompt = "\n".join(line for line in prompt.split("\n") if line.strip() != NO_TIMERS_LINE)
        return llm.APIInstance(
            api=self,
            api_prompt="\n".join(part for part in (prompt, _date_line(tools)) if part),
            llm_context=llm_context,
            tools=tools,
            custom_serializer=_merged_serializer(instances),
        )


def _date_line(tools: list[llm.Tool]) -> str:
    """Today's date with its weekday, when HA leaves out its date line: it does while a GetDateTime tool exists."""
    if not any(tool.name.endswith("GetDateTime") for tool in tools):
        return ""
    # A model that skips the tool guesses the year, and from a date alone it counts weekdays wrong. The line changes
    # once a day, so the prompt stays cacheable.
    today = dt_util.now()
    return f"Today is {WEEKDAYS[today.weekday()]}, {today:%Y-%m-%d}."


class WrappedTool(llm.Tool):
    """A tool under the name the model sees, tagged with the integration that provides it."""

    def __init__(
        self, tool: llm.Tool, name: str, failed: dict[str, Exception] | None = None, api_id: str | None = None
    ) -> None:
        """Take the tool's description and parameters as they are.

        `failed`: the turn's failed calls by `_signature` and the error each raised, shared by all its tools. `api_id`:
        the LLM API that provides it; None for the agent's own tools (the schedule's).
        """
        self.name = name
        self.title = tool.title
        self.description = tool.description
        self.parameters = tool.parameters
        self.annotations = tool.annotations
        # The integration that provides the tool, not AI Suite: HA names it in its report of untagged tools.
        self.integration = tool.integration or _providing_domain(tool)
        self.tool = tool
        self.api_id = api_id
        self._failed = {} if failed is None else failed

    def forget_failed(self) -> None:
        """Let every call run again: the record of failed calls is one turn's."""
        self._failed.clear()

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult | JsonObjectType:
        """Run the wrapped tool under its own name."""
        step = recorder.tool_opened(tool_input.id, self.name, tool_input.tool_args)
        try:
            result = await self._call_once(hass, tool_input, llm_context)
        except Exception as err:
            recorder.step_failed(step, err)
            raise
        recorder.tool_returned(step, result)
        return result

    async def _call_once(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult | JsonObjectType:
        """The tool's result; every failure a HomeAssistantError, and a failed call is not run again in the turn.

        HA's chat log hands the model only a HomeAssistantError (or invalid arguments); others end the turn in silence.
        """
        signature = _signature(self.name, tool_input.tool_args)
        # A model repeats a failed call unchanged: the repeat gets the first error as it was, without a slow tool's wait.
        if (earlier := self._failed.get(signature)) is not None:
            raise earlier
        own_input = llm.ToolInput(tool_name=self.tool.name, id=tool_input.id,
                                  tool_args=lists_unquoted(self.tool.parameters, tool_input.tool_args))
        try:
            return await self.tool.async_call(hass, own_input, llm_context)
        except (HomeAssistantError, probatio.Invalid) as err:
            self._failed[signature] = err
            raise
        except Exception as err:
            # The model gets it as an error; the log keeps the traceback of what is likely a bug in the tool.
            _LOGGER.warning("Tool %s failed", self.name, exc_info=err)
            failure = HomeAssistantError(f"{type(err).__name__}: {err}")
            self._failed[signature] = failure
            raise failure from err


def visible_tools(instances: list[llm.APIInstance]) -> list[WrappedTool]:
    """Every tool of the APIs under its plain name, without the standard ones ours replace.

    A name two tools would share gets its API's id in front. The tools share one record of the turn's failed calls.
    """
    offered = [(instance.api, tool) for instance in instances for tool in instance.tools]
    replaced = set().union(*(REPLACES.get(tool.name, set()) for _, tool in offered))
    kept = [(api, tool) for api, tool in offered if tool.name not in replaced]
    shared = {name for name, count in Counter(plain_name(tool.name) for _, tool in kept).items() if count > 1}
    failed: dict[str, Exception] = {}
    return [
        WrappedTool(checked(tool), _readable(f"{slugify(api.id)}_{tool.name}") if name in shared else name, failed,
                    api_id=api.id)
        for api, tool in kept
        for name in (plain_name(tool.name),)
    ]


def forget_failed_calls(instance: llm.APIInstance) -> None:
    """Start a turn without the failed calls of the turns before: HA keeps an agent's tools for a conversation."""
    for tool in instance.tools:
        if isinstance(tool, WrappedTool):
            tool.forget_failed()


def _signature(name: str, args: Mapping[str, Any]) -> str:
    return f"{name} {json.dumps(args, sort_keys=True, default=str)}"


def lists_unquoted(parameters: Callable[[Any], Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments, each JSON list written as a string passed as that list where the tool's schema takes it there.

    Models write a list as text ('["light"]') where a schema takes a string or a list; a string-only parameter keeps it.
    """
    lists = {key: listed for key, value in args.items() if (listed := _json_list(value)) is not None}
    if not lists:
        return dict(args)
    if _accepts(parameters, {**args, **lists}):
        return {**args, **lists}
    return _unquoted_one_by_one(parameters, args, lists)


def _unquoted_one_by_one(
    parameters: Callable[[Any], Any], args: Mapping[str, Any], lists: dict[str, list[Any]]
) -> dict[str, Any]:
    """The arguments with each of `lists` the schema takes in its place, beside those taken before."""
    unquoted = dict(args)
    # Until a pass takes none: one taken may let another fit. A pass tries every list left, so order does not matter.
    while taken := {key: listed for key, listed in lists.items() if _accepts(parameters, {**unquoted, key: listed})}:
        unquoted |= taken
        lists = {key: listed for key, listed in lists.items() if key not in taken}
    return unquoted


def _json_list(value: Any) -> list[Any] | None:
    """The list a string holds as JSON; None for any other value."""
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _accepts(parameters: Callable[[Any], Any], args: Mapping[str, Any]) -> bool:
    """Whether the schema takes the arguments; any failure of the check counts as no."""
    try:
        parameters(args)
    except Exception:  # noqa: BLE001 - probatio.Invalid, or a validator not made for a list failing another way
        return False
    return True


def plain_name(name: str) -> str:
    """The tool's name without the prefixes in front of it, in characters every parser reads as a name."""
    return _readable(name.rpartition("__")[2])


def _readable(name: str) -> str:
    return _NOT_IN_A_NAME.sub("_", name)


def _merged_serializer(instances: list[llm.APIInstance]) -> Callable[[Any], Any] | None:
    """The APIs' schema serializers as one, the first that knows a schema answering, as HA's merge does."""
    serializers = [instance.custom_serializer for instance in instances if instance.custom_serializer is not None]
    if not serializers:
        return None

    def serialize(schema: Any) -> Any:
        for serializer in serializers:
            if (result := serializer(schema)) is not None:
                return result
        return schema

    return serialize


def _providing_domain(tool: llm.Tool) -> str | None:
    """The domain of the integration whose module defines the tool, through HA's namespace wrapper."""
    while isinstance(tool, llm.NamespacedTool):
        tool = tool.tool
    module = type(tool).__module__
    for prefix in ("custom_components.", "homeassistant.components."):
        if module.startswith(prefix):
            return module.removeprefix(prefix).partition(".")[0]
    return None
