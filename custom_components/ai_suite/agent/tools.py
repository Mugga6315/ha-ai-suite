"""An agent's tools: the LLM APIs it selects, each tool under a plain name, some left out.

Tool names come with prefixes: HA's merge puts the API's name in front (`<api>__<tool>`), HA's own tools their domain.
The model gets each tool by its plain name, the part after the last prefix, of letters, digits and underscores only,
which small models read and call more reliably; a name two tools would share gets its API's id in front. Each tool
is wrapped to run under its own name and to name the integration that provides it.
"""

from collections import Counter
from collections.abc import Callable, Iterable
import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util, slugify
from homeassistant.util.json import JsonObjectType

from ..const import DOMAIN
from ..monitor import recorder
from ..tools.when import WEEKDAYS

# What a plain name holds: a model's tool-call parser may read anything else as an operator.
_NOT_IN_A_NAME = re.compile(r"[^A-Za-z0-9_]")

# Where one of AI Suite's tools does a Home Assistant tool's job better, the agent has only ours: the model gets only
# the tools it needs. Only Home Assistant's own tools are replaced; which tools of other integrations an agent uses is
# the user's choice (its APIs and "Tools left out").
REPLACES = {
    # Every entry with its weekday; HA's gives ISO dates only, and a model that uses it alone names weekdays wrong.
    "get_calendar_events": {"calendar__get_events"},
}


class AgentAPI(llm.API):
    """The tools of one agent. Not registered: an agent hands it to its chat log itself."""

    def __init__(self, hass: HomeAssistant, agent_id: str, sources: Iterable[str], hidden: Iterable[str] = ()) -> None:
        """Merge the tools of `sources`, leaving out the `hidden` ones."""
        super().__init__(hass=hass, id=f"{DOMAIN}.{agent_id}", name="AI Suite")
        self._sources = list(sources)
        self._hidden = set(hidden)

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """The merged tools and prompts for this request."""
        instances = [await llm.async_get_api(self.hass, api_id, llm_context) for api_id in self._sources]
        tools: list[llm.Tool] = [tool for tool in visible_tools(instances) if tool.name not in self._hidden]
        prompt = "\n\n".join(instance.api_prompt for instance in instances if instance.api_prompt)
        return llm.APIInstance(
            api=self,
            api_prompt="\n".join(part for part in (prompt, _date_line(tools)) if part),
            llm_context=llm_context,
            tools=tools,
            custom_serializer=_merged_serializer(instances),
        )


def _date_line(tools: list[llm.Tool]) -> str:
    """Today's date, where HA would write its date and time line, when HA leaves that line out.

    HA writes no date while a GetDateTime tool exists (conversation/chat_log.py). A model that skips the tool then
    guesses the year for date ranges ahead. Code writes the weekday: from the date alone a model counts weekdays wrong
    ("the day after tomorrow" a day off). The line changes once a day, so the prompt stays cacheable.
    """
    if not any(tool.name.endswith("GetDateTime") for tool in tools):
        return ""
    today = dt_util.now()
    return f"Today is {WEEKDAYS[today.weekday()]}, {today:%Y-%m-%d}."


class WrappedTool(llm.Tool):
    """A tool under the name the model sees, tagged with the integration that provides it."""

    def __init__(self, tool: llm.Tool, name: str) -> None:
        """Take the tool's description and parameters as they are."""
        self.name = name
        self.title = tool.title
        self.description = tool.description
        self.parameters = tool.parameters
        self.annotations = tool.annotations
        # The integration that provides the tool, not AI Suite: HA names it in its report of untagged tools.
        self.integration = tool.integration or _providing_domain(tool)
        self.tool = tool

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult | JsonObjectType:
        """Run the wrapped tool under its own name."""
        own_input = llm.ToolInput(tool_name=self.tool.name, tool_args=tool_input.tool_args, id=tool_input.id)
        step = recorder.tool_opened(tool_input.id, self.name, tool_input.tool_args)
        try:
            result = await self.tool.async_call(hass, own_input, llm_context)
        except Exception as err:
            recorder.step_failed(step, err)
            raise
        recorder.tool_returned(step, result)
        return result


def visible_tools(instances: list[llm.APIInstance]) -> list[WrappedTool]:
    """Every tool of the APIs under its plain name, without the standard ones ours replace.

    A name two tools would share would leave the model unable to tell them apart, so those get their API's id and
    their full name instead.
    """
    offered = [(instance.api, tool) for instance in instances for tool in instance.tools]
    replaced = set().union(*(REPLACES.get(tool.name, set()) for _, tool in offered))
    kept = [(api, tool) for api, tool in offered if tool.name not in replaced]
    shared = {name for name, count in Counter(plain_name(tool.name) for _, tool in kept).items() if count > 1}
    return [
        WrappedTool(tool, _readable(f"{slugify(api.id)}_{tool.name}") if name in shared else name)
        for api, tool in kept
        for name in (plain_name(tool.name),)
    ]


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
