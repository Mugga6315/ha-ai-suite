"""An LLM tool as a record and an async function, and the one adapter that makes it a tool HA can call.

HA's tool list takes `llm.Tool` objects only. Every tool of AI Suite, its own tool sets' and Memory's alike, is a
`ToolSpec` turned into one by `SpecTool`: the shape of HA's own KNX integration (components/knx/llm_api.py).

A tool answers the model with {"result": ...}, or {"error": ...} for a call it could not answer: the dict the model
has always seen from these tools.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import probatio

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..const import DOMAIN

type RecordFailure = Callable[[str, str], Awaitable[None]]
type ToolArgs = dict[str, Any]

# A failure log line names the call's arguments; a memory's whole text would bury the failure.
MAX_LOGGED_ARGUMENT = 80

# What a tool that only reads declares: the reply check retries only turns that ran such tools.
READ_ONLY = llm.ToolAnnotations(read_only=True, destructive=False, idempotent=True, open_world=False)
READ_ONLY_REMOTE = llm.ToolAnnotations(read_only=True, destructive=False, idempotent=True, open_world=True)
NOTHING_DECLARED = llm.ToolAnnotations()  # HA's default for a tool that declares nothing


class ToolError(Exception):
    """A call the tool could not answer; its message goes back to the model as the tool's error.

    `learn`: the call was the model's mistake (a range it got wrong, an entity without statistics). It goes into
    Memory's failure log when Memory is there, so self-improvement can learn from it.
    """

    def __init__(self, message: str, *, learn: bool = False) -> None:
        """Fail the call with `message`."""
        super().__init__(message)
        self.learn = learn


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSpec[D]:
    """One tool: what the model sees of it, and the function that answers.

    `run` gets the tool's dependency (a client, the store), the validated arguments and the request's context, and
    returns the answer's text or raises ToolError.
    """

    name: str
    description: str
    parameters: probatio.Schema
    run: Callable[[HomeAssistant, D, ToolArgs, llm.LLMContext], Awaitable[str]]
    annotations: llm.ToolAnnotations = NOTHING_DECLARED


class SpecTool[D](llm.Tool):
    """A ToolSpec as HA calls it: the arguments checked, the answer wrapped, a learnable failure recorded."""

    integration = DOMAIN

    def __init__(self, spec: ToolSpec[D], deps: D, record_failure: RecordFailure | None = None) -> None:
        """Serve `spec` with its dependency; failures go to `record_failure` when one is given."""
        self.name = spec.name
        self.description = spec.description
        self.parameters = spec.parameters
        self.annotations = spec.annotations
        self._run = spec.run
        self._deps = deps
        self._record_failure = record_failure

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult:
        """Answer the call; HA's chat log reports arguments that do not fit the schema to the model."""
        args = self.parameters(tool_input.tool_args)
        try:
            answer = await self._run(hass, self._deps, args, llm_context)
        except ToolError as err:
            if err.learn and self._record_failure is not None:
                await self._record_failure(self.name, f"{_described(tool_input.tool_args)} -> {err}")
            return llm.ToolResult(data={"error": str(err)}, error=True)
        return llm.ToolResult(data={"result": answer})


def spec_tools[D](specs: tuple[ToolSpec[D], ...], deps: D, record_failure: RecordFailure | None) -> list[llm.Tool]:
    """The tools of a set of specs that share one dependency."""
    return [SpecTool(spec, deps, record_failure) for spec in specs]


def _described(args: Mapping[str, Any]) -> str:
    """The call's arguments as the failure log shows them: `key=value` pairs, each value's repr cut short."""
    return " ".join(f"{key}={_short(repr(value))}" for key, value in args.items())


def _short(text: str) -> str:
    return text if len(text) <= MAX_LOGGED_ARGUMENT else text[: MAX_LOGGED_ARGUMENT - 1] + "…"
