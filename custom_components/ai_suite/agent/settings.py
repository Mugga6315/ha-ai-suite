"""An agent's loop settings, from its conversation subentry."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT

from ..const import CONF_REPLY_CHECK, DEFAULT_REPLY_CHECK

CONF_TEMPERATURE = "temperature"
CONF_THINKING = "thinking"  # "on", "off", or absent: the server's default (no chat_template_kwargs sent)
CONF_REASONING_EFFORT = "reasoning_effort"  # absent: none sent
CONF_THINKING_BUDGET = "thinking_token_budget"  # vLLM: the most reasoning tokens per request
CONF_STRIP_EMOJIS = "strip_emojis"
CONF_PARALLEL_TOOL_CALLS = "parallel_tool_calls"
CONF_MAX_HISTORY = "max_message_history"  # earlier rounds kept; absent: all
CONF_HIDDEN_TOOLS = "hidden_tools"  # tool names, as the model sees them, the agent leaves out
# The model's earlier reasoning goes back with its earlier messages (`reasoning_content`, which vLLM, llama.cpp). Without it a thinking model may answer after a tool call inside an unclosed think block, and HA then
# speaks nothing.
CONF_SEND_REASONING = "send_reasoning"

THINKING_ON, THINKING_OFF = "on", "off"
REASONING_EFFORTS = ["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class LoopSettings:
    """How the agent loop calls the text model."""
    model: str
    prompt: str | None
    llm_apis: tuple[str, ...]
    temperature: float | None
    thinking: str | None
    reasoning_effort: str | None
    thinking_budget: int | None
    strip_emojis: bool
    parallel_tool_calls: bool
    max_history: int | None
    hidden_tools: tuple[str, ...] = ()
    send_reasoning: bool = True
    reply_check: bool = DEFAULT_REPLY_CHECK
    thinking_switch: bool = True  # the server takes chat_template_kwargs (llm_client.LLMServer)

    @classmethod
    def from_subentry(cls, data: Mapping[str, Any], *, thinking_switch: bool = True) -> LoopSettings:
        """The settings as the subentry stores them, with the defaults for what it leaves out.

        `thinking_switch`: the LLM server's (LLMServer.thinking_switch).
        """
        budget = data.get(CONF_THINKING_BUDGET)
        history = data.get(CONF_MAX_HISTORY)
        return cls(
            model=data[CONF_MODEL],
            prompt=data.get(CONF_PROMPT),
            llm_apis=tuple(data.get(CONF_LLM_HASS_API) or ()),
            temperature=data.get(CONF_TEMPERATURE),
            thinking=data.get(CONF_THINKING),
            reasoning_effort=data.get(CONF_REASONING_EFFORT) or None,
            thinking_budget=None if budget is None else int(budget),
            strip_emojis=data.get(CONF_STRIP_EMOJIS, False),
            parallel_tool_calls=data.get(CONF_PARALLEL_TOOL_CALLS, True),
            max_history=None if history is None else int(history),
            hidden_tools=tuple(data.get(CONF_HIDDEN_TOOLS) or ()),
            send_reasoning=data.get(CONF_SEND_REASONING, True),
            reply_check=data.get(CONF_REPLY_CHECK, DEFAULT_REPLY_CHECK),
            thinking_switch=thinking_switch,
        )

    def server_fields(self) -> dict[str, Any]:
        """The request's fields the OpenAI API does not define.

        These are the chat template's thinking switch (vLLM, llama.cpp) and vLLM's thinking budget, only the ones set.
        The switch only goes to a server that takes it, as for every other request (llm_client.ChatClient).
        """
        fields: dict[str, Any] = {}
        if self.thinking is not None and self.thinking_switch:
            fields["chat_template_kwargs"] = {"enable_thinking": self.thinking == THINKING_ON}
        if self.thinking_budget is not None:
            fields["thinking_token_budget"] = self.thinking_budget
        return fields
