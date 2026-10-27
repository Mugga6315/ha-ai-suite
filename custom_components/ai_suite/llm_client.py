"""Single chat-completions requests to the LLM server: everything but the agent loop.

The answer model, the rule compiler and Second Brain's background jobs all go through ChatClient, so the server's
settings (the thinking switch) and its quirks (fields it rejects) are handled in one place.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import re
from typing import Any

import aiohttp

from .const import CONF_LLM_URL, CONF_THINKING_OFF_KWARG, DEFAULT_THINKING_OFF_KWARG
from .monitor import recorder

_LOGGER = logging.getLogger(__name__)
# Statuses that say the server does not take a field of the request. Others (401, 404, 429) say nothing about the
# fields, so they never drop one.
FIELD_REJECTED = (400, 422)
# Optional fields by group, dropped in this order when the server rejects a request that has them.
_OPTIONAL = {"json": ("response_format",), "thinking": ("reasoning_effort", "chat_template_kwargs")}
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMError(Exception):
    """The LLM server gave no usable answer."""


@dataclass(frozen=True, slots=True)
class LLMServer:
    """An OpenAI-compatible server.

    `thinking_switch`: it takes `chat_template_kwargs` (vLLM with the Mistral tokenizer answers it with HTTP 400).
    """

    url: str
    thinking_switch: bool = True

    @classmethod
    def from_entry_data(cls, data: Mapping[str, Any]) -> LLMServer:
        """The LLM server of AI Suite's config entry."""
        return cls(data[CONF_LLM_URL], data.get(CONF_THINKING_OFF_KWARG, DEFAULT_THINKING_OFF_KWARG))


class ChatClient:
    """One model on the LLM server. An optional field the server rejects is left out of later requests too."""

    def __init__(self, session: aiohttp.ClientSession, server: LLMServer, model: str) -> None:
        """Ask `model` on `server`."""
        self.url = server.url.rstrip("/") + "/chat/completions"
        self.model = model
        self._session = session
        self._thinking_switch = server.thinking_switch
        self._rejected: set[str] = set()

    async def complete(
        self,
        system: str,
        user: str,
        *,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """The model's answer to one system and one user message.

        `thinking`: on or off where the server has the switch, None for its default; `effort`: a reasoning effort, sent
        with thinking on.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format
        if effort is not None:
            body["reasoning_effort"] = effort
        if thinking is not None and self._thinking_switch:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        return await self._post(body, timeout)

    async def complete_json(self, system: str, user: str, **options: Any) -> dict[str, Any]:
        """The answer read as a JSON object; a code fence around it is allowed."""
        text = await self.complete(system, user, **options)
        try:
            value = json.loads(_CODE_FENCE.sub("", text.strip()))
        except ValueError as err:
            raise LLMError(f"no JSON in the answer: {text[:200]!r}") from err
        if not isinstance(value, dict):
            raise LLMError(f"the answer is no JSON object: {text[:200]!r}")
        return value

    async def _post(self, body: dict[str, Any], timeout: float) -> str:
        for group in self._rejected:
            for name in _OPTIONAL[group]:
                body.pop(name, None)
        recorder.llm_request(body)
        try:
            async with self._session.post(self.url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status in FIELD_REJECTED and (dropped := _droppable(body, self._rejected)):
                    _LOGGER.info("%s rejected %s (HTTP %s), asking without", self.url, dropped, resp.status)
                    self._rejected.add(dropped)
                    return await self._post(body, timeout)
                if resp.status != 200:
                    raise LLMError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
                data = await resp.json()
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise LLMError(str(err) or type(err).__name__) from err
        recorder.llm_response(data)
        return _answer_text(data)


def _droppable(body: dict[str, Any], rejected: set[str]) -> str | None:
    """The first group of optional fields the request still has."""
    return next(
        (group for group, names in _OPTIONAL.items() if group not in rejected and any(n in body for n in names)), None
    )


def _answer_text(data: Any) -> str:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise LLMError(f"no answer in {str(data)[:200]}") from err
    if not isinstance(text, str) or not text.strip():
        raise LLMError("empty answer")
    return text.strip()
