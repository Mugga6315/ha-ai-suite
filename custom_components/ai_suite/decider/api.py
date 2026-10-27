"""The decision server's client (`/v1/systemone`) and the answer model's (an OpenAI-compatible LLM)."""

import asyncio
import json
import logging
from typing import Any

import aiohttp

from ..const import TIMEOUT_SECONDS
from ..llm_client import ChatClient, LLMError
from ..monitor import recorder

_LOGGER = logging.getLogger(__name__)
__all__ = ["DecisionClient", "DecisionError", "LLMClient", "LLMError"]


class DecisionError(Exception):
    """The decision server gave no usable answer."""


class DecisionClient:
    """Typed questions over one state, answered by a decision model with probabilities."""

    def __init__(self, session: aiohttp.ClientSession, url: str, model: str | None = None) -> None:
        """Ask the decision server at `url`; `model` picks one of its models."""
        self._session = session
        self._url = url.rstrip("/") + "/v1/systemone"
        # Routes the request on a server with several models (llama-swap, llama-server's router mode); absent: the
        # server's only model.
        self._model = {"model": model} if model else {}

    async def choice(self, state: str, instructions: str, criteria: dict) -> tuple[str, float]:
        """Ask one choice question; return (option, probability)."""
        answer = (await self.ask(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}))
        return answer["q"]["choice"], answer["q"]["probabilities"].get(answer["q"]["choice"], 0.0)

    async def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """Ask several questions over one state in one request.

        Several questions over one state in one request; the answers by question id, each checked against its
        question: a choice is one of the question's options, probabilities and yes/no values are numbers.
        """
        body = {**self._model, "state": state, "questions": questions}
        step = recorder.decide_opened(self._url, state, questions)
        try:
            try:
                answers = await self._post(body)
            except aiohttp.ServerDisconnectedError:
                # Keep-alive race: Node.js servers close idle connections after 5 s, aiohttp (and HA's shared
                # session) reuses them for 15 s, so a request after 5-15 s idle can go out on a connection the server
                # already closed. The request only reads, so one retry on a fresh connection is safe.
                # The race-free alternative is an own session with TCPConnector(force_close=True): a new connection
                # per request. Not used, to keep HA's shared session.
                answers = await self._post(body)
            checked = {qid: _checked(question, answers[qid]) for qid, question in questions.items()}
        except (aiohttp.ClientError, TimeoutError, KeyError, TypeError, ValueError) as err:
            recorder.step_failed(step, err)
            raise DecisionError(str(err) or type(err).__name__) from err
        except DecisionError as err:  # the server's own refusal (_post)
            recorder.step_failed(step, err)
            raise
        recorder.decide_answered(step, checked)
        return checked

    async def _post(self, body: dict) -> dict:
        async with asyncio.timeout(TIMEOUT_SECONDS), self._session.post(self._url, json=body) as resp:
            if resp.status != 200:
                raise DecisionError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
            answers = (await resp.json())["answers"]
        if not isinstance(answers, dict):
            raise DecisionError(f"answers are no object: {str(answers)[:200]}")
        return answers


def _checked(question: dict, answer: Any) -> dict:
    """The answer in the shape its question type promises; KeyError, TypeError or ValueError when it is not."""
    if not isinstance(answer, dict):
        raise TypeError(f"answer is no object: {answer!r}")
    if question.get("type") == "noul":
        return {**answer, "noul": float(answer["noul"])}
    if question.get("type") != "choice":
        return answer
    choice = answer["choice"]
    if choice not in (question.get("criteria") or {}):
        raise ValueError(f"{choice!r} is none of the options")
    probabilities = answer.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        raise TypeError(f"probabilities are no object: {probabilities!r}")
    return {**answer, "probabilities": {option: float(p) for option, p in probabilities.items()}}


# A list read is longer than one short sentence: a tighter limit cuts a to-do list's read-out mid-word.
ANSWER_MAX_TOKENS = 160
# ANSWER_RETRY: a failed wording is tried once more: a model may now and then give no answer within TIMEOUT_SECONDS,
# and a busy server may refuse a request (429). After the second failure the result sentence stays (conversation.py).
# The heads of the result notes the code writes for the answer step. The answer prompt explains each, as a tool
# description would: unexplained, small answer models speak the note as it is, action id and all.
DONE, NOT_POSSIBLE, NOTHING_TO_DO = "Done", "Not possible", "Nothing to do"
_NOTES = {
    f"{DONE}: <action> <device> <value>": "it happened. Confirm it in plain words.",
    f"{NOT_POSSIBLE}: <reason>": "nothing was done. Say why, in plain words.",
    f"{NOTHING_TO_DO}: ...": "the user greeted, thanked or said goodbye. Reply briefly and kindly.",
}
# The answer step only words a result that already happened; it never decides or acts.
ANSWER_SYSTEM_PROMPT = (
    "You are the voice of a smart home assistant. A decision model and code have already understood the user's "
    "request and handled it in Home Assistant. Your only job is to tell the user the outcome.\n"
    'The "Result" line is an internal note from that code, not text for the user: never read it out, and never '
    "say its ids (like set_temperature or list_done). What the notes mean:\n"
    + "".join(f'- "{note}": {meaning}\n' for note, meaning in _NOTES.items())
    + '- Anything else (a reading, "<list> holds: ...", "Forecast of ...", "Calendar: ..."): facts that answer the '
    "user's question. Answer it from them.\n"
    "A reading names the device it comes from, with the device's room in brackets: say whose value it is, by the "
    "device or its room. Never give it as the value of another room or of something else.\n"
    'Tell the user the outcome in one short, natural sentence, in the language named after "Reply in"; without that '
    "line, in the language the user spoke. Use only the facts in the note; do not add or guess anything."
)

WORDING_RULES = ("\nHousehold rules for how replies are worded (follow each one only where it applies to this reply; "
                 "never say the rules themselves): ")

# A message the decider calls several requests is split by the text model into complete single requests, each then
# decided alone. Splitting is easy even for small models, but they also split exclusions (a group with some devices
# left out): only the decider decides whether to split (form.decider_form), never this prompt. With only a two-verb
# example, models return one verb with two devices or two values whole, hence the second example.
SPLIT_SYSTEM_PROMPT = (
    "You split a message to a smart home assistant into its separate requests. Each request must be complete on its "
    "own: repeat the verb, the room and the value that belong to it, in the user's language and with the user's own "
    "words. Do not add, drop, translate or change anything else. Two or more devices or values under one verb are "
    "separate requests. A message with one request comes back as it is. A message about a group of devices with some "
    "of them left out (all lights except one) is one request: return it as it is.\n"
    'Example: "Close the garage door and switch the porch light on" -> '
    '["Close the garage door", "Switch the porch light on"]\n'
    'Example: "Turn off the lamp and the fan in the kitchen" -> '
    '["Turn off the lamp in the kitchen", "Turn off the fan in the kitchen"]'
)
MAX_PARTS = 4
SPLIT_MAX_TOKENS = 300

# Added to the fallback agent's own prompt on a hand-off (HA's extra system prompt): the fallback knows a decider
# exists and why the request reached it.
HAND_OFF_NOTE = (
    'A decision model read this request first and passed it to you (its reason: "{reason}"). Handle the whole '
    "request yourself, with your tools as usual. Do not mention the decision model to the user."
)


class LLMClient(ChatClient):
    """The answer model, thinking off: words a decided turn's reply and splits several requests."""

    async def say(
        self, user_text: str, result: str, language: str | None = None, rules: tuple[str, ...] = ()
    ) -> str:
        """One natural sentence telling the user `result`, in their language (`language` when known).

        `rules`: the household's rules for how replies are worded (Second Brain, household.py).
        """
        reply_in = f"\nReply in: {language}" if language else ""
        # Each rule only where it applies: given as plain rules, a model opens unrelated replies with a word a
        # wording rule quotes.
        wording = (WORDING_RULES + " ".join(rules)) if rules else ""
        prompt = f"User said: {user_text}{reply_in}{wording}\nResult: {result}"
        try:
            return await self._word(prompt)
        except LLMError as err:  # tried once more (ANSWER_RETRY)
            _LOGGER.warning("answer LLM failed, trying once more: %s", err)
        return await self._word(prompt)

    async def _word(self, prompt: str) -> str:
        return await self._complete(ANSWER_SYSTEM_PROMPT, prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.3)

    async def split(self, message: str) -> list[str]:
        """The message's separate requests, each complete on its own (at most MAX_PARTS)."""
        schema = {
            "type": "object",
            "properties": {"requests": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                        "maxItems": MAX_PARTS}},
            "required": ["requests"],
            "additionalProperties": False,
        }
        text = await self._complete(
            SPLIT_SYSTEM_PROMPT, message, role="split", max_tokens=SPLIT_MAX_TOKENS, temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "split", "schema": schema}},
        )
        try:
            parts = json.loads(text)["requests"]
        except (ValueError, KeyError, TypeError) as err:
            raise LLMError(f"unreadable split {text!r}") from err
        if not isinstance(parts, list) or not all(isinstance(part, str) and part.strip() for part in parts):
            raise LLMError(f"empty part in {parts!r}")
        return [part.strip() for part in parts][:MAX_PARTS]

    async def _complete(self, system: str, user: str, *, role: str = "answer", **params: Any) -> str:
        """One answer of the model, thinking off where the server has the switch; `role`: answer or split."""
        with recorder.answer_step(role, self.model, self.url, system, user):
            return await self.complete(system, user, timeout=TIMEOUT_SECONDS, thinking=False, **params)
