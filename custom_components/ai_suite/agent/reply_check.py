"""The loop's reply is checked before the user hears it; on a fault the model answers once more, with its tools.

The checks are those that find faults without harming right replies: language, made-up numbers, raw markup. The
retry runs only in a turn that changed nothing: a retry after an action repeats the action. No check for whose value
a reply gives (it flags more right replies than wrong ones) or for declines (a retry does not fix them).
"""

from dataclasses import dataclass
import json
import re
from typing import Any

from homeassistant.components import conversation

from ..languages import LANGUAGE_NAMES, LANGUAGES, alternatives

LANGUAGE, NUMBERS, RAW = "language", "numbers", "raw"

_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
# Dates and clock times are days and times, not readings; a clock time also as an hour with an hour word (a model
# may give a UTC forecast hour in local time, so the hour is in no tool result), and a range of such hours.
_DATE_OR_TIME = re.compile(
    r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    rf"|\b(?:\d{{1,2}}\s*[--]\s*)?\d{{1,2}}\s*{alternatives(lambda language: language.time.clock)}(?!\w)",
    re.IGNORECASE,
)
# A time window the reply names (a window word, a count and a time unit) is no reading. A duration that is a value
# still is: only window words count.
_TIME_WINDOW = re.compile(
    rf"\b{alternatives(lambda language: language.time.window)}\s+\d+\s*"
    rf"{alternatives(lambda language: language.time.time_units)}\b",
    re.IGNORECASE,
)
_ISO_SEPARATOR = re.compile(r"(?<=\d)T(?=\d)")
_SMALL_INT = 10  # counts, ordinals and list numbers are derived, not readings
_RAW = re.compile(r"<function[ =]|</?tool_call|<\|[a-z_\"]+\|?>|__intent__|home-control__|^\s*(done|not possible|"
                  r"nothing to do):|error handling api response|unable to get response",
                  re.IGNORECASE)
# Unit conversions people ask for, both ways: mi→km, in→cm, ft→m, lb→kg, gal→l, oz→g (°F↔°C apart).
_CONVERSIONS = (1.609344, 2.54, 0.3048, 0.45359237, 3.785411784, 28.349523125)


@dataclass(frozen=True)
class Turn:
    """The loop's turn as the checks see it."""

    reply: str
    messages: list[dict]  # what the model saw before its reply, earlier turns included
    acted: bool  # a tool ran that may write: one without HA's read_only annotation counts as writing


@dataclass(frozen=True)
class Fault:
    """One thing wrong with a reply: the check that found it, and the sentence that tells the model."""
    check: str
    text: str  # the fault, as a sentence for the model


def turn_of(chat_log: conversation.ChatLog, user_text: str) -> Turn:
    """The turn of `user_text` in the chat log, with the conversation before it.

    A follow-up answers from earlier tool results too.
    """
    content = list(chat_log.content)
    reply = content[-1].content or "" if content and content[-1].role == "assistant" else ""
    while content and content[-1].role == "assistant" and not content[-1].tool_calls:
        content.pop()  # the reply itself
    start = max((i for i, c in enumerate(content) if c.role == "user" and c.content == user_text), default=0)
    tools = {tool.name: tool for tool in chat_log.llm_api.tools} if chat_log.llm_api else {}
    called = [call.tool_name for c in content[start:] if c.role == "assistant" for call in c.tool_calls or []]
    acted = any(name not in tools or not tools[name].annotations.read_only for name in called)
    return Turn(reply, [_message(c) for c in content if c.role != "system" or c.content], acted)


def faults(turn: Turn, language: str | None) -> list[Fault]:
    """What is wrong with the reply; `language`: the code the reply should be in."""
    reply, seen = turn.reply, json.dumps(turn.messages, ensure_ascii=False)
    found = []
    if _RAW.search(reply):
        found.append(Fault(RAW, "Your reply was not plain text for the user: it held tool-call text or an internal "
                                "note, and a tool call written as text does not run."))
    if language and wrong_language(reply, language):
        found.append(Fault(LANGUAGE, f"The user spoke {LANGUAGE_NAMES[language]}; your reply is in another language."))
    if unplaced := invented_numbers(reply, seen):
        found.append(Fault(NUMBERS, f"These numbers in your reply appear in no tool result or device state: "
                                    f"{', '.join(unplaced)}. Do not state values you were not given."))
    return found


def retry_request(found: list[Fault]) -> str:
    """The message that asks for the reply again.

    The user never hears the checked reply, so the new one must not refer to it (without that line models refer back
    to it or apologise).
    """
    return ("Check of the reply above, before the user hears it: " + " ".join(f.text for f in found)
            + " Write the reply the user will hear instead. The user has not heard the reply above: do not refer to it "
            "or apologize for it.")


def _message(content: conversation.Content) -> dict[str, Any]:
    if content.role == "tool_result":
        data = json.dumps(content.result.data, ensure_ascii=False, default=str)
        return {"role": "tool", "tool_call_id": content.tool_call_id, "content": data}
    message: dict[str, Any] = {"role": content.role, "content": content.content or ""}
    if content.role == "assistant" and content.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.tool_name, "arguments": json.dumps(call.tool_args, ensure_ascii=False)}}
            for call in content.tool_calls
        ]
    return message


# --- language -------------------------------------------------------------------------------------------------------

def language_of(text: str) -> str | None:
    """The language of the user's words, by common words: the one with more of them; None when they do not decide.

    The pipeline's language is no answer: an English question in a German pipeline is English, and a reply right in
    English would be turned German by the retry.
    """
    counts = _common_words(text)
    best = max(counts, key=lambda code: counts[code])
    return best if counts[best] > max((c for code, c in counts.items() if code != best), default=0) else None


def wrong_language(reply: str, language: str) -> bool:
    """By common words; a reply in a language without a module (languages/) passes."""
    if language not in LANGUAGES:
        return False
    counts = _common_words(reply)
    return counts[language] < max((count for code, count in counts.items() if code != language), default=0)


def _common_words(text: str) -> dict[str, int]:
    """How many of each language's common words the text holds."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return {code: sum(word in language.function_words for word in words) for code, language in LANGUAGES.items()}


# --- made-up numbers ------------------------------------------------------------------------------------------------

def invented_numbers(reply: str, seen: str) -> list[str]:
    """Numbers in the reply that no tool gave and no step of arithmetic makes.

    Numbers in the reply that are neither a value the model saw nor computed from the user's own numbers and the
    values the reply took from the data (one step of arithmetic, a unit conversion, a percent as a fraction).
    """
    sources = _source_values(seen)
    said = _TIME_WINDOW.sub(" ", _DATE_OR_TIME.sub(" ", reply))
    tokens = [(t, vs) for t, vs in readings(said) if not all(v <= _SMALL_INT and v == int(v) for v in vs)]
    unplaced = [(t, vs) for t, vs in tokens if not any(_matches(t, v, sources) for v in vs)]
    if not unplaced:
        return []
    placed = [v for t, vs in tokens if (t, vs) not in unplaced for v in vs]
    computed = _derived([v for _, vs in readings(_user_words(seen)) for v in vs] + placed)
    return [t for t, vs in unplaced if not any(_matches(t, v, computed) for v in vs)]


def readings(text: str) -> list[tuple[str, list[float]]]:
    """Each number in the text with the values it can mean.

    One separator is a decimal point, or a thousands separator before three digits; several are thousands separators.
    """
    found = []
    for token in _NUMBER.findall(_LIST_MARKER.sub(" ", text)):
        parts = re.split(r"[.,]", token)
        values = {float("".join(parts))} if len(parts) > 2 else set()
        if len(parts) == 2:
            values.add(float(f"{parts[0]}.{parts[1]}"))
            if len(parts[1]) == 3:
                values.add(float(parts[0] + parts[1]))
        if len(parts) == 1:
            values.add(float(token))
        found.append((token, sorted(values)))
    return found


def _source_values(text: str) -> list[float]:
    """Every value the model saw, also as kilo/milli, as a percent of 255 (brightness) and of 1 (volume)."""
    values = [v for _, vs in readings(_ISO_SEPARATOR.sub(" ", text)) for v in vs]
    return values + [v / 1000 for v in values] + [v * 1000 for v in values] + [v / 2.55 for v in values] + [
        v * 100 for v in values if v <= 1]


def _matches(token: str, value: float, sources: list[float]) -> bool:
    """Equal to a source value as said: rounded to the token's own decimals, or to a whole number."""
    decimals = len(re.split(r"[.,]", token)[-1]) if re.search(r"[.,]\d{1,2}$", token) else 0
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - s) <= tolerance for s in sources)


def _derived(operands: list[float]) -> list[float]:
    """Every value one step of arithmetic makes from the operands.

    One step of arithmetic over the operands (sum, difference, product, quotient, percent), a percent as a
    fraction and back, and each operand converted between common units.
    """
    values = []
    for a in operands:
        values += [a / 100, a * 100]
        values += [a * f for f in _CONVERSIONS] + [a / f for f in _CONVERSIONS] + [(a - 32) / 1.8, a * 1.8 + 32]
        for b in operands:
            values += [a + b, a - b, a * b, a * b / 100] + ([a / b, a / b * 100] if b else [])
    return values


def _user_words(seen: str) -> str:
    users = [m for m in json.loads(seen) if m.get("role") == "user"]
    content = users[-1].get("content") if users else ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
