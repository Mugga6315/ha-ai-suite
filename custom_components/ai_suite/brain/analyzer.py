"""Per-turn self-improvement analyzer.

Runs in the background after each assist turn that used this integration. It
reviews the turn (thinking effort configurable) and, when it finds a misfire,
appends one line to failures.md - the same input the nightly learner turns into
a rule. It never blocks the assist path and never writes a rule itself; that
judgement (does a repeated pattern deserve a rule) stays with the learner.

Signal it looks for, all decidable from the turn alone:
  - redundant/unused tool calls (efficiency),
  - denied-but-present: an answer says a device is missing while it is exposed,
  - over-broad action: a whole area toggled when one device was named AND the
    area holds more than one matching device,
  - rule adherence: a violation of an active rule,
  - language: an answer in a different language than the question,
  - detour: the answer was right but took more calls than the task needs, and
    the shorter path can be named. This is what makes the assistant faster over
    time, not only more correct.
"""
import asyncio
import json

import probatio

from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TimeSelector,
)

from ..llm_client import ChatClient
from .background import effort_options
from .const import (
    CONF_LEARN_TIME,
    CONF_SELF_IMPROVE_EFFORT,
    DEFAULT_LEARN_TIME,
    DEFAULT_SELF_IMPROVE_EFFORT,
    LOGGER,
    SELF_IMPROVE_EFFORTS,
)
from .store import PROMPT_MARKER, Store

# Our standing context is injected on every turn that used this API; its presence
# in the system prompt is how we know the turn was ours and worth analyzing.
_MARKER = PROMPT_MARKER

_SYSTEM_CONTENT_LIMIT = 8000
_LLM_TIMEOUT_SECONDS = 120

# Quiet time after the last chat-log event before the turn is read, and how many
# times that read may be retried while the turn is still running. Long on
# purpose: the review has nobody waiting for it, and a late read is a complete
# turn, while an early one is a turn read twice.
SETTLE_SECONDS = 45
MAX_READS = 6

_SYSTEM = (
    "You review ONE completed smart-home assistant turn and decide if it "
    "misfired. Return ONLY valid JSON, no markdown fences."
)

_RUBRIC = """Judge only against the evidence given; do not invent problems. Check five things:

1. Redundant/unused tool calls (efficiency): the SAME call repeated, or a call whose result clearly did not contribute to the answer. A single reasonable preparatory call is NOT redundant (e.g. GetDateTime to resolve "today" before a date-scoped read).
2. Structural misfire, checkable against the exposed entities in the assistant's context:
   - denied-but-present: the answer says a device/sensor is missing, yet it is exposed (and no read tool was called to check).
   - over-broad action: an action targeted a whole area/domain AND that area/domain holds MORE THAN ONE matching device, so it hit devices beyond the one named. If exactly one matches, targeting the area is fine - do NOT flag it.
3. Rule adherence: did the turn violate an active rule shown in the assistant's context (e.g. answered in the wrong language)?
4. Detour on a CORRECT turn (the answer was right, the path was not): the same result was reachable with fewer calls or in one step - searching or listing to find an entity the context already exposed, reading state before an action that does not need it, or probing to work out how a tool behaves. Say which single call (with its arguments) would have been enough. Only flag a detour you can name that shorter path for.
   A RECOVERY call is never a detour: when a call failed or came back empty and the next call tries another way to get the same thing (a search after a note was not found, a broader read after a narrow one missed), that call is the assistant doing its job. It is not waste just because it happened to find nothing this time - on another turn it is what saves the answer. Judge only the calls the turn could have skipped knowing what it knew BEFORE it made them.

5. Language: the answer must be in the language the question was asked in - a German question gets a German answer. Flag only a different language, never wording or tone.

Return ONLY JSON:
{"verdict": "clean" | "misfire" | "improvable",
 "categories": ["redundant" | "denied_but_present" | "over_action" | "rule_violation" | "detour" | "wrong_language" ...],
 "evidence": "one sentence citing the specific calls/answer",
 "failure_entry": "<one concise line for failures.md that the nightly learner could turn into a rule>" | null}
"improvable" is for point 4: right answer, wasteful path. The entry then states the shortcut, e.g. "answering 'how warm is X' took search_brain then GetLiveContext; GetLiveContext alone answers it".
If the turn is fine and efficient: verdict "clean", categories [], failure_entry null."""


def _tool_call_fields(tc: object) -> tuple[str, object] | None:
    """Pull (name, args) from a ToolInput object or its dict form. None if neither."""
    name = getattr(tc, "tool_name", None)
    args = getattr(tc, "tool_args", None)
    if name is None and isinstance(tc, dict):
        name = tc.get("tool_name")
        args = tc.get("tool_args")
    return (name, args) if name else None


def is_our_turn(content: list[dict]) -> bool:
    """Did this conversation use our API? Read from the injected system prompt.

    Separate from summarize_turn because the two "no" answers differ: a turn
    that is not ours will never become ours, while an unfinished turn is worth
    reading again.
    """
    system = next(
        (c.get("content", "") for c in content if c.get("role") == "system"), ""
    )
    return _MARKER in (system or "")


def summarize_turn(content: list[dict]) -> dict | None:
    """Reduce a chat_log's content list to the pieces the analyzer needs.

    Returns None when the turn is not ours or has no final answer yet.
    """
    if not content:
        return None
    system = next(
        (c.get("content", "") for c in content if c.get("role") == "system"), ""
    )
    if _MARKER not in (system or ""):
        return None
    last = content[-1]
    if last.get("role") != "assistant" or not last.get("content"):
        return None  # turn not finished with a spoken answer
    if last.get("tool_calls"):
        return None  # spoke, but is still calling tools — not the final answer

    user = next(
        (c.get("content", "") for c in reversed(content) if c.get("role") == "user"),
        "",
    )
    calls: list[tuple[str, object]] = []
    for c in content:
        if c.get("role") != "assistant":
            continue
        for tc in c.get("tool_calls") or []:
            fields = _tool_call_fields(tc)
            if fields:
                calls.append(fields)
    # Count of finished assistant answers, so each turn is analyzed exactly once.
    n_answers = sum(
        1 for c in content if c.get("role") == "assistant" and c.get("content")
    )
    return {
        "system": system,
        "user": user,
        "calls": calls,
        "final": last.get("content", ""),
        "n_answers": n_answers,
    }


class TurnAnalyzer:
    """Reviews each finished turn the text model answered and notes a misfire or a detour in failures.md."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, store: Store, client: ChatClient, effort: str
    ) -> None:
        """Review the turns of `entry`'s agents into `store`, with the background model and its effort."""
        self._hass = hass
        self._entry = entry
        self._store = store
        self._client = client
        self._effort = effort
        # One review at a time: reviews queue behind each other instead of
        # competing with the assistant itself for the LLM endpoint's slots.
        self._running = asyncio.Lock()
        # conversation_id -> answers already analyzed, so a multi-turn chat does
        # not re-analyze earlier turns.
        self._seen: dict[str, int] = {}
        # conversation_id -> the pending settle timer's cancel callback.
        self._pending: dict[str, CALLBACK_TYPE] = {}

    @callback
    def on_chat_log(self, conversation_id: str, event_type, data: dict) -> None:
        """Chat-log subscriber. Never raises into HA's conversation path.

        The events are only a wake-up: HA emits no event for the assistant's
        final answer when the agent streams it (`async_add_assistant_content`
        appends without notifying), so the turn is read from the live chat log
        once the events stop coming.

        TODO: a workaround for HA's behaviour as checked in HA 2026.10. Check it again
        on HA updates; once HA notifies for the final answer, read it from the event.
        """
        try:
            if not data:  # DELETED: the conversation is over
                self._cancel(conversation_id)
                self._seen.pop(conversation_id, None)
                return
            self._schedule_read(conversation_id, 0)
        except Exception:  # noqa: BLE001 - logged; a review must never break a real conversation
            LOGGER.exception("analyzer: failed to handle chat log")

    @callback
    def _schedule_read(self, conversation_id: str, attempt: int) -> None:
        self._cancel(conversation_id)

        @callback
        def _read(_now) -> None:
            self._pending.pop(conversation_id, None)
            self._read_turn(conversation_id, attempt)

        self._pending[conversation_id] = async_call_later(
            self._hass, SETTLE_SECONDS, _read
        )

    @callback
    def _cancel(self, conversation_id: str) -> None:
        cancel = self._pending.pop(conversation_id, None)
        if cancel is not None:
            cancel()

    @callback
    def async_shutdown(self) -> None:
        """Drop every pending timer — the entry is unloading."""
        for conversation_id in list(self._pending):
            self._cancel(conversation_id)

    @callback
    def _read_turn(self, conversation_id: str, attempt: int) -> None:
        # The live ChatLog HA keeps for the session, the only place the finished turn exists in full.
        logs = self._hass.data.get(DATA_CHAT_LOGS) or {}
        chat_log = logs.get(conversation_id)
        if chat_log is None:
            return
        content = chat_log.as_dict().get("content") or []
        if not is_our_turn(content):
            return  # another agent's conversation - nothing to wait for
        turn = summarize_turn(content)
        if turn is None:  # still mid-turn — wait for it to finish
            if attempt + 1 < MAX_READS:
                self._schedule_read(conversation_id, attempt + 1)
            return
        if self._seen.get(conversation_id, 0) >= turn["n_answers"]:
            return  # already analyzed this turn
        self._seen[conversation_id] = turn["n_answers"]
        # Owned by the entry: an unload cancels a review still running, so none writes to the store after it.
        self._entry.async_create_background_task(self._hass, self._analyze(turn), "ai_suite_turn_review")

    async def _analyze(self, turn: dict) -> None:
        try:
            accepted = await self._store.async_read_not_a_defect()
            async with self._running:
                verdict = await self._call_llm(self._build_prompt(turn, accepted))
            if verdict.get("verdict") not in ("misfire", "improvable"):
                return
            entry = str(verdict.get("failure_entry") or "").strip()
            if not entry:
                return
            cats = ",".join(verdict.get("categories") or []) or "misfire"
            await self._store.async_record_failure(f"analyzer:{cats}", entry)
            LOGGER.debug("analyzer: recorded misfire (%s): %s", cats, entry)
        except Exception:  # noqa: BLE001 - logged; a failed review costs only its note
            LOGGER.exception("analyzer: turn analysis failed")

    def _build_prompt(self, turn: dict, accepted: str = "") -> str:
        calls = "\n".join(f"CALL {n}({json.dumps(a)})" for n, a in turn["calls"])
        system = turn["system"]
        if len(system) > _SYSTEM_CONTENT_LIMIT:
            system = system[:_SYSTEM_CONTENT_LIMIT] + "\n...(truncated)"
        # The human's veto: behaviour reviewed and judged fine. It outranks the
        # rubric, so a case that keeps being flagged can be settled by hand.
        veto = (
            f"\n\n## Reviewed and accepted by the user - NEVER flag these, "
            f"whatever the rubric says\n{accepted.strip()}"
            if accepted.strip()
            else ""
        )
        return (
            f"{_RUBRIC}{veto}\n\n## Assistant's own context this turn "
            f"(exposed entities and active rules)\n{system}\n\n"
            f"## The turn\nUser said: {turn['user']!r}\n\nTool calls in order:\n"
            f"{calls or '(none)'}\n\nFinal answer to user: {turn['final']!r}"
        )

    async def _call_llm(self, prompt: str) -> dict:
        return await self._client.complete_json(
            _SYSTEM, prompt, timeout=_LLM_TIMEOUT_SECONDS, temperature=0.3,
            response_format={"type": "json_object"}, **effort_options(self._effort),
        )


# --- seams: the feature's subentry interface (see features.py) ----------------


def subentry_schema(data: dict) -> dict:
    """The fields of the self-improvement subentry form.

    The feature is on because the subentry exists — removing it is the off
    switch, as for every other feature. What is left to choose is how hard the
    review thinks, and when the nightly pass turns the notes into rules.
    """
    return {
        probatio.Required(
            CONF_SELF_IMPROVE_EFFORT,
            default=data.get(CONF_SELF_IMPROVE_EFFORT, DEFAULT_SELF_IMPROVE_EFFORT),
        ): SelectSelector(SelectSelectorConfig(
            options=SELF_IMPROVE_EFFORTS, mode=SelectSelectorMode.DROPDOWN, translation_key="effort"
        )),
        probatio.Required(
            CONF_LEARN_TIME,
            default=data.get(CONF_LEARN_TIME, DEFAULT_LEARN_TIME),
        ): TimeSelector(),
    }


async def async_validate(hass, data: dict) -> str | None:
    """Nothing to validate — the LLM is configured and checked on the parent."""
    return None
