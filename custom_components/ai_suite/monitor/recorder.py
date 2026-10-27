"""What a turn did, kept while it runs as references and monotonic stamps; the record is built later (record.py).

No Home Assistant imports: decider/ calls it.

`open_turn` (around `async_process`) puts the turn's TurnTrace into a ContextVar, so the tasks the turn creates (the
reply-language task, the chat log's tool tasks) record into it too. Every hook is a no-op without an open turn and
after the turn closed (a task the turn created may outlive it). Hooks keep references and `time.monotonic()` stamps
only: hashing, JSON and copying happen when the record is built, off the turn's path. A hook that takes a `step` takes
what its opening hook returned, None included. Times spent in hooks add up in `TurnTrace.hook_ns`.

Turn (conversation.py and its wrapper)
  open_turn(recorder, start) -> context manager of TurnTrace | None   around async_process; None: not recorded
                                       (a turn already open in the context: the steps join it; MAX_OPEN_TURNS held)
  turn_path(path, handoff_reason=None) decided | handoff | loop | error; the last call wins
  conversation(conversation_id)        the first id given wins: the chat session's before the result's
  turn_output(speech, continue_conversation)
  turn_error(stage, err)               errors[]; the same exception given twice is kept once
Decider (decider/api.py, decider/form.py, llm_client.py)
  decide_opened(url, state, questions) -> DecideStep | None   DecisionClient.ask, before its request
  decide_answered(step, answers)       the checked answers by question id
  bar(qid, value)                      the bar a probability was compared with: set on the newest decide step that
                                       asked `qid` (one the current task opened first), unless that step has a bar
                                       for it already
  answer_step(role, model, url, system, user) -> context manager of AnswerStep | None
                                       around LLMClient._complete (role answer | split); an exception leaving it is
                                       the step's error. Only inside it do the next two hooks fill a step.
  llm_request(body)                    ChatClient._post: the body as sent, after dropped fields
  llm_response(data)                   ChatClient._post: the parsed JSON answer
Loop (agent/)
  system_parts(api_prompt, extra_prompt)  agent/turn.py, after async_provide_llm_data
  round_opened(call)                   agent/loop.py, the call's keyword arguments, before the request; role round,
                                       or retry after `retrying`
  stream_token()                       agent/reply.py, each chunk with reasoning, reply or a tool-call piece; the
                                       first one stamps the ttft
  stream_finish(reason)                agent/reply.py, a chunk's finish_reason
  stream_usage(usage)                  agent/reply.py, a chunk's usage
  stream_end()                         agent/reply.py, right after the stream's last chunk: the round's end
  round_content(content)               agent/loop.py, each content the chat log yields for the round
  round_failed(err)                    agent/loop.py, the round's server error (record the LLMServerError raised for
                                       it with turn_error("llm", ...))
  tool_opened(call_id, name, args) -> ToolStep | None   WrappedTool.async_call; parent: the newest round
  tool_returned(step, result)
  reply_check_opened() -> ReplyCheckStep | None         agent/turn.py, before the check
  reply_checked(step, *, language, reply, acted, faults)
  retrying()                           agent/turn.py, the check asks for the reply again: later rounds are retries
Actions (home_actions.py)
  intent_opened(intent, slots, target) -> ActionStep | None   HomeActions._handle_intent
  service_opened(domain, service, data) -> ActionStep | None  HomeActions._call
  action_returned(step, result)        the IntentResponse, or the action's response
Any step
  step_failed(step, err)               the step's error; ends it
Effects (the bus listener, outside the turn's context)
  add_effect(trace, effect)            kept until EFFECTS_WINDOW_SECONDS after the turn closed
Exporter
  release(recorder, trace)             the turn is built: no longer held, its context no longer matched
"""

from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from itertools import chain
from time import monotonic, perf_counter_ns, time
from typing import Any, ClassVar, Concatenate, Literal

from .const import EFFECTS_WINDOW_SECONDS, MAX_OPEN_TURNS

# Steps are mutable: each is created when it opens and filled by later hooks. All hooks run on the event loop, so
# nothing is shared across threads.


@dataclass(frozen=True, slots=True)
class TurnStart:
    """What a turn is known by when it opens."""

    id: str  # the record's id (ULID)
    context_id: str  # the turn's HA context: state changes under it are the turn's effects
    agent_entity: str
    language: str
    text: str
    satellite_id: str | None = None
    device_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnError:
    """An error of the turn (errors[]): where it happened and the exception."""

    stage: str
    error: BaseException


@dataclass(frozen=True, slots=True)
class Effect:
    """A state change caused by the turn: HA's states as they were, not copied."""

    time_fired: float  # the event's wall time (Event.time_fired_timestamp)
    entity_id: str
    old_state: Any  # homeassistant.core.State | None
    new_state: Any  # homeassistant.core.State
    via: Literal["own", "child"]  # the turn's context, or a context it caused


@dataclass(slots=True)
class DecideStep:
    """One request to the decision server: its questions over one state."""

    kind: ClassVar[str] = "decide"
    id: int
    start: float
    url: str
    state: str
    questions: dict[str, dict]  # by question id, as sent
    answers: dict[str, dict] | None = None  # by question id, as checked
    bars: dict[str, float] = field(default_factory=dict)  # by question id, where a comparison was made
    end: float | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class AnswerStep:
    """One request to the answer model on the decided path (kind llm, role answer or split)."""

    kind: ClassVar[str] = "llm"
    id: int
    start: float
    role: str  # answer | split
    model: str
    url: str
    system: str
    user: str
    request: dict[str, Any] | None = None  # the body as sent
    response: Any = None  # the parsed JSON answer
    end: float | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class RoundStep:
    """One round of the agent loop (kind llm, role round or retry)."""

    kind: ClassVar[str] = "llm"
    id: int
    start: float
    role: str  # round | retry
    call: dict[str, Any]  # the chat-completions call's keyword arguments: model, messages, tools, params
    first_token: float | None = None
    finish: str | None = None
    usage: Any = None  # the stream's usage (openai CompletionUsage)
    contents: list[Any] = field(default_factory=list)  # what the chat log yielded: assistant content, tool results
    end: float | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class ToolStep:
    """One tool call of a round."""

    kind: ClassVar[str] = "tool"
    id: int
    parent: int | None  # the round that called it
    start: float
    call_id: str
    name: str
    args: dict[str, Any]
    result: Any = None  # llm.ToolResult or the tool's JSON
    end: float | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class ActionStep:
    """One HA intent or action the decided path ran: `intent` with its slots, or `service` with its data."""

    kind: ClassVar[str] = "action"
    id: int
    start: float
    target: str | None  # the intent's target, the action's data["entity_id"]
    data: dict[str, Any]  # the intent's slots, the action's data
    intent: str | None = None
    domain: str | None = None
    service: str | None = None
    result: Any = None  # intent.IntentResponse, or the action's response
    end: float | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class ReplyCheckStep:
    """The check of the loop's reply before the user hears it."""

    kind: ClassVar[str] = "reply_check"
    id: int
    start: float
    language: str | None = None
    reply: str = ""
    acted: bool = False
    faults: Sequence[Any] = ()  # agent.reply_check.Fault: .check, .text
    retried: bool = False
    end: float | None = None
    error: BaseException | None = None


type Step = DecideStep | AnswerStep | RoundStep | ToolStep | ActionStep | ReplyCheckStep


@dataclass(slots=True)
class TurnTrace:
    """One turn as recorded: from open to close (the agent's start and end), then its effects."""

    start: TurnStart
    opened: float  # monotonic
    opened_wall: float  # wall time at `opened`
    closed: float | None = None  # monotonic; None while open
    path: str | None = None
    handoff_reason: str | None = None
    conversation_id: str | None = None
    speech: str | None = None
    continue_conversation: bool = False
    api_prompt: str | None = None  # the system prompt's parts HA joins (agent/turn.py)
    extra_prompt: str | None = None
    steps: list[Step] = field(default_factory=list)  # in open order; a step's id is its place, from 1
    effects: list[Effect] = field(default_factory=list)
    errors: list[TurnError] = field(default_factory=list)
    current_round: RoundStep | None = None
    round_role: str = "round"
    hook_ns: int = 0  # time spent in hooks (perf_counter_ns)
    hook_calls: int = 0


@dataclass(slots=True)
class Recorder:
    """The turns held, recording or waiting to be built, by context id (the effects listener's lookup)."""

    turns: dict[str, TurnTrace] = field(default_factory=dict)
    recorded: int = 0
    skipped: int = 0  # turns not recorded: MAX_OPEN_TURNS held, or their context already held


_turn: ContextVar[TurnTrace | None] = ContextVar("ai_suite_monitor_turn", default=None)
# The answer step being filled. Task-local apart from the turn: only ChatClient requests inside answer_step fill it,
# never another job on the same client (Second Brain) in a task that shares the turn.
_answer: ContextVar[AnswerStep | None] = ContextVar("ai_suite_monitor_answer", default=None)
# The decide steps the current task opened (a task starts with those of the task that created it). A bar goes to one
# of them first: the reply-language task and ask_sure both ask `q`, and with both in flight the newest step that
# asked `q` may be the other task's.
_own_decides: ContextVar[tuple[DecideStep, ...]] = ContextVar("ai_suite_monitor_own_decides", default=())
_NOT_RECORDED = nullcontext(None)


@contextmanager
def open_turn(recorder: Recorder, start: TurnStart) -> Generator[TurnTrace | None]:
    """Record the turn inside; closed on leaving. None: not recorded (see Recorder.skipped), or joined to the open one."""
    if (outer := _turn.get()) is not None and outer.closed is None:
        yield None
        return
    if len(recorder.turns) >= MAX_OPEN_TURNS or start.context_id in recorder.turns:
        recorder.skipped += 1
        yield None
        return
    trace = TurnTrace(start, opened=monotonic(), opened_wall=time())
    recorder.turns[start.context_id] = trace
    recorder.recorded += 1
    token, decides_token = _turn.set(trace), _own_decides.set(())
    try:
        yield trace
    finally:
        trace.closed = monotonic()
        _own_decides.reset(decides_token)
        _turn.reset(token)


def release(recorder: Recorder, trace: TurnTrace) -> None:
    """Stop holding the turn: it was built, or will not be."""
    recorder.turns.pop(trace.start.context_id, None)


def add_effect(trace: TurnTrace, effect: Effect) -> None:
    """Keep a state change of the turn, while its window is open."""
    began = perf_counter_ns()
    if trace.closed is None or monotonic() <= trace.closed + EFFECTS_WINDOW_SECONDS:
        trace.effects.append(effect)
    _spent(trace, began)


def _hook[**P, R](record: Callable[Concatenate[TurnTrace, P], R]) -> Callable[P, R | None]:
    """The hook: `record` with the open turn, timed; nothing without one."""

    @wraps(record)
    def hook(*args: P.args, **kwargs: P.kwargs) -> R | None:
        trace = _turn.get()
        if trace is None or trace.closed is not None:
            return None
        began = perf_counter_ns()
        recorded = record(trace, *args, **kwargs)
        _spent(trace, began)
        return recorded

    return hook


@_hook
def turn_path(trace: TurnTrace, path: str, handoff_reason: str | None = None) -> None:
    """How the turn was answered."""
    trace.path, trace.handoff_reason = path, handoff_reason


@_hook
def conversation(trace: TurnTrace, conversation_id: str | None) -> None:
    """The turn's conversation id; the first one given wins."""
    if trace.conversation_id is None:
        trace.conversation_id = conversation_id


@_hook
def turn_output(trace: TurnTrace, speech: str, continue_conversation: bool) -> None:
    """What the agent answered."""
    trace.speech, trace.continue_conversation = speech, continue_conversation


@_hook
def turn_error(trace: TurnTrace, stage: str, err: BaseException) -> None:
    """An error of the turn; recorded once where several layers report the same exception."""
    if all(known.error is not err for known in trace.errors):
        trace.errors.append(TurnError(stage, err))


@_hook
def decide_opened(trace: TurnTrace, url: str, state: str, questions: dict[str, dict]) -> DecideStep:
    """A request to the decision server is about to go out."""
    step = _added(trace, DecideStep(_next_id(trace), monotonic(), url, state, questions))
    _own_decides.set((*_own_decides.get(), step))
    return step


@_hook
def decide_answered(trace: TurnTrace, step: DecideStep | None, answers: dict[str, dict]) -> None:
    """The decision server's checked answers."""
    if step is not None:
        step.answers = answers
        _end(step)


@_hook
def bar(trace: TurnTrace, qid: str, value: float) -> None:
    """The bar the answer to `qid` was compared with."""
    newest_first = chain(reversed(_own_decides.get()), _decide_steps_newest_first(trace))
    if (step := next((step for step in newest_first if qid in step.questions), None)) is not None:
        step.bars.setdefault(qid, value)


def answer_step(
    role: str, model: str, url: str, system: str, user: str
) -> AbstractContextManager[AnswerStep | None]:
    """A request to the answer model, filled by llm_request and llm_response inside."""
    trace = _turn.get()
    if trace is None or trace.closed is not None:
        return _NOT_RECORDED
    began = perf_counter_ns()
    step = _added(trace, AnswerStep(_next_id(trace), monotonic(), role, model, url, system, user))
    _spent(trace, began)
    return _answering(step)


@contextmanager
def _answering(step: AnswerStep) -> Generator[AnswerStep]:
    token = _answer.set(step)
    try:
        yield step
    except Exception as err:
        step.error = err
        raise
    finally:
        _answer.reset(token)
        _end(step)


@_hook
def llm_request(trace: TurnTrace, body: dict[str, Any]) -> None:
    """The answer model's request body as sent."""
    if (step := _answer.get()) is not None:
        step.request = body


@_hook
def llm_response(trace: TurnTrace, data: Any) -> None:
    """The answer model's parsed answer."""
    if (step := _answer.get()) is not None:
        step.response = data


@_hook
def system_parts(trace: TurnTrace, api_prompt: str | None, extra_prompt: str | None) -> None:
    """The parts HA joins into the loop's system prompt, for splitting it at build time."""
    trace.api_prompt, trace.extra_prompt = api_prompt, extra_prompt


@_hook
def round_opened(trace: TurnTrace, call: dict[str, Any]) -> None:
    """A round of the loop is about to be requested."""
    trace.current_round = _added(trace, RoundStep(_next_id(trace), monotonic(), trace.round_role, call))


@_hook
def stream_token(trace: TurnTrace) -> None:
    """The round's stream brought reasoning, reply or a tool-call piece."""
    if (step := trace.current_round) is not None and step.first_token is None:
        step.first_token = monotonic()


@_hook
def stream_finish(trace: TurnTrace, reason: str) -> None:
    """Why the round's stream ended."""
    if (step := trace.current_round) is not None:
        step.finish = reason


@_hook
def stream_usage(trace: TurnTrace, usage: Any) -> None:
    """The round's token counts, as the server reports them."""
    if (step := trace.current_round) is not None:
        step.usage = usage


@_hook
def stream_end(trace: TurnTrace) -> None:
    """The round's stream ended."""
    if (step := trace.current_round) is not None:
        _end(step)


@_hook
def round_content(trace: TurnTrace, content: Any) -> None:
    """A content the chat log made of the round."""
    if (step := trace.current_round) is not None:
        step.contents.append(content)


@_hook
def round_failed(trace: TurnTrace, err: BaseException) -> None:
    """The round's request or stream failed."""
    if (step := trace.current_round) is not None:
        _fail(step, err)


@_hook
def tool_opened(trace: TurnTrace, call_id: str, name: str, args: dict[str, Any]) -> ToolStep:
    """A tool call of the newest round starts."""
    parent = trace.current_round.id if trace.current_round is not None else None
    return _added(trace, ToolStep(_next_id(trace), parent, monotonic(), call_id, name, args))


@_hook
def tool_returned(trace: TurnTrace, step: ToolStep | None, result: Any) -> None:
    """The tool's result."""
    if step is not None:
        step.result = result
        _end(step)


@_hook
def reply_check_opened(trace: TurnTrace) -> ReplyCheckStep:
    """The reply check starts."""
    return _added(trace, ReplyCheckStep(_next_id(trace), monotonic()))


@_hook
def reply_checked(
    trace: TurnTrace,
    step: ReplyCheckStep | None,
    *,
    language: str | None,
    reply: str,
    acted: bool,
    faults: Sequence[Any],
) -> None:
    """What the reply check found."""
    if step is not None:
        step.language, step.reply, step.acted, step.faults = language, reply, acted, faults
        _end(step)


@_hook
def retrying(trace: TurnTrace) -> None:
    """The reply check asks for the reply again: the newest check retried, the rounds from now are retries."""
    trace.round_role = "retry"
    checks = (step for step in reversed(trace.steps) if isinstance(step, ReplyCheckStep))
    if (check := next(checks, None)) is not None:
        check.retried = True


@_hook
def intent_opened(trace: TurnTrace, intent: str, slots: dict[str, Any], target: str | None) -> ActionStep:
    """An HA intent of the decided path starts."""
    return _added(trace, ActionStep(_next_id(trace), monotonic(), target, slots, intent=intent))


@_hook
def service_opened(trace: TurnTrace, domain: str, service: str, data: dict[str, Any]) -> ActionStep:
    """An HA action of the decided path starts."""
    step = ActionStep(_next_id(trace), monotonic(), data.get("entity_id"), data, domain=domain, service=service)
    return _added(trace, step)


@_hook
def action_returned(trace: TurnTrace, step: ActionStep | None, result: Any) -> None:
    """The intent's response, or the action's."""
    if step is not None:
        step.result = result
        _end(step)


@_hook
def step_failed(trace: TurnTrace, step: Step | None, err: BaseException) -> None:
    """The step failed with `err`."""
    if step is not None:
        _fail(step, err)


def _decide_steps_newest_first(trace: TurnTrace) -> Iterator[DecideStep]:
    return (step for step in reversed(trace.steps) if isinstance(step, DecideStep))


def _next_id(trace: TurnTrace) -> int:
    return len(trace.steps) + 1


def _added[S: Step](trace: TurnTrace, step: S) -> S:
    trace.steps.append(step)
    return step


def _end(step: Step) -> None:
    if step.end is None:
        step.end = monotonic()


def _fail(step: Step, err: BaseException) -> None:
    step.error = err
    _end(step)


def _spent(trace: TurnTrace, began: int) -> None:
    trace.hook_ns += perf_counter_ns() - began
    trace.hook_calls += 1
