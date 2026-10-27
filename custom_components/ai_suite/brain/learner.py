"""The self-improver's learning pass: failures.md -> self_improving.md.

The per-turn analyzer writes what went wrong or could be shorter; this reads
that backlog and turns a *repeated* pattern into a rule the assistant follows
on every turn. It is the second half of the self-improvement feature, kept
apart from the librarian: the librarian curates knowledge (wiki, rules.md
hygiene), this one changes behaviour.

Guardrails, in order of how much they matter:
  - only a pattern seen twice or more earns a rule,
  - nothing listed in not_a_defect.md may ever become one (the human's veto),
  - at most MAX_ADDS rules a run, at most MAX_REMOVES retractions,
  - rules land in self_improving.md, never in the human-owned rules.md.
"""
import asyncio

from homeassistant.core import HomeAssistant

from ..llm_client import ChatClient, LLMError
from .background import RunFailed, effort_options
from .const import LEARNER_GIT_EMAIL, LEARNER_GIT_NAME, LOGGER
from .store import SELF_IMPROVING_FILE, Store

MAX_ADDS = 3
MAX_REMOVES = 5
LLM_TIMEOUT_SECONDS = 300

_SYSTEM = (
    "You turn a smart-home assistant's recorded mistakes into rules it reads "
    "before it acts. Return ONLY valid JSON, no markdown fences."
)

_PROMPT = """Below is the assistant's open failure backlog: turns that misfired, and turns
that were answered correctly but took more tool calls than they needed
(`analyzer:detour`). Nobody reads this file during a conversation - it exists so
you can turn a repeated problem into a rule the assistant *does* read.

Only act on a pattern, never on a one-off. Two or more entries of the same shape,
with a fix you can state in one sentence, earn a rule. Every rule must carry an
`evidence` list: two or more snippets copied VERBATIM from two DIFFERENT entries
below. A rule whose evidence does not match the text below is dropped, so do not
paraphrase, and do not pad the list with entries of another shape to reach two -
a one-off simply does not earn a rule yet.

- `get_statistics entity_id='sensor.energy_today_1' -> No long-term
  statistics` three times, and the store knows the real meter is
  `sensor.energy_1` -> rule: "for energy production use sensor.energy_1, the
  template sensor has no statistics".
- two `analyzer:detour` entries about the same question shape -> rule naming the
  short path: "to answer current energy production, call GetLiveContext for
  'Energy 1' only".

A rule that saves a round trip is worth as much as one that prevents a wrong
answer: it makes the assistant faster, and a faster assistant is a better one.

Do not write a rule that restates a tool description, one you cannot support
with entries below, or one that is really a fact (facts belong in the wiki).

**Retract auto-rules that went wrong.** self_improving.md is yours to prune: if a
current rule is contradicted by the backlog or by the accepted list, emit a
`rules_to_remove` item with a `containing` snippet identifying that one bullet.
Never touch rules.md - it is human-owned.

**Acknowledge what you handled.** For every entry you turned into a rule, emit a
`failures_acknowledged` item with a `containing` snippet and a short `note`. The
entry is marked, never deleted - it stays as the debugging record, it just drops
out of the next run's backlog. Never acknowledge an entry you did nothing about.

**At most {max_adds} rules to add and {max_removes} to remove per run.** A run that asks for more
changes nothing at all. With more patterns than that, pick the {max_adds} with the most
evidence and acknowledge only their entries: the rest stay open for the next run.

Return JSON only, no markdown fences:
{
  "rules_to_add": [
    {"text": "for energy production use sensor.energy_1, the template sensor has no statistics",
     "evidence": ["entity_id='sensor.energy_today_1'", "No long-term statistics"]}
  ],
  "rules_to_remove": [
    {"containing": "always call GetLiveContext before every answer"}
  ],
  "failures_acknowledged": [
    {"containing": "sensor.energy_today_1", "note": "rule added: use sensor.energy_1"}
  ]
}
If nothing repeats, return {"rules_to_add": [], "rules_to_remove": [], "failures_acknowledged": []}."""


class RuleLearner:
    """The nightly pass of self-improvement: repeated failures become rules, wrong rules are retracted."""

    def __init__(self, hass: HomeAssistant, store: Store, client: ChatClient, effort: str) -> None:
        """Learn into `store` with the background model at the given thinking effort."""
        self._hass = hass
        self._store = store
        self._client = client
        self._effort = effort
        self._running = asyncio.Lock()

    async def async_run(self) -> str:
        """Read the failure backlog, learn rules, retract stale ones.

        The summary of what changed; RunFailed when the run changed nothing because the model's answer was missing or
        over the caps.
        """
        if self._running.locked():
            LOGGER.info("Learner: already running, skipping")
            return "Learning already running."
        async with self._running, self._store.async_locked():
            return await self._run_impl()

    async def _run_impl(self) -> str:
        failures = await self._store.async_read_failures()
        if not failures.strip():
            return "No open failures to learn from."
        learned = await self._store.async_read_self_improving()
        accepted = await self._store.async_read_not_a_defect()

        try:
            plan = await self._client.complete_json(
                _SYSTEM, self._build_prompt(failures, learned, accepted), timeout=LLM_TIMEOUT_SECONDS,
                temperature=0.3, response_format={"type": "json_object"}, **effort_options(self._effort),
            )
        except LLMError as err:
            raise RunFailed(f"the model gave no plan: {err}") from err

        to_add, unsupported = _supported_rules(
            plan.get("rules_to_add", []), failures
        )
        to_remove = [
            str(r.get("containing", "")).strip()
            for r in plan.get("rules_to_remove", [])
            if str(r.get("containing", "")).strip()
        ]
        acknowledged = [
            f for f in plan.get("failures_acknowledged", [])
            if str(f.get("containing", "")).strip()
        ]
        if not to_add and not to_remove and not acknowledged:
            return "Nothing to learn." if not unsupported else (
                f"Dropped {unsupported} rule(s) without two matching failures."
            )

        # Caps are the whole safety story here: the rules land in every prompt,
        # so a confused model must not be able to rewrite the assistant in one
        # night. Over the cap, nothing is applied at all.
        if len(to_add) > MAX_ADDS:
            raise RunFailed(f"the model asked to add {len(to_add)} rules, at most {MAX_ADDS} are allowed")
        if len(to_remove) > MAX_REMOVES:
            raise RunFailed(f"the model asked to remove {len(to_remove)} rules, at most {MAX_REMOVES} are allowed")

        added = []
        for rule in to_add:
            if await self._store.async_add_self_improving_rule(rule):
                added.append(rule)
            else:
                LOGGER.debug("Learner: rule already present, skipping: %s", rule)

        removed = [
            containing for containing in to_remove
            if await self._store.async_clear_memory(f"memories/{SELF_IMPROVING_FILE}", containing)
        ]

        # Acked last: an entry is only handled once its rule actually landed.
        acked = 0
        for item in acknowledged:
            acked += await self._store.async_acknowledge_failure(
                str(item["containing"]), str(item.get("note", ""))
            )

        await self._store.async_append_log(self._log_entry(added, removed, acked))
        await self._store.async_commit_held(
            f"learn: {len(added)} rules added, {len(removed)} retracted",
            name=LEARNER_GIT_NAME,
            email=LEARNER_GIT_EMAIL,
        )
        LOGGER.info(
            "Learner: added %d rules, retracted %d, acknowledged %d failures",
            len(added), len(removed), acked,
        )
        summary = (
            f"Learned {len(added)} rules, retracted {len(removed)}, "
            f"acknowledged {acked} failures."
        )
        if unsupported:
            summary += f" Dropped {unsupported} rule(s) without two matching failures."
        return summary

    def _build_prompt(self, failures: str, learned: str, accepted: str) -> str:
        # replace, not format: the prompt's JSON examples are full of braces.
        rules = _PROMPT.replace("{max_adds}", str(MAX_ADDS)).replace("{max_removes}", str(MAX_REMOVES))
        parts = [rules, f"\n\n## Open failures (failures.md):\n\n```\n{failures}\n```"]
        if accepted.strip():
            parts.append(
                "\n\n## Reviewed and accepted by the user (not_a_defect.md) - "
                "NEVER turn any of these into a rule, and retract any current "
                f"rule that contradicts them:\n\n```\n{accepted}\n```"
            )
        if learned.strip():
            parts.append(
                "\n\n## Current auto-learned rules (self_improving.md):\n"
                f"\n```\n{learned}\n```"
            )
        return "".join(parts)

    def _log_entry(self, added: list[str], removed: list[str], acked: int) -> str:
        lines = []
        if added:
            lines.append("Learned from failures (added to self_improving.md):")
            lines += [f"- {a}" for a in added]
        if removed:
            lines.append("Retracted from self_improving.md:")
            lines += [f"- {r}" for r in removed]
        if acked:
            lines.append(f"Marked {acked} failure(s) in failures.md as handled.")
        return "\n".join(lines)


def _supported_rules(items: list, failures: str) -> tuple[list[str], int]:
    """Keep the rules whose evidence is really in the backlog.

    The "two or more entries" bar is what stops one odd turn from becoming a
    standing rule, and a prompt cannot enforce it: asked for a pattern, a
    model will present a single entry as one, and a detour seen once becomes
    a rule. So each rule cites snippets, and they have to match.
    """
    entries = [line.lower() for line in failures.splitlines() if line.startswith("- ")]
    kept, dropped = [], 0
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        quotes = {
            q.strip().lower()
            for q in item.get("evidence", [])
            if isinstance(q, str) and len(q.strip()) >= 8
        }
        # Two quotes from the SAME entry are still one occurrence, so count
        # the entries the evidence lands in, not the quotes that matched.
        hit_entries = {
            i for i, entry in enumerate(entries) if any(q in entry for q in quotes)
        }
        if len(hit_entries) < 2:
            LOGGER.info(
                "Learner: dropping rule backed by %d entr(ies): %s",
                len(hit_entries), text,
            )
            dropped += 1
            continue
        kept.append(text)
    return kept, dropped
