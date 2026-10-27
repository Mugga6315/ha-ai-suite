"""The librarian: merges the raw memories into the curated wiki pages, once a night or when asked."""

import asyncio
import re

import probatio

from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TimeSelector,
)

from ..llm_client import ChatClient, LLMError
from .background import RunFailed, effort_options
from .const import (
    CONF_CONSOLIDATE_EFFORT,
    CONF_CONSOLIDATE_ENABLED,
    CONF_CONSOLIDATE_TIME,
    CONSOLIDATOR_GIT_EMAIL,
    CONSOLIDATOR_GIT_NAME,
    DEFAULT_CONSOLIDATE_TIME,
    DEFAULT_SELF_IMPROVE_EFFORT,
    LOGGER,
    SELF_IMPROVE_EFFORTS,
)
from .store import Store, strip_bullet_prefix, word_in

MAX_DELETE_LINES = 100
MAX_RULE_MOVES = 5

# The consolidator runs nightly and nothing waits on it, so it gets a long leash:
# a big triage prompt and a slow local model are both fine here. Only the voice
# path has to be snappy.
LLM_TIMEOUT_SECONDS = 600


def _fact_is_in_wiki(fact: str, wiki_text: str) -> bool:
    """Is this fact present in the wiki? Word containment, not substring.

    The consolidator rewrites as it merges (the same words in another order),
    so an exact-substring gate never matches and the raw bullet is stranded
    forever. Requiring every significant word still means a fact that was
    never merged cannot pass: its distinctive words are simply absent.
    """
    fact = strip_bullet_prefix(fact).lower()
    words = {w for w in re.findall(r"\w+", fact) if len(w) > 3}
    if not words:  # too short to have distinctive words - fall back to substring
        return fact in wiki_text
    # Whole words, not substrings: a word must not be satisfied by a longer
    # word that contains it on an unrelated page. Stricter means a clear is refused
    # rather than wrongly allowed, which is the safe direction for this gate.
    return all(word_in(w, wiki_text) for w in words)


_SYSTEM = "You are a JSON-producing consolidation agent. Return ONLY valid JSON, no markdown fences."


class Consolidator:
    """The librarian: merges the raw memories into the curated wiki, once a night or when asked."""

    def __init__(self, hass: HomeAssistant, store: Store, client: ChatClient, effort: str) -> None:
        """Consolidate `store` with the background model at the given thinking effort."""
        self._hass = hass
        self._store = store
        self._client = client
        self._effort = effort
        self._running = asyncio.Lock()

    async def async_run(self) -> str:
        """Run consolidation: read memories, call LLM, write wiki, clear memories.

        The summary of what changed; RunFailed when the run changed nothing because the model's plan was missing or
        unsafe.
        """
        if self._running.locked():
            LOGGER.info("Consolidator: already running, skipping")
            return "Consolidation already running."

        # The store lock is held for the whole run, the LLM call included (up to LLM_TIMEOUT_SECONDS): a voice
        # add_memory queues behind it
        async with self._running, self._store.async_locked():
            return await self._async_run_impl()

    async def _async_run_impl(self) -> str:
        """Inner implementation of async_run (caller holds store lock)."""
        memories = await self._store.async_read_all_memories()
        rules = await self._store.async_read_rules()
        if not memories and not rules:
            LOGGER.info("Consolidator: no memories to process")
            return "No memories to consolidate."

        wiki = await self._store.async_read_all_wiki()
        prompt = _prompt(memories, wiki, rules)

        try:
            plan = await self._client.complete_json(
                _SYSTEM, prompt, timeout=LLM_TIMEOUT_SECONDS, temperature=0.3,
                response_format={"type": "json_object"},
                # Thinking is worth it here: the triage reasons about failure patterns in the background, with
                # nobody waiting.
                **effort_options(self._effort),
            )
        except LLMError as err:
            raise RunFailed(f"the model gave no plan: {err}") from err

        wiki_updates = plan.get("wiki_updates", [])
        memories_to_clear = plan.get("memories_to_clear", [])
        rules_to_move = plan.get("rules_to_move", [])
        lint_findings = [
            str(f) for f in plan.get("lint_findings", []) if str(f).strip()
        ]

        if (
            not wiki_updates and not memories_to_clear and not rules_to_move
            and not lint_findings
        ):
            LOGGER.info("Consolidator: nothing to do")
            return "Nothing to consolidate."

        for item in wiki_updates:
            if "path" not in item or "content" not in item:
                raise RunFailed("malformed plan: a wiki_update without path or content")

        for item in memories_to_clear:
            if "path" not in item or "containing" not in item:
                raise RunFailed("malformed plan: a memories_to_clear item without path or containing")

        for item in rules_to_move:
            if "containing" not in item or "to_topic" not in item:
                raise RunFailed("malformed plan: a rules_to_move item without containing or to_topic")

        # rules.md is the one file a human maintains by hand and the only one read
        # on every turn. A confused model may not drain it in a single night.
        if len(rules_to_move) > MAX_RULE_MOVES:
            raise RunFailed(f"the plan moves {len(rules_to_move)} rules, at most {MAX_RULE_MOVES} are allowed")

        # Write wiki pages first so clears are validated against the wiki on disk
        written, cleared = [], []
        for update in wiki_updates:
            try:
                await self._store.async_write_note(update["path"], update["content"])
                written.append(update["path"])
            except ValueError as e:
                LOGGER.warning("Consolidator: %s, skipping", e)

        # A memory may only be cleared once its fact is actually in
        # the wiki. Checked against the wiki as it stands *after* this run's
        # writes, not against this run's writes alone - a fact merged on an
        # earlier night is still in the wiki, and gating on "written just now"
        # would leave those raw bullets in memories/ forever.
        # The snippet is usually copied straight out of the memory file, so it
        # carries the stored "- <timestamp> " prefix the wiki page does not -
        # strip it, or every clear is skipped for the wrong reason.
        wiki_now = await self._store.async_read_all_wiki()
        all_wiki_content = " ".join(wiki_now.values()).lower()
        skipped_clears = []
        valid_clears = []
        for item in memories_to_clear:
            if _fact_is_in_wiki(item["containing"], all_wiki_content):
                valid_clears.append(item)
            else:
                skipped_clears.append(item)
        memories_to_clear = valid_clears

        # Count matched bullets once per distinct path
        path_containings: dict[str, list[str]] = {}
        for item in memories_to_clear:
            path_containings.setdefault(item["path"], []).append(item["containing"])
        total_lines = sum(
            _count_matched_bullets(memories.get(path, ""), containings)
            for path, containings in path_containings.items()
        )
        if total_lines > MAX_DELETE_LINES:
            raise RunFailed(f"the plan clears {total_lines} lines, at most {MAX_DELETE_LINES} are allowed")

        for item in memories_to_clear:
            try:
                await self._store.async_clear_memory(item["path"], item["containing"])
                cleared.append(f"{item['path']} ({item['containing']})")
            except ValueError as e:
                LOGGER.warning("Consolidator: %s, skipping", e)

        moved = []
        for item in rules_to_move:
            try:
                slug = await self._store.async_move_rule(
                    item["containing"], item["to_topic"]
                )
                moved.append(f"{item['containing']} -> memories/{slug}.md")
            except ValueError as e:
                LOGGER.warning("Consolidator: %s, skipping", e)

        await self._store.async_append_log(
            _log_entry(written, cleared, lint_findings, skipped_clears, moved)
        )

        await self._store.async_commit_held(
            f"consolidate: {len(written)} wiki updates, {len(cleared)} memories cleared",
            name=CONSOLIDATOR_GIT_NAME,
            email=CONSOLIDATOR_GIT_EMAIL,
        )
        LOGGER.info(
            "Consolidator: wrote %d wiki pages, cleared %d memory entries, "
            "moved %d rules",
            len(written), len(cleared), len(moved),
        )
        return (
            f"Consolidated {len(written)} wiki pages, cleared {len(cleared)} "
            f"memory entries, moved {len(moved)} rules."
        )


def _prompt(memories: dict[str, str], wiki: dict[str, str], rules: str = "") -> str:
    """The instructions with the store's memories, wiki pages and rules."""
    parts = [_INSTRUCTIONS, "\n\n## Current memories:\n"]
    for path, content in sorted(memories.items()):
        parts.append(f"\n### {path}\n```\n{content}\n```")
    parts.append("\n\n## Current wiki pages:\n")
    for path, content in sorted(wiki.items()):
        parts.append(f"\n### {path}\n```\n{content}\n```")
    if rules.strip():
        parts.append(
            "\n\n## Current rules (memories/rules.md) - triage only:\n"
            f"\n```\n{rules}\n```"
        )
    return "".join(parts)


def _log_entry(
    written: list[str],
    cleared: list[str],
    lint_findings: list[str],
    skipped_clears: list[dict] | None = None,
    moved: list[str] | None = None,
) -> str:
    """One readable entry: what the run changed, what lint fixed, what was skipped."""
    lines = []
    if written:
        lines.append("Updated:")
        lines += [f"- {p}" for p in written]
    if cleared:
        lines.append("Cleared from memories:")
        lines += [f"- {c}" for c in cleared]
    if moved:
        lines.append("Moved out of rules (not a behaviour rule):")
        lines += [f"- {m}" for m in moved]
    if skipped_clears:
        lines.append("Skipped (fact not written to any wiki page):")
        lines += [
            f"- {s['path']} ({s['containing']})" for s in skipped_clears
        ]
    if lint_findings:
        lines.append("Lint:")
        lines += [f"- {f}" for f in lint_findings]
    return "\n".join(lines) or "No changes."


def _count_matched_bullets(text: str, containings: list[str]) -> int:
    count = 0
    for line in text.splitlines():
        if line.lstrip().startswith("- "):
            line_lower = line.lower()
            if any(c.lower() in line_lower for c in containings):
                count += 1
    return count


# --- librarian subentry interface (see features.py) ---------------------------
# The model is Memory's background model, so this subentry only carries the schedule and the effort.


def subentry_schema(data: dict) -> dict:
    """The librarian's settings: the nightly run, its time and its thinking effort."""
    return {
        probatio.Required(
            CONF_CONSOLIDATE_ENABLED, default=data.get(CONF_CONSOLIDATE_ENABLED, True)
        ): BooleanSelector(),
        probatio.Required(
            CONF_CONSOLIDATE_TIME,
            default=data.get(CONF_CONSOLIDATE_TIME, DEFAULT_CONSOLIDATE_TIME),
        ): TimeSelector(),
        probatio.Required(
            CONF_CONSOLIDATE_EFFORT,
            default=data.get(CONF_CONSOLIDATE_EFFORT, DEFAULT_SELF_IMPROVE_EFFORT),
        ): SelectSelector(SelectSelectorConfig(
            options=SELF_IMPROVE_EFFORTS, mode=SelectSelectorMode.DROPDOWN, translation_key="effort"
        )),
    }


async def async_validate(hass, data: dict) -> str | None:
    """Nothing to validate here — the LLM is configured and checked on the parent."""
    return None


_INSTRUCTIONS = """\
# Consolidation instructions

You keep the memory store of a household's voice assistant. Your job: organize
raw memories into curated wiki pages.

## Input
You receive all memory files (memories/*.md) and all existing wiki pages
(wiki/*.md).

## Task
1. Merge new memory bullets into the matching wiki/ page (create if missing).
2. Drop exact duplicates.
3. For every memory bullet you merged, emit a `memories_to_clear` item naming
   its file and a snippet identifying that one bullet. Merging is only half the
   job: a bullet merged but not cleared stays in memories/ **and** in the wiki,
   forever - nothing retries the clear on a later run. Clearing is safe, because
   every clear is checked against the wiki on disk first and refused if the fact
   is not actually there; a bullet you did not merge simply fails that check. A
   memory file whose last bullet is cleared is deleted for you.
4. Mark superseded facts (append "(superseded YYYY-MM-DD)" to the old line).
5. Write frontmatter on every wiki page you write: `title`, `tags`, and
   `load_when` - one short line saying when this page is worth reading, e.g.
   `load_when: questions about Energy 1, its meter readings or tariff`. The index
   shows it, so it is how the assistant decides which page answers a question.
   `tags` must be a YAML list in brackets - `tags: [household, tv]`. A
   bare `tags: household, tv` is a single tag containing a comma, an invalid
   tag name.
6. Link related pages with `[[name]]`, where name is the target page's filename
   without the `.md` (so `[[energy]]` points at `wiki/energy.md`). Add a link where
   one page genuinely refers to another - a meter page mentioning the tariff,
   an appliance mentioning its maintenance page. Only link pages that exist (in
   the wiki you were given or one you are writing this run); do not invent
   targets. Search follows these links, so a query that matches one page also
   surfaces the pages it points at.
7. Never touch CORE.md, INDEX.md or log.md. You cannot write memories/rules.md
   either - the only thing you may do with it is the triage pass below.

## Rules triage

`memories/rules.md` is shown to the assistant on **every single turn**, before it
has searched anything. That budget is small and it is the most expensive space in
the store, so only one kind of entry belongs there:

**A rule is an instruction about how the assistant should answer or behave.**
It is something the assistant itself can carry out while replying.

- "give times in 24-hour format"
- "for temperatures give whole degrees, no decimals"
- "for energy production use sensor.energy_1, not the template sensor"
- "when asked about energy, always call GetDateTime first"

**Everything else is a fact**, even when it sounds like a standing arrangement.
If a *human* is the one who has to act, or if it is something true about the
household rather than something the assistant should do, it is a fact and it
belongs in a normal topic where search will find it when it is relevant:

- "the bin goes out on Tuesdays" -> a human puts the bin out. Fact.
- "the hedge is cut in September" -> a human cuts it. Fact.
- "the boiler is serviced every October" -> fact.
- "the wifi password is banana123" -> fact.

The test, in one line: *if the assistant did nothing differently when replying,
would this entry be pointless?* If yes it is a rule. If the entry describes the
world rather than the reply, it is a fact.

The assistant files these itself while talking to the user, and it gets this
wrong: anything phrased as "always" or "every Tuesday" tends to land in rules.
Fixing that is your job, because nothing else ever reads rules.md.

For each entry in rules.md that is a fact and not a rule, emit a `rules_to_move`
item with a `containing` snippet that identifies exactly that one bullet, and a
`to_topic` naming the normal topic it belongs in (a short slug, for example `bins`,
`garden`, `heating`). The bullet is written to its new topic and read back before
it is removed from rules.md, so nothing is lost - but a snippet matching two
bullets is refused, so make the snippet specific. Move at most 5 per run. When
in doubt, leave it: a fact in rules.md costs a little context, a rule wrongly
moved out changes how the assistant behaves.

Do not use `rules_to_move` to delete anything, to reword an entry, or to move a
genuine rule to a "better" topic. Its only purpose is fact-out-of-rules.

## Lint
While merging, also check the wiki you were given and fix what is wrong:
- Two pages stating the same fact differently: keep the newer, mark the older
  superseded.
- A fact contradicted by a newer memory: correct it in place.
- Duplicated bullets inside one page: remove.
- A page that is now empty or has no real content: leave the file alone and
  report it instead of deleting.
Report every fix you made in `lint_findings`, one short sentence each. It is
written to log.md so a human can review what you changed unattended.

## Output format
Return JSON only, no markdown fences:
{
  "wiki_updates": [
    {"path": "wiki/energy.md", "content": "full file content including frontmatter"}
  ],
  "memories_to_clear": [
    {"path": "memories/inbox.md", "containing": "text snippet to identify the bullet"}
  ],
  "rules_to_move": [
    {"containing": "bins go out", "to_topic": "bins"}
  ],
  "lint_findings": [
    "wiki/energy.md: marked the 2025 tariff superseded by the June entry"
  ]
}

If nothing needs consolidating, return:
{"wiki_updates": [], "memories_to_clear": [], "rules_to_move": [], "lint_findings": []}
"""
