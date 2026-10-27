"""Memory's LLM API: the memory tools, the store's standing prompt, and the household rules for decision agents."""

import asyncio

import probatio

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from ..decider.household import PENDING
from ..home import satellite_area
from ..tools.spec import READ_ONLY, ToolArgs, ToolError, ToolSpec, spec_tools
from .background import background_client
from .const import BRAIN_API_ID, BRAIN_API_NAME, LOGGER
from .rule_compiler import CompileError, async_compile_rule, home_context
from .store import Store

RECORD_FIELDS = ("text", "kind", "names", "triggers", "scopes", "effect", "examples")
NO_CONTEXT = {"room": None, "told": None}  # a hand-edited rule, or one told before a decision agent asked
# Compiling every rule at once can exceed what a server takes in flight (HTTP 429), losing a rule's compile.
MAX_PARALLEL_COMPILES = 4


class BrainAPI(llm.API):
    """The memory tools and the store's standing prompt, for every conversation agent that selects it."""

    def __init__(self, hass: HomeAssistant, store: Store, entry: ConfigEntry) -> None:
        """Serve the memory tools of `store`; `entry`'s Memory names the background model that compiles rules."""
        super().__init__(hass=hass, id=BRAIN_API_ID, name=BRAIN_API_NAME)
        self._store = store
        self._entry = entry
        self._compiling = asyncio.Lock()  # one compile of a new rule, however many turns ask at once

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """The standing prompt and the memory tools for this request."""
        return llm.APIInstance(
            api=self,
            api_prompt=await self._store.async_get_standing_context(),
            llm_context=llm_context,
            tools=spec_tools(MEMORY_TOOLS, self._store, self._store.async_record_failure),
        )

    async def async_get_compiled_rules(self) -> list[dict]:
        """The household's rules compiled for small decision models (rule_compiler.py).

        It holds one dict per rule in rules.md with the compiled fields: text, kind, names, triggers, scopes, effect,
        examples, context. A rule changed since the last call is compiled now by the text model; one that cannot be
        compiled has kind "pending".
        """
        records = await self._compiled_records()
        told = await self._store.async_read_rules_told()
        return [
            {**{field: record[field] for field in RECORD_FIELDS if field in record},
             "context": told.get(record["text"], NO_CONTEXT)}
            for record in records
        ]

    async def _compiled_records(self) -> list[dict]:
        """The records as stored, with the compile check's reason ("check") that only diagnostics show."""
        async with self._compiling:
            compiled, missing = await self._store.async_read_compiled_rules()
            if not missing:
                return compiled
            client = background_client(self.hass, self._entry)
            if client is None:  # Memory has no background model: no rule can be compiled
                return compiled + [{"text": rule, "kind": PENDING} for rule in missing]
            home = home_context(self.hass)
            slots = asyncio.Semaphore(MAX_PARALLEL_COMPILES)

            async def compile_one(rule: str) -> dict:
                async with slots:
                    return await async_compile_rule(client, home, rule)

            results = await asyncio.gather(*(compile_one(rule) for rule in missing), return_exceptions=True)
            for result in results:
                if isinstance(result, CompileError):
                    LOGGER.warning("could not compile a household rule: %s", result)
                elif isinstance(result, BaseException):
                    raise result
            new = [r for r in results if isinstance(r, dict)]
            if new:
                await self._store.async_save_compiled_rules(compiled + new)
            failed = [
                {"text": rule, "kind": PENDING}
                for rule, result in zip(missing, results, strict=True) if not isinstance(result, dict)
            ]
            return compiled + new + failed

    async def async_compile_ahead(self) -> None:
        """Compile changed rules right after they are written, so a decision agent's next turn does not wait.

        Only where a decision agent has asked for compiled rules before: other homes never pay the extra text
        model calls.
        """
        if await self._store.async_has_compiled_rules():
            await self._compiled_records()



async def search_brain(_hass: HomeAssistant, store: Store, args: ToolArgs, _context: llm.LLMContext) -> str:
    """The matching notes with snippets, or the notes there are when nothing matches."""
    results = await store.async_search(args["query"])
    if not results:
        notes = await store.async_list_notes()
        if notes:
            return "No matches. Available notes:\n" + "\n".join(f"- {note}" for note in notes)
        return "No results found."
    lines = ["Search results:"]
    for result in results:
        if result.get("linked_from"):
            lines.append(f"- {result['path']} (linked from {result['linked_from']})\n  {result['snippet']}")
        else:
            lines.append(f"- {result['path']} (score: {result['score']})\n  {result['snippet']}")
    return "\n".join(lines)


async def read_note(_hass: HomeAssistant, store: Store, args: ToolArgs, _context: llm.LLMContext) -> str:
    """The note's text; an unknown path is the model's mistake."""
    try:
        return await store.async_read_note(args["path"])
    except (ValueError, FileNotFoundError) as err:
        raise ToolError(str(err), learn=True) from err


async def add_memory(hass: HomeAssistant, store: Store, args: ToolArgs, context: llm.LLMContext) -> str:
    """Store the memory, with the room a rule was told in."""
    slug = await store.async_add_memory(
        args["text"], args.get("topic"), why=args.get("why", ""), told_in=satellite_area(hass, context.device_id)
    )
    return f"Saved to memories/{slug}.md"


async def update_memory(hass: HomeAssistant, store: Store, args: ToolArgs, context: llm.LLMContext) -> str:
    """Replace the topic; a refused update is the model's mistake."""
    try:
        slug = await store.async_update_memory(
            args["topic"], args["text"], why=args.get("why", ""), told_in=satellite_area(hass, context.device_id)
        )
    except ValueError as err:
        raise ToolError(str(err), learn=True) from err
    return f"Replaced memories/{slug}.md with the new text"


async def forget(_hass: HomeAssistant, store: Store, args: ToolArgs, _context: llm.LLMContext) -> str:
    """Delete the memories; a refused delete is the model's mistake."""
    try:
        return await store.async_forget(args["topic"], args.get("containing"), why=args.get("why", ""))
    except ValueError as err:
        raise ToolError(str(err), learn=True) from err


MEMORY_TOOLS: tuple[ToolSpec[Store], ...] = (
    ToolSpec(
        name="search_brain",
        description=(
            "Search stored notes and memories by keywords. Returns note paths with "
            "snippets. Use before answering questions about the household or "
            "previously remembered facts. NOT for live or historical device data - "
            "sensor readings, energy, statistics and history come from get_statistics "
            "and get_history."
        ),
        parameters=probatio.Schema({probatio.Required("query"): str}),
        run=search_brain,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="read_note",
        description="Read the full content of a note by path",
        parameters=probatio.Schema({probatio.Required("path"): str}),
        run=read_note,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="add_memory",
        description=(
            "Add a NEW memory. THIS is the tool for 'remember that ...' - the only "
            "way to store something new, including into a topic that already exists "
            "(writing a new rule into 'rules' is add_memory, not update_memory). "
            "Appends a timestamped entry to memories/<topic>.md; "
            "existing entries are never changed or removed. If the user corrects or "
            "removes something already remembered, "
            "call update_memory instead - do NOT call add_memory. Pick the topic from "
            "what the fact is about, for example 'bins', 'garden', 'wifi'. Topic 'rules' is "
            "special and narrow: it is read on every turn and is ONLY for how YOU "
            "should answer or behave, for example 'give times in 24-hour format', 'give "
            "temperatures in whole degrees', 'for energy use sensor.energy_1'. Something a HUMAN does is a "
            "normal fact even when it repeats: 'the bin goes out on Tuesdays' and "
            "'the hedge is cut in September' are topics 'bins' and 'garden', NOT "
            "rules. Test: if you would not answer any differently because of it, it "
            "is not a rule."
            " Always pass 'why': one short clause on what in the conversation made you store this. It goes into the "
            "git history so a human can later see what prompted it."
        ),
        parameters=probatio.Schema(
            {probatio.Required("text"): str, probatio.Optional("topic"): str, probatio.Optional("why"): str}
        ),
        run=add_memory,
    ),
    ToolSpec(
        name="update_memory",
        description=(
            "NOT for storing something new - use add_memory for that, even when the "
            "topic already exists. This REPLACES ALL stored memories for a topic with "
            "new text, so everything you do not repeat is lost. Use ONLY when an "
            "existing entry must change or go away. A call that keeps every stored "
            "entry and appends one is refused. One call fully handles the update - do not also call "
            "add_memory. Pass plain text, one item per line - no bullets, no timestamps."
            " Always pass 'why': one short clause on what in the conversation made you change it. It goes into the "
            "git history so a human can later see what prompted it."
        ),
        parameters=probatio.Schema(
            {probatio.Required("topic"): str, probatio.Required("text"): str, probatio.Optional("why"): str}
        ),
        run=update_memory,
    ),
    ToolSpec(
        name="forget",
        description=(
            "Delete stored memories. Deletes the whole topic file, or only the "
            "entries containing the optional 'containing' text. Use when the user "
            "asks to forget or delete something remembered."
            " Always pass 'why': one short clause on what in the conversation made you delete it. It goes into the "
            "git history so a human can later see what prompted it."
        ),
        parameters=probatio.Schema(
            {probatio.Required("topic"): str, probatio.Optional("containing"): str, probatio.Optional("why"): str}
        ),
        run=forget,
    ),
)
