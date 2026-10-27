"""The markdown store: notes, memories, rules and the wiki, in a git repository."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from homeassistant.core import HomeAssistant

from ..languages import ascii_spelling
from .const import (
    CORE_CHARS,
    DEFAULT_GIT_EMAIL,
    DEFAULT_GIT_NAME,
    INDEX_CHARS,
    LOGGER,
    NOTE_CHARS,
    RULES_CHARS,
    SEARCH_RESULTS,
    SEARCH_SCORE_BODY,
    SEARCH_SCORE_FILENAME,
    SEARCH_SCORE_HEADING,
    SEARCH_SCORE_TAG,
)
from .rule_compiler import COMPILER_VERSION


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML-like frontmatter from markdown text."""
    body = text
    front = {}
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            raw = text[4:end]
            body = text[end + 5 :]
            for line in raw.splitlines():
                m = re.match(r"(\w+):\s*(.+)", line)
                if m:
                    front[m.group(1)] = m.group(2).strip()
    return front, body


# Store machinery, not knowledge: INDEX.md is generated, CONSOLIDATE.md is a copy
# of the consolidator's prompt that earlier versions wrote into the store. Both
# scored on searches and sent the model reading example paths out of the prompt.
_MACHINERY_FILES = {"INDEX.md", "CONSOLIDATE.md"}


def _is_hidden(rel: Path) -> bool:
    """Is this path inside a dot-folder? Those are tooling, never knowledge.

    A markdown editor may keep its own files there: Obsidian, for one, moves a
    deleted note to .trash/ instead of deleting it, and such a note must not come
    back in the INDEX or in search. So every dot-folder is out, not only .git.
    """
    return any(p.startswith(".") for p in rel.parts)


# Indexed and readable, but not searchable: a changelog contains every word that
# ever passed through the store, so it matches every query and buries real notes.
# failures.md is the same shape - it is the self-improver's input, not knowledge.
# The phrase that tells the analyzer a turn used this integration. It lives in
# the injected prompt below; both sides must read it from here or a wording
# change would silently stop every review.
PROMPT_MARKER = "search_brain finds notes"

_LOG_FILE = "log.md"
_RULES_FILE = "rules.md"
# A git call that takes longer hangs (a lock file, a stalled share): it is given up so the store lock is freed.
GIT_TIMEOUT_SECONDS = 30
_FAILURES_FILE = "failures.md"
# self_improving.md holds auto-learned rules; it is injected into every prompt
# like rules.md, so searching it would surface it against itself.
SELF_IMPROVING_FILE = "self_improving.md"
# not_a_defect.md is the human's veto over the self-improver: anything listed
# there was reviewed and judged fine, so the analyzer must not flag it again and
# no rule may be learned from it. Written by hand only - no agent writes here.
_NOT_A_DEFECT_FILE = "not_a_defect.md"
# rules.md compiled for decision agents (rule_compiler.py), next to it in memories/.
COMPILED_RULES_FILE = "rules.compiled.json"
_UNSEARCHABLE = {
    _LOG_FILE, _FAILURES_FILE, SELF_IMPROVING_FILE, _NOT_A_DEFECT_FILE,
}

# Seeded once, then owned by the human. The header explains the file to whoever
# opens it months later, because an empty file explains nothing.
_NOT_A_DEFECT_SEED = """---
title: not_a_defect
tags: [not_a_defect]
load_when: never - this file is for the self-improver, not for answering
---

Behaviour that was reviewed and judged correct. The per-turn review must not
flag anything listed here, and no rule may be learned from it. One bullet per
case, in plain words - say what the assistant did and why it is fine.

- Repeating a tool call the user explicitly asked to repeat ("check again") is
  correct, not a redundant call.
"""
# failures.md is never pruned. It is the debugging record of what the assistant
# got wrong, and a mistake stays interesting long after it was fixed - "when did
# this start" is only answerable if the entry survives. Handled entries are
# MARKED, not removed, and the marker is what keeps them out of the nightly
# prompt so the file can grow without the prompt growing with it.
_ACK_MARKER = "[ack "

_LINE_CRUFT = re.compile(
    r"^\s*(?:[-*]\s+)?(?:\d{4}-\d{2}-\d{2}[T ][\d:.]+(?:[+-]\d{2}:\d{2}|Z)?\s+)?"
)

# Notes are related by hand with [[wikilinks]]
# ([[target]] or [[target|alias]]). Search follows them one hop so a note the
# query missed can still surface. Capped so a hub note can't flood the results.
_WIKILINK = re.compile(r"\[\[([^\]|#\n]+)")
_MAX_LINKED = 5


def word_in(word: str, text: str) -> bool:
    """Whole-word match.

    Bare substrings scored every note holding a short query word inside a longer word - the query word has to
    stand alone.
    """
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def _clean_lines(text: str) -> list[str]:
    """Strip model-supplied bullets/timestamps; store adds its own format."""
    lines = [_LINE_CRUFT.sub("", line, count=1).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def strip_bullet_prefix(text: str) -> str:
    """Drop a leading '- ' and timestamp.

    Public: the consolidator compares the `containing` snippets an LLM copied out of a memory file, and those carry the
    stored prefix while the wiki page it was merged into does not.
    """
    return _LINE_CRUFT.sub("", text, count=1).strip()


def _clip(text: str, limit: int, what: str) -> str:
    """Truncate to a char budget and say so.

    A silent clip is worse than a short one: rules past the cutoff are invisible
    while looking present, so the model answers as if it had read them all and a
    human reading the same file cannot tell why it did not follow one. Naming
    the omission turns "the assistant ignores my rule" into a file to shorten.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... {len(text) - limit} more chars of {what} not shown]"


def _commit_message(subject: str, why: str = "") -> str:
    """The commit message: the subject and the model's stated reason.

    Commit subject plus the model's stated reason, so `git log` answers not
    only what the assistant stored but why it thought it should.
    """
    why = " ".join(why.split())
    return f"{subject}\n\nwhy: {why}" if why else subject


def _norm_text(text: str) -> str:
    """Letters and digits only, for comparing a stored bullet against a re-sent one.

    A model that rewrites a whole topic re-emits the text it was shown, and it
    drifts: JSON escaping puts a backslash before each quote, whitespace
    collapses, punctuation wanders. None of that means the entry changed.

    Scope, so nobody trusts it further than it goes: it normalises escaping,
    whitespace and punctuation, NOT transliteration - non-ASCII is dropped, so
    a word with an umlaut does not match its re-emitted transliteration. It also
    collapses short umlaut words hard, so an unrelated bullet could in principle
    be mistaken for a duplicate and silently skipped. Both need a model that
    transliterates its own stored text.
    """
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _tags_of(front: dict[str, str]) -> str:
    """The raw `tags` value without YAML list brackets.

    Written as a flow sequence (`tags: [a, b]`), read here and in older files as
    a bare `a, b` - both parse the same once the brackets are off.
    """
    return front.get("tags", "").strip("[] ")


def _subcommand(args: tuple[str, ...]) -> str:
    """The git command named in args, after any "-c name=value" options."""
    while args[:1] == ("-c",):
        args = args[2:]
    return args[0] if args else ""


def _make_frontmatter(title: str, tags: list[str], load_when: str = "") -> str:
    lines = ["---"]
    lines.append(f"title: {title}")
    # A YAML list, not a bare string: `tags: a, b` is ONE tag containing a comma
    # and a space, an invalid tag name. `[a, b]` is two tags.
    lines.append(f"tags: [{', '.join(tags)}]")
    if load_when:
        lines.append(f"load_when: {load_when}")
    lines.append(f"created: {datetime.now(UTC).isoformat()}")
    lines.append("---\n")
    return "\n".join(lines)


class Store:
    """The markdown store: notes, memories, rules and the wiki, in a git repository. Every write is a commit."""

    def __init__(
        self,
        hass: HomeAssistant,
        root: Path,
        git_name: str = DEFAULT_GIT_NAME,
        git_email: str = DEFAULT_GIT_EMAIL,
        core_chars: int = CORE_CHARS,
        rules_chars: int = RULES_CHARS,
        index_chars: int = INDEX_CHARS,
        note_chars: int = NOTE_CHARS,
    ) -> None:
        """A store at `root`; git authorship and the prompt limits default to Second Brain's."""
        self._hass = hass
        # Resolved by the caller, off the event loop: resolving follows links, and the store may be on a share.
        self._root = Path(root)
        self._memories = self._root / "memories"
        self._git_name = git_name
        self._git_email = git_email
        self._core_chars = core_chars
        self._rules_chars = rules_chars
        self._index_chars = index_chars
        self._note_chars = note_chars
        self._lock = asyncio.Lock()
        # failures.md alone: written by voice turns and the review, which must not wait for a nightly job's store
        # lock, and rewritten by the learner when it marks entries as handled.
        self._failures_lock = asyncio.Lock()
        self._git_available = False  # looked up in async_setup
        self._rules_listeners: list[Callable[[], None]] = []

    def listen_rules_changed(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call listener after each write to rules.md made through the store. Returns the unsubscribe."""
        self._rules_listeners.append(listener)
        return lambda: self._rules_listeners.remove(listener)

    def _rules_changed(self) -> None:
        for listener in list(self._rules_listeners):
            listener()

    async def async_setup(self) -> None:
        """Create store directory structure and seed files."""

        def _setup():
            self._root.mkdir(parents=True, exist_ok=True)
            for d in ("memories", "wiki"):
                (self._root / d).mkdir(exist_ok=True)
            core = self._root / "CORE.md"
            if not core.exists():
                core.write_text("# Second Brain\n\nYour persistent knowledge store.\n")
            index = self._root / "INDEX.md"
            if not index.exists():
                index.write_text("# INDEX\n\n")
            accepted = self._memories / _NOT_A_DEFECT_FILE
            if not accepted.exists():
                accepted.write_text(_NOT_A_DEFECT_SEED, encoding="utf-8")
            self._git_available = shutil.which("git") is not None
            self._init_git()

        await self._hass.async_add_executor_job(_setup)
        LOGGER.info("Second Brain store ready at %s", self._root)

    @property
    def root(self) -> Path:
        """Store root. Public so callers stop reaching for _root."""
        return self._root

    @property
    def compiled_rules_path(self) -> Path:
        """rules.md as compiled for decision agents."""
        return self._memories / COMPILED_RULES_FILE

    async def async_exists(self) -> bool:
        """Has the store been initialized (CORE.md present)?"""
        return await self._hass.async_add_executor_job(
            lambda: (self._root / "CORE.md").exists()
        )

    def _init_git(self) -> None:
        if not self._git_available:
            LOGGER.warning("git not found; AI writes will not be committed")
            return
        if (self._root / ".git").exists():
            return
        self._git("init")
        self._git("add", "-A")
        self._git(
            "-c", f"user.name={self._git_name}", "-c", f"user.email={self._git_email}",
            "commit", "--allow-empty", "-m", "chore: initialize second brain store",
        )

    def _git(self, *args: str) -> bool:
        """One git command in the store; False, with a warning, when it fails or hangs."""
        command = ["git", "-c", f"safe.directory={self._root}", *args]
        try:
            result = subprocess.run(
                command, cwd=self._root, capture_output=True, check=False, timeout=GIT_TIMEOUT_SECONDS
            )
        except (subprocess.TimeoutExpired, OSError) as err:
            LOGGER.warning("git %s failed: %s", _subcommand(args), err)
            return False
        if result.returncode != 0:
            LOGGER.warning(
                "git %s failed (%s): %s", _subcommand(args), result.returncode,
                result.stderr.decode(errors="replace").strip(),
            )
            return False
        return True

    def _search_sync(self, query: str) -> list[dict]:
        """Scored scan of all .md files, then one hop along [[wikilinks]]."""
        # O(filesxlines) scan; inverted index only if store outgrows ~1k files
        words = [w for w in query.lower().split() if w]
        results: list[dict[str, Any]] = []
        by_key: dict[str, str] = {}  # stem/title (lowercased) -> rel path, for [[links]]
        bodies: dict[str, str] = {}  # rel path -> body, for snippets + link extraction
        for fpath in self._root.rglob("*.md"):
            if fpath.name in _MACHINERY_FILES or fpath.name in _UNSEARCHABLE:
                continue
            if _is_hidden(fpath.relative_to(self._root)):
                continue
            try:
                text = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as err:
                LOGGER.debug("search skips %s: %s", fpath, err)
                continue
            front, body = _parse_frontmatter(text)
            rel = str(fpath.relative_to(self._root))
            stem = fpath.stem.lower()
            # A [[link]] must resolve to the curated page, not the raw memory
            # dump, and rglob order is arbitrary - so wiki/ wins the key and
            # anything else only fills a gap.
            in_wiki = rel.startswith("wiki/")
            if in_wiki or stem not in by_key:
                by_key[stem] = rel
            if title := front.get("title", "").strip().lower():
                if in_wiki or title not in by_key:
                    by_key[title] = rel
            bodies[rel] = body
            tags = [t.strip().lower() for t in _tags_of(front).split(",") if t.strip()]
            headings = [h.lower() for h in re.findall(r"^#{1,6}\s+(.+)", body, re.MULTILINE)]
            body_lower = body.lower()
            score = 0
            for w in words:
                if word_in(w, stem):
                    score += SEARCH_SCORE_FILENAME
                if any(word_in(w, t) for t in tags):
                    score += SEARCH_SCORE_TAG
                if any(word_in(w, h) for h in headings):
                    score += SEARCH_SCORE_HEADING
                if word_in(w, body_lower):
                    score += SEARCH_SCORE_BODY
            if score > 0:
                results.append(
                    {
                        "path": rel,
                        "score": score,
                        "snippet": _snippet(body, _anchor(words, body_lower), 200),
                    }
                )
        results.sort(key=lambda r: -r["score"])
        results = results[:SEARCH_RESULTS]
        return results + _linked_notes(results, by_key, bodies)

    async def async_search(self, query: str) -> list[dict]:
        """The notes matching the query, best first, with snippets."""
        return await self._hass.async_add_executor_job(self._search_sync, query)

    def _list_notes_sync(self, limit: int = 50) -> list[str]:
        """Return relative paths of all user-facing .md files, capped at limit.

        The unsearchable files are the self-improver's own bookkeeping, not
        notes: offering them here had the assistant reading its failure log back
        to the user as an available note.
        """
        notes: list[str] = []
        total = 0
        for fpath in sorted(self._root.rglob("*.md")):
            if fpath.name in _MACHINERY_FILES or fpath.name in _UNSEARCHABLE:
                continue
            if _is_hidden(fpath.relative_to(self._root)):
                continue
            total += 1
            if len(notes) < limit:
                notes.append(str(fpath.relative_to(self._root)))
        if total > limit:
            notes.append(f"…and {total - limit} more")
        return notes

    async def async_list_notes(self) -> list[str]:
        """The notes there are, for a search that found nothing."""
        return await self._hass.async_add_executor_job(self._list_notes_sync)

    async def async_read_note(self, path: str) -> str:
        """Read a note with path traversal protection."""

        def _read():
            # resolve()/exists()/is_file() all hit the filesystem, so they belong
            # in the executor with the read - a stalled network share must not
            # block the event loop.
            target = (self._root / path).resolve()
            if not target.is_relative_to(self._root):
                raise ValueError(f"Path traversal denied: {path}")
            if not target.is_file():
                raise FileNotFoundError(f"Note not found: {path}")
            return _clip(
                target.read_text(encoding="utf-8"), self._note_chars, path
            )

        return await self._hass.async_add_executor_job(_read)

    def _read_dir_sync(self, subdir: str, exclude: set[str] | None = None) -> dict[str, str]:
        """Read all .md files in a subdirectory. Returns {relative_path: content}."""
        result: dict[str, str] = {}
        base = self._root / subdir
        if not base.is_dir():
            return result
        for fpath in base.rglob("*.md"):
            if exclude and fpath.name in exclude:
                continue
            try:
                result[str(fpath.relative_to(self._root))] = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as err:
                LOGGER.debug("skipping unreadable %s: %s", fpath, err)
        return result

    async def async_read_all_memories(self) -> dict[str, str]:
        """Read all memory files except the injected-not-searched ones."""
        return await self._hass.async_add_executor_job(
            self._read_dir_sync,
            "memories",
            {_RULES_FILE, SELF_IMPROVING_FILE, _NOT_A_DEFECT_FILE},
        )

    async def async_read_all_wiki(self) -> dict[str, str]:
        """Read all wiki pages."""
        return await self._hass.async_add_executor_job(
            self._read_dir_sync, "wiki"
        )

    def _write_note_sync(self, path: str, content: str) -> None:
        """Write a wiki note with path traversal protection. No commit."""
        wiki_root = (self._root / "wiki").resolve()
        target = (self._root / path).resolve()
        if not target.is_relative_to(wiki_root):
            raise ValueError(f"Path outside wiki/ denied: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def async_write_note(self, path: str, content: str) -> None:
        """Write a note (consolidator use). No commit — caller commits in batch."""
        await self._hass.async_add_executor_job(self._write_note_sync, path, content)

    def _clear_memory_lines_sync(self, path: str, containing: str) -> int:
        """Remove bullets containing a substring from a memory file. Returns lines removed."""
        mem_root = (self._memories).resolve()
        target = (self._root / path).resolve()
        if not target.is_relative_to(mem_root):
            raise ValueError(f"Path outside memories/ denied: {path}")
        if target.name == _RULES_FILE:
            # No agent reads rules.md as its own file, so none may delete from
            # it: it is the one file a human maintains by hand.
            LOGGER.warning("an agent tried to clear rules.md - ignored")
            return 0
        if not target.exists():
            return 0
        text = target.read_text(encoding="utf-8")
        needle = containing.lower()
        original_lines = text.splitlines(keepends=True)
        kept = [
            line for line in original_lines
            if not (line.lstrip().startswith("- ") and needle in line.lower())
        ]
        removed = len(original_lines) - len(kept)
        remaining_bullets = sum(1 for line in kept if line.lstrip().startswith("- "))
        if remaining_bullets:
            target.write_text("".join(kept), encoding="utf-8")
        else:
            target.unlink()
        return removed

    async def async_clear_memory(self, path: str, containing: str) -> int:
        """Clear matching bullets from a memory file (consolidator use).

        No lock, no commit, no index: the caller holds async_locked() for the whole run and commits once at the
        end. Called bare from a service handler it would race a voice add_memory; use async_forget for that path
        instead.
        """
        return await self._hass.async_add_executor_job(
            self._clear_memory_lines_sync, path, containing
        )

    def _append_log_sync(self, entry: str) -> None:
        """Append one dated entry to log.md (consolidator use). No commit."""
        path = self._root / "log.md"
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        if not path.exists():
            # The index routes on load_when, so the log has to say what it is for.
            path.write_text(
                _make_frontmatter(
                    "log",
                    ["log"],
                    load_when=(
                        "the user asks what changed recently, what was updated "
                        "or cleaned up, or what the consolidator did"
                    ),
                ),
                encoding="utf-8",
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {stamp} UTC\n{entry.rstrip()}\n")

    async def async_append_log(self, entry: str) -> None:
        """Record what changed, so a session can start from the delta."""
        await self._hass.async_add_executor_job(self._append_log_sync, entry)

    def _record_failure_sync(self, tool: str, detail: str) -> None:
        path = self._root / _FAILURES_FILE
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        detail = " ".join(detail.split())
        if not path.exists():
            path.write_text(
                _make_frontmatter(
                    "failures",
                    ["failures"],
                    load_when=(
                        "the user asks why an answer failed, or what the "
                        "assistant has been getting wrong"
                    ),
                ),
                encoding="utf-8",
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(f"- {stamp} {tool}: {detail}\n")

    async def async_record_failure(self, tool: str, detail: str) -> None:
        """Note a tool call that could not be answered.

        This is the raw signal for self-improvement: the nightly learner reads
        it and may compile a repeated failure into a rule. No commit - a failed voice turn must not cost a git
        write; the next real write picks the file up.
        """
        try:
            async with self._failures_lock:
                await self._hass.async_add_executor_job(self._record_failure_sync, tool, detail)
        except OSError:  # never let bookkeeping break a tool's own error path
            LOGGER.exception("could not record failure for %s", tool)

    async def async_read_failures(self, only_open: bool = True) -> str:
        """failures.md as text. The learner's input.

        `only_open` hides entries already marked as handled, so a fixed mistake
        is not learned from twice and the prompt stays the size of the backlog
        rather than the size of the history.
        """
        def _read():
            path = self._root / _FAILURES_FILE
            if not path.exists():
                return ""
            text = path.read_text(encoding="utf-8")
            if not only_open:
                return text
            kept = [
                line
                for line in text.splitlines(keepends=True)
                if not (line.startswith("- ") and _ACK_MARKER in line)
            ]
            return "".join(kept)

        return await self._hass.async_add_executor_job(_read)

    async def async_acknowledge_failure(self, containing: str, note: str) -> int:
        """Mark the failures matching `containing` as handled. Never deletes.

        Returns how many entries were marked. The entry keeps its original text
        and gains `[ack <date>: <note>]`, so the debugging record stays complete
        and a human can see both the mistake and what was done about it.
        """
        def _ack() -> int:
            path = self._root / _FAILURES_FILE
            if not path.exists():
                return 0
            needle = containing.lower()
            stamp = datetime.now(UTC).strftime("%Y-%m-%d")
            clean_note = " ".join(note.split()) or "handled"
            out, marked = [], 0
            for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
                if (
                    line.startswith("- ")
                    and _ACK_MARKER not in line
                    and needle in line.lower()
                ):
                    out.append(f"{line.rstrip()} {_ACK_MARKER}{stamp}: {clean_note}]\n")
                    marked += 1
                else:
                    out.append(line)
            if marked:
                path.write_text("".join(out), encoding="utf-8")
            return marked

        async with self._failures_lock:
            return await self._hass.async_add_executor_job(_ack)

    async def async_add_rule(self, text: str) -> bool:
        """Append one rule to rules.md. Returns False if it is already there.

        Append-only on purpose: rules.md is the human's file. Auto-learned rules
        go to self_improving.md via async_add_self_improving_rule - nothing here
        may edit or delete a rule a human wrote.

        No index regeneration, no commit: the caller holds async_locked() and does both once at the end of its
        run. A direct caller would write a rule that is absent from INDEX.md until the next nightly pass.
        """
        def _add() -> bool:
            text_clean = " ".join(_clean_lines(text))
            if not text_clean:
                return False
            path = self._memories / _RULES_FILE
            if path.exists() and _has_bullet(path, text_clean):
                return False
            self._append_bullet("rules", text_clean)
            return True

        added = await self._hass.async_add_executor_job(_add)
        if added:
            self._rules_changed()
        return added

    async def async_add_self_improving_rule(self, text: str) -> bool:
        """Append one auto-learned rule to self_improving.md. False if duplicate.

        Unlike rules.md (human-owned, append-only), this file is fully the
        learner's: it may add here and also retract via async_clear_memory, so a
        rule that later proves harmful can be removed by the loop itself.
        No commit - the learner holds async_locked() and commits at the end.
        """
        def _add() -> bool:
            text_clean = " ".join(_clean_lines(text))
            if not text_clean:
                return False
            path = self._memories / SELF_IMPROVING_FILE
            if path.exists() and _has_bullet(path, text_clean):
                return False
            self._append_bullet("self_improving", text_clean)
            return True

        return await self._hass.async_add_executor_job(_add)

    async def async_read_not_a_defect(self) -> str:
        """not_a_defect.md as text: the human's list of accepted behaviour."""
        def _read():
            path = self._memories / _NOT_A_DEFECT_FILE
            return path.read_text(encoding="utf-8") if path.exists() else ""

        return await self._hass.async_add_executor_job(_read)

    async def async_read_self_improving(self) -> str:
        """self_improving.md as text, so the learner can retract stale rules."""
        def _read():
            path = self._memories / SELF_IMPROVING_FILE
            return path.read_text(encoding="utf-8") if path.exists() else ""

        return await self._hass.async_add_executor_job(_read)

    @asynccontextmanager
    async def async_locked(self) -> AsyncGenerator[None]:
        """Acquire the store lock for the caller's full scope; commit inside it with async_commit_held.

        The consolidator holds this lock across its entire write phase so that
        a concurrent async_add_memory cannot add a fact that the clear step then
        deletes because it shares containing text with an already-read memory.
        """
        async with self._lock:
            yield

    async def async_commit_held(self, message: str, name: str | None = None, email: str | None = None) -> None:
        """Regenerate the index and commit, with an optional author, for a caller inside async_locked()."""
        if not self._lock.locked():
            raise RuntimeError("async_commit_held needs the store lock (async_locked)")
        await self._hass.async_add_executor_job(self._generate_index_sync)
        await self._hass.async_add_executor_job(
            self._commit_sync, message, name, email
        )

    async def async_add_memory(
        self, text: str, topic: str | None = None, why: str = "", told_in: str | None = None
    ) -> str:
        """Append a timestamped bullet to a memory file. told_in: the room a rule was told in, if known."""
        async with self._lock:  # global lock, household-scale; per-slug locks if throughput matters
            slug = await self._hass.async_add_executor_job(
                self._write_noting_told_rules, topic, told_in, self._add_memory_sync, text, topic, why
            )
        if slug == "rules":
            self._rules_changed()
        return slug

    def _append_bullet(self, slug: str, text: str) -> None:
        """Append one bullet to memories/<slug>.md, creating it with frontmatter."""
        path = self._memories / f"{slug}.md"
        bullet = f"- {_bullet_prefix(slug)}{text}\n"
        if path.exists():
            with path.open("a", encoding="utf-8") as f:
                f.write(bullet)
        else:
            path.write_text(
                _make_frontmatter(slug, [slug]) + bullet, encoding="utf-8"
            )

    def _topic_slug_sync(self, topic: str | None) -> str:
        """The topic's file name; a file an earlier version named without the topic's umlauts stays its file."""
        if not topic:
            return "inbox"
        slug, old = _slugify(topic), _old_slug(topic)
        if old != slug and not (self._memories / f"{slug}.md").exists() and (self._memories / f"{old}.md").exists():
            return old
        return slug

    def _add_memory_sync(self, text: str, topic: str | None, why: str = "") -> str:
        slug = self._topic_slug_sync(topic)
        text = " ".join(_clean_lines(text))
        path = self._memories / f"{slug}.md"
        # A model that emits the same tool call twice in one turn is common. A
        # duplicate is also fatal to rules triage: every snippet identifying it
        # matches two bullets, so the move is refused as ambiguous.
        if path.exists() and _has_bullet(path, text):
            LOGGER.debug("add_memory: memories/%s.md already has this bullet", slug)
            return slug
        self._append_bullet(slug, text)
        self._generate_index_sync()
        self._commit_sync(_commit_message(f"add_memory({slug}): {text}", why))
        return slug

    async def async_read_rules(self) -> str:
        """rules.md as text. Read-only input for the consolidator's triage pass."""
        def _read():
            path = self._memories / _RULES_FILE
            return path.read_text(encoding="utf-8") if path.exists() else ""

        return await self._hass.async_add_executor_job(_read)

    async def async_move_rule(self, containing: str, to_topic: str) -> str:
        """Move one bullet out of rules.md into a normal memory topic.

        rules.md is injected into every conversation, so a household fact filed
        there costs prompt weight on every turn and tells the assistant to do
        something it cannot do. The consolidator triages those out - but it may
        only *move* them: the bullet is written to its new home and read back
        before it is removed from rules.md, so a failed write cannot lose a rule.

        No index regeneration, no commit: see async_add_rule.
        """
        result = await self._hass.async_add_executor_job(
            self._move_rule_sync, containing, to_topic
        )
        self._rules_changed()
        return result

    def _move_rule_sync(self, containing: str, to_topic: str) -> str:
        rules = self._memories / _RULES_FILE
        if not rules.exists():
            raise ValueError("memories/rules.md does not exist")
        slug = self._topic_slug_sync(to_topic)
        if slug == "rules":
            raise ValueError("to_topic must not be 'rules'")
        text = rules.read_text(encoding="utf-8")
        needle = containing.lower()
        matches = [
            line
            for line in text.splitlines(keepends=True)
            if line.lstrip().startswith("- ") and needle in line.lower()
        ]
        if not matches:
            raise ValueError(f"No rule contains {containing!r}")
        if len(matches) > 1:
            raise ValueError(
                f"{len(matches)} rules contain {containing!r} - be more specific"
            )
        body = _LINE_CRUFT.sub("", matches[0], count=1).strip()

        # Write first, read back, only then remove. Reverse that order and a
        # rejected write silently deletes a rule.
        self._append_bullet(slug, body)
        landed = (self._memories / f"{slug}.md").read_text(encoding="utf-8")
        if body not in landed:
            raise ValueError(f"Refused: {body!r} did not land in memories/{slug}.md")

        out, removed = [], False
        for line in text.splitlines(keepends=True):
            if not removed and line.lstrip().startswith("- ") and needle in line.lower():
                removed = True
                continue
            out.append(line)
        # Keep rules.md even when the last bullet leaves: an absent file reads as
        # "no rules were ever set", a present empty one as "they were triaged".
        rules.write_text("".join(out), encoding="utf-8")
        return slug

    async def async_update_memory(
        self, topic: str, text: str, why: str = "", told_in: str | None = None
    ) -> str:
        """Replace all memories for a topic with new text. told_in: the room a rule was told in, if known."""
        async with self._lock:
            slug = await self._hass.async_add_executor_job(
                self._write_noting_told_rules, topic, told_in, self._update_sync, topic, text, why
            )
        if slug == "rules":
            self._rules_changed()
        return slug

    def _update_sync(self, topic: str, text: str, why: str = "") -> str:
        slug = self._topic_slug_sync(topic)
        path = self._memories / f"{slug}.md"
        lines = _clean_lines(text)
        if slug == "rules":
            self._refuse_learned_rules(lines, path)
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if slug == "rules" and len(old) > self._rules_chars:
                raise ValueError(
                    "rules.md is longer than what was shown to you, so replacing "
                    "it would delete rules you cannot see. Edit the file directly "
                    "or use forget with 'containing'."
                )
            old_texts = [
                _LINE_CRUFT.sub("", line, count=1).strip()
                for line in old.splitlines()
                if line.lstrip().startswith("- ")
            ]
            old_bullets = len(old_texts)
            # Every stored entry still present and at least one added is an
            # append wearing a replace costume. Sending it through here re-emits
            # the whole topic as one tool argument, which can be cut off
            # mid-string and kill the turn, and re-escapes the text it echoes
            # back, mangling lines that were fine.
            new_norm = {_norm_text(line) for line in lines}
            if (
                old_bullets
                and len(lines) > old_bullets
                and all(_norm_text(t) in new_norm for t in old_texts)
            ):
                raise ValueError(
                    f"This only adds to memories/{slug}.md - nothing existing "
                    "changed. Use add_memory(text, topic) for a new fact; "
                    "update_memory is for changing or removing what is stored."
                )
            # bullet-count heuristic, no content diffing. A replace that
            # halves a topic is the model forgetting to write back entries it was
            # shown, which loses them.
            if old_bullets and len(lines) * 2 < old_bullets:
                raise ValueError(
                    f"Refused: this would cut memories/{slug}.md from {old_bullets} "
                    f"entries to {len(lines)}. Repeat ALL entries you want to keep, "
                    "or use forget to delete specific ones."
                )
        prefix = _bullet_prefix(slug)
        bullets = "".join(f"- {prefix}{line}\n" for line in lines)
        path.write_text(_make_frontmatter(slug, [slug]) + bullets, encoding="utf-8")
        self._generate_index_sync()
        self._commit_sync(_commit_message(f"update({slug}): {text[:60]}", why))
        return slug

    def _refuse_learned_rules(self, lines: list[str], rules: Path) -> None:
        """Learned rules belong to self_improving.md, never to the human's rules.md.

        Told to remember one rule, a model may write rules.md with the learned
        rules copied in; compiled for the decider, such a copy can make it hand
        off a whole kind of command. Copies already in rules.md pass: refusing
        them too would block every rewrite of such a store.
        """
        learned = self._memories / SELF_IMPROVING_FILE
        if not learned.exists():
            return
        copied = [
            line for line in lines
            if _has_bullet(learned, line)
            and not (rules.exists() and _has_bullet(rules, line))
        ]
        if copied:
            raise ValueError(
                "Refused: these are auto-learned rules, they stay out of "
                "'rules' - leave them out: " + "; ".join(copied)
            )

    async def async_forget(
        self, topic: str, containing: str | None = None, why: str = ""
    ) -> str:
        """Delete a topic's memories, or only entries containing a substring."""
        async with self._lock:
            result = await self._hass.async_add_executor_job(
                self._forget_sync, topic, containing, why
            )
        if topic and _slugify(topic) == "rules":
            self._rules_changed()
        return result

    def _forget_sync(self, topic: str, containing: str | None, why: str = "") -> str:
        slug = self._topic_slug_sync(topic)
        path = self._memories / f"{slug}.md"
        if not path.exists():
            return f"No memories stored for topic '{slug}'."
        if slug == "rules" and not containing:
            raise ValueError(
                "Refused: deleting all rules at once. Pass 'containing' with text "
                "from the single rule to delete."
            )
        if containing:
            text = path.read_text(encoding="utf-8")
            needle = containing.lower()
            kept = [
                line
                for line in text.splitlines(keepends=True)
                if not (line.lstrip().startswith("- ") and needle in line.lower())
            ]
            remaining_bullets = sum(1 for line in kept if line.lstrip().startswith("- "))
            if remaining_bullets:
                path.write_text("".join(kept), encoding="utf-8")
                result = f"Deleted matching entries from memories/{slug}.md."
            else:
                path.unlink()
                result = f"Deleted memories/{slug}.md (no entries left)."
        else:
            path.unlink()
            result = f"Deleted memories/{slug}.md."
        self._generate_index_sync()
        self._commit_sync(_commit_message(f"forget({slug}): {containing or 'all'}", why))
        return result

    def _generate_index_sync(self) -> None:
        lines = ["# INDEX\n\n"]
        for fpath in sorted(self._root.rglob("*.md")):
            if fpath.name in _MACHINERY_FILES:
                continue
            if _is_hidden(fpath.relative_to(self._root)):
                continue
            try:
                text = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as err:
                LOGGER.debug("index skips %s: %s", fpath, err)
                continue
            front, _ = _parse_frontmatter(text)
            rel = str(fpath.relative_to(self._root))
            title = front.get("title", fpath.stem)
            tags = _tags_of(front)
            mtime = datetime.fromtimestamp(
                fpath.stat().st_mtime, tz=UTC
            ).strftime("%Y-%m-%d")
            entry = f"- **{title}** `{rel}` tags: {tags} modified: {mtime}"
            # An index is a routing table, not an inventory: the consolidator
            # writes `load_when:` so the model can tell which page answers what.
            if load_when := front.get("load_when"):
                entry += f"\n  load when: {load_when}"
            lines.append(entry + "\n")
        (self._root / "INDEX.md").write_text("".join(lines), encoding="utf-8")

    def _commit_sync(self, message: str, name: str | None = None, email: str | None = None) -> None:
        if not self._git_available:
            return
        git_name = name or self._git_name
        git_email = email or self._git_email
        if self._git("add", "-A"):
            self._git(
                "-c", f"user.name={git_name}", "-c", f"user.email={git_email}",
                "commit", "--author", f"{git_name} <{git_email}>", "-m", message,
            )

    def _rule_bullets_sync(self) -> list[str]:
        rules = self._memories / _RULES_FILE
        if not rules.exists():
            return []
        body = _parse_frontmatter(rules.read_text(encoding="utf-8"))[1]
        return [strip_bullet_prefix(line) for line in body.splitlines() if line.lstrip().startswith("- ")]

    def _write_noting_told_rules(self, topic: str | None, told_in: str | None, write, *args) -> str:
        """Run a memory write; for 'rules', note where and when each new rule was told (diagnostics only)."""
        if (_slugify(topic) if topic else "inbox") != "rules":
            return write(*args)
        before = set(self._rule_bullets_sync())
        slug = write(*args)
        self._note_told_sync([rule for rule in self._rule_bullets_sync() if rule not in before], told_in)
        return slug

    def _note_told_sync(self, rules: list[str], room: str | None) -> None:
        """Kept in rules.compiled.json, so only where a decision agent asked for compiled rules before."""
        path = self.compiled_rules_path
        if not rules or not path.exists():
            return
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            told = stored.setdefault("told", {})
        except (ValueError, AttributeError):
            return  # the next compile rewrites the unreadable file
        when = datetime.now(UTC).isoformat(timespec="seconds")
        for rule in rules:
            told[rule] = {"room": room, "told": when}
        path.write_text(json.dumps(stored, ensure_ascii=False, indent=1), encoding="utf-8")

    async def async_read_rules_told(self) -> dict[str, dict]:
        """Where and when each rule in rules.md was told, by rule text; hand-edited rules have none."""
        return await self._hass.async_add_executor_job(
            self._rules_told_sync, self.compiled_rules_path
        )

    def _rules_told_sync(self, path: Path) -> dict[str, dict]:
        """The told notes of the rules still in rules.md."""
        try:
            told = json.loads(path.read_text(encoding="utf-8")).get("told", {})
        except (OSError, ValueError, AttributeError):
            return {}
        return {rule: told[rule] for rule in self._rule_bullets_sync() if rule in told}

    async def async_read_compiled_rules(self) -> tuple[list[dict], list[str]]:
        """The compiled rules (rule_compiler) for the rules in rules.md now, and the rules not compiled yet.

        A rule edited by hand is a new rule.
        """

        def _read():
            bullets = self._rule_bullets_sync()
            path = self.compiled_rules_path
            known = {}
            if path.exists():
                try:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    if stored.get("version") == COMPILER_VERSION:
                        known = {r["text"]: r for r in stored["rules"]}
                    else:
                        LOGGER.info("%s is from another compiler version, compiling the rules again",
                                    COMPILED_RULES_FILE)
                except (ValueError, KeyError, TypeError, AttributeError):
                    LOGGER.warning("%s is unreadable, compiling the rules again", COMPILED_RULES_FILE)
            return [known[b] for b in bullets if b in known], [b for b in bullets if b not in known]

        return await self._hass.async_add_executor_job(_read)

    async def async_save_compiled_rules(self, compiled: list[dict]) -> None:
        """Store the compiled rules; derived from rules.md, so not committed on its own."""

        def _write():
            path = self.compiled_rules_path
            stored: dict[str, Any] = {"version": COMPILER_VERSION, "rules": compiled}
            if told := self._rules_told_sync(path):
                stored["told"] = told
            path.write_text(json.dumps(stored, ensure_ascii=False, indent=1), encoding="utf-8")

        await self._hass.async_add_executor_job(_write)

    async def async_has_compiled_rules(self) -> bool:
        """True once rules were compiled here, which a decision agent asking for them causes."""
        return await self._hass.async_add_executor_job(
            (self.compiled_rules_path).exists
        )

    async def async_get_standing_context(self) -> str:
        """Return CORE.md + INDEX summary for the system prompt."""

        def _read():
            parts = []
            core = self._root / "CORE.md"
            if core.exists():
                parts.append(
                    _clip(core.read_text(encoding="utf-8"), self._core_chars, "CORE.md")
                )
            rules = self._memories / _RULES_FILE
            if rules.exists():
                parts.append(
                    "## Active rules - always follow these when answering:\n"
                    + _clip(
                        # Body only: the frontmatter is the store's bookkeeping,
                        # and every header line would cost prompt budget on
                        # every single turn.
                        _parse_frontmatter(rules.read_text(encoding="utf-8"))[1],
                        self._rules_chars,
                        "rules.md - shorten rules.md to see them",
                    )
                )
            learned = self._memories / "self_improving.md"
            if learned.exists():
                parts.append(
                    "## Learned rules (auto, from past turns) - follow these too:\n"
                    + _clip(
                        _parse_frontmatter(learned.read_text(encoding="utf-8"))[1],
                        self._rules_chars,
                        "self_improving.md",
                    )
                )
            idx = self._root / "INDEX.md"
            if idx.exists():
                parts.append(
                    _clip(idx.read_text(encoding="utf-8"), self._index_chars, "INDEX.md")
                )
            parts.append(
                f"Memory tools: {PROMPT_MARKER}; read_note reads one; "
                "add_memory ADDS a new fact; update_memory REPLACES a topic's stored "
                "facts (use for corrections); forget DELETES memories. "
                "Pick exactly one write tool per request - add_memory for new facts, "
                "update_memory for changes, forget for deletions. "
                "Topic 'rules' is only for instructions about HOW YOU answer or "
                "behave - it is the one topic always active without searching. "
                "A recurring thing a human does ('bin out on Tuesdays') is a "
                "normal fact under its own topic, not a rule. "
                "IMPORTANT: when the user corrects your behavior or contradicts a "
                "stored rule or fact - even without being asked to remember it - "
                "persist the "
                "correction in the same turn, then answer. A correction applied only "
                "in your answer is forgotten in the next conversation. "
                "update_memory REPLACES the whole topic: when updating 'rules', "
                "write back every rule from the 'always follow these' section "
                "(never the auto-learned ones), with the corrected rule changed."
            )
            return "\n\n".join(parts)

        return await self._hass.async_add_executor_job(_read)


def _anchor(words: list[str], body_lower: str) -> str:
    """The first query word present in the body.

    Anchoring on words[0] unconditionally showed the head of the file whenever
    the first word was absent: a query whose first word the note lacks pointed
    at the frontmatter instead of the hit.
    """
    return next((w for w in words if word_in(w, body_lower)), words[0] if words else "")


def _snippet(text: str, query: str, width: int = 200) -> str:
    idx = text.lower().find(query)
    if idx == -1:
        return text[:width]
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(query) + width // 2)
    snip = text[start:end].replace("\n", " ")
    if start > 0:
        snip = "..." + snip
    if end < len(text):
        snip = snip + "..."
    return snip


def _linked_notes(
    results: list[dict], by_key: dict[str, str], bodies: dict[str, str]
) -> list[dict]:
    """Notes reached by following [[wikilinks]] out of the matched notes.

    One hop only, deduped against the matches, capped at _MAX_LINKED. Each is
    marked with the note it was linked from and gets a short preview, so the
    model can tell a curated link from a keyword hit and read_note it if useful.
    """
    seen = {r["path"] for r in results}
    linked: list[dict] = []
    for r in results:
        for target in _WIKILINK.findall(bodies.get(r["path"], "")):
            key = target.strip().lower()
            path = by_key.get(key) or by_key.get(_slugify(target)) or by_key.get(_old_slug(target))
            if not path or path in seen:
                continue
            seen.add(path)
            preview = " ".join(bodies.get(path, "").split())[:160]
            linked.append(
                {"path": path, "score": 0, "linked_from": r["path"], "snippet": preview}
            )
            if len(linked) >= _MAX_LINKED:
                return linked
    return linked


def _has_bullet(path: Path, text: str) -> bool:
    """Is this text already stored as a bullet in this file?"""
    norm = _norm_text(text)
    return any(
        line.lstrip().startswith("- ")
        and _norm_text(strip_bullet_prefix(line)) == norm
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _bullet_prefix(slug: str) -> str:
    """Timestamp prefix for memory bullets; injected rule files get none (git has dates)."""
    if slug in ("rules", "self_improving"):
        return ""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M ")


def _slugify(text: str) -> str:
    """A topic's file name: its words in ASCII letters (languages.ascii_spelling), joined by "-"."""
    return _slug(ascii_spelling(text.lower().strip()))


def _old_slug(text: str) -> str:
    """The file name earlier versions gave a topic: every letter outside ASCII dropped ("Müll" became "mll")."""
    return _slug(text.lower().strip())


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    # A topic of pure punctuation stripped to "" and wrote memories/.md - a
    # dotfile, hidden from the store listing.
    return text[:60].strip("-") or "inbox"
