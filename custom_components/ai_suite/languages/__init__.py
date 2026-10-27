"""The languages AI Suite understands: one module each, all filled the same way (words.py).

A request is matched against the words of every language at once, as people mix them (a German request with an
English title). A new language is one module like de.py and its entry in LANGUAGES. No HA imports.
"""

from collections.abc import Callable, Iterable
import re
import unicodedata

from . import de, en
from .words import Language

LANGUAGES: dict[str, Language] = {"de": de.LANGUAGE, "en": en.LANGUAGE}
LANGUAGE_NAMES = {code: language.name for code, language in LANGUAGES.items()}
# For a language without a module: its replies' sentences and decimal point.
FALLBACK = LANGUAGES["en"]


def alternatives(part: Callable[[Language], str]) -> str:
    """The part of every language as one group; a group that never matches when no language has the part."""
    found = [f"(?:{text})" for language in LANGUAGES.values() if (text := part(language))]
    return f"(?:{'|'.join(found)})" if found else "(?!)"


def pattern(part: Callable[[Language], str]) -> re.Pattern[str]:
    """The part of every language as one case-insensitive pattern."""
    return re.compile(alternatives(part), re.IGNORECASE)


def merged(part: Callable[[Language], Iterable[str]]) -> frozenset[str]:
    """The words of every language."""
    return frozenset(word for language in LANGUAGES.values() for word in part(language))


def language_of_tag(tag: str) -> Language:
    """The language of an HA language tag ("de-DE"); FALLBACK for one without a module."""
    return LANGUAGES.get(tag.split("-", 1)[0].lower(), FALLBACK)


def ascii_spelling(text: str) -> str:
    """Lowercase text in ASCII letters: every language's transliteration first, then accents dropped."""
    for language in LANGUAGES.values():
        for letter, spelling in language.transliteration.items():
            text = text.replace(letter, spelling)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
