"""Pure decision logic: candidates and their labels, slot parsing, speech. No Home Assistant imports."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
import re

from ..languages import LANGUAGES, alternatives, language_of_tag, merged
from ..languages.words import Language

NO_AREA_LABEL = "no room"  # a device label's area part for a device without one


@dataclass(slots=True)
class Candidate:
    """A device as the decider sees it: id, name, domain, area, what it measures, aliases."""
    entity_id: str
    name: str
    domain: str
    area: str | None
    measures: str | None = None  # device class and unit: what a sensor reads, so a question by quantity can match it
    aliases: tuple[str, ...] = ()  # HA entity aliases (often the German names)
    # A media player that plays now; not part of what the device is (a follow-up still finds it when it stopped).
    playing: bool = field(default=False, compare=False)


@dataclass(slots=True, frozen=True)
class RoomGroup:
    """All devices of one domain in one area: HA's area + domain slots."""

    domain: str
    area: str

    @property
    def label(self) -> str:
        """The group's name in a reply."""
        return f"all {self.domain} devices in {self.area}"


Target = Candidate | RoomGroup


STATE_HEAD = "Smart home voice assistant."


def state_text(text: str, previous: str | None = None) -> str:
    """The `state` sent to the decision model. `previous` = the last exchange of this conversation."""
    head = STATE_HEAD
    if previous:
        head += f" Previous turn: {previous}."
    return f'{head} The user said: "{text}"'


def with_rules(state: str, rules: str | None) -> str:
    """The state with the household's rules from Second Brain after its head.

    Only for a question where they help: the media type, which a rule may set for a name. In every question they
    break rooms and reply languages: an alias rule that gives a room another name pulls a request for another room to
    that room's lights.
    """
    if not rules:
        return state
    return state.replace(STATE_HEAD, f"{STATE_HEAD} Household rules: {' '.join(rules.split())}", 1)


def previous_turn(text: str, answer: str) -> str:
    """The previous exchange, as the state of a follow-up's questions."""
    return f'the user said "{text}", the assistant answered "{answer.strip()}"'


# Most candidates to hand the model in one pick; above this the server needs a two-round pick.
MAX_ONE_ROUND = 16

# Words that carry no target information; without them every "the"/"die" would match everything.
STOPWORDS = merged(lambda language: language.stopwords)


def tokens(text: str) -> list[str]:
    """Lower-case words without stopwords, repeats kept (BM25 counts them)."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


def words(text: str) -> set[str]:
    """The text's lowercase words without stopwords."""
    return set(tokens(text))


def known_text(c: Candidate, area_aliases: dict[str, list[str]]) -> str:
    """Everything a request may call the candidate: name, area, area aliases, entity aliases."""
    return " ".join([c.name, c.area or "", *area_aliases.get(c.area or "", []), *c.aliases])


def narrow(
    cands: list[Candidate], text: str, area_aliases: dict[str, list[str]], limit: int = MAX_ONE_ROUND
) -> list[Candidate]:
    """The candidates sharing the most words with the request.

    Candidates sharing the most words with the request (name, area, area aliases, entity aliases),
    normally at most `limit`. No shared word at all = no signal: all candidates come back unchanged.
    """
    asked = words(text)

    def overlap(c: Candidate) -> int:
        return len(asked & words(known_text(c, area_aliases)))

    scored = sorted(((overlap(c), c) for c in cands), key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return cands
    # Keep whole tie groups: cutting inside one could drop the right entity (then >limit -> two-round pick).
    cutoff = max(scored[min(limit, len(scored)) - 1][0], 1)
    return [c for score, c in scored if score >= cutoff]


def measures(attributes: dict) -> str | None:
    """What a device measures (device class and unit), or None."""
    parts = [attributes.get("device_class"), attributes.get("unit_of_measurement")]
    return " ".join(str(x) for x in parts if x) or None


def labelled(cands: list[Candidate]) -> dict[str, Candidate]:
    """Unique option labels "Name (domain, measures, area)" -> candidate, in candidate order.

    A device without an area says so: an area is not mandatory in HA, and without the word the decider cannot tell it
    from a label that leaves the area out.
    """
    out: dict[str, Candidate] = {}
    for c in cands:
        detail = ", ".join(x for x in (c.domain, c.measures, c.area or NO_AREA_LABEL) if x)
        base = f"{c.name} ({detail})"
        key, n = base, 1
        while key in out:
            n += 1
            key = f"{base} #{n}"
        out[key] = c
    return out


_NUM = r"(\d{1,3}(?:[.,]\d+)?)"
_PERCENT = alternatives(lambda language: language.reads.percent)


def parse_brightness(text: str) -> int | None:
    """A percent from the text, or None."""
    m = re.search(rf"{_NUM}\s*(?:%|{_PERCENT})", text, re.IGNORECASE)
    if not m:
        return None
    value = round(float(m.group(1).replace(",", ".")))
    return value if 0 <= value <= 100 else None


def _alternatives(numbers: Iterable[str]) -> str:
    return "|".join(sorted(numbers, key=len, reverse=True))  # longest first: "achtzehn" before "acht"


def _compound(language: Language) -> tuple[re.Pattern[str], dict[str, int]] | None:
    """The language's compound number words as a pattern, with the value of each word in them."""
    numbers = language.numbers
    if not numbers.compound:
        return None
    units = {**numbers.units, numbers.one: 1}  # "one" counts only inside a compound: alone it is an article
    found = numbers.compound.format(units=_alternatives(units), tens=_alternatives(numbers.tens))
    return re.compile(found, re.IGNORECASE), {**units, **numbers.tens}


_COMPOUNDS = [compound for language in LANGUAGES.values() if (compound := _compound(language))]
_SINGLE = {
    word: value
    for language in LANGUAGES.values()
    for word, value in {**language.numbers.units, **language.numbers.teens, **language.numbers.tens}.items()
}
_SINGLE_WORD = re.compile(rf"\b({_alternatives(_SINGLE)})\b", re.IGNORECASE)
_DEGREES = alternatives(lambda language: language.reads.degrees)


def _compound_value(match: re.Match[str], values: dict[str, int]) -> str:
    return str(values[match["unit"].lower()] + values[match["tens"].lower()])


def spelled_out(text: str) -> str:
    """The text with number words written as digits, so the digit parsers read them."""
    for compound, values in _COMPOUNDS:
        text = compound.sub(partial(_compound_value, values=values), text)
    return _SINGLE_WORD.sub(lambda m: str(_SINGLE[m[1].lower()]), text)


def parse_temperature(text: str) -> float | None:
    """A temperature from the text, or None."""
    # Only called once the intent is climate_set, so a bare number is the temperature.
    m = re.search(rf"{_NUM}\s*(?:°|{_DEGREES})", text, re.IGNORECASE) or re.search(_NUM, text)
    return float(m.group(1).replace(",", ".")) if m else None


def format_value(state: str, unit: str | None, language: str) -> str:
    """The state as HA reports it, with the language's decimal separator.

    The state as HA reports it (only the decimal separator follows the language); the answer LLM
    decides how to say it.
    """
    if language_of_tag(language).replies.decimal_comma:
        state = state.replace(".", ",")
    return f"{state} {unit}" if unit else state


def located(c: Candidate) -> str:
    """The device's name with its room, for a result note: the answer step says whose value it gives.

    Without the room a single device's reading may be worded as the whole home's.
    """
    return f"{c.name} (in {c.area})" if c.area else c.name


def read_answer(
    c: Candidate, state: str, attributes: dict, language: str, temperature_unit: str, media_current: bool = True
) -> str:
    """Spoken answer for a read.

    A thermostat's state is its mode and a weather entity's its condition, so both also speak their temperatures; a
    player also says what it plays, unless its media is not current (`media_current`: an idle player keeps its last
    title). A stale title is worded as playing now, even when the note says it played last.
    """
    current, target = attributes.get("current_temperature"), attributes.get("temperature")
    # Labelled: unlabelled, the answer LLM takes an album named like one of its songs for the song.
    labels = language_of_tag(language).replies.media_labels
    playing = ", ".join(f'{label} "{attributes[key]}"' for key, label in labels.items() if attributes.get(key))
    if c.domain == "media_player" and playing and media_current:
        return speech("read_media", language, name=located(c), state=state, playing=playing)
    if c.domain == "weather" and (outside := attributes.get("temperature")) is not None:
        outside_text = format_value(str(outside), attributes.get("temperature_unit", temperature_unit), language)
        return speech("read_weather", language, name=located(c), condition=state, temperature=outside_text)
    if c.domain == "climate" and current is not None and target is not None:
        return speech(
            "read_climate",
            language,
            name=located(c),
            current=format_value(str(current), temperature_unit, language),
            target=format_value(str(target), temperature_unit, language),
        )
    value = format_value(state, attributes.get("unit_of_measurement"), language)
    return speech("read_state", language, name=located(c), value=value)


def speech(intent: str, language: str, **slots: object) -> str:
    """A reply sentence for the intent in the language, its slots filled."""
    shown = {key: f"{value:g}" if isinstance(value, float) else value for key, value in slots.items()}
    return language_of_tag(language).replies.speech[intent].format(**shown)


def lights_in(devices: dict[str, Candidate], room: str) -> int:
    """How many lights the room holds."""
    return sum(1 for d in devices.values() if d.domain == "light" and d.area == room)


def only_of_its_kind(device: Candidate, devices: dict[str, Candidate]) -> bool:
    """Whether no other device has the same domain and measures."""
    return sum((d.domain, d.measures) == (device.domain, device.measures) for d in devices.values()) == 1
