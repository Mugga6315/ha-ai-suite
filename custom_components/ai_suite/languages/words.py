"""The words and grammar every language module gives, by the part of AI Suite that reads them. No HA imports.

A `str` field is a regular expression matched as it is: it carries its own word boundaries, except where its comment
says it is a plain word list that the reading module bounds itself. An empty field means the language has no such
words or grammar.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Days:
    """The days a request names (decider/days.py)."""

    weekdays: tuple[str, str, str, str, str, str, str]  # Monday first; plain word lists
    today: str
    tomorrow: str
    day_after_tomorrow: str
    next_days: str  # the next few days, this week
    weekend: str
    next_week: str  # the calendar week after this one
    yesterday: str
    day_before_yesterday: str


@dataclass(frozen=True, slots=True)
class Time:
    """Words that put a question in the past or ask for a time (decider/calendar.py, decider/command.py)."""

    past: str  # last, yesterday, earlier, ago
    past_verbs: str  # the past of be and do
    when: str
    time_question: str  # what time, how long
    window: str  # a window's word before a count and a time unit (the last 3 days); plain word list
    time_units: str  # plain word list
    clock: str  # the word after an hour that makes it a clock time; plain word list


@dataclass(frozen=True, slots=True)
class Reads:
    """What a question about a device's state asks (decider/command.py, decider/decision.py)."""

    temperature: str
    humidity: str
    battery: str
    degrees: str  # the unit after a number; plain word list
    percent: str  # the unit after a number; plain word list
    many: str  # all, every
    outdoors: str
    so_far: str  # an amount so far: today, so far
    now: str  # a rate now: now, currently
    follow_up_opener: str  # a word a follow-up may start with before its "and" (ok); bounded after only
    follow_up_and: str  # the word a follow-up starts with (and, what about); bounded after only
    there: str


@dataclass(frozen=True, slots=True)
class Requests:
    """Words in a request that change what it asks (decider/form.py, decider/household.py)."""

    half: str  # a half in a percent slot
    stop: str  # a request to stop
    negation: str


@dataclass(frozen=True, slots=True)
class Rooms:
    """Room words (decider/rooms.py, decider/media.py)."""

    words: str  # rooms and floors, also inside compounds
    before_room: str  # the words before a room's name (in the); plain word list


@dataclass(frozen=True, slots=True)
class Locks:
    """Whether a text says to lock or to unlock (decider/locks.py)."""

    unlocked: str
    locked: str
    # A command whose verb is split: the lock verb first, its particle last, in the group "particle". Empty: the
    # language does not split its verbs.
    split: str = ""
    unlock_particles: frozenset[str] = frozenset()  # the particles of `split` that unlock


@dataclass(frozen=True, slots=True)
class Lists:
    """How a request names a list and its item (decider/lists.py); plain word lists unless said otherwise."""

    tick_off: str  # a regular expression
    delete: str  # a regular expression
    add: str  # the verbs that start a request to add
    need: str  # the phrases that start a request for something needed (we need)
    to_list: str  # the prepositions before the list's name
    list_owners: str  # the articles and possessives before the list's name
    list: str  # the word the list's name ends in
    item_articles: str  # the article before a new item
    is_on: str  # the words that start a question whether an item is on a list
    still: str
    asked_articles: str  # the article before the item asked about
    pronouns: frozenset[str]  # words that stand for an item instead of naming one


@dataclass(frozen=True, slots=True)
class Media:
    """How a request names what to play (decider/media.py); plain word lists."""

    play: str  # the words that start a request to play
    play_end: str  # the closing particle or verb of a request to play
    kind_articles: str
    kinds: str  # playlist, album, song
    by: str  # between a title and its artist
    anything: frozenset[str]  # music in general, no item
    skip_articles: str
    queue: str  # a place in the queue: next, previous, last
    again: str


@dataclass(frozen=True, slots=True)
class Lights:
    """Light words (decider/naming.py); plain word lists."""

    words: str  # inside compounds too
    generic: str  # light as such, no kind of light
    not_a_kind: frozenset[str]  # words before a light word that make no kind of light


@dataclass(frozen=True, slots=True)
class Numbers:
    """Number words (decider/decision.py)."""

    units: dict[str, int]  # two to nine
    one: str  # the word for one, counted only inside a compound (alone it is an article)
    teens: dict[str, int]  # ten to nineteen
    tens: dict[str, int]  # twenty to ninety, and a hundred
    # A compound number word, with the groups "unit" and "tens" filled from the lists above; empty: none.
    compound: str = ""


@dataclass(frozen=True, slots=True)
class Replies:
    """What the decider says itself (decider/decision.py)."""

    speech: dict[str, str]  # by reply: the sentence, its slots in braces
    media_labels: dict[str, str]  # by media attribute: its label
    decimal_comma: bool


@dataclass(frozen=True, slots=True)
class Language:
    """One language: its name and every word list and grammar rule of it that AI Suite reads."""

    name: str  # in English, as prompts name it
    polite: str  # please; plain word list
    stopwords: frozenset[str]  # words no device name is told by
    function_words: frozenset[str]  # common words only this language has, to tell a reply's language
    days: Days
    time: Time
    reads: Reads
    requests: Requests
    rooms: Rooms
    locks: Locks
    lists: Lists
    media: Media
    lights: Lights
    numbers: Numbers
    replies: Replies
    transliteration: dict[str, str] = field(default_factory=dict)  # letter -> its ASCII spelling, for file names
