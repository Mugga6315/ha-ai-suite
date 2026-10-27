"""English. No HA imports."""

from .words import (
    Days,
    Language,
    Lights,
    Lists,
    Locks,
    Media,
    Numbers,
    Reads,
    Replies,
    Requests,
    Rooms,
    Time,
)

LANGUAGE = Language(
    name="English",
    polite="please",
    stopwords=frozenset({
        "the", "a", "an", "in", "on", "of", "to", "at", "is", "are", "it", "its", "my", "and", "or", "please", "turn",
        "switch", "set", "make", "what", "how", "much",
    }),
    function_words=frozenset({
        "the", "is", "and", "not", "i", "it", "you", "are", "a", "no", "there", "has", "have", "been", "your", "now",
        "on", "of", "to", "today", "sorry",
    }),
    days=Days(
        weekdays=("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
        today=r"\b(?:today|tonight)\b",
        tomorrow=r"\btomorrow\b",
        day_after_tomorrow=r"day after tomorrow",
        next_days=r"\b(?:next|coming) (?:few )?days\b|\bthis week\b",
        weekend=r"weekend",
        next_week=r"\b(?:next|coming) week\b",
        yesterday=r"\byesterday\b",
        day_before_yesterday=r"\bday before yesterday\b",
    ),
    time=Time(
        past=r"\b(?:last|yesterday|earlier|ago)\b",
        past_verbs=r"\b(?:was|were|did)\b",
        when=r"\bwhen\b",
        time_question=r"\b(?:what time|how long)\b",
        window=r"last|past|next|coming",
        time_units=r"seconds?|minutes?|hours?|days?|weeks?|months?|years?",
        clock=r"am|pm|a\.m\.|p\.m\.|o'clock",
    ),
    reads=Reads(
        temperature=r"temperature|\b(?:hot|cold)\b",
        humidity=r"humid",
        battery=r"\bbatter(?:y|ies)\b",
        degrees=r"degrees?",
        percent=r"percent",
        many=r"\b(?:all|every|each)\b",
        outdoors=r"\b(?:outside|outdoors?)\b",
        so_far=r"\b(?:today|so far)\b",
        now=r"\b(?:right now|now|currently|current)\b",
        follow_up_opener=r"ok(?:ay)?\b",
        follow_up_and=r"(?:and|what about)\b",
        there=r"\bthere\b",
    ),
    requests=Requests(
        half=r"\bhalf\b",
        stop=r"\bstop\b",
        negation=r"\b(?:not|never|no|none)\b|n't\b",
    ),
    rooms=Rooms(
        words=(
            r"room|kitchen|bath|toilet|cellar|basement|hall|corridor|stair|attic|loft|laundry|yard|patio|terrace|"
            r"balcony|porch|office|study|nursery|pantry|closet|workshop|shed\b|house\b|upstairs|downstairs|"
            r"\b(?:ground|first|second|top|upper|lower) floor"
        ),
        before_room=r"in the|in|on the",
    ),
    locks=Locks(
        unlocked=r"\bunlock",
        locked=r"\block|deadbolt",
    ),
    lists=Lists(
        tick_off=r"\b(?:check|tick)(?:ed)?\s+off\b|\bmark\w*\b.*\b(?:done|complete)",
        delete=r"\b(?:delete|remove)",
        add=r"add|put",
        need=r"we need|i need",
        to_list=r"to|on|onto|in",
        list_owners=r"the|my|our",
        list=r"list",
        item_articles=r"some|a|an|the",
        is_on=r"is there|are there|is|are|do we have|have we got",
        still=r"still",
        asked_articles=r"the|some|any",
        pronouns=frozenset({"it"}),
    ),
    media=Media(
        play=r"play|put on|i want to (?:hear|listen to)|i'd like to hear",
        play_end=r"",
        kind_articles=r"the|a",
        kinds=r"playlist|album|song|track|radio station",
        by=r"by",
        anything=frozenset({"music", "something", "songs", "some music", "some songs"}),
        skip_articles=r"the",
        queue=r"next|previous|last",
        again=r"again",
    ),
    lights=Lights(
        words=r"light|lamp",
        generic=r"light|lights|lamp|lamps|lighting",
        not_a_kind=frozenset({"the", "all", "my", "our", "these", "those", "turn", "switch", "dim", "set"}),
    ),
    numbers=Numbers(
        units={"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9},
        one="one",
        teens={"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
               "seventeen": 17, "eighteen": 18, "nineteen": 19},
        tens={"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
              "ninety": 90, "hundred": 100},
        compound=r"\b(?P<tens>{tens})[- ](?P<unit>{units})\b",
    ),
    replies=Replies(
        speech={
            "read_state": "{name} is {value}.",
            "read_climate": "{name} is at {current}, set to {target}.",
            "read_weather": "{name}: {condition}, {temperature}.",
            "read_media": "{name} is {state}: {playing}.",
        },
        media_labels={"media_title": "title", "media_artist": "artist", "media_album_name": "album"},
        decimal_comma=False,
    ),
)
