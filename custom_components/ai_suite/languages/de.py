"""German. No HA imports."""

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
    name="German",
    polite="bitte",
    stopwords=frozenset({
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "im", "in", "am", "an", "auf", "aus", "ist", "sind",
        "es", "mach", "mache", "schalte", "stell", "bitte", "und", "oder", "wie", "was", "viel", "hoch",
    }),
    # "was" and "an" are English words too: they would count for the wrong language.
    function_words=frozenset({
        "der", "die", "das", "ist", "und", "nicht", "ich", "im", "es", "wird", "ein", "eine", "kein", "keine", "auf",
        "mit", "gibt", "sind", "habe", "dein", "deine", "wurde", "jetzt", "leider", "heute",
    }),
    days=Days(
        weekdays=("montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag|sonnabend", "sonntag"),
        today=r"\bheute\b",
        tomorrow=r"(?<!heute )\bmorgen\b",  # "heute morgen" is this morning
        day_after_tomorrow=r"übermorgen|uebermorgen",
        next_days=r"\b(?:nächsten|naechsten|kommenden) tagen?\b|\bdiese woche\b",
        weekend=r"wochenende",
        next_week=r"\b(?:nächste[nr]?|naechste[nr]?|kommende[nr]?) woche\b",
        yesterday=r"\bgestern\b",
        day_before_yesterday=r"\bvorgestern\b",
    ),
    time=Time(
        past=r"\b(?:letzte[nrms]?|vorhin|gestern|zuletzt)\b",
        past_verbs=r"\b(?:war|warst|waren)\b",
        when=r"\bwann\b",
        time_question=r"\b(?:um wie viel uhr|wie lange)\b",
        window=r"letzten|vergangenen|nächsten|kommenden",
        time_units=r"sekunden?|minuten?|stunden?|tagen?|tage|wochen?|monaten?|monate|jahren?|jahre",
        clock=r"uhr",
    ),
    reads=Reads(
        temperature=r"temperatur|\b(?:warm|kalt|heiß|heiss)\b",
        humidity=r"feucht",
        battery=r"batterie|akku",
        degrees=r"grad",
        percent=r"prozent",
        many=r"\b(?:alle[nrs]?|jede[nrms]?)\b",
        outdoors=r"außen|aussen|draußen|draussen",
        so_far=r"\b(?:heute|bisher)\b",
        now=r"\b(?:aktuell\w*|jetzt|gerade|momentan)\b",
        follow_up_opener=r"gut\b",
        follow_up_and=r"und\b",
        there=r"\bdort\b",
    ),
    requests=Requests(
        half=r"\b(?:halbe?[nr]?|hälfte)\b",
        stop=r"\b(?:stopp?e?n?|anhalten)\b",
        negation=r"\b(?:nicht|nie|niemals|kein\w*)\b",
    ),
    rooms=Rooms(
        words=(
            r"zimmer|küche|kueche|\bbad|toilette|\bwc\b|keller|garage|flur|diele|\bgang\b|treppe|dachboden|speicher|"
            r"garten|terrasse|balkon|veranda|stube|büro|buero|kammer|werkstatt|haus\b|geschoss|stockwerk|etage|"
            r"\boben\b|\bunten\b"
        ),
        before_room=r"im|in der|in dem|in|auf dem",
    ),
    locks=Locks(
        # By prefix, as "geschlossen" (closed) shares the stem; and the mixed "geunlockt", "gelockt".
        unlocked=r"\b(?:auf|ent)(?:ge)?(?:schlie(?:ß|ss)|schloss|sperr|riegel)|\bgeunlock",
        locked=r"\b(?:ab|zu|ver)(?:ge)?(?:schlie(?:ß|ss)|schloss|sperr|riegel)|\bgelock",
        # The particle counts only at the sentence's end, so "auf" as a preposition inside it says nothing.
        split=r"\b(?:schlie(?:ß|ss)|schloss|sperr|riegel)\w*\b.*\b(?P<particle>ab|zu|auf)\b[\s,]*(?:bitte)?[\s.!?]*$",
        unlock_particles=frozenset({"auf"}),
    ),
    lists=Lists(
        tick_off=r"\bab(?:ge)?hak|\bhak\w*\b.*\bab\b|\berledigt\b",
        delete=r"\b(?:lösch|entfern)",
        add=r"setze?|schreibe?|packe?|trag(?:e)?",
        need=r"wir brauchen|ich brauche",
        to_list=r"auf|zur|in",
        list_owners=r"die|meine|unsere|der|den",
        list=r"liste",
        item_articles=r"etwas|ein|eine|einen",
        is_on=r"ist|sind|steht|stehen|haben wir",
        still=r"noch",
        asked_articles=r"der|die|das",
        pronouns=frozenset({"es", "das"}),
    ),
    media=Media(
        play=r"spiele?|lege?|ich (?:will|möchte)",
        play_end=r"auf|hören|anhören",
        kind_articles=r"die|das|den",
        kinds=r"playlist|album|lied|radiosender",
        by=r"von",
        anything=frozenset({"musik", "was", "etwas", "lieder", "irgendwas", "irgendwelche musik"}),
        skip_articles=r"das|den|dem|die",
        queue=r"nächste[mnrs]?|vorherige[mnrs]?|letzte[mnrs]?",
        again=r"nochmal|noch mal",
    ),
    lights=Lights(
        words=r"licht|leucht",
        generic=r"licht|lichter|lampe|lampen|leuchte|leuchten|beleuchtung",
        not_a_kind=frozenset({"die", "das", "alle", "den", "dem", "der", "meine", "unsere", "schalte", "mach", "dimme"}),
    ),
    numbers=Numbers(
        units={"zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9},
        one="ein",  # alone also a verb's particle ("schalte ... ein")
        teens={"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
               "siebzehn": 17, "achtzehn": 18, "neunzehn": 19},
        tens={"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
              "siebzig": 70, "achtzig": 80, "neunzig": 90, "hundert": 100},
        compound=r"\b(?P<unit>{units})und(?P<tens>{tens})\b",
    ),
    replies=Replies(
        speech={
            "read_state": "{name} ist {value}.",
            "read_climate": "{name} hat {current}, eingestellt auf {target}.",
            "read_weather": "{name}: {condition}, {temperature}.",
            "read_media": "{name} ist {state}: {playing}.",
        },
        media_labels={"media_title": "Titel", "media_artist": "Interpret", "media_album_name": "Album"},
        decimal_comma=True,
    ),
    transliteration={"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"},
)
