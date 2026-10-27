"""Constants of Second Brain inside AI Suite."""

import logging

LOGGER = logging.getLogger(__package__)

# The LLM API's id and name stay Second Brain's: agents select it by this id, and HA prefixes its tool names with the
# name when several APIs are selected ("second-brain__search_brain").
BRAIN_API_ID = "second_brain"
BRAIN_API_NAME = "Second Brain"

CONF_STORE_LOCATION = "store_location"

# Subentry types. Memory holds the store, its prompt budgets and the background model; its features need it.
SUBENTRY_MEMORY = "memory"
SUBENTRY_LIBRARIAN = "librarian"
SUBENTRY_SELF_IMPROVE = "self_improve"

CONF_LLM_MODEL = "llm_model"  # the background model (librarian, rule compiler, self-improver)
CONF_CONSOLIDATE_TIME = "consolidate_time"
CONF_CONSOLIDATE_ENABLED = "consolidate_enabled"
CONF_SELF_IMPROVE_EFFORT = "self_improve_effort"
CONF_LEARN_TIME = "learn_time"
CONF_CONSOLIDATE_EFFORT = "consolidate_effort"

STORE_FOLDER = "second_brain"
DEFAULT_GIT_NAME = "Second Brain Assistant"
DEFAULT_GIT_EMAIL = "second-brain@ha.local"
CONSOLIDATOR_GIT_NAME = "Second Brain Consolidator"
CONSOLIDATOR_GIT_EMAIL = "consolidator@ha.local"
LEARNER_GIT_NAME = "Second Brain Learner"
LEARNER_GIT_EMAIL = "learner@ha.local"

# How much of the store goes into every prompt, and the most one read_note returns: the index grows with every
# note, and every voice turn pays for what it carries. A cut is announced to the model (store._clip).
CORE_CHARS = 4000
RULES_CHARS = 2000
INDEX_CHARS = 2000
NOTE_CHARS = 8000
DEFAULT_CONSOLIDATE_TIME = "03:00"
# After the librarian's run: the two passes touch the same store and there is no
# reason for them to queue behind each other's lock.
DEFAULT_LEARN_TIME = "03:30"
# Thinking effort for the background LLM calls (consolidator triage, turn
# analyzer). They reason with nobody waiting, so effort is affordable there.
# NO_THINKING turns thinking off. Values map to reasoning_effort.
NO_THINKING = "none"
SELF_IMPROVE_EFFORTS = [NO_THINKING, "low", "medium", "high", "xhigh"]
DEFAULT_SELF_IMPROVE_EFFORT = "high"

SEARCH_SCORE_FILENAME = 8
SEARCH_SCORE_TAG = 6
SEARCH_SCORE_HEADING = 4
SEARCH_SCORE_BODY = 1
SEARCH_RESULTS = 5
