"""Constants of the trace monitor exporter: its options, its limits and the record's schema version."""

# Options of the config entry (options flow).
CONF_MONITOR_ENABLED = "monitor_enabled"
CONF_MONITOR_URL = "monitor_url"
CONF_MONITOR_TOKEN = "monitor_token"  # the monitor's ingest bearer token; never logged, redacted in diagnostics
# STT audio needs `assist_pipeline: debug_recording_dir:` in configuration.yaml; each run folder is deleted after
# its upload.
CONF_MONITOR_AUDIO = "monitor_audio"

# The turn record's version; the monitor rejects any other.
SCHEMA_VERSION = 1
# The monitor's ingest of turn records, below its base URL; a record's audio goes to `<path>/<id>/audio/<kind>`.
INGEST_TURNS_PATH = "/api/ingest/turns"

# Turns held at once, recording or waiting to be built; past it a turn is not recorded and counted as skipped.
MAX_OPEN_TURNS = 20
# Built records waiting to be sent; the oldest is dropped first.
QUEUE_SIZE = 100

# State changes count for a turn until this long after the agent's end: HA links an entity's change to the call's
# context for 5 s (CONTEXT_RECENT_TIME_SECONDS, helpers/entity.py), plus a margin.
EFFECTS_WINDOW_SECONDS = 5.5
# The pipeline's run-end is waited for until this long after the agent's end (polled), the satellite's idle and the
# answer's playback on its device's media player until this long after tts-start (state changes).
RUN_END_WAIT_SECONDS = 15
IDLE_WAIT_SECONDS = 90
POLL_SECONDS = 0.25

# Sending: tries per record, the backoff between them (doubling), the timeout of one request.
SEND_TRIES = 5
BACKOFF_FIRST_SECONDS = 5
BACKOFF_MAX_SECONDS = 300
HTTP_TIMEOUT_SECONDS = 10
# Hashes of the blobs the monitor confirmed, kept so a blob is sent once; the oldest forgotten first.
SENT_HASHES_SIZE = 256

# STT audio: a larger WAV is not sent (the monitor takes at most 2 MiB); the run folder must be within this many
# seconds of the expected start.
AUDIO_MAX_BYTES = 2 * 1024 * 1024
AUDIO_MATCH_WINDOW_SECONDS = 2

# Diagnostics: the hook and build times of this many latest turns give their p50 and max.
TIMES_KEPT = 100
