"""What passes between the exporter's parts: the voice part, the audio clips and the built record.

No Home Assistant imports. voice.py makes a VoicePart, audio.py the AudioClips, record.py the BuiltTurn from them and
the recorder's TurnTrace, exporter.py sends the BuiltTurn.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class VoicePart:
    """The pipeline run behind a turn, read from HA's debug store."""

    run_start: float  # wall clock (epoch seconds) of the run's run-start event: the turn's t0
    block: dict[str, Any]  # the turn record's `voice` without `audio`; times in ms from run-start


@dataclass(frozen=True, slots=True)
class AudioClip:
    """One WAV file HA's debug_recording_dir kept for the turn's run."""

    kind: Literal["wake", "stt"]
    file: str  # HA's file name: `<prefix>-<entity id>.wav`, the prefix naming the kind (audio.py)
    path: Path
    folder: Path  # the run folder; deleted once every clip in it was uploaded
    bytes: int  # whole file, header included
    duration_ms: int


@dataclass(frozen=True, slots=True)
class Blob:
    """A large text the record references by its hash."""

    kind: str  # "settings" | "system" | "tools"
    text: str


@dataclass(frozen=True, slots=True)
class BuiltTurn:
    """A turn record ready to send."""

    id: str
    turn: dict[str, Any]  # the turn record, JSON-ready (serialised with HA's json_bytes by the exporter)
    blobs: dict[str, Blob]  # lowercase sha256 hex of the text -> blob; every blob the record references
    audio: tuple[AudioClip, ...]
