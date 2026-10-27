"""A turn's STT audio: the WAV files HA's `assist_pipeline: debug_recording_dir:` kept for the turn's run.

HA writes each run's audio to `<dir>/[<device_id>/]<pipeline name>/<monotonic_ns>/` (assist_pipeline/run.py), the
ns taken just before the run's run-start event: `00_wake-<entity>.wav` and `01_stt-<engine>.wav`, 16 kHz mono s16.
The folder name carries no run id, so the run's folder is the one whose ns lies nearest the run-start's monotonic
time, which follows from the turn's open (same process, same clock). The folder is deleted once its clips were sent;
HA never reads it back.

Every file system access runs in the executor.
"""

import logging
from pathlib import Path
import shutil
from typing import Literal
import wave

from homeassistant.core import HomeAssistant

from .built import AudioClip
from .const import AUDIO_MATCH_WINDOW_SECONDS, AUDIO_MAX_BYTES
from .recorder import TurnTrace
from .voice import RunRef

_LOGGER = logging.getLogger(__name__)

# HA's file name prefixes (assist_pipeline/default_pipeline.py start_debug_recording) -> the clip's kind.
_KINDS: dict[str, Literal["wake", "stt"]] = {"00_wake-": "wake", "01_stt-": "stt"}


async def find_clips(hass: HomeAssistant, ref: RunRef, trace: TurnTrace) -> tuple[AudioClip, ...]:
    """The run's clips; empty without debug_recording_dir, without a folder, or when two folders could be it.

    A clip over AUDIO_MAX_BYTES is left out (the monitor takes no more). Call after the run's run-end: HA closes
    the files before it.
    """
    if ref.recording_dir is None or ref.pipeline_name is None:
        return ()
    runs = Path(ref.recording_dir)
    if ref.device_id is not None:
        runs /= ref.device_id
    runs /= ref.pipeline_name
    run_start_monotonic = trace.opened - (trace.opened_wall - ref.run_start)
    return await hass.async_add_executor_job(_clips, runs, round(run_start_monotonic * 1e9))


async def delete_folder(hass: HomeAssistant, folder: Path) -> None:
    """Delete a run folder whose clips were sent; a failure is logged and leaves the folder."""
    try:
        await hass.async_add_executor_job(shutil.rmtree, folder)
    except OSError as err:
        _LOGGER.warning("Could not delete the sent recording folder %s: %s", folder, err)


def _clips(runs: Path, run_start_ns: int) -> tuple[AudioClip, ...]:
    if (folder := _run_folder(runs, run_start_ns)) is None:
        return ()
    return tuple(clip for file in sorted(folder.iterdir()) if (clip := _clip(folder, file)) is not None)


def _run_folder(runs: Path, run_start_ns: int) -> Path | None:
    """The one run folder within AUDIO_MATCH_WINDOW_SECONDS of the run's start; None for none or several."""
    try:
        folders = [path for path in runs.iterdir() if path.name.isdigit()]
    except FileNotFoundError:  # no run of this pipeline (and device) was recorded yet
        return None
    window_ns = AUDIO_MATCH_WINDOW_SECONDS * 1_000_000_000
    near = [folder for folder in folders if abs(int(folder.name) - run_start_ns) <= window_ns]
    return near[0] if len(near) == 1 else None


def _clip(folder: Path, file: Path) -> AudioClip | None:
    """The file as a clip; None for a file that is not a clip, too large, or not a readable WAV."""
    kind = next((kind for prefix, kind in _KINDS.items() if file.name.startswith(prefix)), None)
    if kind is None or file.suffix != ".wav":
        return None
    size = file.stat().st_size
    if size > AUDIO_MAX_BYTES:
        return None
    try:
        with wave.open(str(file), "rb") as wav:
            duration_ms = round(wav.getnframes() * 1000 / wav.getframerate())
    except (wave.Error, EOFError):  # a run whose recording thread failed leaves a broken file
        return None
    return AudioClip(kind=kind, file=file.name, path=file, folder=folder, bytes=size, duration_ms=duration_ms)
