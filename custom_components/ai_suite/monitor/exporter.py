"""The sender: built records posted to the trace monitor's ingest API one at a time, then their STT audio.

`submit` queues a record from the event loop; `run` is the one sender task (the entry's background task). The sender
takes the oldest record and finishes it before the next: a request the monitor cannot take now (no connection,
timeout, 5xx, 429) is tried again after a doubling pause, up to SEND_TRIES times, then the record is dropped; a
status another try will not change (any other 4xx) drops the record, or the one clip, at once; so does a record
JSON cannot express. A blob is sent until a
record carrying it was accepted. A run folder is deleted once each of its clips was accepted.
"""

from asyncio import Event, sleep
from collections import OrderedDict, deque
from http import HTTPStatus
import logging
from typing import Any

import aiohttp
from aiohttp import hdrs
from yarl import URL

from homeassistant.const import CONTENT_TYPE_JSON
from homeassistant.core import HomeAssistant
from homeassistant.helpers.json import json_bytes

from .audio import delete_folder
from .built import AudioClip, Blob, BuiltTurn
from .const import (
    BACKOFF_FIRST_SECONDS,
    BACKOFF_MAX_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    INGEST_TURNS_PATH,
    QUEUE_SIZE,
    SEND_TRIES,
    SENT_HASHES_SIZE,
)

_LOGGER = logging.getLogger(__name__)

CONTENT_TYPE_WAV = "audio/wav"
_ACCEPTED = frozenset({HTTPStatus.CREATED, HTTPStatus.OK})


class _Unreachable(Exception):
    """The monitor cannot take the request now; another try may succeed."""


class _Unserialisable(Exception):
    """The record holds a value JSON cannot express (a tool result can be any HA object)."""


class _Refused(Exception):
    """The monitor answered with a status another try will not change."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class Exporter:
    """The queue of built records and the one task that sends them; its counters are the diagnostics."""

    def __init__(self, hass: HomeAssistant, session: aiohttp.ClientSession, url: str, token: str) -> None:
        """Send to the monitor at `url` (its base, no path) with its ingest `token`, over HA's shared session."""
        self._hass = hass
        self._session = session
        self._url = url.rstrip("/")
        self._shown_url = str(URL(self._url).with_user(None))  # for logs: never credentials in the URL
        self._auth = {hdrs.AUTHORIZATION: f"Bearer {token}"}
        self._timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        self._queue: deque[BuiltTurn] = deque(maxlen=QUEUE_SIZE)
        self._queued = Event()
        self._sent_hashes: OrderedDict[str, None] = OrderedDict()  # LRU of blobs the monitor confirmed
        self._unavailable = False
        self._warned_statuses: set[int] = set()
        self._warned_unserialisable = False
        self._sent = 0
        self._dropped_full = 0
        self._dropped_failed = 0
        self._dropped_invalid = 0
        self._blobs_sent = 0
        self._audio_sent = 0
        self._last_error: str | None = None

    def submit(self, built: BuiltTurn) -> None:
        """Queue a record; when the queue is full the oldest waiting record is dropped."""
        if len(self._queue) == QUEUE_SIZE:
            self._dropped_full += 1
        self._queue.append(built)
        self._queued.set()

    async def run(self) -> None:
        """Send the queued records, oldest first, until cancelled."""
        while True:
            if not self._queue:
                self._queued.clear()
                await self._queued.wait()
                continue
            await self._deliver(self._queue.popleft())

    def diagnostics(self) -> dict[str, Any]:
        """The sender's counters and its last error; never the token."""
        return {
            "sent": self._sent,
            "dropped_full": self._dropped_full,
            "dropped_failed": self._dropped_failed,
            "dropped_invalid": self._dropped_invalid,
            "blobs_sent": self._blobs_sent,
            "audio_sent": self._audio_sent,
            "last_error": self._last_error,
        }

    async def _deliver(self, built: BuiltTurn) -> None:
        """Post the record, then upload its clips; the whole delivery is tried again while the monitor is away."""
        posted = False
        uploaded: set[AudioClip] = set()
        for attempt in range(SEND_TRIES):
            if attempt:
                await sleep(_backoff(attempt))
            try:
                if not posted:
                    await self._post(built)
                    posted = True
                await self._upload_audio(built, uploaded)
            except _Unreachable as err:
                self._went_unavailable(err)
                continue
            except _Refused as err:
                self._refused(err, f"record {built.id}")
                self._dropped_failed += 1
                return
            except _Unserialisable as err:
                self._unserialisable(err, built.id)
                return
            return
        _LOGGER.debug("Record %s given up after %d tries", built.id, SEND_TRIES)
        if not posted:
            self._dropped_failed += 1

    async def _post(self, built: BuiltTurn) -> None:
        blobs = [
            {"hash": digest, "kind": blob.kind, "text": blob.text}
            for digest, blob in built.blobs.items()
            if digest not in self._sent_hashes
        ]
        try:
            body = json_bytes({"turn": built.turn, "blobs": blobs})
        except TypeError as err:  # orjson's JSONEncodeError is one
            raise _Unserialisable(type(err).__name__) from err
        await self._request(hdrs.METH_POST, INGEST_TURNS_PATH, body, CONTENT_TYPE_JSON)
        self._sent += 1
        self._blobs_sent += len(blobs)
        self._confirm_hashes(built.blobs)

    async def _upload_audio(self, built: BuiltTurn, uploaded: set[AudioClip]) -> None:
        """Upload the clips not uploaded yet; delete each run folder whose clips were all accepted."""
        for clip in built.audio:
            if clip not in uploaded:
                await self._upload_clip(built.id, clip, uploaded)
        for folder in {clip.folder for clip in built.audio}:
            if all(clip in uploaded for clip in built.audio if clip.folder == folder):
                await delete_folder(self._hass, folder)

    async def _upload_clip(self, turn_id: str, clip: AudioClip, uploaded: set[AudioClip]) -> None:
        try:
            wav = await self._hass.async_add_executor_job(clip.path.read_bytes)
        except OSError as err:
            _LOGGER.debug("Clip %s of record %s not read, skipped: %s", clip.file, turn_id, err)
            return
        try:
            path = f"{INGEST_TURNS_PATH}/{turn_id}/audio/{clip.kind}"
            await self._request(hdrs.METH_PUT, path, wav, CONTENT_TYPE_WAV)
        except _Refused as err:
            self._refused(err, f"clip {clip.file} of record {turn_id}")
            return
        uploaded.add(clip)
        self._audio_sent += 1

    async def _request(self, method: str, path: str, body: bytes, content_type: str) -> None:
        """One request; returns when the monitor accepted it (201 new, 200 already there)."""
        headers = {**self._auth, hdrs.CONTENT_TYPE: content_type}
        try:
            async with self._session.request(
                method, self._url + path, data=body, headers=headers, timeout=self._timeout
            ) as resp:
                status = resp.status
        except (aiohttp.ClientError, TimeoutError) as err:
            raise _Unreachable(type(err).__name__) from err
        if status in _ACCEPTED:
            self._came_back()
            return
        if status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            raise _Unreachable(f"HTTP {status}")
        self._came_back()
        raise _Refused(status)

    def _confirm_hashes(self, blobs: dict[str, Blob]) -> None:
        for digest in blobs:
            self._sent_hashes[digest] = None
            self._sent_hashes.move_to_end(digest)
        while len(self._sent_hashes) > SENT_HASHES_SIZE:
            self._sent_hashes.popitem(last=False)

    def _went_unavailable(self, err: _Unreachable) -> None:
        self._last_error = f"{err} from {self._shown_url}"
        if not self._unavailable:
            self._unavailable = True
            _LOGGER.info("Trace monitor at %s is unavailable: %s", self._shown_url, err)

    def _came_back(self) -> None:
        if self._unavailable:
            self._unavailable = False
            _LOGGER.info("Trace monitor at %s is available again", self._shown_url)

    def _unserialisable(self, err: _Unserialisable, turn_id: str) -> None:
        """Drop the record; its content stays out of the log, a value in it may be anything."""
        self._dropped_invalid += 1
        self._last_error = f"record {turn_id} not serialisable: {err}"
        if not self._warned_unserialisable:
            self._warned_unserialisable = True
            _LOGGER.warning("Record %s holds a value JSON cannot express (%s); dropped", turn_id, err)

    def _refused(self, err: _Refused, what: str) -> None:
        self._last_error = f"{err} from {self._shown_url} for {what}"
        if err.status not in self._warned_statuses:
            self._warned_statuses.add(err.status)
            _LOGGER.warning("Trace monitor at %s refused %s with %s; dropped", self._shown_url, what, err)


def _backoff(attempt: int) -> float:
    """The pause before try `attempt + 1`: BACKOFF_FIRST_SECONDS, doubling, at most BACKOFF_MAX_SECONDS."""
    return min(BACKOFF_FIRST_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
