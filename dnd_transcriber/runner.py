"""The work loop: collect, transcribe, send back, archive.

Strictly one session at a time. Whisper saturates the CPU on its own, and two
concurrent runs on a laptop would only make both slower while making the fans
louder.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .contract import ContractError, SessionMetadata, validate_outbox
from .jobs import JobResult, run_session
from .quiet_hours import in_quiet_hours
from .sync import Transport
from .transcription import WhisperTranscriber

log = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    pulled: bool
    transcribed: bool
    pushed: bool

    @property
    def stage(self) -> str:
        if self.pushed:
            return "done"
        if self.transcribed:
            return "transcribed, not sent"
        if self.pulled:
            return "pulled, not transcribed"
        return "waiting on the recorder"


class Runner:
    def __init__(
        self,
        config: Config,
        transport: Transport,
        transcriber: WhisperTranscriber | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._transcriber = transcriber

    # -- lazily loaded model ----------------------------------------------

    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
                download_root=self.config.model_cache_dir or None,
                beam_size=self.config.whisper_beam_size,
                condition_on_previous_text=self.config.whisper_condition_on_previous_text,
                vad_min_silence_ms=self.config.whisper_vad_min_silence_ms,
                filter_hallucinations_enabled=self.config.filter_hallucinations,
            )
        return self._transcriber

    # -- scheduling --------------------------------------------------------

    def may_work(self, now: datetime | None = None) -> bool:
        """Quiet hours are optional here; the operator usually runs by hand."""
        if not self.config.quiet_hours_enabled:
            return True
        now = now or datetime.now(self.config.tz)
        return in_quiet_hours(now, self.config.quiet_hours_start, self.config.quiet_hours_end, True)

    # -- inspection --------------------------------------------------------

    def remote_pending(self) -> list[str]:
        return self.transport.list_ready()

    def local_sessions(self) -> list[SessionState]:
        states: list[SessionState] = []
        for directory in sorted(p for p in self.config.incoming_dir.glob("*") if p.is_dir()):
            session_id = directory.name
            states.append(
                SessionState(
                    session_id=session_id,
                    pulled=True,
                    transcribed=(self.config.outgoing_dir / session_id).is_dir(),
                    pushed=(self.config.archive_dir / session_id).is_dir(),
                )
            )
        for directory in sorted(p for p in self.config.archive_dir.glob("*") if p.is_dir()):
            if not any(s.session_id == directory.name for s in states):
                states.append(
                    SessionState(directory.name, pulled=True, transcribed=True, pushed=True)
                )
        return states

    # -- the three steps ---------------------------------------------------

    def fetch(self, session_id: str | None = None) -> list[str]:
        """Collect ready sessions from the recorder."""
        wanted = [session_id] if session_id else self.remote_pending()
        pulled: list[str] = []
        for sid in wanted:
            local = self.config.incoming_dir / sid
            if local.is_dir() or (self.config.archive_dir / sid).is_dir():
                log.info("Session %s is already here; skipping", sid)
                continue
            log.info("Fetching session %s", sid)
            self.transport.pull(sid, self.config.incoming_dir)
            try:
                validate_outbox(local)
            except ContractError as exc:
                log.error("Session %s arrived unusable: %s", sid, exc)
                shutil.rmtree(local, ignore_errors=True)
                continue
            pulled.append(sid)
        return pulled

    def transcribe(self, session_id: str | None = None) -> list[JobResult]:
        """Transcribe pulled sessions, one at a time."""
        results: list[JobResult] = []
        candidates = (
            [self.config.incoming_dir / session_id]
            if session_id
            else sorted(p for p in self.config.incoming_dir.glob("*") if p.is_dir())
        )
        for directory in candidates:
            sid = directory.name
            if (self.config.outgoing_dir / sid).is_dir():
                log.info("Session %s is already transcribed; skipping", sid)
                continue
            try:
                metadata: SessionMetadata = validate_outbox(directory)
            except ContractError as exc:
                log.error("Refusing to transcribe %s: %s", sid, exc)
                self._move(directory, self.config.failed_dir)
                continue

            log.info("Starting transcription of %s (%s)", sid, metadata.name)
            try:
                result = run_session(
                    directory,
                    metadata,
                    self.transcriber,
                    output_dir=self.config.outgoing_dir / sid,
                    chunk_seconds=self.config.transcribe_chunk_minutes * 60,
                    model_name=self.config.whisper_model,
                )
            except Exception:  # noqa: BLE001 - one bad session must not stop the rest
                log.exception("Transcription failed for %s", sid)
                self._move(directory, self.config.failed_dir)
                continue
            log.info(
                "Finished %s: %s speaker(s), %s words",
                sid,
                result.speaker_count,
                result.word_count,
            )
            results.append(result)
        return results

    def push(self, session_id: str | None = None, *, discard_remote: bool = True) -> list[str]:
        """Send transcripts back, then release the audio held by the recorder."""
        candidates = (
            [self.config.outgoing_dir / session_id]
            if session_id
            else sorted(p for p in self.config.outgoing_dir.glob("*") if p.is_dir())
        )
        pushed: list[str] = []
        for directory in candidates:
            sid = directory.name
            log.info("Sending transcript for %s", sid)
            self.transport.push(sid, directory)
            if discard_remote:
                # The transcript is delivered and the audio is archived here, so
                # the recorder no longer needs to hold it.
                self.transport.discard_remote(sid)
            self._archive(sid)
            pushed.append(sid)
        return pushed

    def session(self, session_id: str | None = None) -> list[str]:
        """The everyday command: fetch, transcribe, send back."""
        self.fetch(session_id)
        self.transcribe(session_id)
        return self.push(session_id)

    # -- housekeeping ------------------------------------------------------

    def _archive(self, session_id: str) -> None:
        """Keep the audio and the transcript together, here, permanently."""
        destination = self.config.archive_dir / session_id
        destination.mkdir(parents=True, exist_ok=True)
        incoming = self.config.incoming_dir / session_id
        if incoming.is_dir():
            for item in incoming.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(destination / item.name))
            shutil.rmtree(incoming, ignore_errors=True)
        outgoing = self.config.outgoing_dir / session_id
        if outgoing.is_dir():
            for item in outgoing.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(destination / item.name))
            shutil.rmtree(outgoing, ignore_errors=True)

    def _move(self, directory: Path, into: Path) -> None:
        into.mkdir(parents=True, exist_ok=True)
        target = into / directory.name
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(directory), str(target))
