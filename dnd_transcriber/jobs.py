"""Turning one collected session directory into a transcript.

The recorder's version of this file read database rows. This one reads
`metadata.json` and nothing else, which is what lets the transcriber run on a
machine that has never heard of Discord or SQLite.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .chunking import CHUNK_DIR_NAME, cleanup_chunks, split_audio
from .contract import (
    TRANSCRIPT_JSON,
    TRANSCRIPT_MD,
    SessionMetadata,
    audio_tracks,
)
from .timeutil import from_iso, utcnow
from .transcription import (
    RawSegment,
    Segment,
    Transcriber,
    build_initial_prompt,
    merge_segments,
    render_json,
    render_markdown,
    word_count,
    write_transcripts,
)

log = logging.getLogger(__name__)


class TranscriptionJobError(RuntimeError):
    """Raised when a session cannot produce a transcript at all."""


@dataclass
class JobResult:
    session_id: str
    transcript_md: Path
    transcript_json: Path
    duration_seconds: float
    speaker_count: int
    word_count: int
    segment_count: int
    warnings: list[str] = field(default_factory=list)


def session_duration_seconds(metadata: SessionMetadata) -> float:
    start = from_iso(metadata.start_time_utc)
    end = from_iso(metadata.end_time_utc) or utcnow()
    if start is None:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def transcribe_one(
    path: Path,
    transcriber: Transcriber,
    language: str,
    initial_prompt: str | None,
    chunk_seconds: float,
) -> list[RawSegment]:
    """Transcribe one speaker track, in chunks, with timestamps re-based.

    Chunking is what keeps peak memory flat: faster-whisper loads whatever file
    it is given entirely into memory first, so a whole four-hour track would
    need several GB before decoding even starts.
    """
    try:
        chunks = split_audio(path, path.parent / CHUNK_DIR_NAME, chunk_seconds)
    except Exception as exc:  # noqa: BLE001 - unreadable input, or no ffmpeg
        log.warning(
            "Could not split %s (%s); transcribing it whole. A long track handled "
            "this way may use a lot of memory.",
            path.name,
            exc,
        )
        return transcriber.transcribe_file(path, language, initial_prompt)

    if len(chunks) == 1 and chunks[0].path == path:
        return transcriber.transcribe_file(path, language, initial_prompt)

    segments: list[RawSegment] = []
    try:
        for index, chunk in enumerate(chunks, start=1):
            log.info("Transcribing %s chunk %s/%s", path.name, index, len(chunks))
            for segment in transcriber.transcribe_file(chunk.path, language, initial_prompt):
                segments.append(
                    RawSegment(
                        start=segment.start + chunk.offset,
                        end=segment.end + chunk.offset,
                        text=segment.text,
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                    )
                )
            # Free each chunk as soon as it is done, not at the end.
            chunk.path.unlink(missing_ok=True)
    finally:
        cleanup_chunks(chunks, path)
    return segments


def transcribe_tracks(
    tracks: list[Path],
    transcriber: Transcriber,
    language: str,
    initial_prompt: str | None,
    chunk_seconds: float,
) -> tuple[dict[str, list[RawSegment]], list[str]]:
    """Transcribe every speaker track, skipping (not failing on) broken ones."""
    per_user: dict[str, list[RawSegment]] = {}
    warnings: list[str] = []
    for path in tracks:
        user_id = path.stem
        try:
            per_user[user_id] = transcribe_one(
                path, transcriber, language, initial_prompt, chunk_seconds
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the session
            log.exception("Transcription failed for %s", path)
            warnings.append(f"Skipped speaker {user_id}: {type(exc).__name__}: {exc}")
    if tracks and not per_user:
        raise TranscriptionJobError(
            "Every speaker track failed to transcribe: " + "; ".join(warnings)
        )
    return per_user, warnings


def build_transcript(
    metadata: SessionMetadata,
    per_user: dict[str, list[RawSegment]],
    *,
    output_dir: Path,
    warnings: list[str],
    model_name: str,
) -> JobResult:
    """Merge, render and write both transcript files."""
    segments: list[Segment] = merge_segments(per_user, metadata.offsets, metadata.participants)
    duration = session_duration_seconds(metadata)
    start = from_iso(metadata.start_time_utc) or datetime.now(UTC)
    tz = ZoneInfo(metadata.timezone)

    markdown = render_markdown(
        segments,
        session_name=metadata.name,
        session_start=start,
        tz=tz,
        duration_seconds=duration,
        warnings=warnings,
    )
    payload = render_json(
        segments,
        session_id=metadata.session_id,
        session_name=metadata.name,
        session_start=start,
        tz=tz,
        duration_seconds=duration,
        language=metadata.language,
        model=model_name,
        warnings=warnings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / TRANSCRIPT_MD
    json_path = output_dir / TRANSCRIPT_JSON
    write_transcripts(md_path, json_path, markdown, payload)

    return JobResult(
        session_id=metadata.session_id,
        transcript_md=md_path,
        transcript_json=json_path,
        duration_seconds=duration,
        speaker_count=len({segment.speaker for segment in segments}),
        word_count=word_count(segments),
        segment_count=len(segments),
        warnings=warnings,
    )


def run_session(
    session_dir: Path,
    metadata: SessionMetadata,
    transcriber: Transcriber,
    *,
    output_dir: Path,
    chunk_seconds: float = 600.0,
    model_name: str = "",
) -> JobResult:
    """Transcribe one collected session directory end to end."""
    tracks = audio_tracks(session_dir, metadata.audio_format)
    warnings: list[str] = []

    # The session's own character names go to Whisper as a decode prompt; this
    # is the single biggest lever on getting proper nouns right.
    initial_prompt = build_initial_prompt(metadata.participants.values(), metadata.prompt_extra)
    log.info(
        "Transcribing session %s (%s track(s), language=%s)",
        metadata.session_id,
        len(tracks),
        metadata.language,
    )

    if not tracks:
        warnings.append("No audio tracks were found for this session.")
        per_user: dict[str, list[RawSegment]] = {}
    else:
        per_user, warnings = transcribe_tracks(
            tracks, transcriber, metadata.language, initial_prompt, chunk_seconds
        )

    return build_transcript(
        metadata,
        per_user,
        output_dir=output_dir,
        warnings=warnings,
        model_name=model_name,
    )


def load_metadata_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
