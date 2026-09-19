"""The collect / transcribe / return loop, end to end against local directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnd_transcriber.config import Config
from dnd_transcriber.contract import (
    DONE_MARKER,
    READY_MARKER,
    SessionMetadata,
    mark,
    write_metadata,
)
from dnd_transcriber.runner import Runner
from dnd_transcriber.sync import LocalTransport, SyncError
from dnd_transcriber.transcription import ModelLoadError, RawSegment

METADATA = SessionMetadata(
    session_id="s1",
    name="Kamp Gecesi",
    start_time_utc="2026-08-29T18:00:00+00:00",
    end_time_utc="2026-08-29T20:00:00+00:00",
    timezone="Europe/Istanbul",
    participants={"10": "Thorin", "11": "Elenya"},
    offsets={"10": 0.0, "11": 12.5},
    language="tr",
)


class FakeTranscriber:
    """Stands in for Whisper: one segment per track, no model needed."""

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.prompts: list[str | None] = []

    def transcribe_file(self, path: Path, language: str, initial_prompt=None):
        self.calls.append(path)
        self.prompts.append(initial_prompt)
        return [RawSegment(0.0, 1.0, f"merhaba from {path.stem}")]


class ExplodingTranscriber:
    def transcribe_file(self, path: Path, language: str, initial_prompt=None):
        raise RuntimeError("model blew up")


@pytest.fixture
def setup(tmp_path: Path):
    outbox, inbox = tmp_path / "remote-outbox", tmp_path / "remote-inbox"
    outbox.mkdir()
    inbox.mkdir()
    config = Config(
        workspace=tmp_path / "workspace",
        remote_outbox=str(outbox),
        remote_inbox=str(inbox),
    )
    config.ensure_dirs()
    transport = LocalTransport(outbox, inbox)
    return config, transport, outbox, inbox


def stage_remote(outbox: Path, session_id: str = "s1", *, ready: bool = True, tracks=("10", "11")):
    directory = outbox / session_id
    directory.mkdir(parents=True, exist_ok=True)
    for user_id in tracks:
        (directory / f"{user_id}.opus").write_bytes(b"fake audio")
    write_metadata(directory, METADATA if session_id == "s1" else METADATA)
    if ready:
        mark(directory, READY_MARKER)
    return directory


# -- listing and fetching --------------------------------------------------


def test_only_ready_sessions_are_visible(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox, "s1")
    stage_remote(outbox, "s2", ready=False)  # still copying

    runner = Runner(config, transport, FakeTranscriber())

    assert runner.remote_pending() == ["s1"]


def test_fetch_pulls_a_ready_session(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())

    assert runner.fetch() == ["s1"]
    assert (config.incoming_dir / "s1" / "metadata.json").is_file()
    assert sorted(p.name for p in (config.incoming_dir / "s1").glob("*.opus")) == [
        "10.opus",
        "11.opus",
    ]


def test_fetch_is_idempotent(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())

    assert runner.fetch() == ["s1"]
    assert runner.fetch() == []  # already here


def test_fetching_an_unready_session_is_refused(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox, "s2", ready=False)
    runner = Runner(config, transport, FakeTranscriber())

    with pytest.raises(SyncError, match="not marked READY"):
        runner.fetch("s2")


def test_a_session_that_arrives_broken_is_discarded(setup):
    config, transport, outbox, _ = setup
    directory = outbox / "s1"
    directory.mkdir()
    write_metadata(directory, METADATA)
    mark(directory, READY_MARKER)  # marked, but no audio at all
    runner = Runner(config, transport, FakeTranscriber())

    assert runner.fetch() == []
    assert not (config.incoming_dir / "s1").exists()


# -- transcription ---------------------------------------------------------


def test_transcribe_produces_both_transcript_files(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    fake = FakeTranscriber()
    runner = Runner(config, transport, fake)
    runner.fetch()

    results = runner.transcribe()

    assert len(results) == 1
    result = results[0]
    assert result.transcript_md.is_file()
    assert result.transcript_json.is_file()
    assert result.speaker_count == 2
    markdown = result.transcript_md.read_text(encoding="utf-8")
    assert "Thorin" in markdown and "Elenya" in markdown


def test_character_names_reach_whisper_as_a_prompt(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    fake = FakeTranscriber()
    runner = Runner(config, transport, fake)
    runner.fetch()
    runner.transcribe()

    assert fake.prompts and "Thorin" in fake.prompts[0]
    assert "Dungeons & Dragons" in fake.prompts[0]


def test_offsets_from_metadata_shift_the_timeline(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    result = runner.transcribe()[0]

    payload = json.loads(result.transcript_json.read_text(encoding="utf-8"))
    elenya = [s for s in payload["segments"] if s["speaker"] == "Elenya"]
    assert elenya and elenya[0]["start"] == 12.5


def test_transcribe_skips_already_finished_sessions(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    fake = FakeTranscriber()
    runner = Runner(config, transport, fake)
    runner.fetch()
    runner.transcribe()
    calls_after_first = len(fake.calls)

    runner.transcribe()

    assert len(fake.calls) == calls_after_first


def test_a_failing_session_is_set_aside_not_retried_forever(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, ExplodingTranscriber())
    runner.fetch()

    assert runner.transcribe() == []
    assert (config.failed_dir / "s1").is_dir()
    assert not (config.incoming_dir / "s1").exists()


class UnloadableTranscriber:
    """A machine problem: the model never loads."""

    def load(self):
        raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    def transcribe_file(self, path: Path, language: str, initial_prompt=None):
        raise AssertionError("nothing may be transcribed without a model")


class OutOfMemoryTranscriber:
    def load(self):
        return None

    def transcribe_file(self, path: Path, language: str, initial_prompt=None):
        raise RuntimeError("CUDA failed with error out of memory")


def test_a_model_that_cannot_load_stops_the_run_and_fails_nothing(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, UnloadableTranscriber())
    runner.fetch()

    with pytest.raises(ModelLoadError, match="libcublas"):
        runner.transcribe()

    assert (config.incoming_dir / "s1").is_dir()
    assert not (config.failed_dir / "s1").exists()


def test_running_out_of_gpu_memory_keeps_the_session_queued(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, OutOfMemoryTranscriber())
    runner.fetch()

    with pytest.raises(ModelLoadError, match="int8_float16"):
        runner.transcribe()

    assert (config.incoming_dir / "s1").is_dir()
    assert not (config.failed_dir / "s1").exists()
    assert not (config.outgoing_dir / "s1").exists()


def test_no_waiting_work_never_loads_the_model(setup):
    config, transport, _, _ = setup
    runner = Runner(config, transport, UnloadableTranscriber())

    assert runner.transcribe() == []


# -- pushing back and archiving -------------------------------------------


def test_push_sends_the_transcript_and_marks_it_done(setup):
    config, transport, outbox, inbox = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    runner.transcribe()

    assert runner.push() == ["s1"]

    delivered = inbox / "s1"
    assert (delivered / "transcript.md").is_file()
    assert (delivered / "transcript.json").is_file()
    assert (delivered / DONE_MARKER).is_file()


def test_push_releases_the_audio_held_by_the_recorder(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    runner.transcribe()
    runner.push()

    assert not (outbox / "s1").exists()


def test_keep_remote_leaves_the_recorder_untouched(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    runner.transcribe()
    runner.push(discard_remote=False)

    assert (outbox / "s1").is_dir()


def test_audio_and_transcript_end_up_together_in_the_archive(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.session()

    archived = config.archive_dir / "s1"
    names = sorted(p.name for p in archived.iterdir())
    assert "transcript.md" in names
    assert "10.opus" in names and "11.opus" in names
    assert not any(config.incoming_dir.iterdir())
    assert not any(config.outgoing_dir.iterdir())


def test_session_command_does_the_whole_round_trip(setup):
    config, transport, outbox, inbox = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())

    assert runner.session() == ["s1"]
    assert (inbox / "s1" / DONE_MARKER).is_file()
    assert not (outbox / "s1").exists()


class OrderRecordingTranscriber(FakeTranscriber):
    """Notes, for every session it starts, which transcripts had already been sent."""

    def __init__(self, inbox: Path) -> None:
        super().__init__()
        self.inbox = inbox
        self.sent_when_started: dict[str, list[str]] = {}

    def transcribe_file(self, path: Path, language: str, initial_prompt=None):
        session_id = path.parent.name
        self.sent_when_started.setdefault(
            session_id, sorted(p.parent.name for p in self.inbox.glob("*/DONE"))
        )
        return super().transcribe_file(path, language, initial_prompt)


def test_each_transcript_is_sent_before_the_next_session_starts(setup):
    config, transport, outbox, inbox = setup
    for sid in ("s1", "s2", "s3"):
        stage_remote(outbox, sid)
    transcriber = OrderRecordingTranscriber(inbox)

    assert Runner(config, transport, transcriber).session() == ["s1", "s2", "s3"]
    assert transcriber.sent_when_started == {
        "s1": [],
        "s2": ["s1"],
        "s3": ["s1", "s2"],
    }


def test_a_run_that_dies_midway_has_already_delivered_what_it_finished(setup):
    config, transport, outbox, inbox = setup
    stage_remote(outbox, "s1")
    stage_remote(outbox, "s2")

    class OutOfMemoryOnSecond(FakeTranscriber):
        def transcribe_file(self, path: Path, language: str, initial_prompt=None):
            if path.parent.name == "s2":
                raise RuntimeError("CUDA failed with error out of memory")
            return super().transcribe_file(path, language, initial_prompt)

    runner = Runner(config, transport, OutOfMemoryOnSecond())
    with pytest.raises(ModelLoadError, match="out of GPU memory"):
        runner.session()

    assert (inbox / "s1" / DONE_MARKER).is_file()
    assert not (outbox / "s1").exists()
    assert (config.incoming_dir / "s2").is_dir()
    assert not (inbox / "s2").exists()


def test_a_transcript_left_over_from_an_interrupted_run_is_sent_first(setup):
    config, transport, outbox, inbox = setup
    stage_remote(outbox, "s1")
    stage_remote(outbox, "s2")
    first = Runner(config, transport, FakeTranscriber())
    first.fetch("s1")
    first.transcribe("s1")  # transcribed, never sent

    transcriber = OrderRecordingTranscriber(inbox)
    assert Runner(config, transport, transcriber).session() == ["s1", "s2"]
    assert transcriber.sent_when_started == {"s2": ["s1"]}


def test_a_failed_session_does_not_stop_the_rest_being_sent(setup):
    config, transport, outbox, inbox = setup
    stage_remote(outbox, "s1")
    stage_remote(outbox, "s2")

    class BreaksOnFirst(FakeTranscriber):
        def transcribe_file(self, path: Path, language: str, initial_prompt=None):
            if path.parent.name == "s1":
                raise RuntimeError("corrupt track")
            return super().transcribe_file(path, language, initial_prompt)

    assert Runner(config, transport, BreaksOnFirst()).session() == ["s2"]
    assert (config.failed_dir / "s1").is_dir()
    assert (inbox / "s2" / DONE_MARKER).is_file()


def test_status_reports_the_stage_of_each_session(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()

    states = {s.session_id: s.stage for s in runner.local_sessions()}
    assert states["s1"] == "pulled, not transcribed"

    runner.transcribe()
    states = {s.session_id: s.stage for s in runner.local_sessions()}
    assert states["s1"] == "transcribed, not sent"

    runner.push()
    states = {s.session_id: s.stage for s in runner.local_sessions()}
    assert states["s1"] == "done"


# -- quiet hours -----------------------------------------------------------


def test_work_is_unrestricted_by_default(setup):
    config, transport, _, _ = setup
    assert Runner(config, transport, FakeTranscriber()).may_work() is True


def test_quiet_hours_gate_work_when_enabled(setup):
    from datetime import datetime

    config, transport, _, _ = setup
    config = Config(**{**config.__dict__, "quiet_hours_enabled": True, "timezone_name": "UTC"})
    runner = Runner(config, transport, FakeTranscriber())

    assert runner.may_work(datetime(2026, 5, 1, 3, 0)) is True
    assert runner.may_work(datetime(2026, 5, 1, 14, 0)) is False


# -- an interrupted run must not be mistaken for a finished one ------------


def test_an_empty_outgoing_directory_is_not_treated_as_transcribed(setup):
    """A run that died between creating the directory and writing to it."""
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    (config.outgoing_dir / "s1").mkdir(parents=True)

    results = runner.transcribe()

    assert [r.session_id for r in results] == ["s1"], "the session must still be transcribed"
    assert (config.outgoing_dir / "s1" / "transcript.md").is_file()


def test_push_refuses_a_directory_with_no_transcript(setup):
    """Pushing retires the session on the recorder, so it must not push nothing."""
    config, transport, outbox, inbox = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    (config.outgoing_dir / "s1").mkdir(parents=True)

    assert runner.push() == []
    assert not (inbox / "s1" / DONE_MARKER).exists(), "nothing may be marked done"
    assert (outbox / "s1" / READY_MARKER).is_file(), "the audio must stay collectable"


def test_status_agrees_with_push_about_what_transcribed_means(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)
    runner = Runner(config, transport, FakeTranscriber())
    runner.fetch()
    (config.outgoing_dir / "s1").mkdir(parents=True)

    state = runner.local_sessions()[0]
    assert not state.transcribed
    assert state.stage == "pulled, not transcribed"


def test_a_failed_run_leaves_no_half_written_output_behind(setup):
    config, transport, outbox, _ = setup
    stage_remote(outbox)

    class Exploding:
        def transcribe_file(self, path, language, initial_prompt=None):
            (config.outgoing_dir / "s1").mkdir(parents=True, exist_ok=True)
            raise RuntimeError("model blew up")

    runner = Runner(config, transport, Exploding())
    runner.fetch()

    assert runner.transcribe() == []
    assert not (config.outgoing_dir / "s1").exists(), "a retry must start clean"
    assert (config.failed_dir / "s1").is_dir()
