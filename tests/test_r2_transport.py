"""Collecting from and returning to Cloudflare R2.

The transport is exercised against an in-memory stand-in for the S3 client, so
these tests stay offline like the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_transcriber.contract import DONE_MARKER, READY_MARKER
from dnd_transcriber.sync import R2Transport, SyncError


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, Filename, Bucket, Key):  # noqa: N803 - boto3 spelling
        self.objects[Key] = Path(Filename).read_bytes()

    def put_object(self, Bucket, Key, Body=b""):  # noqa: N803
        self.objects[Key] = Body

    def download_file(self, Bucket, Key, Filename):  # noqa: N803
        Path(Filename).write_bytes(self.objects[Key])

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix="", **kwargs):  # noqa: N803
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys]}


def make_transport(keep_audio: bool = True) -> R2Transport:
    return R2Transport("acct", "key", "secret", "bucket", keep_audio=keep_audio, client=FakeS3())


def seed_outbox(transport: R2Transport, session_id: str, *, ready: bool = True) -> None:
    base = f"outbox/{session_id}/"
    transport.client.objects[f"{base}metadata.json"] = b'{"schema": 1}'
    transport.client.objects[f"{base}10.opus"] = b"audio"
    if ready:
        transport.client.objects[f"{base}{READY_MARKER}"] = b""


def test_list_ready_sees_only_completed_uploads():
    transport = make_transport()
    seed_outbox(transport, "done")
    seed_outbox(transport, "still-copying", ready=False)

    assert transport.list_ready() == ["done"]


def test_pull_reproduces_the_session_directory(tmp_path):
    transport = make_transport()
    seed_outbox(transport, "s1")

    target = transport.pull("s1", tmp_path)

    assert target == tmp_path / "s1"
    assert (target / "metadata.json").read_bytes() == b'{"schema": 1}'
    assert (target / "10.opus").is_file()
    assert (target / READY_MARKER).is_file()


def test_pull_refuses_a_session_that_is_not_marked_ready(tmp_path):
    transport = make_transport()
    seed_outbox(transport, "s1", ready=False)

    with pytest.raises(SyncError, match=READY_MARKER):
        transport.pull("s1", tmp_path)


def test_pull_replaces_a_stale_local_copy(tmp_path):
    transport = make_transport()
    seed_outbox(transport, "s1")
    stale = tmp_path / "s1"
    stale.mkdir()
    (stale / "leftover.txt").write_text("old", encoding="utf-8")

    target = transport.pull("s1", tmp_path)

    assert not (target / "leftover.txt").exists()


def test_push_writes_the_done_marker_last(tmp_path):
    transport = make_transport()
    source = tmp_path / "outgoing"
    source.mkdir()
    (source / "transcript.md").write_text("# Transcript", encoding="utf-8")
    (source / "transcript.json").write_text("{}", encoding="utf-8")

    transport.push("s1", source)

    keys = list(transport.client.objects)
    assert keys[-1] == f"inbox/s1/{DONE_MARKER}"
    assert set(keys) == {
        "inbox/s1/transcript.md",
        "inbox/s1/transcript.json",
        f"inbox/s1/{DONE_MARKER}",
    }


def test_push_never_copies_a_local_done_marker_early(tmp_path):
    """A DONE left in the outgoing directory must not be uploaded as a file."""
    transport = make_transport()
    source = tmp_path / "outgoing"
    source.mkdir()
    (source / "transcript.md").write_text("# Transcript", encoding="utf-8")
    (source / DONE_MARKER).write_text("", encoding="utf-8")

    transport.push("s1", source)

    assert transport.client.objects[f"inbox/s1/{DONE_MARKER}"] == b""


def test_discard_keeps_the_audio_but_retires_the_ready_marker():
    """The archive survives; the work queue does not keep re-offering it."""
    transport = make_transport(keep_audio=True)
    seed_outbox(transport, "s1")

    transport.discard_remote("s1")

    assert transport.list_ready() == [], "a collected session must leave the queue"
    assert transport.client.objects["outbox/s1/10.opus"] == b"audio"
    assert transport.client.objects["outbox/s1/metadata.json"]


def test_discard_removes_the_audio_when_archiving_is_off():
    transport = make_transport(keep_audio=False)
    seed_outbox(transport, "s1")

    transport.discard_remote("s1")

    assert transport.client.objects == {}


@pytest.mark.parametrize("session_id", ["../etc", "a/b", "", "."])
def test_a_session_id_cannot_escape_its_prefix(session_id, tmp_path):
    transport = make_transport()
    with pytest.raises(SyncError):
        transport.pull(session_id, tmp_path)
