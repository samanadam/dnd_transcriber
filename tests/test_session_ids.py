"""Session ids cross a shell boundary, so they are validated before they do.

The SSH transport interpolates ids into commands that run on the recorder.
The ids come from directory names there, so a hostile one means the recorder
is already compromised - but a name that merely contains a space or a quote
is an ordinary accident, and it should fail here rather than half-run there.
"""

from __future__ import annotations

import pytest

from dnd_transcriber.sync import SyncError, is_safe_session_id, validate_session_id


@pytest.mark.parametrize(
    "session_id",
    [
        "s1",
        "sess-001",
        "2026-08-29T18-00-00",
        "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "kamp_gecesi.2",
        "a" * 128,
    ],
)
def test_ordinary_ids_are_accepted(session_id):
    assert is_safe_session_id(session_id)
    assert validate_session_id(session_id) == session_id


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        "   ",
        "a b",
        "a;rm -rf ~",
        "a$(whoami)",
        "a`id`",
        "a|b",
        "a&b",
        "a'b",
        'a"b',
        "a\nb",
        "../escape",
        "a/b",
        "a\b",
        ".hidden",
        "-startswithdash",
        "a" * 129,
    ],
)
def test_dangerous_or_malformed_ids_are_refused(session_id):
    assert not is_safe_session_id(session_id)
    with pytest.raises(SyncError):
        validate_session_id(session_id)


def test_the_error_names_the_offending_id():
    with pytest.raises(SyncError, match="rm -rf"):
        validate_session_id("x; rm -rf ~")
