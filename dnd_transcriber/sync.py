"""Moving sessions between the recorder and this machine.

Always initiated from here. The recorder never reaches into this network, so
there are no inbound ports at home, no port forwarding, and no VPN - just an
outbound SSH connection.

Two transports share one interface: `SshTransport` for the real thing, and
`LocalTransport` so the whole pipeline can be exercised against a pair of
directories with no server involved.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from .contract import DONE_MARKER, READY_MARKER

log = logging.getLogger(__name__)


class SyncError(RuntimeError):
    """Raised when a transfer fails."""


class Transport(Protocol):
    def list_ready(self) -> list[str]: ...

    def pull(self, session_id: str, destination: Path) -> Path: ...

    def push(self, session_id: str, source: Path) -> None: ...

    def discard_remote(self, session_id: str) -> None: ...


class LocalTransport:
    """Directory-to-directory. Used for tests and single-machine setups."""

    def __init__(self, outbox: Path, inbox: Path) -> None:
        self.outbox = Path(outbox)
        self.inbox = Path(inbox)

    def list_ready(self) -> list[str]:
        if not self.outbox.is_dir():
            return []
        return sorted(
            p.name for p in self.outbox.iterdir() if p.is_dir() and (p / READY_MARKER).is_file()
        )

    def pull(self, session_id: str, destination: Path) -> Path:
        source = self.outbox / session_id
        if not (source / READY_MARKER).is_file():
            raise SyncError(f"{session_id} is not marked {READY_MARKER} on the recorder")
        target = Path(destination) / session_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target

    def push(self, session_id: str, source: Path) -> None:
        target = self.inbox / session_id
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(Path(source).iterdir()):
            # The marker goes last, separately.
            if item.name == DONE_MARKER or not item.is_file():
                continue
            shutil.copy2(item, target / item.name)
        (target / DONE_MARKER).write_text("", encoding="utf-8")

    def discard_remote(self, session_id: str) -> None:
        shutil.rmtree(self.outbox / session_id, ignore_errors=True)


class SshTransport:
    """rsync over SSH, pulling from and pushing to the recorder."""

    def __init__(
        self,
        target: str,
        outbox: str,
        inbox: str,
        *,
        port: int = 22,
        key: str = "",
    ) -> None:
        self.target = target
        self.outbox = outbox.rstrip("/")
        self.inbox = inbox.rstrip("/")
        self.port = port
        self.key = key

    # -- plumbing ----------------------------------------------------------

    def _ssh_command(self) -> list[str]:
        command = ["ssh", "-p", str(self.port), "-o", "BatchMode=yes"]
        if self.key:
            command += ["-i", self.key]
        return command

    def _rsh(self) -> str:
        parts = ["ssh", "-p", str(self.port), "-o", "BatchMode=yes"]
        if self.key:
            parts += ["-i", self.key]
        return " ".join(parts)

    def _run(self, command: list[str], what: str) -> subprocess.CompletedProcess:
        log.debug("%s: %s", what, " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SyncError(f"{what} failed: {result.stderr.strip() or result.stdout.strip()}")
        return result

    # -- operations --------------------------------------------------------

    def list_ready(self) -> list[str]:
        """Ask the recorder which sessions are complete and waiting."""
        remote = (
            f"for d in {self.outbox}/*/; do "
            f'[ -f "$d/{READY_MARKER}" ] && basename "$d"; '
            f"done 2>/dev/null || true"
        )
        result = self._run([*self._ssh_command(), self.target, remote], "Listing remote sessions")
        return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())

    def pull(self, session_id: str, destination: Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # Trailing slash on the source copies the directory's contents into a
        # directory of the same name here.
        self._run(
            [
                "rsync",
                "-az",
                "--partial",
                "--info=progress2",
                "-e",
                self._rsh(),
                f"{self.target}:{self.outbox}/{session_id}",
                str(destination) + "/",
            ],
            f"Pulling session {session_id}",
        )
        return destination / session_id

    def push(self, session_id: str, source: Path) -> None:
        source = Path(source)
        target = f"{self.target}:{self.inbox}/{session_id}/"
        self._run(
            [*self._ssh_command(), self.target, f"mkdir -p {self.inbox}/{session_id}"],
            "Creating remote inbox directory",
        )
        # Everything except the marker first...
        files = [str(p) for p in sorted(source.iterdir()) if p.is_file() and p.name != DONE_MARKER]
        if files:
            self._run(
                ["rsync", "-az", "--partial", "-e", self._rsh(), *files, target],
                f"Pushing transcript for {session_id}",
            )
        # ...then the marker, so a partial push is never acted on.
        self._run(
            [*self._ssh_command(), self.target, f"touch {self.inbox}/{session_id}/{DONE_MARKER}"],
            "Marking transcript complete",
        )

    def discard_remote(self, session_id: str) -> None:
        """Delete the collected audio from the recorder.

        Safe because the transcript has already been pushed and the archive
        copy lives here.
        """
        safe_id = session_id.replace("'", "")
        self._run(
            [*self._ssh_command(), self.target, f"rm -rf '{self.outbox}/{safe_id}'"],
            f"Removing collected session {session_id} from the recorder",
        )


def build_transport(config) -> Transport:  # noqa: ANN001 - Config, avoiding a cycle
    """Pick a transport from configuration."""
    if config.is_remote:
        return SshTransport(
            config.ssh_target,
            config.remote_outbox,
            config.remote_inbox,
            port=config.ssh_port,
            key=config.ssh_key,
        )
    return LocalTransport(Path(config.remote_outbox), Path(config.remote_inbox))
