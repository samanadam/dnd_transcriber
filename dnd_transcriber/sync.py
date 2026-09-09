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


class R2Transport:
    """Cloudflare R2 as the exchange medium.

    Same four operations, same marker-last ordering, no SSH. Both halves reach
    outbound to Cloudflare and neither needs to be reachable, so this machine
    can sit behind a home router with nothing forwarded - and the recorder does
    not need a Unix account for us to log into.

    The object layout mirrors the directory layout exactly:

        outbox/<session_id>/metadata.json, <user_id>.opus, READY
        inbox/<session_id>/transcript.md, transcript.json, DONE
    """

    OUTBOX_PREFIX = "outbox"
    INBOX_PREFIX = "inbox"

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        *,
        keep_audio: bool = True,
        client=None,  # noqa: ANN001 - an S3 client, injected by tests
    ) -> None:
        self.bucket = bucket
        self.keep_audio = keep_audio
        if client is not None:
            self.client = client
            return

        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise SyncError("STORAGE_BACKEND=r2 needs boto3. pip install boto3.") from exc

        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",  # R2 ignores it; the SigV4 signer requires one
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    # -- plumbing ----------------------------------------------------------

    @staticmethod
    def _prefix(prefix: str, session_id: str) -> str:
        session_id = str(session_id)
        if not session_id or "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
            raise SyncError(f"Not a usable session id: {session_id!r}")
        return f"{prefix}/{session_id}/"

    def _keys(self, prefix: str) -> list[str]:
        return sorted(self._sizes(prefix))

    def _sizes(self, prefix: str) -> dict[str, int]:
        """Every key under a prefix with its stored size, in bytes."""
        sizes: dict[str, int] = {}
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                sizes[item["Key"]] = int(item.get("Size", 0))
            token = response.get("NextContinuationToken")
            if not response.get("IsTruncated") or not token:
                break
        return sizes

    # -- operations --------------------------------------------------------

    def list_ready(self) -> list[str]:
        suffix = f"/{READY_MARKER}"
        found = []
        for key in self._keys(f"{self.OUTBOX_PREFIX}/"):
            if key.endswith(suffix):
                session_id = key[len(self.OUTBOX_PREFIX) + 1 : -len(suffix)]
                if session_id and "/" not in session_id:
                    found.append(session_id)
        return sorted(found)

    def pull(self, session_id: str, destination: Path) -> Path:
        base = self._prefix(self.OUTBOX_PREFIX, session_id)
        keys = self._keys(base)
        if f"{base}{READY_MARKER}" not in keys:
            raise SyncError(f"{session_id} is not marked {READY_MARKER} in R2")

        target = Path(destination) / session_id
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for key in keys:
            name = key[len(base) :]
            if not name or "/" in name:
                continue
            self.client.download_file(Bucket=self.bucket, Key=key, Filename=str(target / name))
        return target

    def push(self, session_id: str, source: Path) -> None:
        """Send a transcript back, and prove it arrived before returning.

        The caller retires the session's READY marker as soon as this returns,
        which takes it out of the work queue for good. "The uploads did not
        raise" is not a strong enough guarantee for that, so this re-reads the
        prefix and fails unless every file is present at its own size. A push
        that only half succeeded then stays collectable instead of vanishing.
        """
        base = self._prefix(self.INBOX_PREFIX, session_id)
        files = [p for p in sorted(Path(source).iterdir()) if p.is_file() and p.name != DONE_MARKER]
        for path in files:
            self.client.upload_file(
                Filename=str(path), Bucket=self.bucket, Key=f"{base}{path.name}"
            )
        # The marker last, so a partial push is never acted on.
        self.client.put_object(Bucket=self.bucket, Key=f"{base}{DONE_MARKER}", Body=b"")

        stored = self._sizes(base)
        if f"{base}{DONE_MARKER}" not in stored:
            raise SyncError(f"Pushed {session_id} but R2 has no {DONE_MARKER}")
        for path in files:
            key = f"{base}{path.name}"
            if key not in stored:
                raise SyncError(f"Pushed {session_id} but {path.name} is missing from R2")
            if stored[key] != path.stat().st_size:
                raise SyncError(
                    f"Pushed {session_id} but {path.name} is {stored[key]} bytes in R2, "
                    f"{path.stat().st_size} here"
                )

    def discard_remote(self, session_id: str) -> None:
        """Release a collected session.

        With `keep_audio` on (the default) R2 is the long-term archive, so the
        audio stays and only the READY marker is retired. That distinction
        matters: the marker is what makes a session appear in `list_ready`, so
        leaving it in place would keep every session ever recorded in the work
        queue forever - and would re-pull the whole bucket if this machine's
        workspace were ever rebuilt.

        With `keep_audio` off the objects go entirely, and the workspace
        archive here becomes the only copy.
        """
        base = self._prefix(self.OUTBOX_PREFIX, session_id)
        if self.keep_audio:
            self.client.delete_object(Bucket=self.bucket, Key=f"{base}{READY_MARKER}")
            log.info("Collected %s; audio stays in R2 as the archive", session_id)
            return
        for key in self._keys(base):
            self.client.delete_object(Bucket=self.bucket, Key=key)


def build_transport(config) -> Transport:  # noqa: ANN001 - Config, avoiding a cycle
    """Pick a transport from configuration."""
    if config.uses_r2:
        return R2Transport(
            config.r2_account_id,
            config.r2_access_key_id,
            config.r2_secret_access_key,
            config.r2_bucket,
            keep_audio=config.r2_keep_audio,
        )
    if config.is_remote:
        return SshTransport(
            config.ssh_target,
            config.remote_outbox,
            config.remote_inbox,
            port=config.ssh_port,
            key=config.ssh_key,
        )
    return LocalTransport(Path(config.remote_outbox), Path(config.remote_inbox))
