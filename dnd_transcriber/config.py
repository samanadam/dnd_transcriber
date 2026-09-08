"""Environment-driven configuration for the transcriber.

Deliberately smaller than the recorder's: no Discord, no database, no
retention policy. This process reads audio and writes text.
"""

from __future__ import annotations
# I just needed a commit
import os
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or ""


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_hhmm(value: str, name: str) -> dtime:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ConfigError(f"{name} must be HH:MM, got {value!r}")
    try:
        return dtime(hour=int(parts[0]), minute=int(parts[1]))
    except ValueError as exc:
        raise ConfigError(f"{name} must be HH:MM, got {value!r}") from exc


@dataclass(frozen=True)
class Config:
    # -- where work happens ------------------------------------------------
    workspace: Path = Path("./workspace")

    # -- the recorder we collect from --------------------------------------
    # Empty host means "local directories", which is how the whole pipeline is
    # tested without a server.
    remote_host: str = ""
    remote_user: str = ""
    remote_outbox: str = "/srv/dnd-bot-data/outbox"
    remote_inbox: str = "/srv/dnd-bot-data/inbox"
    ssh_port: int = 22
    ssh_key: str = ""

    # -- or Cloudflare R2 --------------------------------------------------
    # With "r2" neither machine needs to reach the other: both talk outbound to
    # Cloudflare. No SSH account on the recorder, no port forwarding here.
    storage_backend: str = "local"
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    # R2 doubles as the long-term audio archive, so collecting a session leaves
    # its objects in place rather than deleting them from the recorder's outbox.
    r2_keep_audio: bool = True

    # -- Whisper -----------------------------------------------------------
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_condition_on_previous_text: bool = False
    whisper_vad_min_silence_ms: int = 500
    filter_hallucinations: bool = True
    transcribe_chunk_minutes: int = 10
    model_cache_dir: str = ""

    # -- when we are willing to work ---------------------------------------
    quiet_hours_enabled: bool = False
    quiet_hours_start: dtime = field(default_factory=lambda: dtime(0, 0))
    quiet_hours_end: dtime = field(default_factory=lambda: dtime(8, 0))
    timezone_name: str = "Europe/Istanbul"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def incoming_dir(self) -> Path:
        """Sessions pulled from the recorder, waiting to be transcribed."""
        return self.workspace / "incoming"

    @property
    def outgoing_dir(self) -> Path:
        """Finished transcripts, waiting to be sent back."""
        return self.workspace / "outgoing"

    @property
    def archive_dir(self) -> Path:
        """Sessions fully handled. This is your long-term audio archive."""
        return self.workspace / "archive"

    @property
    def failed_dir(self) -> Path:
        """Sessions that could not be transcribed, kept for inspection."""
        return self.workspace / "failed"

    @property
    def uses_r2(self) -> bool:
        return self.storage_backend == "r2"

    @property
    def is_remote(self) -> bool:
        return bool(self.remote_host)

    @property
    def ssh_target(self) -> str:
        return f"{self.remote_user}@{self.remote_host}" if self.remote_user else self.remote_host

    def ensure_dirs(self) -> None:
        for path in (
            self.workspace,
            self.incoming_dir,
            self.outgoing_dir,
            self.archive_dir,
            self.failed_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - optional
        pass

    timezone_name = _get("TIMEZONE", "Europe/Istanbul")
    try:
        ZoneInfo(timezone_name)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Unknown TIMEZONE {timezone_name!r}") from exc

    storage_backend = _get("STORAGE_BACKEND", "local").lower()
    if storage_backend not in {"local", "r2"}:
        raise ConfigError(f"STORAGE_BACKEND must be 'local' or 'r2', got {storage_backend!r}")
    r2_settings = {
        "R2_ACCOUNT_ID": _get("R2_ACCOUNT_ID"),
        "R2_ACCESS_KEY_ID": _get("R2_ACCESS_KEY_ID"),
        "R2_SECRET_ACCESS_KEY": _get("R2_SECRET_ACCESS_KEY"),
        "R2_BUCKET": _get("R2_BUCKET"),
    }
    if storage_backend == "r2":
        missing = sorted(name for name, value in r2_settings.items() if not value)
        if missing:
            raise ConfigError(
                f"STORAGE_BACKEND=r2 needs {', '.join(missing)}. "
                "Use the same bucket the recorder writes to."
            )

    return Config(
        workspace=Path(_get("WORKSPACE_DIR", "./workspace")),
        remote_host=_get("REMOTE_HOST"),
        remote_user=_get("REMOTE_USER"),
        remote_outbox=_get("REMOTE_OUTBOX", "/srv/dnd-bot-data/outbox"),
        remote_inbox=_get("REMOTE_INBOX", "/srv/dnd-bot-data/inbox"),
        ssh_port=_get_int("SSH_PORT", 22),
        ssh_key=_get("SSH_KEY"),
        storage_backend=storage_backend,
        r2_account_id=r2_settings["R2_ACCOUNT_ID"],
        r2_access_key_id=r2_settings["R2_ACCESS_KEY_ID"],
        r2_secret_access_key=r2_settings["R2_SECRET_ACCESS_KEY"],
        r2_bucket=r2_settings["R2_BUCKET"],
        r2_keep_audio=_get_bool("R2_KEEP_AUDIO", True),
        whisper_model=_get("WHISPER_MODEL", "medium"),
        whisper_device=_get("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_get("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_beam_size=_get_int("WHISPER_BEAM_SIZE", 5),
        whisper_condition_on_previous_text=_get_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False),
        whisper_vad_min_silence_ms=_get_int("WHISPER_VAD_MIN_SILENCE_MS", 500),
        filter_hallucinations=_get_bool("FILTER_HALLUCINATIONS", True),
        transcribe_chunk_minutes=_get_int("TRANSCRIBE_CHUNK_MINUTES", 10),
        model_cache_dir=_get("WHISPER_CACHE_DIR"),
        quiet_hours_enabled=_get_bool("QUIET_HOURS_ENABLED", False),
        quiet_hours_start=parse_hhmm(_get("QUIET_HOURS_START", "00:00"), "QUIET_HOURS_START"),
        quiet_hours_end=parse_hhmm(_get("QUIET_HOURS_END", "08:00"), "QUIET_HOURS_END"),
        timezone_name=timezone_name,
    )
