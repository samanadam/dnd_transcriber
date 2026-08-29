"""Environment-driven configuration for the transcriber.

Deliberately smaller than the recorder's: no Discord, no database, no
retention policy. This process reads audio and writes text.
"""

from __future__ import annotations

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

    return Config(
        workspace=Path(_get("WORKSPACE_DIR", "./workspace")),
        remote_host=_get("REMOTE_HOST"),
        remote_user=_get("REMOTE_USER"),
        remote_outbox=_get("REMOTE_OUTBOX", "/srv/dnd-bot-data/outbox"),
        remote_inbox=_get("REMOTE_INBOX", "/srv/dnd-bot-data/inbox"),
        ssh_port=_get_int("SSH_PORT", 22),
        ssh_key=_get("SSH_KEY"),
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
