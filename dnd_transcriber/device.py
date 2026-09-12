"""Choosing where Whisper runs.

GPU is the default whenever one is visible, because on any CUDA card a session
transcribes several times faster than on the CPU. It is not a requirement: the
same build falls back to the CPU on a machine without one, so nobody needs a
separate configuration just to try it.

"Visible" is decided by ctranslate2 itself, so inside a container it reflects
what the container can reach, not what the host has. A host GPU the container
was never given counts as no GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .config import WHISPER_DEVICES, ConfigError

log = logging.getLogger(__name__)


class DeviceError(ConfigError):
    """Raised when the requested device cannot be used on this machine."""


@dataclass(frozen=True)
class Placement:
    """Where the model will actually run, after `auto` has been resolved."""

    device: str
    compute_type: str
    gpu_count: int

    def describe(self, requested: str) -> str:
        if requested == "auto":
            seen = f"{self.gpu_count} GPU visible" if self.gpu_count else "no GPU visible"
            why = f"auto, {seen}"
        else:
            why = "set explicitly"
        return f"{self.device}/{self.compute_type} ({why})"


def cuda_device_count() -> int:
    """How many CUDA devices ctranslate2 can use. No driver means zero."""
    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - a missing driver is an answer, not a crash
        return 0


def resolve_placement(
    requested: str,
    compute_type: str,
    *,
    count_gpus: Callable[[], int] = cuda_device_count,
) -> Placement:
    """Turn WHISPER_DEVICE and WHISPER_COMPUTE_TYPE into a concrete choice."""
    device = (requested or "auto").strip().lower()
    if device not in WHISPER_DEVICES:
        raise DeviceError(
            f"WHISPER_DEVICE must be one of {', '.join(WHISPER_DEVICES)}, got {requested!r}"
        )

    gpus = 0 if device == "cpu" else count_gpus()
    if device == "cuda" and gpus == 0:
        raise DeviceError(
            "WHISPER_DEVICE=cuda but no CUDA device is visible. The host needs the "
            "NVIDIA driver and, under Docker, the NVIDIA Container Toolkit plus a GPU "
            "reservation for the container. Set WHISPER_DEVICE=auto or cpu to run "
            "without one."
        )
    if device == "auto":
        device = "cuda" if gpus else "cpu"

    # float16 is the natural type on a GPU and int8 on a CPU. Explicit values
    # win, which is how a 4 GB card fits a larger model with int8_float16.
    chosen = (compute_type or "auto").strip().lower()
    if chosen == "auto":
        chosen = "float16" if device == "cuda" else "int8"

    return Placement(device=device, compute_type=chosen, gpu_count=gpus)
