"""GPU by default when there is one, CPU otherwise, and loud when forced wrong."""

from __future__ import annotations

import pytest

from dnd_transcriber import device as device_module
from dnd_transcriber.config import ConfigError, load_config
from dnd_transcriber.device import DeviceError, resolve_placement


def gpus(count: int):
    return lambda: count


def test_auto_uses_the_gpu_when_one_is_visible():
    placement = resolve_placement("auto", "auto", count_gpus=gpus(1))
    assert (placement.device, placement.compute_type) == ("cuda", "float16")


def test_auto_falls_back_to_the_cpu_without_one():
    placement = resolve_placement("auto", "auto", count_gpus=gpus(0))
    assert (placement.device, placement.compute_type) == ("cpu", "int8")


def test_cpu_never_probes_for_a_gpu():
    def probe():
        raise AssertionError("cpu must not touch CUDA")

    placement = resolve_placement("cpu", "auto", count_gpus=probe)
    assert (placement.device, placement.compute_type) == ("cpu", "int8")


def test_forcing_cuda_without_a_gpu_is_a_configuration_error():
    with pytest.raises(ConfigError, match="NVIDIA Container Toolkit"):
        resolve_placement("cuda", "auto", count_gpus=gpus(0))


def test_an_explicit_compute_type_wins():
    placement = resolve_placement("auto", "int8_float16", count_gpus=gpus(1))
    assert (placement.device, placement.compute_type) == ("cuda", "int8_float16")


def test_values_are_case_insensitive():
    placement = resolve_placement(" CUDA ", "FLOAT16", count_gpus=gpus(2))
    assert (placement.device, placement.compute_type, placement.gpu_count) == (
        "cuda",
        "float16",
        2,
    )


@pytest.mark.parametrize("bad", ["gpu", "nvidia", "cuda:0", "mps"])
def test_unknown_devices_are_refused(bad):
    with pytest.raises(DeviceError, match="WHISPER_DEVICE"):
        resolve_placement(bad, "auto", count_gpus=gpus(1))


def test_the_description_says_why():
    assert "1 GPU visible" in resolve_placement("auto", "auto", count_gpus=gpus(1)).describe("auto")
    assert "no GPU visible" in resolve_placement("auto", "auto", count_gpus=gpus(0)).describe(
        "auto"
    )
    assert "set explicitly" in resolve_placement("cpu", "auto").describe("cpu")


def test_a_failing_probe_counts_as_no_gpu(monkeypatch):
    ctranslate2 = pytest.importorskip("ctranslate2")

    def broken():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", broken)
    assert device_module.cuda_device_count() == 0


def test_config_rejects_an_unknown_device(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "gpu")
    with pytest.raises(ConfigError, match="WHISPER_DEVICE"):
        load_config()


def test_config_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    config = load_config()
    assert (config.whisper_device, config.whisper_compute_type) == ("auto", "auto")
