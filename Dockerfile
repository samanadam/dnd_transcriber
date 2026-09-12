# The transcriber is a batch job, not a service: it starts, drains the work
# that is waiting, and exits. Everything here is shaped around that.
#
# Targets:
#   gpu   (default)  CUDA libraries included; runs on the CPU too if no GPU
#                    reaches the container, so it is never a dead end.
#   cpu              no CUDA libraries, about a gigabyte smaller.
#   dev              adds pytest and the suite so CI tests the real image.

# ---------------------------------------------------------------- base ----
FROM python:3.11-slim-bookworm AS base

# ffmpeg does all audio decoding and splitting - the chunking code shells out
# to it for every track. Without it the tests skip rather than fail, so pin it
# in the image instead of inheriting whatever the host happens to have.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Unbuffered so `docker logs` shows progress during a long transcription
# instead of nothing followed by everything.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements before source: this layer pulls roughly a gigabyte of
# ctranslate2 and onnxruntime wheels and must survive a source-only change.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Both of these must be volumes in production. The workspace holds the
# permanent audio archive, and the model cache holds gigabytes that would
# otherwise be re-downloaded on every single run.
ENV WORKSPACE_DIR=/data/workspace \
    WHISPER_CACHE_DIR=/data/models \
    HF_HOME=/data/models

# Non-root. The uid is fixed so a bind-mounted host directory can be chowned
# to match; named volumes inherit these permissions on first creation.
RUN useradd --create-home --uid 1000 dndt \
    && mkdir -p /data/workspace /data/models \
    && chown -R dndt:dndt /data /app

# ----------------------------------------------------------------- dev ----
# Built with `--target dev`. Never the default, so no shipped image carries
# pytest or the test suite.
FROM base AS dev
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY pyproject.toml README.md LICENSE ./
COPY dnd_transcriber ./dnd_transcriber
COPY tests ./tests
RUN pip install --no-deps -e . && chown -R dndt:dndt /app
USER dndt
ENTRYPOINT []
CMD ["pytest", "-q"]

# ----------------------------------------------------------------- cpu ----
FROM base AS cpu
COPY pyproject.toml README.md LICENSE ./
COPY dnd_transcriber ./dnd_transcriber
RUN pip install --no-deps -e . && chown -R dndt:dndt /app
USER dndt
ENTRYPOINT ["dndt"]
CMD ["session"]

# ------------------------------------------------------------ gpu-libs ----
# Its own stage so the CUDA layer, well over a gigabyte, is cached separately
# and a source change never re-downloads it.
FROM base AS gpu-libs
COPY requirements-gpu.txt ./
RUN pip install -r requirements-gpu.txt

# ctranslate2 finds cuBLAS and cuDNN through the loader path. The pip wheels
# put them under site-packages rather than a system library directory.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
RUN test -e /usr/local/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12 \
    && test -e /usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9

# Tells the NVIDIA Container Toolkit which driver libraries to mount in.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ----------------------------------------------------------------- gpu ----
# Last stage, so a plain `docker build` produces the GPU image.
FROM gpu-libs AS gpu
COPY pyproject.toml README.md LICENSE ./
COPY dnd_transcriber ./dnd_transcriber
RUN pip install --no-deps -e . && chown -R dndt:dndt /app
USER dndt

# `docker run <image>` with no arguments does the thing you actually want:
# collect, transcribe, send back. Override with list, status, fetch, run, push.
ENTRYPOINT ["dndt"]
CMD ["session"]
