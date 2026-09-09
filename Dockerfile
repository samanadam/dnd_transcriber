# The transcriber is a batch job, not a service: it starts, drains the work
# that is waiting, and exits. Everything here is shaped around that.
#
# Two targets. The default is the production image and carries no test
# tooling. `--target dev` adds pytest and the suite so CI can exercise the
# real image rather than a lookalike.

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
# permanent audio archive, and the model cache holds ~1.5 GB that would
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
# Built with `--target dev`. Never the default, so the production image never
# carries pytest or the test suite.
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

# ------------------------------------------------------------ production ---
# Last stage, so a plain `docker build` produces this one.
FROM base AS production
COPY pyproject.toml README.md LICENSE ./
COPY dnd_transcriber ./dnd_transcriber
RUN pip install --no-deps -e . && chown -R dndt:dndt /app
USER dndt

# `docker run <image>` with no arguments does the thing you actually want:
# collect, transcribe, send back. Override with list, status, fetch, run, push.
ENTRYPOINT ["dndt"]
CMD ["session"]
