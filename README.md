# D&D Transcriber

The offline half of a two-part system. It collects session recordings staged by
[dnd_bot](https://github.com/samanadam/dnd_bot), transcribes them locally with
Whisper, and sends the transcripts back for the bot to post.

**It never talks to Discord.** No token, no gateway, and no inbound connections
at all — it reaches out to collect work and pushes the result back. That is the
point: the machine with the CPU does not need to reach Discord, and the machine
that reaches Discord does not need a CPU.

Two ways to exchange sessions with the recorder, set by `STORAGE_BACKEND`:
a **Cloudflare R2 bucket** both halves talk to (`r2`, recommended — no SSH
account on the recorder, no port forwarding here, and R2 charges no egress on
the gigabytes of audio you pull down), or **SSH** to the recorder (`local`).

```
recorder (VPS)                     this machine
──────────────                     ────────────
records, encodes to Opus
outbox/<id>/ + READY
                       <────────── dndt fetch
                                   transcribe locally
                       ──────────> dndt push
inbox/<id>/ + DONE
posts to Discord
```

Every transfer is initiated from here, so your network needs no inbound ports,
no port forwarding and no VPN.

## Commands

```
dndt list                 what is waiting, on the recorder and here
dndt session              fetch + transcribe + push          <- the usual one
dndt fetch [<id>]         collect ready sessions
dndt run [<id>]           transcribe what has been collected
dndt push [<id>]          send transcripts back
dndt status               current configuration
```

`dndt session` after game night is normally all you need. The individual steps
exist for when something goes wrong, and every one of them is safe to re-run:
work already done is skipped rather than repeated.

## Setup

```bash
git clone https://github.com/samanadam/dnd-transcriber.git
cd dnd-transcriber
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                  # provides the `dndt` command

cp .env.example .env
$EDITOR .env                                      # STORAGE_BACKEND, WORKSPACE_DIR
```

You also need **ffmpeg** on PATH — it does all audio decoding and splitting.

### Reaching the recorder

**Cloudflare R2 (`STORAGE_BACKEND=r2`)** — put the same four values the recorder
uses into `.env` here: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_BUCKET`. Same bucket, both ends. Nothing else to
configure and nothing to open on this network.

R2 doubles as the permanent audio archive, so a collected session is left in the
bucket rather than deleted. Set `R2_KEEP_AUDIO=false` to delete each session
from R2 once it is transcribed — the `archive/` directory here is then your only
copy.

**SSH (`STORAGE_BACKEND=local`)** — set `REMOTE_HOST` and `REMOTE_USER`, then
set up a key so transfers need no password:

```bash
ssh-keygen -t ed25519 -C dnd-transcriber
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-recorder
```

Leaving `REMOTE_HOST` empty uses plain local directories, which is how the whole
pipeline is exercised on one machine.

The first `dndt run` downloads the Whisper model (~1.5 GB for `medium`) and
needs internet. Everything after that is fully offline.

## Running it with Docker

Recommended on a server. The image pins ffmpeg and the Python environment
together, so what you tested is what runs, and the fixed container name keeps
two runs from colliding on the same workspace.

```bash
cp .env.example .env
$EDITOR .env                 # STORAGE_BACKEND and the R2 credentials
docker compose build
docker compose run --rm transcriber status    # check the configuration
docker compose up                             # fetch, transcribe, push
```

Use `docker compose up` rather than `run` for real work. `up` honours the
container name, so a second invocation refuses to start while the first is
still transcribing. `run --rm` is for one-off commands like `status` and
`list`, which are safe to do concurrently.

Any subcommand works as an argument:

```bash
docker compose run --rm transcriber list
docker compose run --rm transcriber fetch
```

Two named volumes matter and neither is optional:

- `workspace` holds `archive/`, your permanent copy of the audio. Losing this
  volume loses your recordings. Back it up like any other data volume.
- `models` holds the Whisper model. Without it, every run re-downloads ~1.5 GB.

Pre-warm the model once so the first real session does not wait on a download:

```bash
docker compose run --rm transcriber run
```

The container runs as uid 1000, not root. If you swap a named volume for a bind
mount, `chown 1000:1000` the host directory or the container cannot write to it.

Scheduling it is a systemd timer or a cron entry calling `docker compose up`;
the container name is the lock, so overlapping invocations are already handled.

The production image carries no test tooling. To run the suite against the same
base, build the `dev` target:

```bash
docker build --target dev -t dnd-transcriber:dev .
docker run --rm dnd-transcriber:dev
```

## Workspace

```
workspace/
  incoming/<id>/     pulled from the recorder, not yet transcribed
  outgoing/<id>/     transcribed, not yet sent back
  archive/<id>/      done: audio and transcript together, kept permanently
  failed/<id>/       could not be transcribed; kept for inspection
```

`archive/` is your permanent library. The recorder deletes its copy of the audio
once a transcript comes back, so this machine holds the only long-term copy —
back it up accordingly.

## Getting good Turkish transcripts

Four things do the heavy lifting, all on by default:

1. **Character names are fed to Whisper as a decode prompt**, taken from the
   `metadata.json` the recorder sends. Invented names are what a transcript gets
   most wrong, so run `/character set` in Discord for every player — it is the
   single biggest accuracy win.
2. **`condition_on_previous_text` is off.** Each speaker has their own track, so
   most of any one file is silence while others talk; carrying decode context
   across those gaps is exactly what sends Whisper into repetition loops.
3. **Hallucination filtering** removes the subtitle boilerplate Whisper emits
   over silence — `Altyazı M.K.`, `Abone olmayı unutmayın`, `Thanks for
   watching` — plus low-confidence noise and stuck repeats.
4. **Temperature fallback and VAD**: a segment that decodes badly is retried
   rather than emitted as garbage, and dead air is trimmed first.

`medium` is the right model for CPU. `large-v3` is better at Turkish but far too
slow; `small` is noticeably worse at Turkish morphology. If transcripts come out
mangled, check microphone quality before changing models — Whisper is much more
sensitive to a bad mic and room echo than to model size.

## Memory and time

faster-whisper loads whatever file it is given **entirely into memory** before
streaming segments out. So each track is split into `TRANSCRIBE_CHUNK_MINUTES`
pieces first, which keeps peak memory near 350 MB regardless of session length
instead of growing to gigabytes. Chunk boundaries are snapped to the quietest
moment near the target time so a cut does not land mid-word, and timestamps are
re-based onto the full session timeline afterwards.

Expect transcription to run roughly 2–5× slower than realtime on CPU with
`medium`. A four-hour session can take most of a night. Measure your own machine
before trusting any estimate.

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
black --check .
```

Tests use a fake transcriber and local directories, so they need neither a model
nor an SSH server. ffmpeg is required for the chunking tests; they skip without
it.

## The contract

`dnd_transcriber/contract.py` is **duplicated verbatim** in the recorder
repository. It defines the exchange format:

```
outbox/<session_id>/          recorder -> here
    metadata.json             speaker labels, offsets, timezone, language
    <user_id>.opus            one track per speaker
    READY                     written last

inbox/<session_id>/           here -> recorder
    transcript.md
    transcript.json
    DONE                      written last
```

Marker files are always written last, so a directory still being copied is
invisible to the other side and an interrupted transfer is harmless. The schema
carries a version number: if the two repos drift apart, the mismatch fails
loudly instead of being silently misread. **Change `contract.py` in both repos
in the same commit.**

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Eren YANGİL.
