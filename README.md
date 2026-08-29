# D&D Transcriber

The offline half of a two-part system. It collects session recordings staged by
[dnd_bot](https://github.com/samanadam/dnd_bot), transcribes them locally with
Whisper, and sends the transcripts back for the bot to post.

**It never talks to Discord.** No token, no gateway, no network access beyond a
single outbound SSH connection to the recorder. That is the point: the machine
with the CPU does not need to reach Discord, and the machine that reaches
Discord does not need a CPU.

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
$EDITOR .env                                      # REMOTE_HOST, REMOTE_USER, WORKSPACE_DIR
```

You also need **ffmpeg** on PATH — it does all audio decoding and splitting.

Set up an SSH key to the recorder so transfers need no password:

```bash
ssh-keygen -t ed25519 -C dnd-transcriber
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@your-recorder
```

The first `dndt run` downloads the Whisper model (~1.5 GB for `medium`) and
needs internet. Everything after that is fully offline.

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
