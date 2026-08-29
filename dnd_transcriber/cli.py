"""`dndt` - the operator's interface.

Everything is a deliberate, manual action. Transcription is slow and noisy, so
it happens when you decide it should, not on a timer you forgot about.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import Config, ConfigError, load_config
from .runner import Runner
from .sync import SyncError, build_transport
from .timeutil import format_duration


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else os.environ.get("LOG_LEVEL", "INFO").upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def build_runner(config: Config) -> Runner:
    config.ensure_dirs()
    return Runner(config, build_transport(config))


# -- commands --------------------------------------------------------------


def cmd_list(runner: Runner, _args: argparse.Namespace) -> int:
    remote = runner.remote_pending()
    print(f"Waiting on the recorder: {len(remote)}")
    for session_id in remote:
        print(f"  {session_id}")

    local = runner.local_sessions()
    print(f"\nHere: {len(local)}")
    for state in local:
        print(f"  {state.session_id:38} {state.stage}")
    return 0


def cmd_fetch(runner: Runner, args: argparse.Namespace) -> int:
    pulled = runner.fetch(args.session_id)
    print(f"Fetched {len(pulled)} session(s)." if pulled else "Nothing new to fetch.")
    for session_id in pulled:
        print(f"  {session_id}")
    return 0


def cmd_run(runner: Runner, args: argparse.Namespace) -> int:
    if not runner.may_work():
        print("Outside the configured quiet hours; not starting. Use --force to override.")
        if not args.force:
            return 1
    results = runner.transcribe(args.session_id)
    if not results:
        print("Nothing to transcribe.")
        return 0
    for result in results:
        print(
            f"{result.session_id}: {result.speaker_count} speaker(s), "
            f"{result.word_count} words, {format_duration(result.duration_seconds)}"
        )
        for warning in result.warnings:
            print(f"    warning: {warning}")
    return 0


def cmd_push(runner: Runner, args: argparse.Namespace) -> int:
    pushed = runner.push(args.session_id, discard_remote=not args.keep_remote)
    print(f"Sent {len(pushed)} transcript(s)." if pushed else "Nothing to send.")
    for session_id in pushed:
        print(f"  {session_id}")
    return 0


def cmd_session(runner: Runner, args: argparse.Namespace) -> int:
    if not runner.may_work() and not args.force:
        print("Outside the configured quiet hours; not starting. Use --force to override.")
        return 1
    done = runner.session(args.session_id)
    print(f"Completed {len(done)} session(s)." if done else "Nothing to do.")
    for session_id in done:
        print(f"  {session_id}")
    return 0


def cmd_status(runner: Runner, _args: argparse.Namespace) -> int:
    config = runner.config
    where = config.ssh_target if config.is_remote else "local directories"
    print(f"Recorder:  {where}")
    print(f"Outbox:    {config.remote_outbox}")
    print(f"Workspace: {config.workspace}")
    print(
        f"Model:     {config.whisper_model} ({config.whisper_device}/{config.whisper_compute_type})"
    )
    print(f"Chunking:  {config.transcribe_chunk_minutes} min")
    if config.quiet_hours_enabled:
        print(
            f"Working hours: {config.quiet_hours_start:%H:%M}-{config.quiet_hours_end:%H:%M} "
            f"{config.timezone_name} (currently {'open' if runner.may_work() else 'closed'})"
        )
    else:
        print("Working hours: unrestricted")
    return 0


COMMANDS = {
    "list": cmd_list,
    "fetch": cmd_fetch,
    "run": cmd_run,
    "push": cmd_push,
    "session": cmd_session,
    "status": cmd_status,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dndt",
        description="Transcribe D&D sessions recorded by the Discord bot.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="show what is waiting, here and on the recorder")
    subparsers.add_parser("status", help="show the current configuration")

    fetch = subparsers.add_parser("fetch", help="collect ready sessions from the recorder")
    fetch.add_argument("session_id", nargs="?", help="one session, instead of all ready ones")

    run = subparsers.add_parser("run", help="transcribe collected sessions")
    run.add_argument("session_id", nargs="?")
    run.add_argument("--force", action="store_true", help="ignore configured quiet hours")

    push = subparsers.add_parser("push", help="send transcripts back to the recorder")
    push.add_argument("session_id", nargs="?")
    push.add_argument(
        "--keep-remote",
        action="store_true",
        help="leave the audio on the recorder instead of releasing it",
    )

    session = subparsers.add_parser("session", help="fetch, transcribe and send back")
    session.add_argument("session_id", nargs="?")
    session.add_argument("--force", action="store_true", help="ignore configured quiet hours")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_config()
        runner = build_runner(config)
        return COMMANDS[args.command](runner, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except SyncError as exc:
        print(f"Transfer failed: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
