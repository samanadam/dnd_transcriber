"""Offline transcription worker for D&D session recordings.

Collects sessions staged by the recorder, transcribes them locally with
Whisper, and sends the transcripts back. Never talks to Discord.
"""

__version__ = "1.1.0"
