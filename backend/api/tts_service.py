from __future__ import annotations

import re
from typing import AsyncIterator

import edge_tts

# Microsoft Edge neural voices. These are the highest-quality free voices available
# without an API key — same engine the Edge browser uses for read-aloud.
# Indian English defaults match the audience; users can pick a different voice via the
# `voice` query param.
DEFAULT_VOICE = "en-IN-NeerjaNeural"

# Stripped before sending to TTS so citation markers and bullet glyphs don't get read
# as "bracket four bracket" or pause awkwardly between bullets.
_CITATION_RE = re.compile(r"\[\d+\]")
_BULLET_RE = re.compile(r"[•·]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    s = _CITATION_RE.sub("", text)
    s = _BULLET_RE.sub(",", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


async def synthesize_mp3(text: str, voice: str = DEFAULT_VOICE) -> AsyncIterator[bytes]:
    """Stream MP3 audio chunks for `text` using Microsoft Edge's neural TTS.

    Free, no API key. Requires outbound internet (talks to speech.platform.bing.com).
    """
    cleaned = clean_for_speech(text)
    if not cleaned:
        return
    communicator = edge_tts.Communicate(cleaned, voice)
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            yield chunk["data"]
