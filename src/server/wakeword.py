"""
Wakeword detection built on top of the existing STT engine.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
from loguru import logger


def normalize_text(text: str) -> str:
    """Normalize text for wakeword matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def strip_wakeword(text: str, phrase: str) -> str:
    """Remove the configured wakeword phrase from the beginning of a transcript."""
    pattern = re.compile(rf"^\s*{re.escape(phrase)}[\s,.:;!?-]*", re.IGNORECASE)
    stripped = pattern.sub("", text, count=1).strip()
    if stripped:
        return stripped

    fallback = re.compile(rf"\b{re.escape(phrase)}\b[\s,.:;!?-]*", re.IGNORECASE)
    return fallback.sub("", text, count=1).strip()


class WakeWordDetector:
    """Detect a wakeword from rolling audio windows using the existing STT engine."""

    def __init__(
        self,
        stt,
        phrase: str,
        sample_rate: int = 16000,
        window_seconds: float = 2.4,
        min_audio_seconds: float = 1.2,
        detect_interval_seconds: float = 0.8,
        cooldown_seconds: float = 2.0,
        preroll_seconds: float = 0.8,
    ):
        self.stt = stt
        self.sample_rate = sample_rate
        self.window_samples = max(int(window_seconds * sample_rate), sample_rate)
        self.min_audio_samples = max(int(min_audio_seconds * sample_rate), 1)
        self.detect_interval_seconds = detect_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        self.preroll_samples = max(int(preroll_seconds * sample_rate), 0)
        self.phrase = phrase.strip()
        self.normalized_phrase = normalize_text(self.phrase)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_check_at = 0.0
        self._last_trigger_at = 0.0

    def reset(self) -> None:
        """Reset rolling audio state."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_check_at = 0.0

    async def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Return preroll audio when the wakeword is detected."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if len(chunk) == 0 or not self.normalized_phrase:
            return None

        if len(self._buffer) == 0:
            self._buffer = chunk.copy()
        else:
            self._buffer = np.concatenate([self._buffer, chunk])

        if len(self._buffer) > self.window_samples:
            self._buffer = self._buffer[-self.window_samples:]

        now = time.monotonic()
        if len(self._buffer) < self.min_audio_samples:
            return None
        if now - self._last_check_at < self.detect_interval_seconds:
            return None

        self._last_check_at = now
        transcript = await self.stt.transcribe(self._buffer)
        normalized = normalize_text(transcript)
        if not normalized:
            return None

        if self.normalized_phrase not in normalized:
            return None
        if now - self._last_trigger_at < self.cooldown_seconds:
            return None

        self._last_trigger_at = now
        preroll = self._buffer[-self.preroll_samples:].copy() if self.preroll_samples else np.zeros(0, dtype=np.float32)
        self.reset()
        logger.info(f"Wakeword detected from transcript: {transcript}")
        return preroll
