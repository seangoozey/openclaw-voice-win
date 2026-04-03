"""
Voice Activity Detection module.
"""

from typing import Optional
import numpy as np
from loguru import logger


class VoiceActivityDetector:
    """Voice Activity Detection."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load VAD model."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
            )
            self.model = model
            self._get_speech_timestamps = utils[0]
            logger.info("✅ Silero VAD loaded")
        except Exception as e:
            logger.warning(f"VAD not available: {e}")
            self.model = None

    def _frame_size(self, sample_rate: int) -> Optional[int]:
        """Return a valid Silero frame size for the given sample rate."""
        if sample_rate == 8000:
            return 256
        if sample_rate == 16000:
            return 512
        return None
    
    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if audio contains speech."""
        if self.model is None:
            return True  # Assume speech if no VAD
        try:
            import torch

            frame_size = self._frame_size(sample_rate)
            if frame_size is None:
                logger.warning(f"Unsupported VAD sample rate: {sample_rate}")
                return True

            audio = np.asarray(audio, dtype=np.float32).flatten()
            if len(audio) == 0:
                return False

            # Silero expects exact frame sizes, so evaluate in chunks and
            # return True if any chunk crosses the speech threshold.
            for start in range(0, len(audio), frame_size):
                chunk = audio[start:start + frame_size]
                if len(chunk) < frame_size:
                    chunk = np.pad(chunk, (0, frame_size - len(chunk)))

                audio_tensor = torch.from_numpy(chunk).float()
                speech_prob = self.model(audio_tensor, sample_rate).item()
                if speech_prob > self.threshold:
                    return True

            return False
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True
