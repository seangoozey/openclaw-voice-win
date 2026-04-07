"""
Speech-to-Text module using Whisper.
"""

import asyncio

import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        allow_mock: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.allow_mock = allow_mock
        self.model = None
        self._backend = "mock"
        self._load_model()

    def _faster_whisper_attempts(self):
        """Return device/compute_type attempts for faster-whisper."""
        if self.device == "auto":
            # Let CTranslate2 choose the best available device. If that fails
            # on a machine with broken GPU/CUDA config, retry a known CPU path.
            return [("auto", "default"), ("cpu", "int8")]
        if self.device == "cuda":
            return [("cuda", "float16"), ("cpu", "int8")]
        if self.device == "mps":
            logger.warning("faster-whisper does not support mps directly; using cpu")
            return [("cpu", "int8")]
        return [(self.device, "int8")]

    def _load_model(self):
        """Load the Whisper model."""
        errors = []

        # Try faster-whisper first.
        try:
            from faster_whisper import WhisperModel

            for model_device, compute_type in self._faster_whisper_attempts():
                try:
                    logger.info(
                        f"Loading faster-whisper {self.model_name} on "
                        f"{model_device} ({compute_type})"
                    )
                    self.model = WhisperModel(
                        self.model_name,
                        device=model_device,
                        compute_type=compute_type,
                    )
                    self.device = model_device
                    self._backend = "faster-whisper"
                    logger.info("faster-whisper loaded")
                    return
                except Exception as e:
                    message = f"faster-whisper {model_device}/{compute_type}: {e}"
                    errors.append(message)
                    logger.warning(message)
        except ImportError as e:
            errors.append(f"faster-whisper import: {e}")
            logger.warning("faster-whisper not available")
        except Exception as e:
            errors.append(f"faster-whisper: {e}")
            logger.warning(f"faster-whisper failed: {e}")

        # Try openai-whisper as a fallback if it is installed.
        try:
            import whisper

            if self.device in {"auto", "mps"}:
                self.device = "cpu"

            logger.info(f"Loading openai-whisper {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("openai-whisper loaded")
            return
        except ImportError as e:
            errors.append(f"openai-whisper import: {e}")
            logger.warning("openai-whisper not available")
        except Exception as e:
            errors.append(f"openai-whisper: {e}")
            logger.warning(f"openai-whisper failed: {e}")

        if self.allow_mock:
            logger.warning("No STT backend - using mock mode")
            self._backend = "mock"
            return

        details = "\n".join(f"- {error}" for error in errors) or "- no details"
        raise RuntimeError(
            "No STT backend could be loaded. Run `uv sync` or "
            "`pip install -r requirements.txt` to install faster-whisper. "
            "If CUDA is unavailable or misconfigured, set OPENCLAW_STT_DEVICE=cpu.\n"
            f"{details}"
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        if self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()

        logger.debug(f"Mock STT: received {len(audio)} samples")
        return "[Mock transcription - install whisper for real STT]"
