"""
Text-to-Speech module using ElevenLabs, Chatterbox, or fallbacks.
"""

import asyncio
import os
import wave
from io import BytesIO
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
from loguru import logger


class ChatterboxTTS:
    """Text-to-Speech using ElevenLabs, Chatterbox, or fallbacks."""

    def __init__(
        self,
        model_name: str = "chatterbox",
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        response_format: Optional[str] = None,
        exaggeration: Optional[float] = None,
    ):
        self.model_name = model_name
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "default"
        self.base_url = base_url or os.environ.get("OPENCLAW_TTS_URL")
        self.api_key = api_key or os.environ.get("OPENCLAW_TTS_API_KEY")
        self.response_format = response_format or os.environ.get("OPENCLAW_TTS_RESPONSE_FORMAT", "wav")
        exaggeration_value = exaggeration
        if exaggeration_value is None:
            exaggeration_env = os.environ.get("OPENCLAW_TTS_EXAGGERATION")
            exaggeration_value = float(exaggeration_env) if exaggeration_env else None
        self.exaggeration = exaggeration_value
        self.model = None
        self._backend = "mock"
        self._elevenlabs_client = None
        self._load_model()

    def _load_model(self):
        """Load the configured TTS backend."""
        if self.base_url:
            self._backend = "chatterbox-http"
            logger.info(f"Remote Chatterbox TTS configured: {self.base_url}")
            return

        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_key and self.model_name in {"auto", "elevenlabs"}:
            try:
                from elevenlabs import ElevenLabs

                self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                self._backend = "elevenlabs"
                logger.info("ElevenLabs TTS ready")
                return
            except ImportError:
                logger.warning("ElevenLabs SDK not installed, trying pip install...")
                try:
                    import subprocess

                    subprocess.check_call(["pip", "install", "elevenlabs", "-q"])
                    from elevenlabs import ElevenLabs

                    self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                    self._backend = "elevenlabs"
                    logger.info("ElevenLabs TTS ready (auto-installed)")
                    return
                except Exception as e:
                    logger.warning(f"ElevenLabs auto-install failed: {e}")
            except Exception as e:
                logger.warning(f"ElevenLabs failed: {e}")

        if self.model_name in {"auto", "chatterbox"}:
            try:
                from chatterbox.tts import ChatterboxTTS as CBModel

                logger.info("Loading Chatterbox TTS...")
                self.model = CBModel.from_pretrained(device=self._get_device())
                self._backend = "chatterbox"
                logger.info("Chatterbox loaded")
                return
            except ImportError:
                logger.warning("Chatterbox not installed")
            except Exception as e:
                logger.warning(f"Chatterbox failed: {e}")

        if self.model_name in {"auto", "xtts"}:
            try:
                from TTS.api import TTS

                logger.info("Loading Coqui XTTS...")
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self._backend = "xtts"
                logger.info("XTTS loaded")
                return
            except ImportError:
                logger.warning("Coqui TTS not installed")
            except Exception as e:
                logger.warning(f"XTTS failed: {e}")

        logger.warning("No TTS backend available, using mock mode (silence)")
        self._backend = "mock"

    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.

        Yields:
            Raw PCM audio chunks (24kHz, 16-bit)
        """
        if self._backend == "elevenlabs":
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                for chunk in audio_generator:
                    yield chunk
                return
            except Exception as e:
                logger.error(f"ElevenLabs streaming error: {e}")

        audio = await self.synthesize(text)
        yield self._array_to_pcm16_bytes(audio)

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis."""
        if self._backend == "elevenlabs":
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                audio_bytes = b"".join(audio_generator)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                return audio_array.astype(np.float32) / 32768.0
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
                return np.zeros(16000, dtype=np.float32)

        if self._backend == "chatterbox-http":
            return self._synthesize_remote(text)

        if self._backend == "chatterbox":
            if self.voice_sample:
                audio = self.model.generate(text, audio_prompt=self.voice_sample)
            else:
                audio = self.model.generate(text)
            return audio.cpu().numpy().astype(np.float32)

        if self._backend == "xtts":
            if self.voice_sample:
                wav = self.model.tts(text=text, speaker_wav=self.voice_sample, language="en")
            else:
                wav = self.model.tts(text=text, language="en")
            return np.array(wav, dtype=np.float32)

        logger.debug(f"Mock TTS: '{text[:50]}...'")
        return np.zeros(12000, dtype=np.float32)

    def _synthesize_remote(self, text: str) -> np.ndarray:
        """Call a remote Chatterbox-compatible HTTP endpoint."""
        headers = {"Accept": "audio/*"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payloads = [{
            "model": "chatterbox",
            "input": text,
            "voice": self.voice_id or "default",
            "response_format": self.response_format,
            "exaggeration": self.exaggeration,
        }]

        last_error: Optional[str] = None
        with httpx.Client(timeout=60.0) as client:
            for payload in payloads:
                try:
                    response = client.post(
                        self.base_url,
                        json={k: v for k, v in payload.items() if v is not None},
                        headers=headers,
                    )
                    response.raise_for_status()
                    return self._decode_audio_bytes(response.content)
                except Exception as e:
                    last_error = str(e)

        logger.error(f"Remote Chatterbox TTS error: {last_error}")
        return np.zeros(12000, dtype=np.float32)

    def _decode_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes or raw PCM16 bytes to float32 audio."""
        if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
            if sample_width == 2:
                pcm = np.frombuffer(frames, dtype=np.int16)
                return pcm.astype(np.float32) / 32768.0

        # Many local Chatterbox servers can emit MPEG/other containerized audio.
        # Try librosa as a fallback decoder when raw PCM/WAV detection does not match.
        try:
            import soundfile as sf

            audio_array, _ = sf.read(BytesIO(audio_bytes), dtype="float32")
            if isinstance(audio_array, np.ndarray):
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                return audio_array.astype(np.float32)
        except Exception:
            pass

        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _array_to_pcm16_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float audio to PCM16 bytes for websocket streaming."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()
