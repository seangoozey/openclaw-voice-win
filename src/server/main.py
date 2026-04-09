"""
OpenClaw Voice Server

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication
- Text-to-Speech via ElevenLabs
- Audio streaming back to browser
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .wakeword import WakeWordDetector, strip_wakeword
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    stt_allow_mock: bool = False
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None  # Path to voice sample for cloning
    tts_url: Optional[str] = None
    tts_api_key: Optional[str] = None
    tts_voice_id: Optional[str] = None
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    
    # OpenClaw Gateway (auto-detected from OPENCLAW_GATEWAY_URL + TOKEN)
    openclaw_gateway_url: Optional[str] = Field(
        default=None,
        validation_alias="OPENCLAW_GATEWAY_URL",
    )
    openclaw_gateway_token: Optional[str] = Field(
        default=None,
        validation_alias="OPENCLAW_GATEWAY_TOKEN",
    )
    
    # Audio
    sample_rate: int = 16000
    client_config_path: str = "voice-ui-config.json"
    wakeword_enabled: bool = False
    wakeword_phrase: str = "hey claw"
    wakeword_window_seconds: float = 2.4
    wakeword_min_audio_seconds: float = 1.2
    wakeword_detect_interval_seconds: float = 0.8
    wakeword_cooldown_seconds: float = 2.0
    wakeword_preroll_seconds: float = 0.8
    
    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.1.0")


class ContinuousModeConfig(BaseModel):
    """Persisted client-side tuning for continuous listening."""

    energy_threshold: float = 0.025
    min_speech_ms: int = 250
    silence_ms: int = 1800
    restart_delay_ms: int = 900
    mic_warmup_ms: int = 1200
    vad_hold_ms: int = 450


def _client_config_path() -> Path:
    """Resolve the on-disk path for the browser tuning config."""
    path = Path(settings.client_config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_client_config() -> ContinuousModeConfig:
    """Load the persisted browser tuning config if present."""
    path = _client_config_path()
    if not path.exists():
        return ContinuousModeConfig()

    try:
        return ContinuousModeConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load client config from {path}: {exc}")
        return ContinuousModeConfig()


def save_client_config(config: ContinuousModeConfig) -> Path:
    """Write the browser tuning config to disk."""
    path = _client_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


async def stream_ai_response(websocket: WebSocket, transcript: str) -> None:
    """Stream the AI response and audio back to the browser."""
    logger.debug("Streaming AI response...")

    full_response = ""
    sentence_buffer = ""

    async for chunk in backend.chat_stream(transcript):
        full_response += chunk
        sentence_buffer += chunk

        await websocket.send_json({
            "type": "response_chunk",
            "text": chunk,
        })

        while any(sep in sentence_buffer for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]):
            earliest_idx = len(sentence_buffer)
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                idx = sentence_buffer.find(sep)
                if idx != -1 and idx < earliest_idx:
                    earliest_idx = idx + len(sep)

            if earliest_idx >= len(sentence_buffer):
                break

            sentence = sentence_buffer[:earliest_idx].strip()
            sentence_buffer = sentence_buffer[earliest_idx:]

            if sentence:
                speech_text = clean_for_speech(sentence)
                if speech_text:
                    logger.debug(f"Synthesizing: {speech_text[:50]}...")
                    async for audio_chunk in tts.synthesize_stream(speech_text):
                        audio_b64 = base64.b64encode(audio_chunk).decode()
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "data": audio_b64,
                            "sample_rate": 24000,
                        })

    if sentence_buffer.strip():
        speech_text = clean_for_speech(sentence_buffer.strip())
        if speech_text:
            async for audio_chunk in tts.synthesize_stream(speech_text):
                audio_b64 = base64.b64encode(audio_chunk).decode()
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": audio_b64,
                    "sample_rate": 24000,
                })

    await websocket.send_json({
        "type": "response_complete",
        "text": full_response,
    })
    logger.info(f"Response complete: {full_response[:100]}...")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad
    
    logger.info("Initializing OpenClaw Voice server...")
    
    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")
    
    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
        allow_mock=settings.stt_allow_mock,
    )
    
    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = ChatterboxTTS(
        model_name=settings.tts_model,
        voice_sample=settings.tts_voice,
        base_url=settings.tts_url,
        api_key=settings.tts_api_key,
        voice_id=settings.tts_voice_id or os.getenv("ELEVENLABS_VOICE_ID"),
    )
    
    # Initialize AI backend
    # Auto-detect OpenClaw gateway
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    
    if gateway_url and gateway_token:
        # Use OpenClaw gateway (connects to Aria!)
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url}")
        backend = AIBackend(
            backend_type="openai",  # Gateway speaks OpenAI API
            url=f"{gateway_url}/v1",
            model="openclaw:voice",  # Maps to 'voice' agent in config
            api_key=gateway_token,
            system_prompt=(
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational — a few sentences "
                "at most unless the topic genuinely needs depth. "
                "No markdown, bullet points, code blocks, or special formatting."
            ),
        )
    else:
        # Fallback to direct OpenAI
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()
    
    logger.info("✅ OpenClaw Voice server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse("src/client/index.html")


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """
    Create a new API key (requires master key).
    
    curl -X POST "http://localhost:8765/api/keys?name=myapp&tier=pro" \
         -H "x-master-key: YOUR_MASTER_KEY"
    """
    # Verify master key
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            # Also check if it's a valid master-tier key
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}
    
    from .auth import PRICING_TIERS
    
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    
    tier_config = PRICING_TIERS[tier]
    
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    
    return {
        "api_key": plaintext_key,  # Only shown once!
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """
    Get usage stats for an API key.
    
    curl "http://localhost:8765/api/usage?api_key=ocv_xxx"
    """
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    
    return token_manager.get_usage(key)


@app.get("/api/client-config")
async def get_client_config():
    """Return the persisted browser tuning config."""
    config = load_client_config()
    return {
        "config": config.model_dump(),
        "path": str(_client_config_path()),
    }


@app.post("/api/client-config")
async def update_client_config(config: ContinuousModeConfig):
    """Persist browser tuning config to disk."""
    path = save_client_config(config)
    return {
        "ok": True,
        "config": config.model_dump(),
        "path": str(path),
    }


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    # Check for API key in query params or headers
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    
    api_key: Optional[APIKey] = None
    
    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        # Dev mode - allow all
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")
    
    await websocket.accept()
    
    audio_buffer = []
    is_listening = False
    listening_mode = "manual"
    wakeword_active = False
    wakeword_detector: Optional[WakeWordDetector] = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["type"] == "start_listening":
                is_listening = True
                audio_buffer = []
                listening_mode = msg.get("mode", "manual")
                wakeword_active = not (settings.wakeword_enabled and listening_mode == "continuous")
                wakeword_detector = None

                if settings.wakeword_enabled and listening_mode == "continuous":
                    wakeword_detector = WakeWordDetector(
                        stt=stt,
                        phrase=settings.wakeword_phrase,
                        sample_rate=settings.sample_rate,
                        window_seconds=settings.wakeword_window_seconds,
                        min_audio_seconds=settings.wakeword_min_audio_seconds,
                        detect_interval_seconds=settings.wakeword_detect_interval_seconds,
                        cooldown_seconds=settings.wakeword_cooldown_seconds,
                        preroll_seconds=settings.wakeword_preroll_seconds,
                    )
                    await websocket.send_json({
                        "type": "wakeword_status",
                        "state": "armed",
                        "phrase": settings.wakeword_phrase,
                    })

                await websocket.send_json({
                    "type": "listening_started",
                    "mode": "active" if wakeword_active else "wakeword",
                })
                logger.debug("Started listening")
                
            elif msg["type"] == "stop_listening":
                is_listening = False
                wakeword_detector = None
                
                if audio_buffer and wakeword_active:
                    # Combine audio chunks
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Transcribe
                    logger.debug("Transcribing audio...")
                    transcript = await stt.transcribe(audio_data)
                    if settings.wakeword_enabled and listening_mode == "continuous":
                        transcript = strip_wakeword(transcript, settings.wakeword_phrase)
                    
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "final": True,
                    })
                    logger.info(f"Transcript: {transcript}")
                    
                    if transcript.strip():
                        await stream_ai_response(websocket, transcript)
                
                audio_buffer = []
                await websocket.send_json({"type": "listening_stopped"})
                logger.debug("Stopped listening")
                
            elif msg["type"] == "audio" and is_listening:
                # Decode base64 audio
                audio_bytes = base64.b64decode(msg["data"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

                if not wakeword_active and wakeword_detector:
                    preroll_audio = await wakeword_detector.process_chunk(audio_np)
                    if preroll_audio is not None:
                        wakeword_active = True
                        if len(preroll_audio) > 0:
                            audio_buffer.append(preroll_audio)
                        await websocket.send_json({
                            "type": "wakeword_detected",
                            "phrase": settings.wakeword_phrase,
                        })
                        await websocket.send_json({
                            "type": "wakeword_status",
                            "state": "active",
                            "phrase": settings.wakeword_phrase,
                        })
                    continue

                audio_buffer.append(audio_np)

                if vad and len(audio_np) > 0:
                    has_speech = vad.is_speech(audio_np)
                    await websocket.send_json({
                        "type": "vad_status",
                        "speech_detected": has_speech,
                    })
                
            elif msg["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Serve static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )



