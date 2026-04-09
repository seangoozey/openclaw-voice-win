"""
Server integration tests for OpenClaw Voice.
"""

import pytest
import asyncio
import json
import base64
import numpy as np
import os
import sys
import subprocess
import time
import socket

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


@pytest.fixture(scope="module")
def server():
    """Start the server for testing."""
    port = 8799  # Use high port to avoid conflicts
    
    # Skip if port already in use
    if is_port_in_use(port):
        pytest.skip(f"Port {port} already in use")
    
    # Start server
    env = os.environ.copy()
    env['OPENCLAW_PORT'] = str(port)
    env['OPENCLAW_STT_MODEL'] = 'tiny'  # Use tiny for fast tests
    env['OPENCLAW_CLIENT_CONFIG_PATH'] = '.test-client-config.json'
    
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'src.server.main:app', 
         '--host', '127.0.0.1', '--port', str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    
    # Wait for server to be ready
    max_wait = 15
    for i in range(max_wait):
        if is_port_in_use(port):
            break
        time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Server did not start in time")
    
    yield f"ws://127.0.0.1:{port}/ws", f"http://127.0.0.1:{port}"
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.test-client-config.json')
    if os.path.exists(config_path):
        os.remove(config_path)


class TestServerHTTP:
    """Test HTTP endpoints."""
    
    def test_index_page(self, server):
        """Test that index page loads."""
        import httpx
        
        ws_url, http_url = server
        response = httpx.get(f"{http_url}/")
        
        assert response.status_code == 200
        assert "OpenClaw Voice" in response.text
        assert "voice-button" in response.text

    def test_client_config_round_trip(self, server):
        """Test persisted continuous-mode config endpoint."""
        import httpx

        _, http_url = server
        config = {
            "energy_threshold": 0.04,
            "min_speech_ms": 400,
            "silence_ms": 2200,
            "restart_delay_ms": 1300,
            "mic_warmup_ms": 1500,
            "vad_hold_ms": 600,
        }

        save_response = httpx.post(f"{http_url}/api/client-config", json=config)
        assert save_response.status_code == 200
        assert save_response.json()["config"] == config

        load_response = httpx.get(f"{http_url}/api/client-config")
        assert load_response.status_code == 200
        assert load_response.json()["config"] == config


class TestServerWebSocket:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connect(self, server):
        """Test WebSocket connection."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Connection successful if we get here
            assert ws is not None
    
    @pytest.mark.asyncio
    async def test_ping_pong(self, server):
        """Test ping/pong."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_start_stop_listening(self, server):
        """Test start/stop listening cycle."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Start
            await ws.send(json.dumps({"type": "start_listening"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "listening_started"
            
            # Stop
            await ws.send(json.dumps({"type": "stop_listening"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "listening_stopped"
    
    @pytest.mark.asyncio
    async def test_audio_flow(self, server):
        """Test sending audio and getting response."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Start listening
            await ws.send(json.dumps({"type": "start_listening"}))
            await ws.recv()  # listening_started
            
            # Send some audio (silence)
            audio = np.zeros(16000, dtype=np.float32)
            audio_b64 = base64.b64encode(audio.tobytes()).decode()
            
            await ws.send(json.dumps({
                "type": "audio",
                "data": audio_b64,
            }))
            
            # Stop listening
            await ws.send(json.dumps({"type": "stop_listening"}))
            
            # Should get transcript first, then listening_stopped
            messages = []
            for _ in range(5):  # Collect up to 5 messages
                try:
                    response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    messages.append(response["type"])
                    if response["type"] == "listening_stopped":
                        break
                except asyncio.TimeoutError:
                    break
            
            # Should have gotten transcript and/or listening_stopped
            assert "transcript" in messages or "listening_stopped" in messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
