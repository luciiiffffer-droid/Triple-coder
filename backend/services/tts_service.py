"""
Text-to-speech service using ElevenLabs API.
"""

import httpx
import base64
from config import settings
from loguru import logger


async def synthesize_speech(text: str, voice_id: str = None) -> bytes:
    """
    Convert text to speech audio bytes using ElevenLabs.
    Returns MP3 audio bytes.
    """
    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID

    if not settings.ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY not set — TTS disabled")
        return b""

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": settings.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.5,
                "use_speaker_boost": True,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"TTS synthesized {len(response.content)} bytes")
            return response.content

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""


async def synthesize_speech_base64(text: str, voice_id: str = None) -> str:
    """Return TTS audio as a base64-encoded string for WebSocket transmission."""
    audio_bytes = await synthesize_speech(text, voice_id)
    if audio_bytes:
        return base64.b64encode(audio_bytes).decode("utf-8")
    return ""
