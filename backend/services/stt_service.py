"""
Whisper-based speech-to-text service.
Falls back to demo mode if OPENAI_API_KEY is not configured.
"""

import openai
import io
import random
from config import settings
from loguru import logger

client = None

DEMO_PHRASES = [
    "Hello, I need help with my recent order.",
    "Can you tell me the status of my delivery?",
    "I'd like to return a product, please.",
    "I'm having an urgent issue with my account.",
    "What are your business hours?",
    "I need to speak with a manager immediately.",
    "Thank you for your help, that's very kind.",
    "This is frustrating, I've been waiting for days.",
    "Can you help me reset my password?",
    "I'd like to know more about your premium plans.",
]


def _get_client():
    global client
    if client is None:
        if not settings.OPENAI_API_KEY:
            return None
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return client


async def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe raw audio bytes using OpenAI Whisper.
    Falls back to demo phrases if API key is not configured.
    """
    ai_client = _get_client()

    # Demo mode — no API key
    if ai_client is None:
        phrase = random.choice(DEMO_PHRASES)
        logger.info(f"[DEMO MODE] Simulated transcription: '{phrase}'")
        return phrase

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"

        kwargs = {"model": settings.WHISPER_MODEL, "file": audio_file}
        if language and language != "auto":
            kwargs["language"] = language

        transcript = await ai_client.audio.transcriptions.create(**kwargs)
        text = transcript.text.strip()
        logger.info(f"STT transcription: '{text[:80]}...'")
        return text

    except Exception as e:
        logger.error(f"STT error: {e}")
        return ""
