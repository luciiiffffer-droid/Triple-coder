"""
Voice biometric verification placeholder.
"""

from loguru import logger


async def enroll_voiceprint(user_id: str, audio_bytes: bytes) -> dict:
    """Enroll a user's voiceprint for future verification."""
    logger.info(f"Voice biometrics enroll placeholder: user={user_id}, audio_size={len(audio_bytes)}")
    return {"status": "enrolled", "user_id": user_id, "confidence": 0.0}


async def verify_voiceprint(user_id: str, audio_bytes: bytes) -> dict:
    """Verify if an audio sample matches the enrolled voiceprint."""
    logger.info(f"Voice biometrics verify placeholder: user={user_id}")
    return {"status": "verified", "match": True, "confidence": 0.95}
