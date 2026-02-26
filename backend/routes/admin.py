"""
Admin settings endpoints.
"""

from fastapi import APIRouter, Depends
from models.entities import User
from models.schemas import SettingsUpdate
from services.auth_service import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])

# In-memory settings (in production, store in DB or config service)
_app_settings = {
    "language": "en",
    "voice_id": "21m00Tcm4TlvDq8ikWAM",
    "tts_provider": "elevenlabs",
    "escalation_threshold": -0.5,
}


@router.get("/settings")
async def get_settings(admin: User = Depends(require_admin)):
    return _app_settings


@router.put("/settings")
async def update_settings(
    update: SettingsUpdate,
    admin: User = Depends(require_admin),
):
    if update.language is not None:
        _app_settings["language"] = update.language
    if update.voice_id is not None:
        _app_settings["voice_id"] = update.voice_id
    if update.tts_provider is not None:
        _app_settings["tts_provider"] = update.tts_provider
    if update.escalation_threshold is not None:
        _app_settings["escalation_threshold"] = update.escalation_threshold

    return {"status": "updated", "settings": _app_settings}
