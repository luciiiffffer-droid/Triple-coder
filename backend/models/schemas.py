"""
Pydantic request / response schemas for the API.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ── Auth ─────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    email: str
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    is_admin: bool


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Conversation ─────────────────────────────────────────
class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sentiment_score: Optional[float] = None
    emotion: Optional[str] = None
    is_urgent: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: str
    channel: str
    language: str
    status: str
    sentiment_avg: float
    started_at: datetime
    ended_at: Optional[datetime] = None
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


class ConversationListItem(BaseModel):
    id: str
    channel: str
    language: str
    status: str
    sentiment_avg: float
    started_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


# ── Analytics ────────────────────────────────────────────
class AnalyticsSummary(BaseModel):
    total_conversations: int = 0
    active_conversations: int = 0
    escalated_conversations: int = 0
    avg_sentiment: float = 0.0
    total_messages: int = 0
    urgent_messages: int = 0
    conversations_today: int = 0
    top_emotions: dict = {}


# ── Knowledge Base ───────────────────────────────────────
class KnowledgeIngestRequest(BaseModel):
    title: str
    content: str
    category: str = "general"


class KnowledgeIngestResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str


# ── Escalation ───────────────────────────────────────────
class EscalationRequest(BaseModel):
    conversation_id: str
    reason: str = "User requested human agent"
    priority: str = "normal"  # low / normal / high / critical


class EscalationResponse(BaseModel):
    status: str
    conversation_id: str
    escalation_id: str
    message: str


# ── Voice ────────────────────────────────────────────────
class VoiceResponse(BaseModel):
    transcript: str
    ai_response: str
    emotion: str
    sentiment_score: float
    is_urgent: bool
    audio_base64: Optional[str] = None


# ── Settings ─────────────────────────────────────────────
class SettingsUpdate(BaseModel):
    language: Optional[str] = None
    voice_id: Optional[str] = None
    tts_provider: Optional[str] = None
    escalation_threshold: Optional[float] = None
