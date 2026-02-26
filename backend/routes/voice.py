"""
Real-time voice WebSocket endpoint + text chat REST endpoint.
"""

import json
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from models.database import async_session
from models.entities import Conversation, Message, AnalyticsEvent
from services.stt_service import transcribe_audio
from services.ai_service import generate_response
from services.tts_service import synthesize_speech_base64
from services.sentiment_service import analyze_sentiment
from services.vector_service import search as vector_search
from integrations.fraud_detection import check_fraud

router = APIRouter(tags=["voice"])

# In-memory chat histories keyed by session_id
_chat_sessions: Dict[str, List[Dict[str, str]]] = {}


# ---- Text Chat Models ----
class TextChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "en"


class TextChatResponse(BaseModel):
    session_id: str
    user_message: str
    ai_response: str
    emotion: str
    sentiment_score: float
    is_urgent: bool
    fraud_alert: bool


# ---- Text Chat Endpoint ----
@router.post("/api/chat/text", response_model=TextChatResponse)
async def text_chat(req: TextChatRequest):
    """
    Text-based chat endpoint. Send a message, get an AI response.
    Works without microphone or Whisper.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # Get or create chat history
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []
        # Create conversation in DB
        async with async_session() as db:
            conv = Conversation(id=session_id, channel="web")
            db.add(conv)
            await db.commit()

    chat_history = _chat_sessions[session_id]

    # 1. Sentiment and urgency
    sentiment = analyze_sentiment(req.message)

    # 2. Fraud check
    fraud = await check_fraud(req.message, session_id)

    # 3. Knowledge base context (RAG)
    kb_context = await vector_search(req.message)

    # 4. Build chat history and get GPT response
    chat_history.append({"role": "user", "content": req.message})
    ai_text = await generate_response(
        chat_history, knowledge_context=kb_context, language=req.language
    )
    chat_history.append({"role": "assistant", "content": ai_text})

    # 5. Persist messages
    async with async_session() as db:
        user_msg = Message(
            conversation_id=session_id,
            role="user",
            content=req.message,
            sentiment_score=sentiment["sentiment_score"],
            emotion=sentiment["emotion"],
            is_urgent=sentiment["is_urgent"],
        )
        ai_msg = Message(
            conversation_id=session_id,
            role="assistant",
            content=ai_text,
        )
        db.add_all([user_msg, ai_msg])

        event = AnalyticsEvent(
            event_type="text_interaction",
            event_data=json.dumps({
                "emotion": sentiment["emotion"],
                "is_urgent": sentiment["is_urgent"],
                "fraud_risk": fraud["risk_level"],
            }),
            conversation_id=session_id,
        )
        db.add(event)
        await db.commit()

    logger.info(f"Text chat [{session_id[:8]}]: '{req.message[:50]}' -> '{ai_text[:50]}'")

    return TextChatResponse(
        session_id=session_id,
        user_message=req.message,
        ai_response=ai_text,
        emotion=sentiment["emotion"],
        sentiment_score=sentiment["sentiment_score"],
        is_urgent=sentiment["is_urgent"],
        fraud_alert=fraud["flagged"],
    )


# ---- WebSocket Voice Endpoint ----
@router.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    logger.info(f"WebSocket connected: session={session_id}")

    conversation_id = session_id or str(uuid.uuid4())
    chat_history: list[dict] = []

    async with async_session() as db:
        conv = Conversation(id=conversation_id, channel="web")
        db.add(conv)
        await db.commit()

    try:
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Received {len(data)} bytes of audio")

            transcript = await transcribe_audio(data)
            if not transcript:
                await websocket.send_json({
                    "type": "error",
                    "message": "Could not transcribe audio. Please try again.",
                })
                continue

            sentiment = analyze_sentiment(transcript)
            fraud = await check_fraud(transcript, conversation_id)
            kb_context = await vector_search(transcript)

            chat_history.append({"role": "user", "content": transcript})
            ai_text = await generate_response(chat_history, knowledge_context=kb_context)
            chat_history.append({"role": "assistant", "content": ai_text})

            audio_b64 = await synthesize_speech_base64(ai_text)

            async with async_session() as db:
                user_msg = Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=transcript,
                    sentiment_score=sentiment["sentiment_score"],
                    emotion=sentiment["emotion"],
                    is_urgent=sentiment["is_urgent"],
                )
                ai_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=ai_text,
                )
                db.add_all([user_msg, ai_msg])

                event = AnalyticsEvent(
                    event_type="voice_interaction",
                    event_data=json.dumps({
                        "emotion": sentiment["emotion"],
                        "is_urgent": sentiment["is_urgent"],
                        "fraud_risk": fraud["risk_level"],
                    }),
                    conversation_id=conversation_id,
                )
                db.add(event)
                await db.commit()

            await websocket.send_json({
                "type": "response",
                "transcript": transcript,
                "ai_response": ai_text,
                "emotion": sentiment["emotion"],
                "sentiment_score": sentiment["sentiment_score"],
                "is_urgent": sentiment["is_urgent"],
                "fraud_alert": fraud["flagged"],
                "audio_base64": audio_b64,
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
