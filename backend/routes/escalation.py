"""
Human agent escalation endpoint.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import get_db
from models.entities import Conversation, AnalyticsEvent, User
from models.schemas import EscalationRequest, EscalationResponse
from services.auth_service import get_current_user
from integrations.crm import push_conversation_to_crm
from integrations.erp import create_ticket
from loguru import logger
import json

router = APIRouter(prefix="/api/escalation", tags=["escalation"])


@router.post("/", response_model=EscalationResponse)
async def escalate(
    req: EscalationRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Find conversation
    result = await db.execute(
        select(Conversation).where(Conversation.id == req.conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Mark as escalated
    conv.status = "escalated"
    conv.ended_at = datetime.utcnow()

    escalation_id = str(uuid.uuid4())

    # Log analytics
    event = AnalyticsEvent(
        event_type="escalation",
        event_data=json.dumps({
            "reason": req.reason,
            "priority": req.priority,
            "escalation_id": escalation_id,
        }),
        conversation_id=req.conversation_id,
    )
    db.add(event)

    # Trigger integrations
    await push_conversation_to_crm(req.conversation_id, req.reason, conv.sentiment_avg)
    await create_ticket(req.conversation_id, f"Escalation: {req.reason}", req.priority)

    logger.info(f"Conversation {req.conversation_id} escalated — id={escalation_id}")

    return EscalationResponse(
        status="escalated",
        conversation_id=req.conversation_id,
        escalation_id=escalation_id,
        message="Conversation has been escalated to a human agent.",
    )
