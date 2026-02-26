"""
Analytics data endpoints.
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import get_db
from models.entities import Conversation, Message, AnalyticsEvent, User
from models.schemas import AnalyticsSummary
from services.auth_service import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Total conversations
    total = await db.execute(select(func.count(Conversation.id)))
    total_conversations = total.scalar() or 0

    # Active
    active = await db.execute(
        select(func.count(Conversation.id)).where(Conversation.status == "active")
    )
    active_conversations = active.scalar() or 0

    # Escalated
    escalated = await db.execute(
        select(func.count(Conversation.id)).where(Conversation.status == "escalated")
    )
    escalated_conversations = escalated.scalar() or 0

    # Avg sentiment
    avg_sent = await db.execute(select(func.avg(Conversation.sentiment_avg)))
    avg_sentiment = round(avg_sent.scalar() or 0.0, 4)

    # Total messages
    total_msg = await db.execute(select(func.count(Message.id)))
    total_messages = total_msg.scalar() or 0

    # Urgent messages
    urgent = await db.execute(
        select(func.count(Message.id)).where(Message.is_urgent == True)
    )
    urgent_messages = urgent.scalar() or 0

    # Today's conversations
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = await db.execute(
        select(func.count(Conversation.id)).where(Conversation.started_at >= today)
    )
    conversations_today = today_count.scalar() or 0

    # Top emotions
    emotions = await db.execute(
        select(Message.emotion, func.count(Message.id))
        .where(Message.emotion.isnot(None))
        .group_by(Message.emotion)
        .order_by(func.count(Message.id).desc())
        .limit(5)
    )
    top_emotions = {row[0]: row[1] for row in emotions.all()}

    return AnalyticsSummary(
        total_conversations=total_conversations,
        active_conversations=active_conversations,
        escalated_conversations=escalated_conversations,
        avg_sentiment=avg_sentiment,
        total_messages=total_messages,
        urgent_messages=urgent_messages,
        conversations_today=conversations_today,
        top_emotions=top_emotions,
    )


@router.get("/timeline")
async def get_timeline(
    days: int = 7,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Conversation count per day for the last N days."""
    data = []
    for i in range(days - 1, -1, -1):
        day_start = (datetime.utcnow() - timedelta(days=i)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)
        count = await db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.started_at >= day_start,
                Conversation.started_at < day_end,
            )
        )
        data.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "conversations": count.scalar() or 0,
        })

    return {"timeline": data}
