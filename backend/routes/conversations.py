"""
Conversation history CRUD endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from models.database import get_db
from models.entities import Conversation, Message, User
from models.schemas import ConversationResponse, ConversationListItem
from services.auth_service import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("/", response_model=list[ConversationListItem])
async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    status: str = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Conversation).order_by(Conversation.started_at.desc())
    if status:
        query = query.where(Conversation.status == status)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    conversations = result.scalars().all()

    items = []
    for conv in conversations:
        msg_count = await db.execute(
            select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        )
        items.append(ConversationListItem(
            id=conv.id,
            channel=conv.channel,
            language=conv.language,
            status=conv.status,
            sentiment_avg=conv.sentiment_avg,
            started_at=conv.started_at,
            message_count=msg_count.scalar() or 0,
        ))

    return items


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conv)
    return {"status": "deleted", "conversation_id": conversation_id}
