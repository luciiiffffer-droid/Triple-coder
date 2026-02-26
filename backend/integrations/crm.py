"""
CRM integration placeholder.
"""

from loguru import logger
from config import settings


async def push_conversation_to_crm(conversation_id: str, summary: str, sentiment: float) -> dict:
    """Push conversation data to external CRM system."""
    if not settings.CRM_API_URL:
        logger.info("CRM integration not configured — skipping")
        return {"status": "skipped", "reason": "CRM not configured"}

    # TODO: Implement actual CRM API call
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         f"{settings.CRM_API_URL}/conversations",
    #         headers={"Authorization": f"Bearer {settings.CRM_API_KEY}"},
    #         json={"conversation_id": conversation_id, "summary": summary, "sentiment": sentiment},
    #     )
    #     return response.json()

    logger.info(f"CRM push placeholder: conversation={conversation_id}")
    return {"status": "ok", "conversation_id": conversation_id}


async def get_customer_profile(phone_number: str) -> dict:
    """Retrieve customer profile from CRM."""
    logger.info(f"CRM lookup placeholder: phone={phone_number}")
    return {"name": "Unknown", "tier": "standard", "history": []}
