"""
ERP system integration placeholder.
"""

from loguru import logger
from config import settings


async def create_ticket(conversation_id: str, subject: str, priority: str = "normal") -> dict:
    """Create a support ticket in the ERP system."""
    if not settings.ERP_API_URL:
        logger.info("ERP integration not configured — skipping")
        return {"status": "skipped"}

    logger.info(f"ERP ticket placeholder: conv={conversation_id}, subj={subject}")
    return {"status": "created", "ticket_id": f"TKT-{conversation_id[:8]}"}


async def get_order_status(order_id: str) -> dict:
    """Look up order status from ERP."""
    logger.info(f"ERP order lookup placeholder: order={order_id}")
    return {"order_id": order_id, "status": "processing", "eta": "2-3 business days"}
