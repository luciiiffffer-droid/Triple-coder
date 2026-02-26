"""
WhatsApp Business API integration placeholder.
"""

from loguru import logger
from config import settings


async def send_whatsapp_message(phone_number: str, message: str) -> dict:
    """Send a WhatsApp message via the Business API."""
    if not settings.WHATSAPP_API_URL:
        logger.info("WhatsApp integration not configured — skipping")
        return {"status": "skipped"}

    # TODO: Implement actual WhatsApp API call
    logger.info(f"WhatsApp send placeholder: to={phone_number}, msg={message[:50]}...")
    return {"status": "sent", "to": phone_number}


async def handle_whatsapp_webhook(payload: dict) -> dict:
    """Process incoming WhatsApp webhook events."""
    logger.info(f"WhatsApp webhook placeholder: {payload}")
    return {"status": "received"}
