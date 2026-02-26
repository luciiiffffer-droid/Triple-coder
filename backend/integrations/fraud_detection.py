"""
Fraud pattern detection placeholder.
"""

from loguru import logger

FRAUD_PATTERNS = [
    "give me your password",
    "social security",
    "credit card number",
    "wire transfer",
    "send money",
    "account number",
    "routing number",
]


async def check_fraud(text: str, conversation_id: str) -> dict:
    """
    Scan text for potential fraud patterns.
    Returns a risk assessment.
    """
    lower = text.lower()
    detected = [p for p in FRAUD_PATTERNS if p in lower]

    risk_level = "low"
    if len(detected) >= 2:
        risk_level = "high"
    elif len(detected) == 1:
        risk_level = "medium"

    result = {
        "risk_level": risk_level,
        "patterns_detected": detected,
        "flagged": risk_level in ("medium", "high"),
        "conversation_id": conversation_id,
    }

    if result["flagged"]:
        logger.warning(f"Fraud alert: {result}")

    return result
