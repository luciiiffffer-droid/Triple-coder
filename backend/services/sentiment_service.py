"""
Sentiment analysis and emotion / urgency detection.
"""

from textblob import TextBlob
from loguru import logger

URGENCY_KEYWORDS = [
    "urgent", "emergency", "immediately", "asap", "critical", "help me",
    "right now", "dying", "danger", "fire", "accident", "police",
    "ambulance", "hospital", "threat", "deadline", "now",
    # Hindi
    "turant", "jaldi", "madad", "bachao", "emergency",
]

EMOTION_MAP = {
    (0.5, 1.0): "very_positive",
    (0.1, 0.5): "positive",
    (-0.1, 0.1): "neutral",
    (-0.5, -0.1): "negative",
    (-1.0, -0.5): "very_negative",
}


def analyze_sentiment(text: str) -> dict:
    """
    Return sentiment score, detected emotion label, and urgency flag.
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1.0 … 1.0

        # Emotion label
        emotion = "neutral"
        for (lo, hi), label in EMOTION_MAP.items():
            if lo <= polarity < hi:
                emotion = label
                break

        # Urgency detection
        lower = text.lower()
        is_urgent = any(kw in lower for kw in URGENCY_KEYWORDS)

        result = {
            "sentiment_score": round(polarity, 4),
            "emotion": emotion,
            "is_urgent": is_urgent,
        }

        logger.debug(f"Sentiment: {result}")
        return result

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"sentiment_score": 0.0, "emotion": "neutral", "is_urgent": False}
