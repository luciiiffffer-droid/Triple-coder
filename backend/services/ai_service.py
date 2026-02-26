"""
GPT-powered conversation engine with multi-turn context memory.
Falls back to demo mode if OPENAI_API_KEY is not configured.
"""

import openai
import random
from typing import List, Dict, Optional
from config import settings
from loguru import logger

client = None

DEMO_RESPONSES = {
    "order": "I can see your recent order #ORD-29471. It was shipped yesterday and should arrive within 2-3 business days. Would you like the tracking link?",
    "delivery": "Your delivery is currently in transit and expected to arrive by tomorrow evening. I'll send you a notification when it's out for delivery.",
    "return": "I'd be happy to help with your return! You can initiate a return within 30 days of purchase. Shall I start the process for you?",
    "urgent": "I understand this is urgent. Let me prioritize your request right away. Can you provide me with your account number so I can look into this immediately?",
    "hours": "Our support team is available 24/7! You can reach us anytime through this chat, by phone at 1-800-SUPPORT, or via email at help@company.com.",
    "manager": "I understand your concern. Let me connect you with a senior representative who can assist you further. Please hold for just a moment.",
    "thank": "You're welcome! Is there anything else I can help you with today? I'm happy to assist!",
    "frustrat": "I sincerely apologize for the inconvenience. I understand how frustrating this must be. Let me look into this right away and find a solution for you.",
    "password": "I can help you reset your password! I've just sent a password reset link to your registered email address. Please check your inbox and spam folder.",
    "premium": "Our Premium plan includes priority support, advanced analytics, and unlimited usage. It's $29.99/month or $299/year. Would you like me to explain the features in detail?",
}

DEFAULT_DEMO_RESPONSE = "Thank you for reaching out! I'm here to help. Could you please provide more details about what you need assistance with?"


def _get_client():
    global client
    if client is None:
        if not settings.OPENAI_API_KEY:
            return None
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return client


SYSTEM_PROMPT = """You are an advanced AI voice assistant for customer support.
You are empathetic, professional, and helpful. You:
- Understand and respond in multiple languages including Hindi, Tamil, Telugu, Bengali, and English
- Detect customer emotions and adjust your tone accordingly
- Identify urgent situations and flag them
- Provide concise, clear answers
- Ask clarifying questions when needed
- Escalate to human agents when you cannot resolve an issue
- Never fabricate information — say "I don't know" when uncertain

Keep responses conversational and under 3 sentences unless detail is needed."""


def _demo_response(user_message: str) -> str:
    """Match user message to a demo response by keyword."""
    lower = user_message.lower()
    for keyword, response in DEMO_RESPONSES.items():
        if keyword in lower:
            return response
    return DEFAULT_DEMO_RESPONSE


async def generate_response(
    messages: List[Dict[str, str]],
    knowledge_context: Optional[str] = None,
    language: str = "en",
) -> str:
    """
    Generate a GPT response given conversation history.
    Falls back to demo responses if API key is not configured.
    """
    ai_client = _get_client()

    # Demo mode — no API key
    if ai_client is None:
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        response = _demo_response(last_user_msg)
        logger.info(f"[DEMO MODE] AI response: '{response[:60]}...'")
        return response

    try:
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if language != "en":
            system_messages.append({
                "role": "system",
                "content": f"The user's preferred language is '{language}'. Respond in that language when appropriate.",
            })

        if knowledge_context:
            system_messages.append({
                "role": "system",
                "content": f"Relevant knowledge base context:\n{knowledge_context}",
            })

        full_messages = system_messages + messages[-20:]  # keep last 20 turns

        response = await ai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=full_messages,
            temperature=0.7,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"AI service error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Could you please try again?"
