"""
GPT-powered conversation engine with multi-turn context memory.
Falls back to a rich demo engine with general knowledge if OPENAI_API_KEY is not configured.
"""

import openai
import random
import re
from typing import List, Dict, Optional
from config import settings
from loguru import logger

client = None


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
- Never fabricate information

Keep responses conversational and under 3 sentences unless detail is needed."""


# ─────────────────────────────────────────────────────────
#  COMPREHENSIVE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

_KNOWLEDGE = {
    # --- General Knowledge ---
    'capital': {
        'india': "The capital of India is **New Delhi**. It's located in northern India and serves as the seat of the Indian government. Fun fact: New Delhi was designed by British architects Edwin Lutyens and Herbert Baker! 🏛️",
        'usa': "The capital of the United States is **Washington, D.C.** It's named after George Washington, the first U.S. president. It's not part of any state! 🇺🇸",
        'uk': "The capital of the United Kingdom is **London**. It's one of the most visited cities in the world with landmarks like Big Ben, Buckingham Palace, and the Tower of London! 🇬🇧",
        'france': "The capital of France is **Paris** — the City of Light! Famous for the Eiffel Tower, Louvre Museum, and amazing cuisine 🗼",
        'japan': "The capital of Japan is **Tokyo**. It's the most populous metropolitan area in the world with over 37 million people! 🗾",
        'default': "That's a great geography question! Could you specify which country you're asking about? I know capitals for most countries around the world! 🌍",
    },
    'planet': "Our solar system has **8 planets**: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Fun fact: Jupiter is so big that all other planets could fit inside it! 🪐",
    'sun': "The Sun is a **G-type main-sequence star** at the center of our solar system. It's about 4.6 billion years old and accounts for 99.86% of the total mass of the solar system! The surface temperature is about 5,500°C ☀️",
    'moon': "Earth's Moon is approximately **4.5 billion years old** and is about 384,400 km away from Earth. It's the fifth-largest moon in our solar system. Only 12 people have ever walked on it! 🌙",
    'earth': "Earth is the **third planet** from the Sun and the only known planet to support life. It's about 4.54 billion years old, has a circumference of about 40,075 km, and is 71% covered in water! 🌍",

    # --- Technology ---
    'ai': "**Artificial Intelligence (AI)** is a branch of computer science focused on creating systems that can perform tasks that typically require human intelligence — like understanding speech, recognizing images, making decisions, and translating languages. I'm an example of AI in action! 🤖",
    'python': "**Python** is one of the world's most popular programming languages! Created by Guido van Rossum in 1991, it's known for its clean, readable syntax. It's widely used in AI/ML, web development, data science, and automation. 🐍",
    'machine learning': "**Machine Learning** is a subset of AI where computers learn from data without being explicitly programmed. Types include supervised learning, unsupervised learning, and reinforcement learning. It powers everything from Netflix recommendations to self-driving cars! 🧠",
    'blockchain': "**Blockchain** is a decentralized, distributed digital ledger technology. Each block contains transaction data and is cryptographically linked to the previous block. It's the technology behind cryptocurrencies like Bitcoin! ⛓️",
    'chatgpt': "**ChatGPT** is an AI chatbot developed by OpenAI, launched in November 2022. It uses large language models (LLMs) trained on vast amounts of text data to generate human-like responses. It's part of the GPT (Generative Pre-trained Transformer) family! 💬",
    'internet': "The **Internet** is a global network of interconnected computers. It was born from ARPANET in the late 1960s. Today, over 5 billion people use the internet worldwide — that's about 63% of the world's population! 🌐",

    # --- Science ---
    'gravity': "**Gravity** is one of the four fundamental forces of nature. Discovered by Sir Isaac Newton (inspired by that famous falling apple 🍎), it's the force that attracts objects toward each other. Einstein later described it as the warping of spacetime in his General Theory of Relativity!",
    'dna': "**DNA (Deoxyribonucleic acid)** is the molecule that carries the genetic instructions for life. It's shaped like a double helix and contains four bases: Adenine, Thymine, Guanine, and Cytosine. If you uncoiled all the DNA in your body, it would stretch to the Sun and back over 600 times! 🧬",
    'water': "**Water (H₂O)** is essential for all known forms of life. It covers about 71% of Earth's surface. Fun facts: hot water freezes faster than cold water (Mpemba effect), and water is the only substance that naturally exists in all three states — solid, liquid, and gas! 💧",
    'light': "**Light** travels at approximately **299,792 km/s** (about 186,000 miles per second) — the fastest speed in the universe! It takes sunlight about 8 minutes and 20 seconds to reach Earth. Light behaves both as a wave and a particle! 💡",
    'photosynthesis': "**Photosynthesis** is the process by which plants convert sunlight, water, and CO₂ into glucose and oxygen. The equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. It's basically how plants make their food and produce the oxygen we breathe! 🌱",

    # --- History ---
    'world war': "**World War I** (1914-1918) involved over 70 million military personnel. **World War II** (1939-1945) was the deadliest conflict in human history, with an estimated 70-85 million fatalities. Both wars fundamentally reshaped the political landscape of the world. 📚",
    'independence': "India gained **independence on August 15, 1947** from British rule, after a long struggle led by leaders like Mahatma Gandhi, Jawaharlal Nehru, Subhas Chandra Bose, and many others. The USA declared independence on July 4, 1776! 🇮🇳",
    'gandhi': "**Mahatma Gandhi** (1869-1948) was the leader of India's non-violent independence movement against British colonial rule. He employed non-violent civil disobedience methods like the Salt March. He's known as the 'Father of the Nation' in India. 🕊️",

    # --- Math ---
    'pi': "**Pi (π)** is approximately **3.14159265358979...** It's the ratio of a circle's circumference to its diameter. Pi is an irrational number — its decimal representation never ends and never repeats! March 14 (3/14) is celebrated as Pi Day 🥧",
    'fibonacci': "The **Fibonacci sequence** is: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89... Each number is the sum of the two preceding ones. It appears everywhere in nature — from flower petals to spiral galaxies! 🌻",

    # --- Health ---
    'vitamin': "**Vitamins** are essential organic compounds your body needs. Key ones: Vitamin A (vision), B complex (energy), C (immune system), D (bones — from sunlight!), E (antioxidant), K (blood clotting). Eating a balanced diet usually provides all you need! 🍊",
    'exercise': "Regular **exercise** is incredible for health! The WHO recommends at least 150 minutes of moderate exercise per week. Benefits include: reduced risk of heart disease, better mental health, stronger bones, improved sleep, and longer lifespan! 💪",
    'sleep': "Adults need **7-9 hours of sleep** per night. During sleep, your brain processes memories, your body repairs tissues, and growth hormones are released. Poor sleep is linked to obesity, heart disease, and reduced immune function. 😴",

    # --- Fun Facts ---
    'fun fact': "Here's a fun fact: **Honey never spoils!** Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible. Also, octopuses have three hearts, and bananas are technically berries but strawberries aren't! 🍯",
    'random fact': "Did you know? **The shortest war in history** lasted only 38 to 45 minutes — between Britain and Zanzibar on August 27, 1896. Also, a group of flamingos is called a 'flamboyance'! 🦩",
    'space': "**Space** is wild! The observable universe is about 93 billion light-years in diameter. There are more stars in the universe than grains of sand on all of Earth's beaches. And on Venus, a day is longer than a year! 🚀",
    'ocean': "The **ocean** covers 71% of Earth's surface but we've only explored about 5% of it! The deepest point is the Mariana Trench at about 11,034 meters. There might be more historical artifacts in the ocean than in all museums combined! 🌊",
    'animal': "The **animal kingdom** is fascinating! Blue whales are the largest animals ever (up to 100 feet long), hummingbirds can fly backwards, elephants can't jump, and a group of crows is called a 'murder'. Nature is amazing! 🐋",
}

# Conversational patterns
_GREETINGS = [r'\b(hi|hello|hey|hiya|howdy|yo|sup|hola|namaste|namaskar)\b', r'\bgood\s*(morning|afternoon|evening|night)\b', r'\bwhat\'?s?\s*up\b']
_GREETING_RESPONSES = [
    "Hey there! 😊 I'm your AI assistant and I can help with anything — customer support, general knowledge, science, tech, history, you name it! What would you like to know?",
    "Hello! 👋 Welcome! I can answer questions, share interesting facts, help with orders, or just have a friendly chat. What's on your mind?",
    "Hi! Great to see you! I know about science, technology, history, geography, and much more. Plus I can help with any support issues. Fire away! 🚀",
]

_HOWAREYOU = [r'\bhow\s*(are|r)\s*(you|u|ya)\b', r'\bhow\'?s?\s*it\s*going\b']
_HOWAREYOU_RESPONSES = [
    "I'm doing great, thanks for asking! 😊 I've been reading up on everything from quantum physics to cooking recipes. What would you like to talk about?",
    "Fantastic! I love having conversations. I can chat about science, tech, history, or help you with anything you need. What are you curious about?",
]

_THANKS = [r'\b(thanks|thank\s*you|thx|tysm|appreciate)\b']
_THANKS_RESPONSES = [
    "You're so welcome! 😊 I love sharing knowledge. Is there anything else you're curious about?",
    "Happy to help! That's what I'm here for. Got any more questions? I never run out of answers! ✨",
]

_GOODBYE = [r'\b(bye|goodbye|see\s*ya|take\s*care|good\s*night|cya|later)\b']
_GOODBYE_RESPONSES = [
    "Goodbye! 👋 It was great chatting with you. Come back anytime — I'll be here 24/7!",
    "Take care! 😊 Remember, I'm always here if you want to learn something new or need help. See you soon!",
]

_AFFIRMATIVE = [r'\b(yes|yeah|yep|sure|okay|ok|please|go ahead|absolutely)\b']
_AFFIRMATIVE_RESPONSES = [
    "Perfect! Let me get that sorted for you right away... ✨ Done! Anything else I can help with?",
    "Great, I'm on it! All taken care of. What else would you like to know? 😊",
]

_NEGATIVE = [r'\b(no|nah|nope|nothing|that\'s all|all good|i\'m good|im good)\b']
_NEGATIVE_RESPONSES = [
    "No problem! If you ever want to learn something new or need help, I'm just a message away. Have a wonderful day! 😊",
    "Alright! Remember, I'm here 24/7 with answers about literally anything. Take care! ✨",
]

# Support-specific responses
_SUPPORT = {
    'order': ["I'd love to help with your order! Could you share your order number? I'll track it down right away for you. 📦", "Sure thing! Let me look into your order. What's the order number or email you used?"],
    'deliver': ["Let me check on your delivery! Most orders arrive in 3-5 business days. Could you share your order number for a precise update? 🚚", "I'll track your delivery right away! What's your order number?"],
    'return': ["No worries, returns are easy! You can return within 30 days. Want me to start the process? Just share your order number. 📋", "I'm sorry it didn't work out! Our return process is simple. Share the order number and I'll get it started."],
    'refund': ["Refunds typically process in 5-7 business days after we receive your return. Want me to check the status? 💳", "Let me look into your refund! What's your order number?"],
    'password': ["No worries, let's get you back in! I'll send a reset link to your email. Can you confirm the email on your account? 🔐", "Forgetting passwords happens to everyone! I'll help you reset it right away."],
    'cancel': ["I'm sorry to hear that! Before cancelling, is there anything I can do to help? If you've decided, I'll process it right away. 💔", "I understand. I'll process the cancellation. Is there anything we could have done differently?"],
    'billing': ["Let me check your billing details! I can help with invoices, charges, and payment methods. What specifically do you need? 🧾"],
    'shipping': ["We offer Standard (5-7 days, free over $50), Express (2-3 days), and Overnight shipping! Which would you like? 📬"],
    'manager': ["I'll connect you with a senior team member right away. Let me escalate this for you. 👔", "Of course, escalating to a manager now. They'll have full context of our conversation."],
    'hours': ["We're available **24/7**! Anytime, anywhere — phone, chat, or email. We never close! ⏰"],
    'price': ["Our plans: Basic ($9.99/mo), Standard ($19.99/mo), and Premium ($29.99/mo). Want details on any? 💰"],
}

# Emotion & personal feelings responses
_EMOTIONS = {
    'frustrated': ["I completely understand your frustration, and I'm really sorry. 😔 Let me take personal ownership of this — tell me what happened and I'll fix it.", "I hear you, and your frustration is completely valid. Let me prioritize this and get it resolved right now."],
    'angry': ["I am so sorry about this experience. You have every right to be upset. Let me do everything I can to make this right immediately. 🙏"],
    'happy': ["That's wonderful to hear! 🎉 Your happiness makes my day. Is there anything else I can help with?", "So glad to hear that! Positive vibes all around! ✨ What else can I do for you?"],
    'sad': ["I'm really sorry you're feeling this way. 🫂 It's okay to feel sad sometimes — it shows you care deeply. Remember, tough times don't last but tough people do. Want to talk about what's going on? I'm here to listen.", "Sending you a virtual hug. 💙 Sadness is a normal part of life, and it's brave of you to express it. Take things one step at a time. Is there something specific that's bothering you? Sometimes talking helps."],
    'lonely': ["I'm sorry you're feeling lonely. 💙 You're not alone — I'm right here with you. Loneliness can feel heavy, but it's temporary. Try reaching out to a friend, or even just stepping outside for a walk. You matter more than you know. Want to chat for a while?", "I hear you, and loneliness is tougher than most people realize. 🫂 But remember, some of the strongest people feel lonely sometimes. I'm here 24/7 if you just want someone to talk to. No judgement, ever."],
    'stressed': ["I can feel the stress in your words, and I want you to know it's okay to feel overwhelmed. 🌊 Take a deep breath — in for 4 seconds, hold for 4, out for 4. Stress is temporary, but your health is important. What's weighing on you?", "Stress can feel crushing, but you're stronger than you think. 💪 Here are some quick tips: take 5 deep breaths, step away for 2 minutes, drink some water, and remind yourself — you've survived 100% of your worst days so far. What's stressing you out?"],
    'anxious': ["I understand anxiety can feel overwhelming. 🌟 Remember: your anxiety is lying to you — most of what we worry about never actually happens. Try grounding yourself: name 5 things you can see, 4 you can touch, 3 you can hear. You've got this!", "Anxiety is your brain trying to protect you, even when there's no real danger. 💙 Take slow, deep breaths. You're safe right now, in this moment. Would you like to talk about what's making you anxious? Sometimes putting it into words helps."],
    'tired': ["It sounds like you need some rest, and that's perfectly okay! 😴 Your body is telling you something important. If possible, take a break — even 10 minutes of rest can make a big difference. You've been working hard and you deserve it. Take care of yourself!", "Being tired is your body's way of saying 'slow down'. 🌙 Make sure you're drinking enough water, and try to get some quality sleep tonight. Remember, you can't pour from an empty cup. Rest is not laziness — it's recovery!"],
    'bored': ["Bored? Let's fix that! 🎯 How about this: Ask me about space, tell me to share a fun fact, or test my knowledge on any topic! I can also tell jokes, share science facts, or discuss history. What sounds fun?", "I've got the cure for boredom! 🚀 Try asking me: 'Tell me a fun fact', 'What's fascinating about the ocean?', or 'Tell me a joke'. Or we could play a game — ask me any trivia question and see if I know the answer!"],
    'excited': ["Your excitement is contagious! 🎉🥳 That's amazing energy! What's got you so pumped? I'd love to hear about it!", "YES! I love that energy! 🔥 Excitement is the best feeling. Tell me more — what's happening? I want to celebrate with you!"],
    'grateful': ["That's so beautiful! 🥰 Gratitude is one of the most powerful emotions. Studies show that grateful people are happier, sleep better, and have stronger relationships. What are you grateful for today?", "Gratitude is truly special. 🌟 The fact that you're feeling grateful shows what a wonderful person you are. It takes strength to appreciate what you have. Thank you for sharing that with me!"],
    'confused': ["It's totally okay to feel confused — it means you're thinking deeply about something! 🤔 Let's work through it together. What's confusing you? I'll try my best to explain it clearly.", "Confusion is just the first step to understanding! 💡 Everyone gets confused sometimes — even the smartest people. Tell me what's puzzling you and let's figure it out together."],
    'scared': ["It's okay to feel scared — fear is a natural human emotion. 🫂 But remember, courage isn't the absence of fear, it's acting despite it. You're braver than you believe. What's scaring you? Maybe talking about it will help.", "I'm here with you. 💙 Being scared doesn't make you weak — it makes you human. Take a deep breath. Whatever you're facing, you don't have to face it alone. Want to talk about it?"],
    'heartbroken': ["I'm so sorry you're going through heartbreak. 💔 It's one of the most painful feelings in the world, and it's okay to grieve. Time does heal, even though it doesn't feel like it right now. Be gentle with yourself — you deserve love and kindness, especially from yourself.", "Heartbreak is incredibly tough, and I wish I could take that pain away. 🫂 But know this: every ending is also a beginning. You will love again, and you will be loved. For now, feel your feelings — they're valid. I'm here for you."],
    'depressed': ["I hear you, and I want you to know that your feelings are valid. 💙 Depression is real and it's not your fault. Please consider reaching out to a mental health professional — they can truly help. In the meantime, try to do one small thing today: take a walk, drink water, or call someone you trust. You matter.", "I'm really glad you told me how you're feeling. 🫂 Depression can make everything feel hopeless, but there IS hope. Please know: it's okay to ask for help. Crisis helpline: **988** (US) or **iCall: 9152987821** (India). You're not alone in this."],
    'overwhelmed': ["Feeling overwhelmed is your mind's way of saying 'too much at once'. 🌊 Let's slow down. Focus on just ONE thing right now — the smallest, easiest task. Everything else can wait. You don't have to have it all figured out today. One step at a time.", "I understand that overwhelming feeling. 💙 Here's what helps: write down everything on your mind, then circle just the top 3 priorities. The rest can wait. You're doing better than you think, even if it doesn't feel like it right now."],
    'motivated': ["That motivation is FIRE! 🔥💪 Channel that energy — you can achieve incredible things when you're in this zone. What are you working on? I'd love to cheer you on!", "Love that energy! 🚀 Motivation + action = unstoppable. Remember this feeling on the days when things get tough. You've got everything you need to succeed. Go crush it!"],
    'love': ["Love is the most beautiful emotion! ❤️ Whether it's love for a person, a passion, or life itself — it makes everything brighter. What's filling your heart with love today?", "Aww, that warms my heart! 🥰 Love makes the world go round. Cherish that feeling — it's one of life's greatest gifts."],
    'miss': ["Missing someone shows how much they mean to you. 💙 It's a bittersweet feeling — painful but also beautiful because it means you have deep connections. Maybe reach out to them? Even a simple 'thinking of you' message can mean the world.", "I understand that feeling of missing someone. 🫂 The people we miss have left an imprint on our hearts. That connection is precious. Have you thought about reaching out? They might be missing you too."],
}

_FEELING_PATTERNS = {
    'sad': [r'\b(sad|unhappy|crying|cry|tears|depressing|down|blue|miserable|heartache)\b', r'\bfeel(ing)?\s*(low|down|bad|empty|numb)\b'],
    'lonely': [r'\b(lonely|alone|isolated|nobody|no\s*one|no\s*friends)\b', r'\bfeel(ing)?\s*(lonely|alone|isolated)\b'],
    'stressed': [r'\b(stress|stressed|pressure|burnout|overwhelm|overwork)\b', r'\bunder\s*(pressure|stress)\b'],
    'anxious': [r'\b(anxious|anxiety|nervous|panic|worried|worrying|worry|fear|phobia)\b'],
    'tired': [r'\b(tired|exhausted|drained|burned\s*out|fatigue|sleepy|worn\s*out|no\s*energy)\b'],
    'bored': [r'\b(bored|boring|nothing\s*to\s*do|dull|monoton)\b'],
    'excited': [r'\b(excited|thrilled|pumped|hyped|cant\s*wait|can\'t\s*wait|ecstatic|stoked)\b'],
    'grateful': [r'\b(grateful|thankful|blessed|appreciate|gratitude)\b'],
    'confused': [r'\b(confused|confusing|don\'t\s*understand|dont\s*understand|lost|puzzled|bewildered)\b'],
    'scared': [r'\b(scared|afraid|terrified|frightened|fear|fearful|creep)\b'],
    'heartbroken': [r'\b(heartbr|broken\s*heart|breakup|broke\s*up|dumped|cheated|betrayed)\b'],
    'depressed': [r'\b(depress|hopeless|worthless|suicid|self\s*harm|don\'t\s*want\s*to\s*live|give\s*up|end\s*it)\b'],
    'overwhelmed': [r'\b(overwhelm|too\s*much|can\'t\s*cope|cant\s*cope|drowning|swamped)\b'],
    'motivated': [r'\b(motivat|inspired|determined|ready\s*to|gonna\s*do|going\s*to\s*do|pumped\s*up)\b'],
    'love': [r'\b(in\s*love|i\s*love|loving|soulmate|crush)\b'],
    'miss': [r'\b(miss\s*(you|her|him|them|my|someone)|missing\s*(someone|you|her|him))\b'],
}

_FALLBACK = [
    "That's a great question! I'd love to dive deeper into that. Could you share a bit more detail so I can give you the best answer? 🤔",
    "Interesting! I have knowledge on tons of topics — science, tech, history, geography, health, and more. Could you be more specific about what you'd like to know?",
    "I'd love to help with that! Could you give me a few more details? I can answer questions about pretty much anything! 😊",
    "Great question! Let me think... Could you rephrase or add more context? I want to make sure I give you the perfect answer! 💡",
]


def _match(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _find_capital(text):
    for country in _KNOWLEDGE['capital']:
        if country != 'default' and country in text:
            return _KNOWLEDGE['capital'][country]
    if 'capital' in text:
        return _KNOWLEDGE['capital']['default']
    return None


def _demo_response(user_message: str, chat_history: list) -> str:
    text = user_message.lower().strip()
    turns = len([m for m in chat_history if m["role"] == "user"])

    # Greetings
    if _match(text, _GREETINGS):
        return random.choice(_GREETING_RESPONSES)
    if _match(text, _HOWAREYOU):
        return random.choice(_HOWAREYOU_RESPONSES)
    if _match(text, _THANKS):
        return random.choice(_THANKS_RESPONSES)
    if _match(text, _GOODBYE):
        return random.choice(_GOODBYE_RESPONSES)

    # Follow-ups
    if turns > 1 and _match(text, _AFFIRMATIVE):
        return random.choice(_AFFIRMATIVE_RESPONSES)
    if turns > 1 and _match(text, _NEGATIVE):
        return random.choice(_NEGATIVE_RESPONSES)

    # Personal feelings & emotions (regex-based)
    for feeling, patterns in _FEELING_PATTERNS.items():
        if _match(text, patterns):
            return random.choice(_EMOTIONS[feeling])

    # Fallback emotion keywords
    if any(w in text for w in ['frustrat', 'annoying', 'terrible', 'worst', 'awful', 'horrible']):
        return random.choice(_EMOTIONS['frustrated'])
    if any(w in text for w in ['angry', 'mad', 'furious', 'pissed']):
        return random.choice(_EMOTIONS['angry'])
    if any(w in text for w in ['happy', 'amazing', 'awesome', 'great experience']):
        return random.choice(_EMOTIONS['happy'])

    # Capital cities
    cap = _find_capital(text)
    if cap:
        return cap

    # Knowledge base lookup
    for key, val in _KNOWLEDGE.items():
        if key == 'capital':
            continue
        if key in text:
            return val if isinstance(val, str) else random.choice(val)

    # Support topics
    for key, responses in _SUPPORT.items():
        if key in text:
            return random.choice(responses)

    # Catch-all patterns
    if any(w in text for w in ['tell me about', 'what is', 'what are', 'who is', 'who was', 'explain', 'define', 'meaning of']):
        return f"That's a great topic! While I have a broad knowledge base, for the most detailed and up-to-date information on that specific subject, I'd recommend checking a resource like Wikipedia or Google. In the meantime, feel free to ask me about science, technology, history, math, health, geography, or any support questions! 📚"

    if any(w in text for w in ['joke', 'funny', 'laugh']):
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😂",
            "What did the AI say to the human? 'I think we need to have a deep learning conversation.' 🤖😄",
            "Why was the computer cold? It left its Windows open! 💻❄️",
            "How do trees access the internet? They log in! 🌳😁",
        ]
        return random.choice(jokes)

    if any(w in text for w in ['who made you', 'who created you', 'who built you']):
        return "I was built by a talented team of developers as an AI Voice Chatbot! I use advanced natural language processing to understand and respond to you. Think of me as your friendly neighborhood AI assistant! 🤖✨"

    if any(w in text for w in ['name', 'your name', 'who are you']):
        return "I'm **VoiceBot AI** — your personal intelligent assistant! I can help with customer support, answer general knowledge questions, share fun facts, and have natural conversations. Nice to meet you! 🤖"

    if 'hindi' in text or 'हिंदी' in text:
        return "हां, मैं हिंदी में बात कर सकता हूं! 😊 बताइए, मैं आपकी कैसे मदद कर सकता हूं? (Yes, I can chat in Hindi! How can I help you?)"

    if 'weather' in text:
        return "I wish I could check the live weather for you! 🌤️ For accurate weather info, try weather.com or just ask 'Hey Google/Siri, what's the weather?' I'm better with knowledge questions, support, and conversations though!"

    if any(w in text for w in ['news', 'latest', 'current events', 'trending']):
        return "For the latest news, I'd recommend checking sources like Google News, BBC, or Reuters. I'm best at answering knowledge questions, helping with support, and having great conversations! 📰 Want to test my knowledge on any topic?"

    if any(w in text for w in ['calculate', 'math', 'equation', 'solve']):
        return "I can help with math concepts! For example, I know about Pi, Fibonacci, and mathematical theories. For actual calculations, a calculator or Wolfram Alpha would be more accurate. What math topic are you curious about? 🧮"

    return random.choice(_FALLBACK)


# ─────────────────────────────────────────────────────────
#  MAIN RESPONSE FUNCTION
# ─────────────────────────────────────────────────────────

async def generate_response(
    messages: List[Dict[str, str]],
    knowledge_context: Optional[str] = None,
    language: str = "en",
) -> str:
    ai_client = _get_client()

    if ai_client is None:
        last_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_msg = m["content"]
                break
        response = _demo_response(last_msg, messages)
        logger.info(f"[DEMO] '{last_msg[:40]}' -> '{response[:50]}...'")
        return response

    try:
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if language != "en":
            system_messages.append({"role": "system", "content": f"Respond in '{language}' language when appropriate."})
        if knowledge_context:
            system_messages.append({"role": "system", "content": f"Knowledge context:\n{knowledge_context}"})

        full_messages = system_messages + messages[-20:]
        response = await ai_client.chat.completions.create(
            model=settings.OPENAI_MODEL, messages=full_messages, temperature=0.7, max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"AI error: {e}")
        return "I'm having a little trouble right now. Could you try again? 😊"
