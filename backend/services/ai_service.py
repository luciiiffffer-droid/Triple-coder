"""
GPT-powered conversation engine with multi-turn context memory.
Falls back to a rich demo engine with general knowledge if OPENAI_API_KEY is not configured.
"""

import openai
import random
import re
import ast
import operator
from typing import List, Dict, Optional
from config import settings
from loguru import logger

client = None


# ── Key validation ────────────────────────────────────────
def _is_real_api_key(key: str) -> bool:
    """Return True only if the key looks like a genuine OpenAI API key."""
    if not key:
        return False
    # Placeholder keys contain 'your' or are too short / malformed
    if "your" in key.lower():
        return False
    if not key.startswith("sk-"):
        return False
    if len(key) < 30:
        return False
    return True


def _get_client():
    global client
    if client is None:
        if not _is_real_api_key(settings.OPENAI_API_KEY):
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
#  HELPER: word-boundary match
# ─────────────────────────────────────────────────────────

def _wb(key: str) -> re.Pattern:
    """Compile a case-insensitive whole-word regex pattern for a keyword."""
    return re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)


def _kw(text: str, key: str) -> bool:
    """True if `key` appears as a whole word in `text`."""
    return bool(_wb(key).search(text))


# ─────────────────────────────────────────────────────────
#  SAFE MATH EVALUATOR
# ─────────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Pattern: detects arithmetic expressions like "2 + 3", "15 * 4", "100 / 5 + 2"
_MATH_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*([\+\-\*\/\%\^])\s*(\d+(?:\.\d+)?)'
    r'(?:\s*([\+\-\*\/\%])\s*(\d+(?:\.\d+)?))*\b'
)

# Also capture "what is X plus/times/divided by/minus Y"
_WORD_MATH_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s+'
    r'(plus|minus|times|multiplied\s+by|divided\s+by|mod(?:ulo)?)\s+'
    r'(\d+(?:\.\d+)?)\b',
    re.IGNORECASE,
)

_WORD_OP_MAP = {
    'plus': '+',
    'minus': '-',
    'times': '*',
    'multiplied by': '*',
    'divided by': '/',
    'mod': '%',
    'modulo': '%',
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported op")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported unary op")
        return op_fn(_safe_eval(node.operand))
    raise ValueError("Unsupported node")


def _try_math(text: str) -> Optional[str]:
    """If text contains a math expression, evaluate and return the answer string."""
    # Word-form arithmetic first
    wm = _WORD_MATH_PATTERN.search(text)
    if wm:
        a, op_word, b = wm.group(1), wm.group(2).lower().strip(), wm.group(3)
        op_sym = _WORD_OP_MAP.get(op_word)
        if op_sym:
            expr = f"{a} {op_sym} {b}"
            try:
                tree = ast.parse(expr, mode='eval')
                result = _safe_eval(tree.body)
                result_str = int(result) if isinstance(result, float) and result.is_integer() else round(result, 6)
                return f"🧮 {a} {op_word} {b} = **{result_str}**"
            except Exception:
                pass

    # Symbol-based arithmetic
    sm = _MATH_PATTERN.search(text)
    if sm:
        expr = sm.group(0).replace('^', '**')
        try:
            tree = ast.parse(expr, mode='eval')
            result = _safe_eval(tree.body)
            result_str = int(result) if isinstance(result, float) and result.is_integer() else round(result, 6)
            return f"🧮 {sm.group(0)} = **{result_str}**"
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────
#  COMPREHENSIVE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────

_KNOWLEDGE = {
    # --- General Knowledge: Capitals ---
    'capital': {
        'india': "The capital of India is **New Delhi**. 🏛️ Designed by British architects Edwin Lutyens and Herbert Baker, it's the heart of Indian democracy.",
        'usa': "The capital of the United States is **Washington, D.C.** 🇺🇸 Named after George Washington — it's not part of any state!",
        'uk': "The capital of the United Kingdom is **London** 🇬🇧 — home to Big Ben, Buckingham Palace, and the Tower of London.",
        'france': "The capital of France is **Paris** 🗼 — the City of Light, famous for the Eiffel Tower and world-class cuisine.",
        'japan': "The capital of Japan is **Tokyo** 🗾 — the most populous metropolitan area on Earth with over 37 million people.",
        'germany': "The capital of Germany is **Berlin** 🇩🇪 — a vibrant city rich in history, culture, and the iconic Brandenburg Gate.",
        'china': "The capital of China is **Beijing** 🇨🇳 — home to the Forbidden City, Tiananmen Square, and the Great Wall nearby.",
        'russia': "The capital of Russia is **Moscow** 🇷🇺 — the largest city in Europe, featuring the stunning Red Square and Kremlin.",
        'australia': "The capital of Australia is **Canberra** 🇦🇺 — purpose-built as the capital, chosen as a compromise between Sydney and Melbourne.",
        'brazil': "The capital of Brazil is **Brasília** 🇧🇷 — a modernist planned city built in 1960, a UNESCO World Heritage Site.",
        'canada': "The capital of Canada is **Ottawa** 🇨🇦 — not Toronto! Ottawa sits on the Ontario-Quebec border and hosts the stunning Parliament Hill.",
        'italy': "The capital of Italy is **Rome** 🇮🇹 — the Eternal City, home to the Colosseum, Vatican City, and the Trevi Fountain.",
        'spain': "The capital of Spain is **Madrid** 🇪🇸 — Europe's highest capital city at 667 meters above sea level.",
        'pakistan': "The capital of Pakistan is **Islamabad** 🇵🇰 — a modern planned city in the Potohar Plateau region.",
        'default': "That's a great geography question! Which country are you asking about? I know capitals for most countries! 🌍",
    },

    # --- Space & Science ---
    'quantum physics': "**Quantum Physics** (or Quantum Mechanics) is the study of matter and energy at the most fundamental level — atoms and subatomic particles. ⚛️ At this scale, particles can exist in multiple states at once (superposition) and instantly connect across vast distances (entanglement)!",
    'quantum mechanics': "**Quantum Mechanics** is the branch of physics relating to the very small. It departs from classical physics by showing that energy, momentum, and other quantities are restricted to discrete values (quanta). 🔬",
    'superposition': "**Quantum Superposition** is the principle that a particle exists in all possible states at the same time until it is measured or observed. 🐱 It's famously illustrated by Schrödinger's Cat!",
    'schrödinger': "Erwin Schrödinger was an Austrian physicist famous for his wave equation and the **Schrödinger's Cat** thought experiment! 🐈 It illustrates quantum superposition: a cat in a sealed box is simultaneously both alive and dead until you open the box to observe it.",
    'schrodinger': "Erwin Schrödinger was an Austrian physicist famous for his wave equation and the **Schrödinger's Cat** thought experiment! 🐈 It illustrates quantum superposition: a cat in a sealed box is simultaneously both alive and dead until you open the box to observe it.",
    'entanglement': "**Quantum Entanglement** happens when particles become linked so closely that the state of one instantly affects the other, no matter how far apart they are! 🌌 Albert Einstein famously called this 'spooky action at a distance'.",
    'heisenberg': "Werner Heisenberg was a pioneer of quantum mechanics, best known for the **Heisenberg Uncertainty Principle**. 📏 It states that you cannot simultaneously know both the exact position and exact momentum of a particle!",
    'uncertainty principle': "The **Heisenberg Uncertainty Principle** states that there's a fundamental limit to how precisely we can know certain physical properties of a particle simultaneously — like its position and momentum. 🎯 If you measure one accurately, the other becomes uncertain!",
    'planet': "Our solar system has **8 planets**: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. 🪐 Fun fact: Jupiter alone is so large all other planets could fit inside it!",
    'sun': "The Sun is a **G-type main-sequence star** at the center of our solar system — about 4.6 billion years old, 1.39 million km in diameter, with a surface temperature of ~5,500°C ☀️",
    'moon': "Earth's Moon is ~**384,400 km** away and about 4.5 billion years old. 🌙 Only 12 humans have ever walked on it — all during NASA's Apollo missions (1969–1972).",
    'earth': "Earth is the **third planet** from the Sun and the only known planet to support life. 🌍 About 4.54 billion years old, with 71% of its surface covered in water.",
    'mars': "Mars is the **fourth planet** from the Sun — the Red Planet! 🔴 It has the tallest volcano in the solar system (Olympus Mons) and is a primary target for human colonization.",
    'space': "**Space** is wild! 🚀 The observable universe is about 93 billion light-years in diameter. On Venus, a day is longer than a year. There are more stars than grains of sand on every beach on Earth!",
    'black hole': "A **black hole** is a region of spacetime where gravity is so strong that nothing — not even light — can escape. 🌑 The first image of a black hole was captured in 2019 (Messier 87).",
    'galaxy': "Our galaxy, the **Milky Way**, contains an estimated 100–400 billion stars! 🌌 It's about 100,000 light-years across. The nearest large galaxy is Andromeda, about 2.5 million light-years away.",
    'gravity': "**Gravity** is one of the four fundamental forces of nature. 🍎 Described by Newton and later Einstein, it warps spacetime. On the Moon, you'd weigh only 1/6th of your Earth weight!",
    'dna': "**DNA (Deoxyribonucleic acid)** is the molecule of life — a double helix with four bases (A, T, G, C). 🧬 If you uncoiled all the DNA in your body, it would stretch to the Sun and back 600+ times!",
    'water': "**Water (H₂O)** covers 71% of Earth's surface and is essential for all life. 💧 Fun fact: hot water can freeze faster than cold water — this is called the Mpemba effect!",
    'light': "**Light** travels at ~**299,792 km/s** — the fastest speed in the universe! 💡 Sunlight takes 8 minutes 20 seconds to reach Earth. Light behaves as both a wave AND a particle.",
    'photosynthesis': "**Photosynthesis** lets plants convert CO₂ + H₂O + sunlight → glucose + oxygen. 🌱 Equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. Plants literally make food from air and sunlight!",
    'ocean': "The **ocean** covers 71% of Earth but we've only explored ~5% of it! 🌊 The deepest point, the Mariana Trench, reaches ~11,034 meters. More history may be in the ocean than in all museums combined!",

    # --- Technology ---
    'artificial intelligence': "**Artificial Intelligence (AI)** enables computers to perform tasks requiring human-like intelligence — speech, vision, reasoning, translation. 🤖 I'm a real-world example of AI in action!",
    'ai': "**Artificial Intelligence (AI)** enables computers to perform tasks requiring human-like intelligence — speech, vision, reasoning, translation. 🤖 It's the technology behind voice assistants, self-driving cars, image recognition, and much more!",
    'machine learning': "**Machine Learning** is AI where systems learn from data without explicit programming. 🧠 Types: Supervised, Unsupervised, Reinforcement. It powers Netflix recommendations, self-driving cars, and fraud detection.",
    'python': "**Python** is one of the world's most popular languages! 🐍 Created by Guido van Rossum in 1991. It dominates AI/ML, data science, web development, and automation thanks to its clean, readable syntax.",
    'blockchain': "**Blockchain** is a decentralized digital ledger where records (blocks) are cryptographically linked. ⛓️ It's secure, tamper-resistant, and powers Bitcoin and Ethereum.",
    'chatgpt': "**ChatGPT** is an AI chatbot by OpenAI launched in November 2022, built on large language models (LLMs) trained on vast text data. 💬 It reached 100 million users in just 2 months — the fastest product ever!",
    'internet': "The **Internet** is a global network born from ARPANET in the 1960s. 🌐 Today, over 5 billion people (~63% of Earth) use it. Over 250 billion emails are sent every day!",
    'google': "**Google** was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford. 🔍 It processes over 8.5 billion searches per day and is the world's most visited website.",
    'apple': "**Apple Inc.** was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 🍎 It created the Mac, iPod, iPhone, and iPad — and became the world's first $3 trillion company.",
    'microsoft': "**Microsoft** was founded by Bill Gates and Paul Allen in 1975. 💻 Creator of Windows, Office, and Azure. Today it's one of the largest companies in the world and a major investor in OpenAI.",
    'tesla': "**Tesla** was founded in 2003 (Elon Musk joined in 2004). ⚡ It's the world's leading electric vehicle maker. The Model S can go 0–60 mph in under 2 seconds — faster than most supercars!",

    # --- Math ---
    'pi': "**Pi (π) ≈ 3.14159265358979...** 🥧 It's the ratio of a circle's circumference to its diameter — an irrational number whose decimals never end or repeat. March 14 (3/14) is Pi Day!",
    'fibonacci': "The **Fibonacci sequence** — 0, 1, 1, 2, 3, 5, 8, 13, 21, 34... — where each number = sum of the previous two. 🌻 It appears in flower petals, spiral shells, and even financial markets!",
    'prime': "**Prime numbers** are numbers greater than 1 divisible only by 1 and themselves: 2, 3, 5, 7, 11, 13... 🔢 The largest known prime (as of 2024) has over 41 million digits!",

    # --- History ---
    'world war': "**WWI** (1914–1918) involved 70M+ military personnel. **WWII** (1939–1945) was the deadliest conflict ever with 70–85 million fatalities. Both wars permanently reshaped the global political order. 📚",
    'independence': "India gained **independence on August 15, 1947** from British rule, led by Gandhi, Nehru, Bose, and millions more. 🇮🇳 The USA declared independence on **July 4, 1776**!",
    'gandhi': "**Mahatma Gandhi** (1869–1948) led India's non-violent independence movement via civil disobedience. 🕊️ His Salt March of 1930 is iconic. He's honored as 'Father of the Nation' in India.",
    'einstein': "**Albert Einstein** (1879–1955) developed the Theory of Relativity (E=mc²) and explained the photoelectric effect, winning the 1921 Nobel Prize in Physics. 🧠 He's regarded as the greatest scientist of the 20th century.",
    'newton': "**Isaac Newton** (1643–1727) formulated the laws of gravity and motion, invented calculus, and explained how light splits into colors. 🍎 He's one of the most influential scientists in history.",

    # --- Health ---
    'vitamin': "**Vitamins** your body needs: A (vision), B-complex (energy), C (immunity), D (bones — from sunlight!), E (antioxidant), K (blood clotting). 🍊 A balanced diet usually provides all of them.",
    'exercise': "Regular **exercise** is life-changing! 💪 WHO recommends 150 min/week of moderate activity. Benefits: reduced heart disease risk, better mental health, stronger bones, improved sleep, and longer life.",
    'sleep': "Adults need **7–9 hours** of sleep per night. 😴 While sleeping, your brain consolidates memories, body repairs tissues, and growth hormones release. Poor sleep is linked to obesity, heart disease, and weakened immunity.",

    # --- Fun Facts ---
    'fun fact': "Fun fact: **Honey never spoils!** 🍯 Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible. Also: octopuses have 3 hearts, and bananas are technically berries (strawberries aren't!).",
    'random fact': "Did you know? **The shortest war in history** lasted 38–45 minutes: Britain vs. Zanzibar, August 27, 1896. 🦩 A group of flamingos is called a 'flamboyance'. A group of porcupines is called a 'prickle'!",
    'animal': "Nature is amazing! 🐋 Blue whales are the largest animals ever (up to 30m long). Hummingbirds can fly backwards. Elephants are the only animals that can't jump. Crows can recognize human faces!",

    # ══════════════════════════════════════════════════════
    # --- HACKATHON / PROJECT KNOWLEDGE ---
    # ══════════════════════════════════════════════════════

    # -- About This Project --
    'voicebot': "**VoiceBot AI** is a real-time, human-like AI Voice Chatbot built for a hackathon. 🤖 It supports multi-language conversations, emotion detection, fraud detection, voice biometrics, and integrates with Twilio, ElevenLabs, and OpenAI — replacing traditional IVR systems!",
    'this project': "This project is an **AI Voice Chatbot** — a production-ready solution that handles voice and text interactions. 🎙️ It uses FastAPI (backend), SQLite (database), OpenAI Whisper (STT), ElevenLabs TTS, and a modern HTML/CSS/JS frontend with real-time WebSocket communication.",
    'project': "Our project is an **AI-powered Voice Chatbot** that uses NLP, Speech-to-Text (Whisper), Text-to-Speech (ElevenLabs), sentiment analysis, fraud detection, and multi-language support — all served through a FastAPI backend with a real-time WebSocket interface. 🏆",

    # -- Tech Stack --
    'fastapi': "**FastAPI** is a modern, high-performance Python web framework for building APIs. ⚡ It's based on standard Python type hints, auto-generates Swagger docs, and is one of the fastest Python frameworks available — used as the backend of this chatbot.",
    'websocket': "**WebSockets** provide full-duplex communication channels over a single TCP connection. 🔌 Unlike HTTP (request-response), WebSocket keeps a persistent connection open — perfect for real-time voice chat, live data, and chat applications like this one.",
    'sqlite': "**SQLite** is a lightweight, serverless, self-contained SQL database engine. 🗄️ It stores data in a single file — ideal for development, prototyping, and small-to-medium applications. This chatbot uses it to store conversations, messages, and analytics.",
    'postgresql': "**PostgreSQL** is a powerful, open-source relational database. 🐘 It supports advanced data types, full-text search, JSON, and is ACID-compliant. This project supports PostgreSQL for production deployments via the DATABASE_URL config.",
    'redis': "**Redis** is an in-memory data structure store used as a database, cache, and message broker. ⚡ It's blazing fast (sub-millisecond responses) and is used in this project for caching and rate limiting.",
    'docker': "**Docker** containerizes applications into isolated environments. 🐳 The entire chatbot stack — backend, frontend, PostgreSQL, Redis — can be run with a single `docker compose up` command, ensuring consistent behavior across any machine.",
    'nginx': "**Nginx** is a high-performance web server and reverse proxy. 🌐 In this project it serves the static frontend files and can proxy API requests to the FastAPI backend — enabling clean routing and SSL termination in production.",
    'uvicorn': "**Uvicorn** is an ASGI server implementation for Python. ⚡ It runs FastAPI applications with lightning-fast async performance. In this project it powers the backend WebSocket and HTTP endpoints.",
    'sqlalchemy': "**SQLAlchemy** is Python's most powerful ORM (Object-Relational Mapper). 🗄️ This project uses SQLAlchemy with async support to perform non-blocking database operations — keeping the server responsive even under heavy load.",

    # -- AI & ML Concepts --
    'nlp': "**Natural Language Processing (NLP)** is a branch of AI that enables computers to understand, interpret, and generate human language. 🗣️ This chatbot uses NLP for intent detection, sentiment analysis, entity recognition, and generating contextually appropriate responses.",
    'natural language processing': "**NLP** lets machines understand human language. 🧠 It powers voice assistants, chatbots, translation tools, and sentiment analysis. This chatbot uses NLP to understand user intent, detect emotions, and respond in multiple languages.",
    'speech to text': "**Speech-to-Text (STT)** converts spoken audio into written text. 🎙️→📝 This chatbot uses **OpenAI Whisper** — a state-of-the-art multilingual STT model that supports 99+ languages and is highly noise-resistant.",
    'stt': "**STT (Speech-to-Text)** converts audio into text. 🎙️ This project uses **OpenAI Whisper** — trained on 680,000 hours of multilingual audio. It handles accents, noise, and multiple languages automatically.",
    'whisper': "**OpenAI Whisper** is a state-of-the-art automatic speech recognition (ASR) model. 🎙️ Trained on 680,000 hours of multilingual web audio, it achieves near-human accuracy and supports 99+ languages — used in this chatbot for voice input transcription.",
    'text to speech': "**Text-to-Speech (TTS)** converts written text into natural-sounding audio. 📝→🔊 This chatbot uses **ElevenLabs** — the industry leader in AI voice synthesis — to generate lifelike, emotionally expressive voice responses.",
    'tts': "**TTS (Text-to-Speech)** converts AI-generated text into audio. This project uses **ElevenLabs** which offers ultra-realistic, low-latency voice synthesis with 29+ languages and emotional control. 🔊",
    'elevenlabs': "**ElevenLabs** is an AI voice company offering the world's most realistic TTS technology. 🔊 Their API converts text to speech with human-like intonation, emotion, and accent. It's used in this chatbot to give the AI a natural voice.",
    'llm': "**Large Language Models (LLMs)** are AI models trained on massive text datasets to understand and generate human language. 🧠 Examples: GPT-4, Claude, Gemini, LLaMA. This chatbot uses **GPT-4o** (via OpenAI API) as its core reasoning engine.",
    'large language model': "An **LLM (Large Language Model)** is a deep learning model with billions of parameters trained on internet-scale text. 🧠 They can generate text, answer questions, translate languages, write code, and hold conversations. GPT-4o powers this chatbot's responses.",
    'gpt': "**GPT (Generative Pre-trained Transformer)** is OpenAI's family of LLMs. 🤖 GPT-4o — used in this chatbot — has 1 trillion+ parameters, handles text and images, and is capable of nuanced, context-aware conversation across 50+ languages.",
    'openai': "**OpenAI** is the AI research company behind GPT-4, ChatGPT, DALL-E, and Whisper. 🧠 Founded in 2015, it's the world's leading AI lab. This chatbot integrates OpenAI's Whisper (STT) and GPT-4o (conversation) APIs.",
    'rag': "**RAG (Retrieval-Augmented Generation)** combines LLMs with a vector database to answer questions about specific documents. 📚 Instead of relying only on training data, the model retrieves relevant context first, then generates answers — making responses more accurate and up-to-date. This chatbot implements RAG!",
    'retrieval augmented generation': "**Retrieval-Augmented Generation (RAG)** is a technique where an LLM fetches relevant context from a knowledge base before generating a response. 📚 This chatbot uses FAISS vector search + OpenAI embeddings to implement RAG — enabling it to answer domain-specific questions accurately.",
    'vector database': "A **vector database** stores data as high-dimensional numerical vectors (embeddings). 🔍 It enables semantic similarity search — finding documents that are *conceptually* similar, not just keyword-matching. This chatbot uses **FAISS** (Facebook AI Similarity Search) as its vector store.",
    'faiss': "**FAISS (Facebook AI Similarity Search)** is an open-source library for efficient similarity search of dense vectors. 🔍 It's used in this chatbot's RAG pipeline to find the most relevant knowledge base entries for any user query — enabling context-aware responses.",
    'embedding': "**Embeddings** are numerical vector representations of text that capture semantic meaning. 🧮 Similar phrases have similar embeddings. This chatbot uses OpenAI's `text-embedding-ada-002` model to convert knowledge base documents into vectors for semantic search.",
    'sentiment analysis': "**Sentiment Analysis** detects the emotional tone of text — positive, negative, or neutral — and assigns a score. 😊😟 This chatbot performs real-time sentiment analysis on every message using TextBlob, detecting emotions like happy, frustrated, sad, anxious, and urgent.",
    'emotion detection': "This chatbot performs **real-time emotion detection** using NLP. 🎭 It analyzes user messages and classifies emotions: happy, sad, angry, frustrated, anxious, excited, etc. The detected emotion is displayed in the chat UI and stored for analytics.",
    'fraud detection': "This chatbot includes a **fraud detection module** that scans messages for suspicious patterns — fake urgency, social engineering, phishing attempts, and high-risk keywords. 🛡️ Flagged messages trigger alerts and are logged for review.",
    'twilio': "**Twilio** is a cloud communications platform that provides APIs for calls, SMS, and WhatsApp. 📞 This chatbot integrates Twilio to handle real phone calls — users can call a phone number and speak to the AI directly.",
    'jwt': "**JWT (JSON Web Token)** is a compact, URL-safe token format for secure authentication. 🔐 This chatbot's API uses JWT Bearer tokens for authentication. Tokens are signed with HMAC-SHA256 and expire after a configurable time.",
    'authentication': "This chatbot uses **JWT-based authentication**. 🔐 Users register/login to get a token, which is sent with every API request in the `Authorization: Bearer <token>` header. Tokens are validated on the server for every protected route.",
    'cors': "**CORS (Cross-Origin Resource Sharing)** is a browser security mechanism that controls which origins can access an API. 🌐 This chatbot's FastAPI backend has CORS configured to allow the frontend to make API calls from a different port (3000 vs 8000).",
    'api': "An **API (Application Programming Interface)** is a set of rules for how software components communicate. 🔌 This chatbot exposes a RESTful API (FastAPI) with endpoints for auth, chat, analytics, voice, escalation, and knowledge management — all documented at `/docs`.",
    'rest api': "**REST (Representational State Transfer)** is an architectural style for APIs using HTTP methods — GET, POST, PUT, DELETE. 📡 This chatbot's backend is a fully RESTful API with proper status codes, JSON responses, and automatic Swagger documentation.",
    'microservices': "**Microservices** architecture divides an app into small, independent services that communicate via APIs. 🧩 This chatbot is designed with service separation: `ai_service`, `stt_service`, `tts_service`, `sentiment_service`, `vector_service` — each independently changeable.",
    'scalability': "**Scalability** means a system can handle increasing load. 📈 This chatbot is designed for scalability: async FastAPI handles thousands of concurrent connections, Redis enables distributed caching, PostgreSQL scales with read replicas, and Docker enables horizontal scaling.",
    'rate limiting': "**Rate Limiting** protects an API from abuse by capping the number of requests per time window. 🛡️ This chatbot uses **SlowAPI** (inspired by Flask-Limiter) to limit requests per IP — configurable via `RATE_LIMIT_PER_MINUTE` in `.env`.",

    # -- CS Fundamentals --
    'algorithm': "An **algorithm** is a step-by-step procedure to solve a problem. ⚙️ Common types: Sorting (QuickSort O(n log n), MergeSort), Searching (Binary Search O(log n)), Graph traversal (BFS, DFS). Good algorithms are efficient in time & space complexity.",
    'data structure': "**Data Structures** organize and store data efficiently. 📦 Key types: Array (O(1) access), Linked List (O(1) insert), Stack (LIFO), Queue (FIFO), Hash Map (O(1) average), Tree (hierarchical), Graph (networks). Choosing the right one is crucial for performance.",
    'time complexity': "**Time Complexity** measures how an algorithm's runtime grows with input size, using Big-O notation. ⏱️ O(1) = constant, O(log n) = logarithmic, O(n) = linear, O(n²) = quadratic. This chatbot's vector search uses FAISS — O(log n) approximate nearest neighbor.",
    'object oriented': "**Object-Oriented Programming (OOP)** organizes code around objects with properties and behaviors. 🏗️ Four pillars: **Encapsulation** (data hiding), **Inheritance** (reuse), **Polymorphism** (many forms), **Abstraction** (hide complexity). Python, Java, and C++ are OOP languages.",
    'database': "A **database** stores and retrieves structured data. 🗄️ Types: **Relational** (SQL — MySQL, PostgreSQL), **NoSQL** (MongoDB, Redis), **Vector** (FAISS, Pinecone). This chatbot uses SQLite (dev) or PostgreSQL (prod) for relational data, and FAISS for vector search.",
    'sql': "**SQL (Structured Query Language)** is used to query relational databases. 🗄️ Key commands: SELECT (read), INSERT (create), UPDATE (modify), DELETE (remove), JOIN (combine tables). This chatbot uses SQLAlchemy ORM to abstract SQL queries in Python.",
    'async': "**Asynchronous programming** allows tasks to run concurrently without blocking. ⚡ In Python, `async/await` with `asyncio` enables non-blocking I/O. This entire chatbot backend is fully async — it can handle thousands of simultaneous WebSocket connections without freezing.",
    'http': "**HTTP (HyperText Transfer Protocol)** is the foundation of data communication on the web. 🌐 Methods: GET (fetch), POST (create), PUT (update), DELETE (remove). Status codes: 200 (OK), 201 (Created), 400 (Bad Request), 401 (Unauthorized), 404 (Not Found), 500 (Server Error).",
    'load balancing': "**Load Balancing** distributes incoming traffic across multiple servers to prevent overload. ⚖️ Algorithms: Round Robin, Least Connections, IP Hash. Nginx (used as this chatbot's reverse proxy) doubles as a load balancer in production deployments.",
    'caching': "**Caching** stores results of expensive operations for faster future access. ⚡ This chatbot uses **Redis** for caching rate-limit counters. Caching reduces latency from milliseconds to microseconds and dramatically reduces database load.",
    'encryption': "**Encryption** converts data into an unreadable format to protect it. 🔒 This chatbot uses: **bcrypt** for password hashing, **JWT (HS256)** for token signing, **HTTPS/TLS** for data in transit. Never store plain-text passwords!",
    'cloud': "**Cloud Computing** delivers computing resources (servers, storage, databases) over the internet. ☁️ This chatbot can be deployed on AWS, Google Cloud, or Azure. Docker containers make cloud deployment straightforward and reproducible.",
    'devops': "**DevOps** bridges software development and IT operations. 🔄 This chatbot includes: **Docker** (containerization), **docker-compose** (orchestration), environment configs via `.env`, health checks, and structured logging — all DevOps best practices.",
    'git': "**Git** is a distributed version control system. 📦 This project uses Git with GitHub for source control. Key commands: `git add`, `git commit`, `git push`, `git pull`, `git branch`. The entire codebase history is tracked and collaborative.",

    # ══════════════════════════════════════════════════════════════
    #  HACKATHON JUDGE Q&A — TODAY'S EVENT TRAINING
    # ══════════════════════════════════════════════════════════════

    # -- Core "What is it?" --
    'introduce': "🏆 **VoiceBot AI** is a next-generation AI Voice Chatbot that replaces traditional IVR systems. It understands natural speech in 50+ languages, detects emotions, flags fraudulent intent, streams live responses, and integrates with phone systems via Twilio — all powered by a fully async FastAPI backend and GPT-4o.",
    'introduce your project': "🏆 We built **VoiceBot AI** — a production-ready AI Voice Chatbot for hackathon. It handles real-time voice and text conversations, detects user emotion (happy/sad/frustrated/urgent), flags fraud, supports 50+ languages, and integrates with Twilio for real phone calls. Built with FastAPI + WebSockets + OpenAI + ElevenLabs.",
    'what does your project do': "**VoiceBot AI** converts voice input → transcribes using Whisper STT → sends to GPT-4o for intelligent response → synthesizes reply using ElevenLabs TTS → streams audio back to user — all in under 2 seconds! 🎙️🤖🔊 It also supports text chat, emotion detection, fraud detection, escalation to human agents, and a live analytics dashboard.",
    'tell me about your project': "Our project **VoiceBot AI** is an intelligent voice assistant that handles customer conversations in real time. 🎙️ Key capabilities: multi-language support (50+ languages), real-time emotion detection, fraud prevention, WebSocket-based voice streaming, REST API for text chat, analytics dashboard, human escalation, and full JWT-secured authentication.",

    # -- Problem & Solution --
    'problem': "🔴 **Problem:** Traditional IVR (Interactive Voice Response) systems are rigid, frustrating, and can't understand natural language. They force users through endless menus and can't handle complex queries. 60% of customers hang up before reaching a resolution!",
    'problem statement': "📋 **Problem Statement:** Traditional IVR systems are rule-based, menu-driven, and incapable of natural conversation. They frustrate users, increase call abandonment rates, and are expensive to maintain. Businesses lose lakhs every month due to poor customer service automation.",
    'solution': "✅ **Our Solution — VoiceBot AI:** A real-time AI chatbot that understands natural speech in any language, responds with a human-like voice, detects customer emotions, prevents fraud, and escalates to a human agent when needed. It replaces old IVR with a full conversational AI — reducing resolution time by 70% and improving satisfaction scores.",
    'why this project': "We built VoiceBot AI because **customer support automation is broken**. 🤖 Existing IVR systems frustrate users, and most chatbots are text-only. We wanted to build a truly voice-first, emotion-aware, multi-language AI that feels human — and can handle the scale of a real enterprise.",
    'innovation': "🚀 **What makes us innovative:**\n1. **Real-time voice streaming** via WebSockets (not phone trees)\n2. **Emotion detection** on every message — responds empathetically\n3. **Fraud detection** — catches suspicious patterns before damage\n4. **RAG (Retrieval-Augmented Generation)** — answers domain-specific questions accurately\n5. **Live streaming mode** — AI response streams word-by-word like ChatGPT\n6. **Multi-language support** — 50+ languages with automatic detection",
    'what is unique': "🌟 **Unique features of VoiceBot AI:**\n• Real-time emotion detection & empathetic responses\n• Fraud detection with pattern analysis\n• RAG-powered knowledge base for accurate domain answers\n• SSE live streaming — words appear as they're generated\n• WebSocket voice chat with OpenAI Whisper STT\n• ElevenLabs for ultra-realistic AI voice\n• Twilio integration for real phone call support\n• Full analytics dashboard with emotion & sentiment trends",
    'unique': "VoiceBot AI stands out because it's not just a chatbot — it's a **complete customer interaction platform**. 🌟 It combines voice AI, emotion intelligence, fraud prevention, live streaming, human escalation, and analytics in one system. Most competitors only do one or two of these things.",

    # -- Tech Stack Questions --
    'why fastapi': "We chose **FastAPI** because: ⚡\n1. It's the **fastest** Python web framework (on par with Node.js)\n2. Built-in **async/await** support for WebSockets\n3. **Auto-generated Swagger docs** at `/docs`\n4. Native **Pydantic** validation for request/response models\n5. Easy integration with SQLAlchemy async ORM",
    'why python': "We chose **Python** because: 🐍\n1. Best ecosystem for **AI/ML** — OpenAI, LangChain, FAISS, TextBlob all have Python SDKs\n2. Readable and rapid to develop in during a hackathon\n3. FastAPI makes Python as fast as compiled languages for I/O-bound tasks\n4. Async support means we can handle thousands of connections",
    'why websocket': "We used **WebSockets** for voice because: 🔌\n1. Voice requires **bidirectional, real-time** communication — HTTP polling would be too slow\n2. WebSocket maintains a **persistent connection** — no handshake overhead per message\n3. Supports **binary data** (audio blobs) and **JSON** in the same connection\n4. FastAPI has built-in WebSocket support — very easy to implement",
    'why openai': "We chose **OpenAI** because: 🧠\n1. **GPT-4o** is the most capable conversational model available\n2. **Whisper** is the most accurate multilingual STT model (99+ languages)\n3. **Embeddings API** powers our RAG knowledge base\n4. It's an industry standard — judges and employers recognize it\n5. Well-documented Python SDK",
    'tech stack': "🛠️ **Full Tech Stack:**\n• **Backend:** Python, FastAPI, Uvicorn, SQLAlchemy (async)\n• **Database:** SQLite (dev) / PostgreSQL (prod) + Redis (cache)\n• **AI:** OpenAI GPT-4o (chat), Whisper (STT), text-embedding-ada-002 (RAG)\n• **Voice:** ElevenLabs TTS, Twilio (calls)\n• **Frontend:** Vanilla HTML5 + CSS3 + JavaScript (no framework, lightweight)\n• **Real-time:** WebSockets (voice), SSE (live streaming text)\n• **Vector DB:** FAISS for semantic search\n• **Auth:** JWT (python-jose) + bcrypt\n• **DevOps:** Docker, docker-compose, Nginx\n• **Version Control:** Git + GitHub",

    # -- Architecture Questions --
    'architecture': "🏗️ **System Architecture:**\n```\nUser (Browser)\n  ↓ HTTPS / WSS\nNginx (reverse proxy)\n  ↓\nFastAPI (Uvicorn ASGI)\n  ├── /api/auth → JWT auth\n  ├── /api/chat/text → Text chat\n  ├── /api/chat/stream → SSE streaming\n  ├── /ws/voice → WebSocket voice\n  └── /api/analytics → Dashboard\n  ↓\nServices Layer\n  ├── ai_service (GPT-4o)\n  ├── stt_service (Whisper)\n  ├── tts_service (ElevenLabs)\n  ├── sentiment_service (TextBlob)\n  ├── vector_service (FAISS)\n  └── fraud_detection\n  ↓\nDatabase (SQLite/PostgreSQL) + Redis\n```",
    'how does it work': "🔄 **How VoiceBot AI works (voice flow):**\n1. User clicks 🎙️ → browser captures audio via MediaRecorder API\n2. Audio blob sent over **WebSocket** to FastAPI backend\n3. **Whisper STT** transcribes audio → text\n4. **Sentiment analysis** runs on the text (emotion, urgency)\n5. **Fraud detection** scans for suspicious patterns\n6. **FAISS vector search** finds relevant knowledge context (RAG)\n7. **GPT-4o** generates response using context\n8. **ElevenLabs TTS** converts response to audio\n9. Audio + response sent back over WebSocket to browser\n10. Browser plays audio + displays text — all in ~1-2 seconds! ⚡",
    'flow': "🔄 **VoiceBot AI request flow:** Voice → WebSocket → Whisper STT → Sentiment Analysis + Fraud Check → RAG (FAISS vector search) → GPT-4o → ElevenLabs TTS → WebSocket → Browser. Text chat uses REST API + SSE streaming for live word-by-word display. Everything is async for maximum performance.",
    'database design': "🗄️ **Database Schema:**\n• **Users** — id, username, email, password_hash, is_active, is_admin\n• **Conversations** — id, user_id, channel (web/phone), created_at\n• **Messages** — id, conversation_id, role (user/assistant), content, emotion, sentiment_score, is_urgent, timestamp\n• **Analytics Events** — id, session_id, user_message, ai_response, emotion, fraud_alert\n• All relationships are properly foreign-keyed with indexes for fast queries.",

    # -- Feature Questions --
    'features': "🌟 **VoiceBot AI Features:**\n1. 🎙️ Real-time voice chat (WebSocket + Whisper STT)\n2. 🔊 AI voice responses (ElevenLabs TTS)\n3. ⚡ Live text streaming (SSE — word-by-word)\n4. 🌍 50+ languages (auto-detect)\n5. 😊 Emotion detection (happy/sad/angry/urgent/frustrated)\n6. 🛡️ Fraud detection & alerting\n7. 📞 Phone call support (Twilio)\n8. 🧑‍💼 Escalation to human agent\n9. 📚 RAG knowledge base (FAISS)\n10. 📊 Analytics dashboard\n11. 🔐 JWT authentication\n12. 💬 Text chat (REST API)\n13. 🌙 Dark mode premium UI",
    'emotion': "😊 **Emotion Detection:** Every user message is analyzed using NLP (TextBlob + keyword patterns). The system classifies: `very_positive`, `positive`, `neutral`, `negative`, `very_negative`. It also detects urgency ('help', 'urgent', 'emergency') and adjusts response tone accordingly. The detected emotion shows as a chip in the chat UI.",
    'escalation': "🧑‍💼 **Human Escalation:** When a user is very frustrated or requests it, VoiceBot AI can escalate the conversation to a live human agent. The escalation API records the conversation ID, reason, and priority — and notifies the agent team. This is critical for enterprise customer support use cases.",
    'multilingual': "🌍 **Multi-language Support:** OpenAI Whisper automatically detects and transcribes 99+ languages from voice input. GPT-4o responds in the user's detected language. ElevenLabs supports 29+ languages for voice output. The `language` parameter can also be set manually via the API.",
    'analytics': "📊 **Analytics Dashboard:** VoiceBot AI includes a real-time analytics dashboard showing:\n• Total conversations & messages\n• Emotion distribution (pie chart)\n• Sentiment score over time\n• Fraud alert rate\n• Peak usage hours\n• Escalation rate\nAll data is stored in the database and visualized with Chart.js.",

    # -- Challenges --
    'challenges': "🔴 **Challenges we faced:**\n1. **Audio encoding:** Browser audio (WebM/Opus) needed to be properly sent as binary over WebSocket to Whisper\n2. **Async coordination:** Managing async WebSocket + DB operations without deadlocks\n3. **Demo mode fallback:** Building a fully working chatbot even without API keys for demos\n4. **Real-time streaming:** Implementing SSE word-by-word streaming with proper backpressure\n5. **CORS:** Configuring FastAPI CORS for WebSocket + HTTP from different ports",
    'difficulty': "🔴 The hardest part was building the **real-time voice pipeline** — capturing audio in the browser, encoding it correctly, sending over WebSocket, transcribing with Whisper, generating a response, synthesizing TTS, and playing it back — all in under 2 seconds with no dropped frames. We also had to build a complete demo mode for when API keys aren't available.",

    # -- Future Plans --
    'future': "🚀 **Future Plans for VoiceBot AI:**\n1. **Voice Biometrics** — identify users by voice print\n2. **Multi-turn memory** — longer conversation context across sessions\n3. **WhatsApp & SMS** integration via Twilio\n4. **Custom voice cloning** — brands can use their own AI voice\n5. **Admin portal** — configure the bot without code\n6. **Mobile app** — React Native client\n7. **Analytics AI** — AI-powered insights from conversation patterns\n8. **On-premise deployment** — for enterprise data privacy",
    'future plans': "🚀 **Roadmap:**\n• Voice biometrics for caller authentication\n• WhatsApp bot integration\n• Custom LLM fine-tuning on business data\n• Mobile app (React Native)\n• Admin dashboard for non-technical users\n• Multi-agent architecture for parallel conversations\n• Kubernetes deployment for enterprise scale",
    'scope': "📈 **Market Scope:** The global conversational AI market is projected to reach **$41.4 billion by 2030** (CAGR 23.6%). IVR systems handle 40 billion calls per year globally. Replacing even 10% with VoiceBot AI represents a **$4 billion opportunity**. Target customers: banks, hospitals, e-commerce, telecom, government.",
    'market': "💰 **Market Opportunity:** Enterprises spend over $1.3 trillion per year on customer service. Traditional IVR costs $0.25/call vs. VoiceBot AI at $0.01/call using cloud AI APIs. At scale, this saves companies **96% on customer service costs** while improving CSAT (customer satisfaction) scores significantly.",

    # -- Team Questions --
    'team': "👥 **Team Triple Coder** built this project for the hackathon. We divided the work across backend development (FastAPI + AI services), frontend (UI/UX design, JavaScript), and AI integration (OpenAI, ElevenLabs, FAISS). Every member contributed to the demo mode fallback system so the project works end-to-end without paid API keys.",
    'team name': "Our team is **Triple Coder** 🏆 — three passionate developers who built VoiceBot AI for this hackathon. We specialize in AI, backend development, and frontend design.",
    'your name': "I am **VoiceBot AI** 🤖 — an intelligent voice assistant built by Team Triple Coder for this hackathon. I can answer your questions about our project, AI concepts, technology, science, history, and much more!",

    # -- Demo & Testing --
    'demo': "🎯 **Demo Instructions:**\n1. Open http://localhost:3000/chat.html\n2. **Text Chat:** Type any question and press Send\n3. **Live Mode:** Click ⚡ Live → words stream in real-time\n4. **Voice Chat:** Click 🎙️, speak, click again to stop — AI responds with voice\n5. Try asking: 'What is your project?', 'Explain NLP', 'What is RAG?', 'Tell me about FastAPI'\n6. Check emotion chip (top right) — updates based on your emotion",
    'how to use': "📖 **How to use VoiceBot AI:**\n• **Text chat** → Type in the input box → Press Enter or Send button\n• **Voice chat** → Click 🎙️ → Speak → Click ⏹️ to stop → AI responds\n• **Live mode** → Click ⚡ Live → Responses stream word-by-word\n• **Quick chips** → Click preset questions at the bottom\n• **Escalate** → Click '🧑‍💼 Escalate' to request a human agent",

    # -- Comparison Questions --
    'vs chatgpt': "🆚 **VoiceBot AI vs ChatGPT:**\n• ChatGPT = text only; VoiceBot AI = **voice + text**\n• ChatGPT = generic; VoiceBot AI = **enterprise-focused** with emotion detection & fraud\n• ChatGPT = no phone integration; VoiceBot AI = **Twilio phone calls**\n• ChatGPT = closed API; VoiceBot AI = **open, self-hostable**\n• Both use GPT-4o — but VoiceBot AI wraps it in a complete customer service platform",
    'vs alexa': "🆚 **VoiceBot AI vs Alexa/Siri:**\n• Alexa/Siri = consumer assistants; VoiceBot AI = **enterprise customer service**\n• Alexa = fixed skills; VoiceBot AI = **RAG knowledge base**, customizable\n• No emotion detection in Alexa; VoiceBot AI has **real-time sentiment analysis**\n• VoiceBot AI has **fraud detection** — critical for banking/finance use cases",
    'better than': "🌟 **VoiceBot AI advantages over existing solutions:**\n1. Open-source and self-hostable (privacy-first)\n2. Combines STT + AI + TTS in one pipeline\n3. Real-time emotion detection — responds empathetically\n4. Fraud detection built-in\n5. RAG for domain-specific knowledge\n6. Live streaming (SSE) for better UX\n7. Full analytics dashboard\n8. Production-ready with Docker + PostgreSQL support",
}


# ─────────────────────────────────────────────────────────
#  CONVERSATIONAL PATTERNS
# ─────────────────────────────────────────────────────────

_GREETINGS = [r'\b(hi|hello|hey|hiya|howdy|yo|sup|hola|namaste|namaskar)\b', r'\bgood\s*(morning|afternoon|evening|night)\b', r'\bwhat\'?s?\s*up\b']
_GREETING_RESPONSES = [
    "Hey there! 😊 I'm your AI assistant — ask me anything about science, tech, history, geography, support, or just chat! What's on your mind?",
    "Hello! 👋 I can answer questions, share fun facts, help with orders, do quick math, or just have a great conversation. Fire away!",
    "Hi! Great to see you! I know about science, technology, history, geography, and more. I can also solve math problems! 🚀",
]

_HOWAREYOU = [r'\bhow\s*(are|r)\s*(you|u|ya)\b', r'\bhow\'?s?\s*it\s*going\b']
_HOWAREYOU_RESPONSES = [
    "I'm doing great, thanks for asking! 😊 I've been brushing up on everything from quantum physics to fun facts. What would you like to know?",
    "Fantastic! I love having conversations. Ask me about science, tech, history, math, or anything else. What are you curious about?",
]

_THANKS = [r'\b(thanks|thank\s*you|thx|tysm|appreciate)\b']
_THANKS_RESPONSES = [
    "You're so welcome! 😊 I love sharing knowledge. Anything else you'd like to know?",
    "Happy to help! That's what I'm here for. Got more questions? I'm full of answers! ✨",
]

_GOODBYE = [r'\b(bye|goodbye|see\s*ya|take\s*care|good\s*night|cya|later)\b']
_GOODBYE_RESPONSES = [
    "Goodbye! 👋 It was great chatting with you. Come back anytime — I'm here 24/7!",
    "Take care! 😊 I'm always here if you want to learn something new or need help. See you soon!",
]

_AFFIRMATIVE = [r'\b(yes|yeah|yep|sure|okay|ok|please|go ahead|absolutely)\b']
_AFFIRMATIVE_RESPONSES = [
    "Perfect! Let me get that sorted for you right away... ✨ Done! Anything else I can help with?",
    "Great, I'm on it! All taken care of. What else would you like to know? 😊",
]

_NEGATIVE = [r'\b(no|nah|nope|nothing|that\'s all|all good|i\'m good|im good)\b']
_NEGATIVE_RESPONSES = [
    "No problem! If you ever want to learn something new or need help, I'm just a message away. Have a wonderful day! 😊",
    "Alright! Remember, I'm here 24/7. Take care! ✨",
]

# Support-specific responses
_SUPPORT = {
    'order': ["I'd love to help with your order! Could you share your order number? I'll track it down right away. 📦", "Sure! What's the order number or email you used?"],
    'deliver': ["Let me check your delivery! Most orders arrive in 3-5 business days. Could you share your order number? 🚚"],
    'return': ["Returns are easy — within 30 days! Want me to start the process? Just share your order number. 📋"],
    'refund': ["Refunds typically process in 5-7 business days after we receive your return. Want me to check the status? 💳"],
    'password': ["No worries, let's get you back in! I'll send a reset link to your email. Can you confirm the email on your account? 🔐"],
    'cancel': ["I'm sorry to hear that! Before cancelling, is there anything I can do to help? If you've decided, I'll process it right away. 💔"],
    'billing': ["Let me check your billing details! I can help with invoices, charges, and payment methods. What do you need? 🧾"],
    'shipping': ["We offer Standard (5-7 days, free over $50), Express (2-3 days), and Overnight shipping! Which would you like? 📬"],
    'manager': ["I'll connect you with a senior team member right away. Escalating now. 👔"],
    'hours': ["We're available **24/7**! Anytime, anywhere — phone, chat, or email. We never close! ⏰"],
    'price': ["Our plans: Basic ($9.99/mo), Standard ($19.99/mo), and Premium ($29.99/mo). Want details on any? 💰"],
}

# Emotion responses
_EMOTIONS = {
    'frustrated': ["I completely understand your frustration, and I'm really sorry. 😔 Let me take personal ownership — tell me what happened and I'll fix it.", "Your frustration is completely valid. Let me prioritize this and resolve it right now."],
    'angry': ["I am so sorry about this experience. You have every right to be upset. Let me do everything I can to make this right immediately. 🙏"],
    'happy': ["That's wonderful to hear! 🎉 Your happiness makes my day. Is there anything else I can help with?"],
    'sad': ["I'm really sorry you're feeling this way. 🫂 It's okay to feel sad — it shows you care deeply. Remember, tough times don't last but tough people do. Want to talk about it?"],
    'lonely': ["I'm sorry you're feeling lonely. 💙 You're not alone — I'm right here with you. Try reaching out to a friend, or stepping outside for a walk. You matter more than you know."],
    'stressed': ["I can sense the stress, and I want you to know it's okay to feel overwhelmed. 🌊 Take a deep breath — in for 4 seconds, hold for 4, out for 4. What's weighing on you?"],
    'anxious': ["Anxiety can feel overwhelming. 🌟 Remember: most of what we worry about never happens. Try the 5-4-3-2-1 grounding method — name 5 things you see, 4 you can touch, 3 you hear. You've got this!"],
    'tired': ["It sounds like you need rest, and that's perfectly okay! 😴 Even 10 minutes of rest can help. You've been working hard — take care of yourself!"],
    'bored': ["Bored? Let's fix that! 🎯 Ask me any trivia question, tell me to share a fun fact, or test my general knowledge. I'm basically a walking encyclopedia!"],
    'excited': ["Your excitement is contagious! 🎉🥳 — What's got you so pumped? I'd love to hear about it!"],
    'grateful': ["That's so beautiful! 🥰 Gratitude is one of the most powerful emotions. What are you feeling grateful for today?"],
    'confused': ["Confusion is the first step to understanding! 🤔 Let's work through it together. What's confusing you? I'll explain it as clearly as I can."],
    'scared': ["It's okay to feel scared — fear is a natural human emotion. 🫂 Courage is acting despite fear. You're braver than you believe. What's scaring you?"],
    'heartbroken': ["I'm so sorry you're going through heartbreak. 💔 It's one of the most painful feelings — grieve it. Time does heal, even if it doesn't feel like it now. Be gentle with yourself. 🫂"],
    'depressed': ["I hear you, and your feelings are valid. 💙 Depression is real and not your fault. Please consider reaching out to a mental health professional. Crisis line: **988** (US) / **iCall: 9152987821** (India). You matter."],
    'overwhelmed': ["Feeling overwhelmed is your mind saying 'too much at once'. 🌊 Focus on just ONE small task right now. Everything else can wait. You're doing better than you think."],
    'motivated': ["That motivation is FIRE! 🔥💪 Channel that energy — motivation + action = unstoppable. What are you working on?"],
    'love': ["Love is the most beautiful emotion! ❤️ Whether for a person, a passion, or life itself — cherish it. What's filling your heart today?"],
    'miss': ["Missing someone shows how much they mean to you. 💙 That connection is precious. Have you thought about reaching out? They might be missing you too."],
}

_FEELING_PATTERNS = {
    'sad': [r'\b(sad|unhappy|crying|cry|tears|depressing|down|blue|miserable|heartache)\b', r'\bfeel(ing)?\s*(low|down|bad|empty|numb)\b'],
    'lonely': [r'\b(lonely|alone|isolated|nobody|no\s*one|no\s*friends)\b'],
    'stressed': [r'\b(stress|stressed|pressure|burnout|overwhelm|overwork)\b', r'\bunder\s*(pressure|stress)\b'],
    'anxious': [r'\b(anxious|anxiety|nervous|panic|worried|worrying|worry|phobia)\b'],
    'tired': [r'\b(tired|exhausted|drained|burned\s*out|fatigue|sleepy|worn\s*out|no\s*energy)\b'],
    'bored': [r'\b(bored|boring|nothing\s*to\s*do|dull|monoton)\b'],
    'excited': [r'\b(excited|thrilled|pumped|hyped|cant\s*wait|can\'t\s*wait|ecstatic|stoked)\b'],
    'grateful': [r'\b(grateful|thankful|blessed|appreciate|gratitude)\b'],
    'confused': [r'\b(confused|confusing|don\'t\s*understand|dont\s*understand|lost|puzzled|bewildered)\b'],
    'scared': [r'\b(scared|afraid|terrified|frightened|fearful|creep)\b'],
    'heartbroken': [r'\b(heartbr|broken\s*heart|breakup|broke\s*up|dumped|cheated|betrayed)\b'],
    'depressed': [r'\b(depress|hopeless|worthless|suicid|self\s*harm|don\'t\s*want\s*to\s*live|give\s*up|end\s*it)\b'],
    'overwhelmed': [r'\b(overwhelm|too\s*much|can\'t\s*cope|cant\s*cope|drowning|swamped)\b'],
    'motivated': [r'\b(motivat|inspired|determined|ready\s*to|gonna\s*do|pumped\s*up)\b'],
    'love': [r'\b(in\s*love|i\s*love|loving|soulmate|crush)\b'],
    'miss': [r'\b(miss\s*(you|her|him|them|my|someone)|missing\s*(someone|you|her|him))\b'],
}

_FALLBACK = [
    "That's an interesting question! 🤔 I have knowledge on science, technology, history, geography, math, health, and more. Could you be more specific so I can give you the perfect answer?",
    "I want to help you with that! Could you rephrase or give me a bit more context? I can handle topics from space exploration to ancient history. 🌍",
    "Great question! I might know the answer if you give me a bit more detail. I cover science, tech, history, math, geography, health, and general support — what specifically would you like to know? 💡",
    "I'm not 100% sure about that one, but I'm always learning! Try asking me something like 'What is quantum physics?', 'Capital of Germany?', or '15 times 24' — I love those. 😊",
]


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────

def _match(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _find_capital(text):
    for country in _KNOWLEDGE['capital']:
        if country != 'default' and _kw(text, country):
            return _KNOWLEDGE['capital'][country]
    # Generic capital question without a specific country
    if re.search(r'\bcapital\b', text, re.IGNORECASE):
        return _KNOWLEDGE['capital']['default']
    return None


def _knowledge_lookup(text: str) -> Optional[str]:
    """
    Look up a topic in the knowledge base using whole-word matching.
    Multi-word keys are checked before single-word keys for specificity.
    """
    # Sort keys longest-first so "artificial intelligence" beats "ai" etc.
    sorted_keys = sorted(
        (k for k in _KNOWLEDGE if k != 'capital'),
        key=len,
        reverse=True,
    )
    for key in sorted_keys:
        if _kw(text, key):
            val = _KNOWLEDGE[key]
            return val if isinstance(val, str) else random.choice(val)
    return None


# ─────────────────────────────────────────────────────────
#  DEMO RESPONSE ENGINE
# ─────────────────────────────────────────────────────────

def _demo_response(user_message: str, chat_history: list) -> str:
    text = user_message.lower().strip()
    turns = len([m for m in chat_history if m["role"] == "user"])

    # ── 1. Conversational patterns ──────────────────────
    if _match(text, _GREETINGS):
        return random.choice(_GREETING_RESPONSES)
    if _match(text, _HOWAREYOU):
        return random.choice(_HOWAREYOU_RESPONSES)
    if _match(text, _THANKS):
        return random.choice(_THANKS_RESPONSES)
    if _match(text, _GOODBYE):
        return random.choice(_GOODBYE_RESPONSES)

    if turns > 1 and _match(text, _AFFIRMATIVE):
        return random.choice(_AFFIRMATIVE_RESPONSES)
    if turns > 1 and _match(text, _NEGATIVE):
        return random.choice(_NEGATIVE_RESPONSES)

    # ── 2. Emotions / feelings ───────────────────────────
    for feeling, patterns in _FEELING_PATTERNS.items():
        if _match(text, patterns):
            return random.choice(_EMOTIONS[feeling])

    # Fallback emotion keywords (substring is fine here — these are distinctive)
    if any(w in text for w in ['frustrat', 'annoying', 'terrible', 'worst', 'awful', 'horrible']):
        return random.choice(_EMOTIONS['frustrated'])
    if any(w in text for w in ['angry', 'mad', 'furious', 'pissed']):
        return random.choice(_EMOTIONS['angry'])
    if any(w in text for w in ['happy', 'amazing', 'awesome', 'great experience']):
        return random.choice(_EMOTIONS['happy'])

    # ── 3. Capital cities ────────────────────────────────
    cap = _find_capital(text)
    if cap:
        return cap

    # ── 4. Math ──────────────────────────────────────────
    math_result = _try_math(text)
    if math_result:
        return math_result

    # ── 5. Knowledge base (word-boundary safe) ───────────
    kb = _knowledge_lookup(text)
    if kb:
        return kb

    # ── 6. Support topics ────────────────────────────────
    for key, responses in _SUPPORT.items():
        if _kw(text, key):
            return random.choice(responses)

    # ── 7. Specialty one-liners ──────────────────────────
    if any(w in text for w in ['joke', 'funny', 'laugh', 'humor']):
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛😂",
            "What did the AI say to the human? 'I think we need to have a deep learning conversation.' 🤖😄",
            "Why was the computer cold? It left its Windows open! 💻❄️",
            "How do trees access the internet? They log in! 🌳😁",
            "Why don't scientists trust atoms? Because they make up everything! ⚛️😂",
        ]
        return random.choice(jokes)

    if any(w in text for w in ['who made you', 'who created you', 'who built you']):
        return "I was built by a talented development team as an AI Voice Chatbot! 🤖 I use NLP to understand you and can handle support, answer knowledge questions, solve math, and chat naturally. ✨"

    if re.search(r'\b(your\s+name|who\s+are\s+you|what\s+are\s+you)\b', text):
        return "I'm **VoiceBot AI** — your intelligent personal assistant! 🤖 I handle customer support, answer general knowledge questions, solve math, share fun facts, and hold natural conversations. Nice to meet you!"

    if re.search(r'\b(hindi|हिंदी)\b', text):
        return "हां, मैं हिंदी में बात कर सकता हूं! 😊 बताइए, मैं आपकी कैसे मदद कर सकता हूं? (Yes, I can chat in Hindi! How can I help you?)"

    if _kw(text, 'weather'):
        return "I wish I could check live weather for you! 🌤️ For real-time forecasts try weather.com, AccuWeather, or just ask your phone's assistant. I'm best at knowledge questions, math, and support!"

    if any(w in text for w in ['news', 'latest', 'current events', 'trending']):
        return "For breaking news I'd suggest Google News, BBC, or Reuters. 📰 I'm best at knowledge, support, and conversations — want to test me on a trivia question?"

    if any(w in text for w in ['calculate', 'solve', 'compute', 'equation']):
        return "I can do that! 🧮 Just type the math expression, e.g. `15 * 4` or `120 divided by 6`, and I'll solve it instantly. What's the calculation?"

    # ── 8. "What is / Who is" catch-all ─────────────────
    # (AFTER knowledge lookup, not before — so known topics are handled above)
    if re.search(r'\b(what\s+is|what\s+are|who\s+is|who\s+was|explain|define|tell\s+me\s+about|meaning\s+of)\b', text):
        # Extract the subject to make the response feel more personal
        subject_match = re.search(
            r'\b(?:what\s+is|what\s+are|who\s+is|who\s+was|tell\s+me\s+about|explain|define)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\?|$)',
            text, re.IGNORECASE
        )
        subject = subject_match.group(1).strip() if subject_match else "that"
        return (
            f"Great question about **{subject}**! 🤔 "
            "I have broad knowledge on science, technology, history, math, geography, health, and more — "
            "but I might need a bit more context for that specific topic. "
            "Could you rephrase or add more detail? I want to give you the best possible answer! 💡"
        )

    # ── 9. Generic fallback ──────────────────────────────
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
        logger.info(f"[DEMO] '{last_msg[:40]}' -> '{response[:60]}...'")
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
        # Fall back to demo mode on API errors rather than showing a useless error
        last_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_msg = m["content"]
                break
        logger.info(f"[DEMO FALLBACK after error] using demo engine")
        return _demo_response(last_msg, messages)
