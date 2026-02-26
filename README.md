# 🤖 AI Voice Chatbot

A **production-ready Real-Time Human-Like AI Voice Chatbot** built with FastAPI, vanilla HTML/CSS/JS, WebSockets, OpenAI GPT, Whisper STT, ElevenLabs TTS, and Docker.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Nginx)                         │
│  index.html │ chat.html │ dashboard.html │ history │ settings   │
│  ┌──────────┐ ┌─────────────┐ ┌─────────────┐                  │
│  │ Login UI │ │ Voice Chat  │ │  Dashboard  │                  │
│  │ JWT Auth │ │ MediaRecorder│ │  Chart.js   │                  │
│  └──────────┘ └──────┬──────┘ └──────┬──────┘                  │
└──────────────────────┼───────────────┼──────────────────────────┘
                       │ WebSocket     │ REST API
┌──────────────────────┼───────────────┼──────────────────────────┐
│                  BACKEND (FastAPI / Uvicorn)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      API Routes                            │ │
│  │  /auth  /ws/voice  /conversations  /analytics  /escalation │ │
│  │  /knowledge  /admin  /twilio                               │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│  ┌────────────────────────┼───────────────────────────────────┐ │
│  │                    Services Layer                          │ │
│  │  AI (GPT) │ STT (Whisper) │ TTS (ElevenLabs) │ Sentiment │ │
│  │  Vector (FAISS/RAG) │ Auth (JWT/bcrypt)                   │ │
│  └────────────────────────┬───────────────────────────────────┘ │
│  ┌────────────────────────┼───────────────────────────────────┐ │
│  │                   Integrations                             │ │
│  │  CRM │ WhatsApp │ ERP │ Twilio │ Biometrics │ Fraud       │ │
│  └────────────────────────┼───────────────────────────────────┘ │
└──────────────────────────┼──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌──────┴──────┐
        │ PostgreSQL │ │ Redis │ │ FAISS Index │
        │  (Data)    │ │(Cache)│ │  (Vectors)  │
        └───────────┘ └───────┘ └─────────────┘
```

### Voice Flow

```
User speaks → MediaRecorder → WebSocket → Whisper STT → Sentiment Analysis
→ FAISS Knowledge Lookup → GPT-4o Response → ElevenLabs TTS → Audio Playback
```

---

## 📁 Project Structure

```
ai-voice-bot/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Pydantic Settings
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile
│   ├── models/
│   │   ├── database.py         # SQLAlchemy async engine
│   │   ├── entities.py         # ORM models
│   │   └── schemas.py          # Pydantic schemas
│   ├── services/
│   │   ├── ai_service.py       # GPT conversation engine
│   │   ├── stt_service.py      # Whisper speech-to-text
│   │   ├── tts_service.py      # ElevenLabs text-to-speech
│   │   ├── sentiment_service.py# Emotion + urgency detection
│   │   ├── vector_service.py   # FAISS RAG knowledge base
│   │   └── auth_service.py     # JWT + bcrypt auth
│   ├── integrations/
│   │   ├── crm.py              # CRM API placeholder
│   │   ├── whatsapp.py         # WhatsApp API placeholder
│   │   ├── erp.py              # ERP API placeholder
│   │   ├── voice_biometrics.py # Voice biometric placeholder
│   │   ├── fraud_detection.py  # Fraud pattern detection
│   │   └── twilio_handler.py   # Twilio voice webhooks
│   ├── routes/
│   │   ├── auth.py             # Register, login, users
│   │   ├── voice.py            # WebSocket voice streaming
│   │   ├── conversations.py    # Conversation CRUD
│   │   ├── analytics.py        # Dashboard analytics
│   │   ├── knowledge.py        # Knowledge base ingestion
│   │   ├── escalation.py       # Human agent handoff
│   │   └── admin.py            # Admin settings
│   └── middleware/
│       ├── error_handler.py    # Global exception handler
│       └── logging_middleware.py# Request logging
├── frontend/
│   ├── index.html              # Login page
│   ├── chat.html               # Voice assistant UI
│   ├── dashboard.html          # Analytics dashboard
│   ├── history.html            # Conversation history
│   ├── settings.html           # Admin settings
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── css/
│   │   └── style.css           # Design system
│   └── js/
│       ├── app.js              # Auth, WebSocket, VoiceChat
│       └── dashboard.js        # Chart.js rendering
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **Node.js** (optional, for local dev)
- **Docker & Docker Compose** (recommended)
- **PostgreSQL 16+** (or use Docker)
- **Redis 7+** (or use Docker)

### Option 1: Docker (Recommended)

```bash
# 1. Clone the project
cd ai-voice-bot

# 2. Create environment file
cp .env.example .env
# Edit .env with your API keys

# 3. Build and run
docker-compose up --build -d

# 4. Access the app
#    Frontend: http://localhost:3000
#    Backend:  http://localhost:8000
#    API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

```bash
# 1. Start PostgreSQL and Redis (or use Docker for just these)
docker run -d --name pg -e POSTGRES_DB=voicebot -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Create .env
cp ../.env.example .env
# Edit .env with your API keys

# 4. Run backend
python main.py

# 5. Open frontend
# Simply open frontend/index.html in a browser
# Or serve with: python -m http.server 3000 --directory ../frontend
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | ✅ | JWT signing secret (change in production!) |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `OPENAI_API_KEY` | ✅ | OpenAI API key for GPT + Whisper |
| `OPENAI_MODEL` | ❌ | GPT model (default: `gpt-4o`) |
| `ELEVENLABS_API_KEY` | ✅ | ElevenLabs API key for TTS |
| `ELEVENLABS_VOICE_ID` | ❌ | Voice ID (default: Rachel) |
| `TWILIO_ACCOUNT_SID` | ❌ | Twilio SID for phone calls |
| `TWILIO_AUTH_TOKEN` | ❌ | Twilio auth token |
| `CRM_API_URL` | ❌ | CRM integration endpoint |
| `WHATSAPP_API_URL` | ❌ | WhatsApp Business API endpoint |
| `ERP_API_URL` | ❌ | ERP system endpoint |

---

## 📡 API Documentation

Once running, visit **http://localhost:8000/docs** for interactive Swagger UI.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login, returns JWT |
| `GET` | `/api/auth/me` | Current user info |
| `WS` | `/ws/voice/{session_id}` | WebSocket voice streaming |
| `GET` | `/api/conversations/` | List conversations |
| `GET` | `/api/conversations/{id}` | Get conversation detail |
| `GET` | `/api/analytics/summary` | Dashboard stats |
| `GET` | `/api/analytics/timeline` | Conversations over time |
| `POST` | `/api/escalation/` | Escalate to human agent |
| `POST` | `/api/knowledge/ingest` | Add to knowledge base |
| `GET` | `/api/admin/settings` | Get app settings |
| `PUT` | `/api/admin/settings` | Update app settings |
| `POST` | `/api/twilio/voice` | Twilio voice webhook |
| `GET` | `/api/health` | Health check |

### WebSocket Protocol

```
 Client → Server:  Binary (audio/webm bytes)
 Server → Client:  JSON {
   "type": "response",
   "transcript": "user said...",
   "ai_response": "AI reply...",
   "emotion": "positive",
   "sentiment_score": 0.65,
   "is_urgent": false,
   "fraud_alert": false,
   "audio_base64": "base64-encoded MP3"
 }
```

---

## ☁️ Deployment Guide

### AWS Deployment

```bash
# 1. Launch EC2 instance (t3.medium+ recommended)
# 2. Install Docker & Docker Compose
# 3. Clone repo, configure .env
# 4. Run:
docker-compose up --build -d

# 5. Configure ALB/nginx for HTTPS
# 6. Point domain to ALB
```

### GCP Deployment

```bash
# Option A: Cloud Run
gcloud run deploy ai-voicebot \
  --source ./backend \
  --set-env-vars="$(cat .env | tr '\n' ',')"

# Option B: GKE
# Use docker-compose → kompose to generate K8s manifests
kompose convert -f docker-compose.yml
kubectl apply -f .
```

### Production Checklist

- [ ] Set strong `SECRET_KEY`
- [ ] Enable HTTPS (TLS termination at load balancer)
- [ ] Set `CORS_ORIGINS` to your domain
- [ ] Use managed PostgreSQL (RDS / Cloud SQL)
- [ ] Use managed Redis (ElastiCache / Memorystore)
- [ ] Set `DEBUG=false`
- [ ] Configure monitoring (Datadog / CloudWatch)
- [ ] Set up log aggregation

---

## 📈 Scaling Strategy

| Component | Strategy |
|-----------|----------|
| **Backend** | Horizontal scaling with multiple Uvicorn workers behind a load balancer |
| **Database** | Read replicas, connection pooling (PgBouncer) |
| **Redis** | Redis Cluster for high availability session management |
| **Vector Store** | Migrate from FAISS to Pinecone/Weaviate for distributed vector search |
| **TTS/STT** | Queue-based processing with Celery for high concurrency |
| **WebSockets** | Use Redis pub/sub for cross-instance message routing |

---

## 🔒 Security

| Feature | Implementation |
|---------|----------------|
| **Authentication** | OAuth2 + JWT with bcrypt password hashing |
| **Authorization** | Role-based access (admin / user) |
| **Rate Limiting** | SlowAPI middleware (configurable per-minute) |
| **CORS** | Configurable allowed origins |
| **Input Validation** | Pydantic schema validation on all endpoints |
| **SQL Injection** | SQLAlchemy ORM parameterized queries |
| **XSS Protection** | Content-Security-Policy headers via Nginx |
| **Fraud Detection** | Pattern matching on conversation content |
| **Voice Biometrics** | Placeholder for voiceprint verification |

---

## ✨ Features

| Feature | Status |
|---------|--------|
| Real-time voice chat (WebSocket) | ✅ |
| Speech-to-text (Whisper) | ✅ |
| Text-to-speech (ElevenLabs) | ✅ |
| Multi-turn conversation (GPT-4o) | ✅ |
| Multilingual support (Indian languages) | ✅ |
| Sentiment & emotion detection | ✅ |
| Urgency detection | ✅ |
| Knowledge base (FAISS RAG) | ✅ |
| Analytics dashboard | ✅ |
| Conversation history | ✅ |
| Human agent escalation | ✅ |
| JWT authentication | ✅ |
| CRM integration | 🔲 Placeholder |
| WhatsApp integration | 🔲 Placeholder |
| ERP integration | 🔲 Placeholder |
| Voice biometrics | 🔲 Placeholder |
| Fraud detection | ✅ Basic |
| Twilio phone calls | ✅ Webhook ready |
| Docker deployment | ✅ |

---

## 🗺️ Future Roadmap

1. **v1.1** — Streaming TTS (real-time audio chunks instead of full response)
2. **v1.2** — Multi-agent support (route to specialized AI agents by topic)
3. **v1.3** — Voice biometric enrollment + verification with speaker diarization
4. **v1.4** — Full WhatsApp Business API integration with media support
5. **v1.5** — CRM connectors (Salesforce, HubSpot, Zoho)
6. **v2.0** — On-premise deployment with local Whisper + LLM (Llama 3)
7. **v2.1** — Real-time dashboards with WebSocket push updates
8. **v2.2** — A/B testing for conversation flows
9. **v3.0** — Video call support with emotion detection from facial expressions

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
