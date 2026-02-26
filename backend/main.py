"""
AI Voice Chatbot — FastAPI entry point.
"""

import sys
import os

# Ensure backend dir is on path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from config import settings
from models.database import init_db
from services.vector_service import load_index
from middleware.error_handler import global_exception_handler
from middleware.logging_middleware import logging_middleware

# ── Routes ───────────────────────────────────────────────
from routes.auth import router as auth_router
from routes.voice import router as voice_router
from routes.conversations import router as conversations_router
from routes.analytics import router as analytics_router
from routes.knowledge import router as knowledge_router
from routes.escalation import router as escalation_router
from routes.admin import router as admin_router
from integrations.twilio_handler import router as twilio_router


# ── Lifespan ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    load_index()
    logger.info("Database initialized, vector index loaded")
    yield
    logger.info("Shutting down")


# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-Time Human-Like AI Voice Chatbot API",
    lifespan=lifespan,
)

# ── Rate Limiter ─────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ─────────────────────────────────────────────────
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Custom Middleware ────────────────────────────────────
app.middleware("http")(global_exception_handler)
app.middleware("http")(logging_middleware)

# ── Register Routers ────────────────────────────────────
app.include_router(auth_router)
app.include_router(voice_router)
app.include_router(conversations_router)
app.include_router(analytics_router)
app.include_router(knowledge_router)
app.include_router(escalation_router)
app.include_router(admin_router)
app.include_router(twilio_router)


# ── Health Check ─────────────────────────────────────────
@app.get("/api/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ── Run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
