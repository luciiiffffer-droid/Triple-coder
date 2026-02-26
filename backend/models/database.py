"""
Async SQLAlchemy engine, session factory, and base model.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

_engine_kwargs = {
    "echo": settings.DEBUG,
}

# SQLite doesn't support pool_size/max_overflow
if "sqlite" not in settings.DATABASE_URL:
    _engine_kwargs.update({"pool_size": 20, "max_overflow": 10, "pool_pre_ping": True})

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async DB session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables (for development convenience)."""
    async with engine.begin() as conn:
        from models.entities import User, Conversation, Message, AnalyticsEvent  # noqa
        await conn.run_sync(Base.metadata.create_all)
