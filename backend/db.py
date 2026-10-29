"""
db.py
-----
Minimal PostgreSQL logging layer for the moderation API.

Every call to /moderate-comment is persisted as a ModerationLog row:
timestamp, comment length (not the raw comment — see privacy note below),
and the three classification outputs with their per-task probabilities. This gives the app
a real queryable history instead of the in-memory-only /metrics counters,
and is the basis for any future analytics/dashboard work.

Privacy note: the raw comment text is intentionally NOT stored by default
(only its length + a truncated preview) to avoid persisting user-submitted
text without a documented retention policy — consistent with SECURITY.md.
Set STORE_RAW_COMMENT=true in the environment to store full text if a
retention policy for it exists in your deployment.

Schema note: v2 replaced the single averaged `confidence` column with per-task
probabilities. Delete an old local moderation_logs.db (or migrate it) before
running v2 against it.

Falls back to SQLite locally if DATABASE_URL isn't set, so the app still
runs without a Postgres instance during quick local development.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./moderation_logs.db")
STORE_RAW_COMMENT = os.getenv("STORE_RAW_COMMENT", "false").lower() == "true"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    comment_preview = Column(String(80), nullable=True)   # first 80 chars, or null if raw storage off
    comment_length = Column(Integer)
    toxicity = Column(String(20))
    spam = Column(String(20))
    sentiment = Column(String(20))
    toxicity_prob = Column(Float)
    spam_prob = Column(Float)
    sentiment_prob = Column(Float)
    model_version = Column(String(20))
    latency_ms = Column(Float)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def log_moderation(comment: str, result: dict, latency_ms: float) -> None:
    """Persist one moderation result. Never raises — logging failures
    should not break the API response."""
    session = SessionLocal()
    try:
        entry = ModerationLog(
            comment_preview=comment[:80] if STORE_RAW_COMMENT else None,
            comment_length=len(comment),
            toxicity=result["toxicity"],
            spam=result["spam"],
            sentiment=result["sentiment"],
            toxicity_prob=result["scores"]["toxicity"]["probability"],
            spam_prob=result["scores"]["spam"]["probability"],
            sentiment_prob=result["scores"]["sentiment"]["probability"],
            model_version=result.get("model_version"),
            latency_ms=round(latency_ms, 2),
        )
        session.add(entry)
        session.commit()
    except Exception as exc:
        print(f"[db] Failed to log moderation result: {exc}")
        session.rollback()
    finally:
        session.close()


def get_recent_logs(limit: int = 20) -> list[dict]:
    session = SessionLocal()
    try:
        rows = (
            session.query(ModerationLog)
            .order_by(ModerationLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "comment_preview": r.comment_preview,
                "comment_length": r.comment_length,
                "toxicity": r.toxicity,
                "spam": r.spam,
                "sentiment": r.sentiment,
                "toxicity_prob": r.toxicity_prob,
                "spam_prob": r.spam_prob,
                "sentiment_prob": r.sentiment_prob,
                "model_version": r.model_version,
                "latency_ms": r.latency_ms,
            }
            for r in rows
        ]
    finally:
        session.close()
