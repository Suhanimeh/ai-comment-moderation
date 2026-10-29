"""
main.py
-------
FastAPI backend for the AI Comment Moderation System.

Run locally (from backend/):
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /moderate-comment   classify a single comment
    GET  /health             liveness (process is up)
    GET  /ready              readiness (models loaded, 503 otherwise)
    GET  /metrics            process-local counters (reset on restart)
    GET  /logs/recent        recent persisted results (database-backed)
    GET  /model-info         model version, dataset and evaluation summary

Configuration (environment variables):
    CORS_ORIGINS       comma-separated allowed origins
                       (default: http://localhost:3000, http://127.0.0.1:3000)
    RATE_LIMIT         requests per client IP per window (default 30)
    RATE_LIMIT_WINDOW  window length in seconds (default 60)
    MODEL_PATH         path to the trained model artifact
    DATABASE_URL       SQLAlchemy URL (default: local SQLite file)
    STORE_RAW_COMMENT  "true" to persist a comment preview (default false)
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from db import get_recent_logs, init_db, log_moderation
from model import InsufficientTextError, ModelNotAvailableError, load_models, predict_comment
from preprocess import NLTKResourceError

logger = logging.getLogger("moderation")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

bundle: dict = {}

# Process-local counters: they live in this process's memory, are NOT shared
# between workers/instances and reset on every restart. Use /logs/recent (the
# database) for history that survives restarts.
_metrics = {"requests_total": 0, "errors_total": 0, "rejected_total": 0,
            "rate_limited_total": 0, "total_latency_ms": 0.0}
_metrics_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    try:
        bundle = load_models()
        logger.info("Loaded model %s", bundle["metadata"]["model_version"])
    except (ModelNotAvailableError, NLTKResourceError) as exc:
        # Stay alive so /health works and /ready reports the reason.
        bundle = {}
        app.state.load_error = str(exc)
        logger.error("Model not ready: %s", exc)
    init_db()
    yield
    bundle.clear()


app = FastAPI(
    title="AI Comment Moderation API",
    description="Toxicity, spam and sentiment classification for short comments "
                "(TF-IDF + scikit-learn). Predictions can be wrong - see /model-info.",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.load_error = None

# ---------------------------------------------------------------------------
# CORS - explicit origins from the environment, never "*"
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
_allow_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
                  if o.strip() and o.strip() != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Rate limiting - in-memory sliding window per client IP.
# Process-local: with several instances each has its own counters, so a real
# multi-instance deployment should use a shared store (e.g. Redis) or enforce
# limits at the gateway. Behind a proxy, configure the proxy to pass the real
# client IP (request.client.host is the proxy otherwise).
# ---------------------------------------------------------------------------

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
_hits: Dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limited(client: str) -> bool:
    now = time.monotonic()
    with _hits_lock:
        q = _hits[client]
        while q and now - q[0] > RATE_LIMIT_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        return False


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/moderate-comment":
        client = request.client.host if request.client else "unknown"
        if _rate_limited(client):
            with _metrics_lock:
                _metrics["rate_limited_total"] += 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CommentRequest(BaseModel):
    comment: str

    @field_validator("comment")
    @classmethod
    def comment_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("comment must not be empty")
        if len(v) > 5000:
            raise ValueError("comment must be 5000 characters or fewer")
        return v.strip()


class TaskScore(BaseModel):
    label: str
    probability: float      # model-estimated probability of the predicted class
    uncertain: bool         # True when probability is below the task's evaluated threshold


class ModerationResponse(BaseModel):
    comment: str
    status: str                       # "ok"
    toxicity: str                     # "Toxic" | "Non-Toxic"
    spam: str                         # "Spam" | "Not Spam"
    sentiment: str                    # "Positive" | "Negative" | "Neutral"
    scores: Dict[str, TaskScore]      # per-task predicted-class probability
    all_scores: Optional[Dict[str, Dict[str, float]]] = None   # full class breakdown
    model_version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness: the process is running. Says nothing about model readiness."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: models are loaded and the API can serve predictions."""
    if not bundle:
        return JSONResponse(status_code=503, content={"status": "not_ready", "models_loaded": False})
    return {"status": "ready", "models_loaded": True,
            "model_version": bundle["metadata"]["model_version"]}


@app.get("/metrics")
async def metrics():
    """Process-local counters. They reset when the process restarts."""
    with _metrics_lock:
        total = _metrics["requests_total"]
        avg = round(_metrics["total_latency_ms"] / total, 2) if total else 0.0
        snapshot = dict(_metrics)
    return {
        "scope": "process-local (resets on restart; not shared across instances)",
        "requests_total": total,
        "errors_total": snapshot["errors_total"],
        "rejected_total": snapshot["rejected_total"],
        "rate_limited_total": snapshot["rate_limited_total"],
        "avg_latency_ms": avg,
        "models_loaded": bool(bundle),
    }


@app.get("/model-info")
async def model_info():
    """Model version and held-out evaluation summary (measured, not claimed)."""
    if not bundle:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    meta = bundle["metadata"]
    tasks = {}
    for task, t in meta["tasks"].items():
        tm = t["test_metrics"]
        tasks[task] = {
            "selected_model": t["selected_model"],
            "preprocessing": t["preprocessing"],
            "dataset": t["dataset"]["file"],
            "n_train": t["dataset"]["n_train"],
            "n_test": t["dataset"]["n_test"],
            "test_accuracy": round(tm["accuracy"], 4),
            "test_macro_f1": round(tm["macro_f1"], 4),
            "uncertainty_threshold": t["uncertainty_threshold"],
        }
    return {
        "model_version": meta["model_version"],
        "trained_at": meta["trained_at"],
        "evaluation": "held-out 20% stratified test split; see experiments/ in the repository",
        "tasks": tasks,
        "notice": "Automated moderation makes mistakes; keep human review for consequential decisions.",
    }


@app.post("/moderate-comment", response_model=ModerationResponse)
async def moderate_comment(req: CommentRequest):
    if not bundle:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start = time.perf_counter()
    with _metrics_lock:
        _metrics["requests_total"] += 1
    try:
        result = predict_comment(req.comment, bundle)
    except InsufficientTextError:
        with _metrics_lock:
            _metrics["rejected_total"] += 1
        raise HTTPException(
            status_code=422,
            detail="Comment contains no analysable text (e.g. only punctuation or symbols).",
        )
    except Exception:
        # Full traceback goes to the server log; the comment text is NOT logged.
        logger.exception("Prediction failed (input length=%d)", len(req.comment))
        with _metrics_lock:
            _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail="Internal error while analysing the comment.")
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        with _metrics_lock:
            _metrics["total_latency_ms"] += latency_ms

    log_moderation(req.comment, result, latency_ms)
    return ModerationResponse(**result)


@app.get("/logs/recent")
async def recent_logs(limit: int = 20):
    """Recent moderation results from the database (survives restarts)."""
    limit = max(1, min(limit, 100))
    return {"logs": get_recent_logs(limit=limit)}
