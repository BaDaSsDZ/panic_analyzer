"""
CASI AI — FastAPI inference server.

Endpoints:
  POST /predict          — suggest tags for a panic (completed or active)
  POST /feedback         — record human corrections for retraining
  GET  /health           — health check

Run:
  python -m uvicorn serving.main:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from serving.schemas import (
    TagRequest, TagResponse, TagSuggestion,
    FeedbackRequest, FeedbackResponse,
    HealthResponse,
)
from serving.predictor import get_predictor
from serving.panic_context import fetch_panic_context
from feedback.collector import record_feedback

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_SAVE_DIR = os.getenv("MODEL_SAVE_DIR", "./model/saved")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")

app = FastAPI(
    title="CASI AI Tagging Service",
    description="Suggests tags for CASI security incidents using a trained DistilBERT classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if API_SECRET_KEY and key != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return key


@app.on_event("startup")
async def startup():
    log.info("Loading model...")
    get_predictor(MODEL_SAVE_DIR)
    log.info("CASI AI ready")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        predictor = get_predictor(MODEL_SAVE_DIR)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=predictor.model_version,
            num_tags=predictor.num_labels,
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {e}",
            model_loaded=False,
            model_version="none",
            num_tags=0,
        )


@app.post("/predict", response_model=TagResponse)
def predict(request: TagRequest, _key: str = Security(verify_api_key)):
    """
    Suggest tags for a panic.

    mode="completed"  — fetches full incident data including form answers (default)
    mode="active"     — fetches whatever data exists right now (partial)
                        Use this for live incident analysis during response.
                        The model will do its best with available logs/comments.
    """
    predictor = get_predictor(MODEL_SAVE_DIR)

    try:
        text = fetch_panic_context(panic_id=request.panic_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("DB error fetching panic %d: %s", request.panic_id, e)
        raise HTTPException(status_code=500, detail="Database error fetching panic context")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No usable context found for this panic")

    suggestions_raw = predictor.predict(text)
    suggestions = [TagSuggestion(**s) for s in suggestions_raw]

    log.info(
        "Panic %d [mode=%s]: %d tags suggested",
        request.panic_id, request.mode, len(suggestions)
    )

    return TagResponse(
        panic_id=request.panic_id,
        mode=request.mode or "completed",
        suggestions=suggestions,
        model_version=predictor.model_version,
        threshold_used=predictor.threshold,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest, _key: str = Security(verify_api_key)):
    """
    Record human corrections to AI tag suggestions.
    This data accumulates for the next retraining cycle.
    """
    try:
        record_feedback(request.dict())
        return FeedbackResponse(
            panic_id=request.panic_id,
            recorded=True,
            message="Feedback recorded"
        )
    except Exception as e:
        log.error("Failed to record feedback for panic %d: %s", request.panic_id, e)
        raise HTTPException(status_code=500, detail="Failed to record feedback")
