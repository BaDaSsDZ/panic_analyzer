"""
Request/response schemas for the CASI AI tagging API.
"""

from typing import List, Optional
from pydantic import BaseModel


class TagSuggestion(BaseModel):
    tag_id: str
    tag_name: str
    controller_advice: str
    confidence: float


class TagRequest(BaseModel):
    panic_id: int
    # Optional: for active incident mode — pass partial data already assembled
    # If not provided, the API fetches from DB using panic_id
    mode: Optional[str] = "completed"  # "completed" | "active"


class TagResponse(BaseModel):
    panic_id: int
    mode: str
    suggestions: List[TagSuggestion]
    model_version: str
    threshold_used: float


class FeedbackRequest(BaseModel):
    panic_id: int
    accepted_tag_ids: List[str] = []
    rejected_tag_ids: List[str] = []
    added_tag_ids: List[str] = []
    corrected_by_user_id: int


class FeedbackResponse(BaseModel):
    panic_id: int
    recorded: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    num_tags: int
