from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# API request models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    limit: int = Field(10, gt=0, le=100, description="The maximum number of recommendations to return.")
    content_type: Optional[str] = Field(None, description="Filter recommendations by content type (e.g., 'video', 'text').")

class UserInteraction(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    point_id: str = Field(..., description="The unique identifier of the content point in Qdrant.")
    interaction_type: str = Field("like", description="The type of interaction (e.g., 'like', 'view').")

# API response models
class RecommendationResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="The unique identifier of the source document.")
    filename: Optional[str] = Field(None, description="The original filename of the recommended content (PDF, video, etc.).")
    type: str = Field(..., description="The type of content (e.g., 'document', 'video').")
    similarity_score: float = Field(..., description="The highest cosine similarity score that led to this recommendation.")
    best_matching_chunk_payload: Dict[str, Any] = Field({}, description="Payload of the best matching chunk inside the document.")


class UserProfileResponse(BaseModel):
    user_id: str
    interaction_history: List[str] = Field(..., description="List of point_ids the user has interacted with.")
    learning_path: List[str] = Field([], description="An ordered list of content for a structured learning path (future feature).")
    preferences: Dict[str, Any] = Field({}, description="User-defined preferences (future feature).")

class HealthCheckResponse(BaseModel):
    api_status: str
    qdrant_status: str
    error: Optional[str] = None