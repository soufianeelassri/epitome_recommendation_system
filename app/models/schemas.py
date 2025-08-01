from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class CourseMetadata(BaseModel):
    title: str = Field(..., description="Title of the course.")
    description: str = Field(..., description="General description of the course.")
    category: str = Field(..., description="Category of the course (e.g., Consulting, Management).")
    language: str = Field(..., description="Language of the course (e.g., FRENCH, ENGLISH).")
    level: str = Field(..., description="Level of the course (e.g., INTERMEDIATE, BEGINNER).")
    tags: List[str] = Field([], description="List of tags associated with the course (e.g., UI, UX).")
    video_url: Optional[str] = Field(None, description="URL of the main video if it exists.")

class UserPreferences(BaseModel):
    areas_of_interest: List[str] = Field([], description="Areas of interest, e.g., ['Management & Leadership', 'IT & Cybersecurity']")
    preferred_content_types: List[str] = Field([], description="Preferred content types, e.g., ['Video', 'Document']")
    learning_objectives: List[str] = Field([], description="Learning objectives, e.g., ['Certification preparation']")

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    limit: int = Field(10, gt=0, le=100, description="The maximum number of recommendations to return.")

class KeywordsRecommendationRequest(BaseModel):
    keywords: List[str] = Field(..., description="List of keywords to search for similar content.")
    per_keyword_limit: int = Field(5, gt=0, le=20, description="Number of recommendations to find for each keyword.")
    final_limit: int = Field(5, gt=0, le=50, description="Final number of top recommendations to return.")

class UserInteraction(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    point_id: str = Field(..., description="The unique identifier of the content point in Qdrant.")
    interaction_type: str = Field("like", description="The type of interaction (e.g., 'like', 'view').")

class RecommendationResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="The unique identifier of the source document or course.")
    title: str = Field(..., description="Title of the recommended course.")
    filename: Optional[str] = Field(None, description="The original filename of the recommended content.")
    type: str = Field(..., description="The type of content (e.g., 'document', 'video').")
    similarity_score: float = Field(..., description="The final similarity score, including boosts from preferences.")
    best_matching_chunk_payload: Dict[str, Any] = Field({}, description="Payload of the best matching chunk for context.")

class KeywordsRecommendationResponse(BaseModel):
    doc_id: Optional[str] = Field(None, description="The unique identifier of the source document.")
    filename: Optional[str] = Field(None, description="The original filename of the recommended content.")
    type: str = Field(..., description="The type of content (e.g., 'document', 'video').")
    similarity_score: float = Field(..., description="The highest cosine similarity score that led to this recommendation.")
    keyword: str = Field(..., description="The keyword that matched this content.")
    best_matching_chunk_payload: Dict[str, Any] = Field({}, description="Payload of the best matching chunk inside the document.")

class UserProfileResponse(BaseModel):
    user_id: str
    preferences: UserPreferences = Field(..., description="The user's preferences.")
    interaction_history: List[str] = Field(..., description="List of point_ids the user has interacted with.")

class HealthCheckResponse(BaseModel):
    api_status: str
    qdrant_status: str
    error: Optional[str] = None