"""
Translation API Schemas

Pydantic models for translation-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Vector3(BaseModel):
    """3D vector representation."""
    x: float
    y: float
    z: float


class PoseKeyframe(BaseModel):
    """Keyframe for 3D pose animation."""
    timestamp: float = Field(..., description="Timestamp in milliseconds")
    joints: Dict[str, Vector3] = Field(..., description="Joint positions/rotations")


class FacialExpressionKeyframe(BaseModel):
    """Keyframe for facial expression animation."""
    timestamp: float = Field(..., description="Timestamp in milliseconds")
    expression: str = Field(..., description="Expression name")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Expression intensity")


class TranslationRequest(BaseModel):
    """Request schema for real-time translation."""
    
    text: str = Field(..., description="Text to translate to sign language")
    source_language: str = Field(default="en-US", description="Source language code")
    target_sign_language: str = Field(default="ASL", description="Target sign language")
    emotion_intensity: int = Field(default=100, ge=0, le=200, description="Emotion intensity percentage")
    signing_speed: int = Field(default=100, ge=50, le=200, description="Signing speed percentage")
    avatar_id: Optional[int] = Field(None, description="Avatar ID to use for translation")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class TranslationResponse(BaseModel):
    """Response schema for translation results."""
    
    session_id: str = Field(..., description="Session ID for tracking")
    original_text: str = Field(..., description="Original input text")
    normalized_text: str = Field(..., description="Text normalized for sign language")
    
    # Avatar animation data
    pose_sequence: List[PoseKeyframe] = Field(..., description="3D pose keyframes for avatar")
    facial_expressions: List[FacialExpressionKeyframe] = Field(..., description="Facial expression keyframes")
    duration_ms: int = Field(..., description="Total animation duration in milliseconds")
    
    # Metadata
    emotion_detected: Optional[str] = Field(None, description="Detected emotion")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Translation confidence")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TranslationSessionCreate(BaseModel):
    """Schema for creating a new translation session."""
    
    user_id: Optional[int] = Field(None, description="User ID (optional for anonymous sessions)")
    source_language: str = Field(default="en-US", description="Source language code")
    target_sign_language: str = Field(default="ASL", description="Target sign language")
    avatar_id: int = Field(..., description="Avatar ID to use")
    client_platform: Optional[str] = Field(None, description="Client platform")
    client_version: Optional[str] = Field(None, description="Client version")


class TranslationSessionResponse(BaseModel):
    """Response schema for translation session."""
    
    id: int
    session_id: str
    user_id: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[int]
    source_language: str
    target_sign_language: str
    avatar_id: int
    total_words_translated: int
    average_latency_ms: Optional[float]
    error_count: int
    user_satisfaction_score: Optional[int]
    
    class Config:
        from_attributes = True


class FeedbackCreate(BaseModel):
    """Schema for creating translation feedback."""
    
    session_id: int = Field(..., description="Translation session ID")
    original_text: str = Field(..., description="Original text that was translated")
    feedback_type: str = Field(..., description="Type of feedback: accuracy, speed, expression, other")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Additional feedback comment")
    suggested_correction: Optional[str] = Field(None, description="Suggested correction")
    is_emergency_phrase: bool = Field(default=False, description="Is this an emergency phrase")
    timestamp_in_session: Optional[float] = Field(None, description="Timestamp in session")


class FeedbackResponse(BaseModel):
    """Response schema for feedback."""
    
    id: int
    session_id: int
    original_text: str
    feedback_type: str
    rating: Optional[int]
    comment: Optional[str]
    suggested_correction: Optional[str]
    is_emergency_phrase: bool
    is_processed: bool
    processing_priority: int
    created_at: datetime
    
    class Config:
        from_attributes = True