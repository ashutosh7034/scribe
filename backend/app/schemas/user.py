"""
User API Schemas

Pydantic models for user-related API requests and responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    
    email: Optional[EmailStr] = Field(None, description="User email address")
    username: Optional[str] = Field(None, description="Username")


class UserResponse(BaseModel):
    """Response schema for user information."""
    
    id: int
    email: Optional[str]
    username: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserPreferencesUpdate(BaseModel):
    """Schema for updating user preferences."""
    
    preferred_sign_language: Optional[str] = Field(None, description="Preferred sign language (ASL, BSL, etc.)")
    signing_speed: Optional[int] = Field(None, ge=50, le=200, description="Signing speed percentage")
    emotion_intensity: Optional[int] = Field(None, ge=0, le=200, description="Emotion intensity percentage")
    selected_avatar_id: Optional[int] = Field(None, description="Selected avatar ID")
    high_contrast_mode: Optional[bool] = Field(None, description="High contrast mode enabled")
    font_scaling: Optional[int] = Field(None, ge=50, le=300, description="Font scaling percentage")
    preferred_quality: Optional[str] = Field(None, description="Preferred quality: low, standard, high")


class UserPreferencesResponse(BaseModel):
    """Response schema for user preferences."""
    
    id: int
    user_id: int
    preferred_sign_language: str
    signing_speed: int
    emotion_intensity: int
    selected_avatar_id: Optional[int]
    high_contrast_mode: bool
    font_scaling: int
    preferred_quality: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True