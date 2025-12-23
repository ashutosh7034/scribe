"""
Avatar API Schemas

Pydantic models for avatar-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AvatarResponse(BaseModel):
    """Response schema for avatar information."""
    
    id: int
    name: str
    description: Optional[str]
    mesh_url: str
    texture_url: str
    skeleton_config: Dict[str, Any]
    gender: Optional[str]
    ethnicity: Optional[str]
    age_range: Optional[str]
    is_active: bool
    is_premium: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class AvatarCustomizationCreate(BaseModel):
    """Schema for creating avatar customization."""
    
    avatar_id: int = Field(..., description="Base avatar ID")
    customization_name: str = Field(default="My Avatar", description="Name for this customization")
    
    # Appearance customizations
    skin_tone: Optional[str] = Field(None, description="Skin tone color or preset")
    hair_style: Optional[str] = Field(None, description="Hair style identifier")
    hair_color: Optional[str] = Field(None, description="Hair color")
    eye_color: Optional[str] = Field(None, description="Eye color")
    
    # Clothing and accessories
    clothing_style: str = Field(default="casual", description="Clothing style: casual, professional, cultural")
    clothing_color: Optional[str] = Field(None, description="Clothing color")
    accessories: Optional[List[str]] = Field(None, description="List of accessory identifiers")
    
    # Signing style preferences
    signing_space: str = Field(default="standard", description="Signing space: compact, standard, expansive")
    formality_level: str = Field(default="casual", description="Formality: casual, formal")


class AvatarCustomizationResponse(BaseModel):
    """Response schema for avatar customization."""
    
    id: int
    user_id: int
    avatar_id: int
    customization_name: str
    skin_tone: Optional[str]
    hair_style: Optional[str]
    hair_color: Optional[str]
    eye_color: Optional[str]
    clothing_style: str
    clothing_color: Optional[str]
    accessories: Optional[Dict[str, Any]]
    signing_space: str
    formality_level: str
    created_at: datetime
    updated_at: datetime
    
    # Include base avatar information
    avatar: AvatarResponse
    
    class Config:
        from_attributes = True


class AvatarListResponse(BaseModel):
    """Response schema for avatar list."""
    
    avatars: List[AvatarResponse]
    total: int
    page: int
    page_size: int