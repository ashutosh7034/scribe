"""
Avatar Models

Database models for 3D avatar management and customization.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Avatar(Base):
    """3D Avatar model with base configuration."""
    
    __tablename__ = "avatars"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Avatar assets
    mesh_url = Column(String, nullable=False)  # S3 URL to 3D mesh file
    texture_url = Column(String, nullable=False)  # S3 URL to texture file
    skeleton_config = Column(JSON, nullable=False)  # SMPL-X skeleton configuration
    
    # Avatar properties
    gender = Column(String, nullable=True)  # male, female, neutral
    ethnicity = Column(String, nullable=True)
    age_range = Column(String, nullable=True)  # young, adult, senior
    
    # Availability
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customizations = relationship("AvatarCustomization", back_populates="avatar")


class AvatarCustomization(Base):
    """User-specific avatar customizations."""
    
    __tablename__ = "avatar_customizations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    avatar_id = Column(Integer, ForeignKey("avatars.id"))
    
    # Appearance customizations
    skin_tone = Column(String, nullable=True)  # Hex color or preset name
    hair_style = Column(String, nullable=True)
    hair_color = Column(String, nullable=True)
    eye_color = Column(String, nullable=True)
    
    # Clothing and accessories
    clothing_style = Column(String, default="casual")  # casual, professional, cultural
    clothing_color = Column(String, nullable=True)
    accessories = Column(JSON, nullable=True)  # List of accessory IDs
    
    # Signing style preferences
    signing_space = Column(String, default="standard")  # compact, standard, expansive
    formality_level = Column(String, default="casual")  # casual, formal
    
    # Custom name for this configuration
    customization_name = Column(String, default="My Avatar")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    avatar = relationship("Avatar", back_populates="customizations")
    user = relationship("User")