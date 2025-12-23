"""
User Models

Database models for user management and preferences.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    """User account model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    username = Column(String, unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)
    translation_sessions = relationship("TranslationSession", back_populates="user")


class UserPreferences(Base):
    """User preferences for sign language translation."""
    
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    
    # Sign language preferences
    preferred_sign_language = Column(String, default="ASL")  # ASL, BSL, etc.
    signing_speed = Column(Integer, default=100)  # Percentage of normal speed
    emotion_intensity = Column(Integer, default=100)  # Percentage of emotion expression
    
    # Avatar preferences
    selected_avatar_id = Column(Integer, ForeignKey("avatars.id"), nullable=True)
    
    # Accessibility preferences
    high_contrast_mode = Column(Boolean, default=False)
    font_scaling = Column(Integer, default=100)  # Percentage scaling
    
    # Technical preferences
    preferred_quality = Column(String, default="standard")  # low, standard, high
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    selected_avatar = relationship("Avatar", foreign_keys=[selected_avatar_id])