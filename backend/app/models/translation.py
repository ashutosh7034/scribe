"""
Translation Models

Database models for translation sessions and feedback.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text, Float
from sqlalchemy.orm import relationship

from app.core.database import Base


class TranslationSession(Base):
    """Translation session tracking."""
    
    __tablename__ = "translation_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Anonymous sessions allowed
    session_id = Column(String, unique=True, index=True)  # UUID for session tracking
    
    # Session metadata
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Translation configuration
    source_language = Column(String, default="en-US")
    target_sign_language = Column(String, default="ASL")
    avatar_id = Column(Integer, ForeignKey("avatars.id"))
    
    # Performance metrics
    total_words_translated = Column(Integer, default=0)
    average_latency_ms = Column(Float, nullable=True)
    error_count = Column(Integer, default=0)
    
    # Quality metrics
    user_satisfaction_score = Column(Integer, nullable=True)  # 1-5 rating
    accuracy_feedback_count = Column(Integer, default=0)
    
    # Technical details
    client_platform = Column(String, nullable=True)  # web, ios, android
    client_version = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="translation_sessions")
    avatar = relationship("Avatar")
    feedback_entries = relationship("TranslationFeedback", back_populates="session")


class TranslationFeedback(Base):
    """User feedback on translation quality."""
    
    __tablename__ = "translation_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("translation_sessions.id"))
    
    # Feedback content
    original_text = Column(Text, nullable=False)
    feedback_type = Column(String, nullable=False)  # accuracy, speed, expression, other
    rating = Column(Integer, nullable=True)  # 1-5 rating
    comment = Column(Text, nullable=True)
    
    # Correction data (if provided)
    suggested_correction = Column(Text, nullable=True)
    is_emergency_phrase = Column(Boolean, default=False)
    
    # Context information
    timestamp_in_session = Column(Float, nullable=True)  # Seconds from session start
    emotion_detected = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_priority = Column(Integer, default=1)  # 1=low, 5=high
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("TranslationSession", back_populates="feedback_entries")