"""
Database Models

Import all models here to ensure they are registered with SQLAlchemy.
"""

from app.core.database import Base
from app.models.translation import TranslationSession, TranslationFeedback
from app.models.user import User, UserPreferences
from app.models.avatar import Avatar, AvatarCustomization

__all__ = [
    "Base",
    "TranslationSession",
    "TranslationFeedback", 
    "User",
    "UserPreferences",
    "Avatar",
    "AvatarCustomization",
]