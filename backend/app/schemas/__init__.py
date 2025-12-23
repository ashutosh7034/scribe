"""
Pydantic Schemas

Data validation and serialization schemas for API requests and responses.
"""

from app.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
    TranslationSessionCreate,
    TranslationSessionResponse,
    FeedbackCreate,
    FeedbackResponse,
)
from app.schemas.avatar import (
    AvatarResponse,
    AvatarCustomizationCreate,
    AvatarCustomizationResponse,
)
from app.schemas.user import (
    UserCreate,
    UserResponse,
    UserPreferencesUpdate,
    UserPreferencesResponse,
)

__all__ = [
    "TranslationRequest",
    "TranslationResponse", 
    "TranslationSessionCreate",
    "TranslationSessionResponse",
    "FeedbackCreate",
    "FeedbackResponse",
    "AvatarResponse",
    "AvatarCustomizationCreate",
    "AvatarCustomizationResponse",
    "UserCreate",
    "UserResponse",
    "UserPreferencesUpdate",
    "UserPreferencesResponse",
]