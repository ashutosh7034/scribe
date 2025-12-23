"""
User Endpoints

User management and preferences API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User, UserPreferences
from app.schemas.user import (
    UserCreate,
    UserResponse,
    UserPreferencesUpdate,
    UserPreferencesResponse,
)

router = APIRouter()


@router.post("/", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user account.
    
    Users can be created with email/username or anonymously for guest sessions.
    """
    
    # Check if user already exists
    if user_data.email:
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    if user_data.username:
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create default preferences
    preferences = UserPreferences(user_id=user.id)
    db.add(preferences)
    db.commit()
    
    return user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user information by ID."""
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.get("/{user_id}/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(user_id: int, db: Session = Depends(get_db)):
    """Get user preferences for sign language translation."""
    
    preferences = db.query(UserPreferences).filter(
        UserPreferences.user_id == user_id
    ).first()
    
    if not preferences:
        raise HTTPException(status_code=404, detail="User preferences not found")
    
    return preferences


@router.put("/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    user_id: int,
    preferences_update: UserPreferencesUpdate,
    db: Session = Depends(get_db)
):
    """
    Update user preferences for sign language translation.
    
    Allows users to customize their translation experience including
    sign language variant, speed, emotion intensity, and accessibility options.
    """
    
    preferences = db.query(UserPreferences).filter(
        UserPreferences.user_id == user_id
    ).first()
    
    if not preferences:
        raise HTTPException(status_code=404, detail="User preferences not found")
    
    # Update only provided fields
    update_data = preferences_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(preferences, field, value)
    
    db.commit()
    db.refresh(preferences)
    
    return preferences


@router.delete("/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete a user account and all associated data.
    
    This implements the right to deletion for privacy compliance.
    """
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete user (cascading deletes will handle related records)
    db.delete(user)
    db.commit()
    
    return {"message": "User account deleted successfully"}