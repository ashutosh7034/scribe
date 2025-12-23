"""
Avatar Endpoints

3D avatar management and customization API endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.avatar import Avatar, AvatarCustomization
from app.schemas.avatar import (
    AvatarResponse,
    AvatarListResponse,
    AvatarCustomizationCreate,
    AvatarCustomizationResponse,
)
from app.schemas.translation import PoseKeyframe, FacialExpressionKeyframe
from app.services.avatar_rendering import avatar_rendering_service

router = APIRouter()


@router.get("/", response_model=AvatarListResponse)
async def list_avatars(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    ethnicity: Optional[str] = Query(None, description="Filter by ethnicity"),
    is_premium: Optional[bool] = Query(None, description="Filter by premium status"),
    db: Session = Depends(get_db)
):
    """
    List available avatars with optional filtering.
    
    Returns paginated list of avatars that users can select and customize.
    """
    
    query = db.query(Avatar).filter(Avatar.is_active == True)
    
    # Apply filters
    if gender:
        query = query.filter(Avatar.gender == gender)
    if ethnicity:
        query = query.filter(Avatar.ethnicity == ethnicity)
    if is_premium is not None:
        query = query.filter(Avatar.is_premium == is_premium)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    avatars = query.offset(offset).limit(page_size).all()
    
    return AvatarListResponse(
        avatars=avatars,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{avatar_id}", response_model=AvatarResponse)
async def get_avatar(avatar_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific avatar."""
    
    avatar = db.query(Avatar).filter(
        Avatar.id == avatar_id,
        Avatar.is_active == True
    ).first()
    
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    return avatar


@router.post("/customizations", response_model=AvatarCustomizationResponse)
async def create_customization(
    customization: AvatarCustomizationCreate,
    user_id: int,  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """
    Create a new avatar customization for a user.
    
    Allows users to personalize avatar appearance and signing style.
    """
    
    # Verify avatar exists
    avatar = db.query(Avatar).filter(
        Avatar.id == customization.avatar_id,
        Avatar.is_active == True
    ).first()
    
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    # Create customization
    avatar_customization = AvatarCustomization(
        user_id=user_id,
        avatar_id=customization.avatar_id,
        customization_name=customization.customization_name,
        skin_tone=customization.skin_tone,
        hair_style=customization.hair_style,
        hair_color=customization.hair_color,
        eye_color=customization.eye_color,
        clothing_style=customization.clothing_style,
        clothing_color=customization.clothing_color,
        accessories=customization.accessories,
        signing_space=customization.signing_space,
        formality_level=customization.formality_level,
    )
    
    db.add(avatar_customization)
    db.commit()
    db.refresh(avatar_customization)
    
    return avatar_customization


@router.get("/customizations/{user_id}", response_model=List[AvatarCustomizationResponse])
async def get_user_customizations(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get all avatar customizations for a user."""
    
    customizations = db.query(AvatarCustomization).filter(
        AvatarCustomization.user_id == user_id
    ).all()
    
    return customizations


@router.put("/customizations/{customization_id}", response_model=AvatarCustomizationResponse)
async def update_customization(
    customization_id: int,
    customization_update: AvatarCustomizationCreate,
    user_id: int,  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Update an existing avatar customization."""
    
    customization = db.query(AvatarCustomization).filter(
        AvatarCustomization.id == customization_id,
        AvatarCustomization.user_id == user_id
    ).first()
    
    if not customization:
        raise HTTPException(status_code=404, detail="Customization not found")
    
    # Update fields
    for field, value in customization_update.dict(exclude_unset=True).items():
        setattr(customization, field, value)
    
    db.commit()
    db.refresh(customization)
    
    return customization


@router.delete("/customizations/{customization_id}")
async def delete_customization(
    customization_id: int,
    user_id: int,  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Delete an avatar customization."""
    
    customization = db.query(AvatarCustomization).filter(
        AvatarCustomization.id == customization_id,
        AvatarCustomization.user_id == user_id
    ).first()
    
    if not customization:
        raise HTTPException(status_code=404, detail="Customization not found")
    
    db.delete(customization)
    db.commit()
    
    return {"message": "Customization deleted successfully"}


@router.post("/render/poses")
async def generate_avatar_poses(
    text: str,
    emotion: Optional[str] = None,
    emotion_intensity: float = 0.5,
    signing_speed: float = 1.0
):
    """
    Generate pose sequence for avatar animation from text input.
    
    This endpoint creates the skeletal animation data needed for 3D avatar
    rendering of sign language gestures.
    """
    try:
        # Generate pose sequence
        pose_sequence = avatar_rendering_service.generate_pose_sequence(
            text=text,
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            signing_speed=signing_speed
        )
        
        # Generate facial expressions
        facial_expressions = avatar_rendering_service.generate_facial_expressions(
            text=text,
            emotion=emotion,
            emotion_intensity=emotion_intensity
        )
        
        # Validate the sequence
        is_valid = avatar_rendering_service.validate_pose_sequence(pose_sequence)
        
        # Get performance metrics
        metrics = avatar_rendering_service.get_avatar_performance_metrics(pose_sequence)
        
        return {
            "pose_sequence": pose_sequence,
            "facial_expressions": facial_expressions,
            "is_valid": is_valid,
            "metrics": metrics,
            "text": text,
            "emotion": emotion,
            "emotion_intensity": emotion_intensity,
            "signing_speed": signing_speed
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Avatar rendering failed: {str(e)}")


@router.get("/render/performance/{avatar_id}")
async def get_avatar_performance(
    avatar_id: int,
    db: Session = Depends(get_db)
):
    """Get performance characteristics for a specific avatar."""
    
    avatar = db.query(Avatar).filter(
        Avatar.id == avatar_id,
        Avatar.is_active == True
    ).first()
    
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    # Simulate performance metrics based on avatar complexity
    performance = {
        "avatar_id": avatar_id,
        "estimated_render_fps": 60,
        "memory_usage_mb": 128,
        "gpu_usage_percent": 45,
        "joint_count": 55,  # SMPL-X standard
        "polygon_count": 8000,
        "texture_size_mb": 16,
        "supports_real_time": True,
        "recommended_quality": "standard"
    }
    
    return performance