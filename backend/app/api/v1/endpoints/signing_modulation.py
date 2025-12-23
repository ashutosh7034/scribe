"""
Signing Modulation API endpoints.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from app.services.signing_modulation import SigningModulationService, EmotionCategory
from app.services.emotion_analysis import EmotionVector

logger = logging.getLogger(__name__)

router = APIRouter()

# Global signing modulation service instance
signing_modulation_service = SigningModulationService()


class EmotionVectorRequest(BaseModel):
    """Emotion vector in VAD space."""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Pleasure/displeasure dimension")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Activation/deactivation dimension")
    dominance: float = Field(..., ge=-1.0, le=1.0, description="Control/lack of control dimension")


class SigningModulationRequest(BaseModel):
    """Request for signing modulation generation."""
    emotion_vector: EmotionVectorRequest
    discrete_emotion: str = Field(..., description="Discrete emotion category")
    emotion_intensity: float = Field(..., ge=0.0, le=2.0, description="Emotion intensity")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")


class FACSActionUnitResponse(BaseModel):
    """FACS Action Unit response."""
    au_number: int
    name: str
    intensity: float
    description: str


class SigningModulationResponse(BaseModel):
    """Complete signing modulation response."""
    # Intensity and amplitude
    intensity_scale: float
    gesture_amplitude_scale: float
    
    # Temporal modulation
    speed_multiplier: float
    pause_duration_scale: float
    transition_smoothness: float
    
    # Spatial modulation
    gesture_size_scale: float
    signing_space_scale: float
    hand_separation_scale: float
    
    # Facial expression
    facial_expression_intensity: float
    facs_action_units: list[FACSActionUnitResponse]
    
    # Body posture
    posture_tension: float
    shoulder_elevation: float
    torso_lean: float
    
    # Rhythm and dynamics
    rhythm_regularity: float
    dynamic_range: float
    accent_strength: float


class EmotionBlendRequest(BaseModel):
    """Request for blending two emotions."""
    primary_emotion: tuple[str, float] = Field(..., description="Primary emotion (name, intensity)")
    secondary_emotion: tuple[str, float] = Field(..., description="Secondary emotion (name, intensity)")
    blend_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Blend ratio (primary to secondary)")


@router.post("/generate", response_model=SigningModulationResponse)
async def generate_signing_modulation(request: SigningModulationRequest):
    """
    Generate signing modulation parameters from emotion analysis.
    
    Args:
        request: Emotion analysis data and context
        
    Returns:
        Complete signing modulation parameters
    """
    try:
        # Convert request to internal types
        emotion_vector = EmotionVector(
            valence=request.emotion_vector.valence,
            arousal=request.emotion_vector.arousal,
            dominance=request.emotion_vector.dominance
        )
        
        # Generate modulation parameters
        modulation = signing_modulation_service.generate_signing_modulation(
            emotion_vector=emotion_vector,
            discrete_emotion=request.discrete_emotion,
            emotion_intensity=request.emotion_intensity,
            context=request.context
        )
        
        # Apply contextual adjustments if provided
        if request.context:
            modulation = signing_modulation_service.apply_contextual_adjustments(
                modulation, request.context
            )
        
        # Convert FACS Action Units to response format
        facs_aus = [
            FACSActionUnitResponse(
                au_number=au.au_number,
                name=au.name,
                intensity=au.intensity,
                description=au.description
            )
            for au in modulation.facs_action_units
        ]
        
        return SigningModulationResponse(
            intensity_scale=modulation.intensity_scale,
            gesture_amplitude_scale=modulation.gesture_amplitude_scale,
            speed_multiplier=modulation.speed_multiplier,
            pause_duration_scale=modulation.pause_duration_scale,
            transition_smoothness=modulation.transition_smoothness,
            gesture_size_scale=modulation.gesture_size_scale,
            signing_space_scale=modulation.signing_space_scale,
            hand_separation_scale=modulation.hand_separation_scale,
            facial_expression_intensity=modulation.facial_expression_intensity,
            facs_action_units=facs_aus,
            posture_tension=modulation.posture_tension,
            shoulder_elevation=modulation.shoulder_elevation,
            torso_lean=modulation.torso_lean,
            rhythm_regularity=modulation.rhythm_regularity,
            dynamic_range=modulation.dynamic_range,
            accent_strength=modulation.accent_strength
        )
        
    except Exception as e:
        logger.error(f"Error generating signing modulation: {e}")
        raise HTTPException(status_code=500, detail=f"Signing modulation generation failed: {str(e)}")


@router.post("/blend", response_model=SigningModulationResponse)
async def blend_emotions(request: EmotionBlendRequest):
    """
    Blend signing modulation parameters for mixed emotions.
    
    Args:
        request: Primary and secondary emotions with blend ratio
        
    Returns:
        Blended signing modulation parameters
    """
    try:
        # Generate blended modulation
        modulation = signing_modulation_service.blend_emotions(
            primary_emotion=request.primary_emotion,
            secondary_emotion=request.secondary_emotion,
            blend_ratio=request.blend_ratio
        )
        
        # Convert FACS Action Units to response format
        facs_aus = [
            FACSActionUnitResponse(
                au_number=au.au_number,
                name=au.name,
                intensity=au.intensity,
                description=au.description
            )
            for au in modulation.facs_action_units
        ]
        
        return SigningModulationResponse(
            intensity_scale=modulation.intensity_scale,
            gesture_amplitude_scale=modulation.gesture_amplitude_scale,
            speed_multiplier=modulation.speed_multiplier,
            pause_duration_scale=modulation.pause_duration_scale,
            transition_smoothness=modulation.transition_smoothness,
            gesture_size_scale=modulation.gesture_size_scale,
            signing_space_scale=modulation.signing_space_scale,
            hand_separation_scale=modulation.hand_separation_scale,
            facial_expression_intensity=modulation.facial_expression_intensity,
            facs_action_units=facs_aus,
            posture_tension=modulation.posture_tension,
            shoulder_elevation=modulation.shoulder_elevation,
            torso_lean=modulation.torso_lean,
            rhythm_regularity=modulation.rhythm_regularity,
            dynamic_range=modulation.dynamic_range,
            accent_strength=modulation.accent_strength
        )
        
    except Exception as e:
        logger.error(f"Error blending emotions: {e}")
        raise HTTPException(status_code=500, detail=f"Emotion blending failed: {str(e)}")


@router.get("/emotions")
async def get_supported_emotions():
    """
    Get list of supported emotion categories.
    
    Returns:
        List of supported emotion categories
    """
    return {
        "supported_emotions": [emotion.value for emotion in EmotionCategory],
        "description": "Discrete emotion categories supported by the signing modulation system"
    }


@router.get("/facs-units")
async def get_facs_action_units():
    """
    Get information about FACS Action Units used for facial expression.
    
    Returns:
        Dictionary of FACS Action Units with descriptions
    """
    facs_mapper = signing_modulation_service.facs_mapper
    
    action_units_info = {}
    for au_number, au in facs_mapper.action_units.items():
        action_units_info[au_number] = {
            "name": au.name,
            "description": au.description,
            "category": "upper_face" if au_number <= 7 else "lower_face"
        }
    
    return {
        "facs_action_units": action_units_info,
        "description": "Facial Action Coding System (FACS) Action Units used for facial expression mapping"
    }


@router.post("/test-modulation")
async def test_modulation_with_sample_emotions():
    """
    Test signing modulation with sample emotions for debugging/demonstration.
    
    Returns:
        Modulation parameters for various sample emotions
    """
    try:
        sample_emotions = [
            ("joy", 0.8),
            ("anger", 0.9),
            ("sadness", 0.6),
            ("fear", 0.7),
            ("surprise", 0.8),
            ("neutral", 0.1)
        ]
        
        results = {}
        
        for emotion_name, intensity in sample_emotions:
            # Create sample emotion vector
            if emotion_name == "joy":
                emotion_vector = EmotionVector(valence=0.7, arousal=0.3, dominance=0.5)
            elif emotion_name == "anger":
                emotion_vector = EmotionVector(valence=-0.6, arousal=0.8, dominance=0.7)
            elif emotion_name == "sadness":
                emotion_vector = EmotionVector(valence=-0.8, arousal=-0.4, dominance=-0.5)
            elif emotion_name == "fear":
                emotion_vector = EmotionVector(valence=-0.7, arousal=0.5, dominance=-0.8)
            elif emotion_name == "surprise":
                emotion_vector = EmotionVector(valence=0.2, arousal=0.7, dominance=0.0)
            else:  # neutral
                emotion_vector = EmotionVector(valence=0.0, arousal=0.0, dominance=0.0)
            
            # Generate modulation
            modulation = signing_modulation_service.generate_signing_modulation(
                emotion_vector=emotion_vector,
                discrete_emotion=emotion_name,
                emotion_intensity=intensity
            )
            
            # Get summary
            summary = signing_modulation_service.get_modulation_summary(modulation)
            results[emotion_name] = summary
        
        return {
            "sample_modulations": results,
            "description": "Sample signing modulation parameters for various emotions"
        }
        
    except Exception as e:
        logger.error(f"Error in test modulation: {e}")
        raise HTTPException(status_code=500, detail=f"Test modulation failed: {str(e)}")


@router.get("/health")
async def signing_modulation_health():
    """
    Check signing modulation service health.
    
    Returns:
        Service health status
    """
    try:
        # Test basic functionality
        test_vector = EmotionVector(valence=0.5, arousal=0.3, dominance=0.2)
        test_modulation = signing_modulation_service.generate_signing_modulation(
            emotion_vector=test_vector,
            discrete_emotion="joy",
            emotion_intensity=0.7
        )
        
        return {
            "status": "healthy",
            "facs_mapper_available": signing_modulation_service.facs_mapper is not None,
            "temporal_modulator_available": signing_modulation_service.temporal_modulator is not None,
            "intensity_scaler_available": signing_modulation_service.intensity_scaler is not None,
            "posture_modulator_available": signing_modulation_service.posture_modulator is not None,
            "test_modulation_successful": test_modulation is not None,
            "supported_emotions_count": len(EmotionCategory)
        }
        
    except Exception as e:
        logger.error(f"Signing modulation health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "facs_mapper_available": False,
            "temporal_modulator_available": False,
            "intensity_scaler_available": False,
            "posture_modulator_available": False,
            "test_modulation_successful": False,
            "supported_emotions_count": 0
        }