"""
Signing Modulation Schemas

Pydantic models for signing modulation API requests and responses.
"""

from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class EmotionCategory(str, Enum):
    """Supported emotion categories."""
    NEUTRAL = "neutral"
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    EXCITEMENT = "excitement"


class EmotionVectorSchema(BaseModel):
    """Emotion vector in valence-arousal-dominance space."""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Pleasure/displeasure dimension")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Activation/deactivation dimension")
    dominance: float = Field(..., ge=-1.0, le=1.0, description="Control/lack of control dimension")


class FACSActionUnitSchema(BaseModel):
    """FACS Action Unit schema."""
    au_number: int = Field(..., description="Action Unit number")
    name: str = Field(..., description="Action Unit name")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Activation intensity")
    description: str = Field(..., description="Action Unit description")


class SigningModulationParametersSchema(BaseModel):
    """Complete signing modulation parameters."""
    # Intensity and amplitude
    intensity_scale: float = Field(..., description="Overall gesture intensity scaling")
    gesture_amplitude_scale: float = Field(..., description="Gesture amplitude scaling")
    
    # Temporal modulation
    speed_multiplier: float = Field(..., description="Signing speed multiplier")
    pause_duration_scale: float = Field(..., description="Pause duration scaling")
    transition_smoothness: float = Field(..., description="Transition smoothness factor")
    
    # Spatial modulation
    gesture_size_scale: float = Field(..., description="Gesture size scaling")
    signing_space_scale: float = Field(..., description="Signing space scaling")
    hand_separation_scale: float = Field(..., description="Hand separation scaling")
    
    # Facial expression
    facial_expression_intensity: float = Field(..., description="Facial expression intensity")
    facs_action_units: List[FACSActionUnitSchema] = Field(..., description="FACS Action Units")
    
    # Body posture
    posture_tension: float = Field(..., description="Body posture tension")
    shoulder_elevation: float = Field(..., description="Shoulder elevation")
    torso_lean: float = Field(..., description="Torso lean angle")
    
    # Rhythm and dynamics
    rhythm_regularity: float = Field(..., description="Rhythm regularity factor")
    dynamic_range: float = Field(..., description="Dynamic range scaling")
    accent_strength: float = Field(..., description="Accent strength factor")


class SigningModulationRequest(BaseModel):
    """Request for signing modulation generation."""
    emotion_vector: EmotionVectorSchema = Field(..., description="Emotion vector in VAD space")
    discrete_emotion: EmotionCategory = Field(..., description="Discrete emotion category")
    emotion_intensity: float = Field(..., ge=0.0, le=2.0, description="Emotion intensity")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context information")


class EmotionBlendRequest(BaseModel):
    """Request for blending two emotions."""
    primary_emotion: Tuple[EmotionCategory, float] = Field(..., description="Primary emotion (category, intensity)")
    secondary_emotion: Tuple[EmotionCategory, float] = Field(..., description="Secondary emotion (category, intensity)")
    blend_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Blend ratio (primary to secondary)")


class ContextualAdjustmentRequest(BaseModel):
    """Request for contextual adjustments to modulation."""
    base_modulation: SigningModulationParametersSchema = Field(..., description="Base modulation parameters")
    context: Dict[str, Any] = Field(..., description="Context information")


class SupportedEmotionsResponse(BaseModel):
    """Response with supported emotion categories."""
    supported_emotions: List[str] = Field(..., description="List of supported emotion categories")
    description: str = Field(..., description="Description of supported emotions")


class FACSActionUnitsResponse(BaseModel):
    """Response with FACS Action Units information."""
    facs_action_units: Dict[int, Dict[str, str]] = Field(..., description="FACS Action Units information")
    description: str = Field(..., description="Description of FACS system")


class ModulationSummary(BaseModel):
    """Summary of modulation parameters."""
    intensity_scale: float
    speed_multiplier: float
    gesture_size_scale: float
    facial_expression_intensity: float
    signing_space_scale: float
    active_facs_aus: int
    facs_au_numbers: List[int]
    posture_tension: float
    dynamic_range: float


class TestModulationResponse(BaseModel):
    """Response for test modulation endpoint."""
    sample_modulations: Dict[str, ModulationSummary] = Field(..., description="Sample modulations for various emotions")
    description: str = Field(..., description="Description of test results")


class SigningModulationHealthResponse(BaseModel):
    """Signing modulation service health response."""
    status: str = Field(..., description="Service status")
    facs_mapper_available: bool = Field(..., description="FACS mapper availability")
    temporal_modulator_available: bool = Field(..., description="Temporal modulator availability")
    intensity_scaler_available: bool = Field(..., description="Intensity scaler availability")
    posture_modulator_available: bool = Field(..., description="Posture modulator availability")
    test_modulation_successful: bool = Field(..., description="Test modulation success")
    supported_emotions_count: int = Field(..., description="Number of supported emotions")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class TemporalModulationRequest(BaseModel):
    """Request for temporal modulation of timing sequences."""
    base_timing: List[float] = Field(..., description="Base timing sequence")
    emotion_category: EmotionCategory = Field(..., description="Emotion category")
    emotion_intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity")


class TemporalModulationResponse(BaseModel):
    """Response for temporal modulation."""
    modified_timing: List[float] = Field(..., description="Modified timing sequence")
    temporal_parameters: Dict[str, float] = Field(..., description="Applied temporal parameters")


class IntensityScalingRequest(BaseModel):
    """Request for intensity scaling parameters."""
    emotion_category: EmotionCategory = Field(..., description="Emotion category")
    emotion_intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity")


class IntensityScalingResponse(BaseModel):
    """Response for intensity scaling."""
    scaling_parameters: Dict[str, float] = Field(..., description="Intensity scaling parameters")


class PostureModulationRequest(BaseModel):
    """Request for posture modulation parameters."""
    emotion_category: EmotionCategory = Field(..., description="Emotion category")
    emotion_intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity")


class PostureModulationResponse(BaseModel):
    """Response for posture modulation."""
    posture_parameters: Dict[str, float] = Field(..., description="Posture modulation parameters")