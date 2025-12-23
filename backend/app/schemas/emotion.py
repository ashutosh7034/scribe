"""
Emotion Analysis Schemas

Pydantic models for emotion analysis API requests and responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class EmotionVector(BaseModel):
    """Emotion vector in valence-arousal-dominance space."""
    valence: float = Field(..., ge=-1.0, le=1.0, description="Pleasure/displeasure dimension")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Activation/deactivation dimension")
    dominance: float = Field(..., ge=-1.0, le=1.0, description="Control/lack of control dimension")


class ProsodicFeatures(BaseModel):
    """Prosodic features extracted from audio."""
    pitch_mean: float = Field(..., description="Mean fundamental frequency")
    pitch_std: float = Field(..., description="Standard deviation of pitch")
    pitch_range: float = Field(..., description="Pitch range (max - min)")
    pitch_slope: float = Field(..., description="Pitch trend over time")
    voiced_ratio: float = Field(..., description="Ratio of voiced frames")
    energy_mean: float = Field(..., description="Mean RMS energy")
    energy_std: float = Field(..., description="Standard deviation of energy")
    energy_max: float = Field(..., description="Maximum energy")
    zcr_mean: float = Field(..., description="Mean zero crossing rate")
    zcr_std: float = Field(..., description="Standard deviation of ZCR")
    spectral_centroid_mean: float = Field(..., description="Mean spectral centroid")
    spectral_centroid_std: float = Field(..., description="Standard deviation of spectral centroid")
    spectral_rolloff_mean: float = Field(..., description="Mean spectral rolloff")
    spectral_bandwidth_mean: float = Field(..., description="Mean spectral bandwidth")
    mfcc_mean: float = Field(..., description="Mean MFCC coefficients")
    mfcc_std: float = Field(..., description="Standard deviation of MFCC")
    speaking_rate: float = Field(..., description="Estimated speaking rate")
    pause_ratio: float = Field(..., description="Ratio of silence/pause frames")
    duration: float = Field(..., description="Audio duration in seconds")
    hnr_estimate: float = Field(..., description="Harmonic-to-noise ratio estimate")
    spectral_flatness: float = Field(..., description="Spectral flatness measure")


class SentimentResult(BaseModel):
    """Sentiment classification result."""
    label: str = Field(..., description="Sentiment label (POSITIVE, NEGATIVE, NEUTRAL)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    all_scores: List[Dict[str, Any]] = Field(..., description="All classification scores")


class SigningModulation(BaseModel):
    """Parameters for emotion-aware signing modulation."""
    intensity_scale: float = Field(..., description="Gesture intensity scaling factor")
    speed_multiplier: float = Field(..., description="Signing speed multiplier")
    gesture_size_scale: float = Field(..., description="Gesture size scaling factor")
    facial_expression_intensity: float = Field(..., description="Facial expression intensity")
    signing_space_scale: float = Field(..., description="Signing space scaling factor")


class EmotionAnalysisRequest(BaseModel):
    """Request for emotion analysis."""
    text: str = Field(..., description="Text to analyze for emotion")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")


class EmotionAnalysisResponse(BaseModel):
    """Complete emotion analysis response."""
    timestamp: str = Field(..., description="Analysis timestamp")
    text_input: str = Field(..., description="Input text")
    has_audio: bool = Field(..., description="Whether audio data was provided")
    
    # Text analysis results
    text_sentiment: SentimentResult = Field(..., description="Text sentiment analysis")
    text_emotion_vector: EmotionVector = Field(..., description="Emotion vector from text")
    
    # Audio analysis results
    prosodic_features: ProsodicFeatures = Field(..., description="Prosodic features from audio")
    audio_emotion_vector: EmotionVector = Field(..., description="Emotion vector from audio")
    
    # Combined results
    combined_emotion_vector: EmotionVector = Field(..., description="Combined emotion vector")
    discrete_emotion: str = Field(..., description="Discrete emotion category")
    emotion_intensity: float = Field(..., description="Overall emotion intensity")
    
    # Signing parameters
    signing_modulation: SigningModulation = Field(..., description="Signing modulation parameters")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class EmotionHistoryEntry(BaseModel):
    """Single entry in emotion history."""
    timestamp: str = Field(..., description="Entry timestamp")
    discrete_emotion: str = Field(..., description="Discrete emotion category")
    emotion_vector: EmotionVector = Field(..., description="Emotion vector")
    intensity: float = Field(..., description="Emotion intensity")


class EmotionTrends(BaseModel):
    """Emotion trends analysis."""
    trend: str = Field(..., description="Intensity trend (increasing, decreasing, stable)")
    dominant_emotion: str = Field(..., description="Most frequent recent emotion")
    average_intensity: float = Field(..., description="Average emotion intensity")
    emotion_distribution: Dict[str, int] = Field(..., description="Distribution of emotions")


class EmotionServiceHealth(BaseModel):
    """Emotion service health status."""
    status: str = Field(..., description="Service status (healthy, unhealthy)")
    sentiment_classifier_available: bool = Field(..., description="BERT classifier availability")
    prosodic_extractor_available: bool = Field(..., description="Prosodic extractor availability")
    test_analysis_successful: bool = Field(..., description="Test analysis success")
    error: Optional[str] = Field(None, description="Error message if unhealthy")