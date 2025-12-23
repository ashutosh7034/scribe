"""
Emotion Analysis API endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
import numpy as np
import base64
import logging

from app.services.emotion_analysis import EmotionAnalysisService
from app.schemas.emotion import (
    EmotionAnalysisRequest,
    EmotionAnalysisResponse,
    EmotionTrends,
    EmotionServiceHealth
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global emotion analysis service instance
emotion_service = EmotionAnalysisService()


@router.post("/analyze", response_model=EmotionAnalysisResponse)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """
    Analyze emotion from text and optional audio data.
    
    Args:
        request: Emotion analysis request containing text and optional audio
        
    Returns:
        Comprehensive emotion analysis result
    """
    try:
        # Decode audio data if provided
        audio_data = None
        if request.audio_data:
            try:
                # Decode base64 audio data
                audio_bytes = base64.b64decode(request.audio_data)
                # Convert to numpy array (assuming 16-bit PCM)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                logger.warning(f"Failed to decode audio data: {e}")
                audio_data = None
        
        # Perform emotion analysis
        result = await emotion_service.analyze_emotion(
            text=request.text,
            audio_data=audio_data,
            sample_rate=request.sample_rate
        )
        
        return EmotionAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")


@router.get("/trends", response_model=EmotionTrends)
async def get_emotion_trends():
    """
    Get emotion trends from recent analysis history.
    
    Returns:
        Emotion trends and statistics
    """
    try:
        trends = emotion_service.get_emotion_trends()
        return trends
    except Exception as e:
        logger.error(f"Error getting emotion trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get emotion trends: {str(e)}")


@router.post("/reset-history")
async def reset_emotion_history():
    """
    Reset emotion analysis history.
    
    Returns:
        Success confirmation
    """
    try:
        emotion_service.reset_history()
        return {"message": "Emotion history reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting emotion history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset emotion history: {str(e)}")


@router.get("/health", response_model=EmotionServiceHealth)
async def emotion_service_health():
    """
    Check emotion analysis service health.
    
    Returns:
        Service health status
    """
    try:
        # Test basic functionality
        test_result = await emotion_service.analyze_emotion("Hello world")
        
        return {
            "status": "healthy",
            "sentiment_classifier_available": emotion_service.sentiment_classifier.sentiment_pipeline is not None,
            "prosodic_extractor_available": emotion_service.prosodic_extractor is not None,
            "test_analysis_successful": test_result is not None
        }
    except Exception as e:
        logger.error(f"Emotion service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "sentiment_classifier_available": False,
            "prosodic_extractor_available": False,
            "test_analysis_successful": False
        }