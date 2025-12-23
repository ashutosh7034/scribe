"""
Translation Endpoints

Real-time sign language translation API endpoints with speech processing integration.
"""

import uuid
import json
import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.translation import TranslationSession, TranslationFeedback
from app.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
    TranslationSessionCreate,
    TranslationSessionResponse,
    FeedbackCreate,
    FeedbackResponse,
)
from app.services.speech_processing import SpeechProcessingPipeline
from app.services.webrtc_service import webrtc_service
from app.services.sign_language_translation import get_translation_service, SignLanguage
from app.services.pipeline_integration import pipeline_manager, PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    db: Session = Depends(get_db)
):
    """
    Translate text to sign language animation.
    
    This endpoint processes spoken text and returns 3D avatar animation data
    for real-time sign language display.
    """
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get translation service
    sign_language = SignLanguage.ASL  # Default to ASL, could be from request
    if hasattr(request, 'target_sign_language') and request.target_sign_language == "BSL":
        sign_language = SignLanguage.BSL
    
    translation_service = get_translation_service(sign_language)
    
    # Perform translation
    translation_result = translation_service.translate_text(
        request.text, 
        emotion_intensity=request.emotion_intensity
    )
    
    # Convert to API response format
    response = TranslationResponse(
        session_id=session_id,
        original_text=translation_result.original_text,
        normalized_text=translation_result.normalized_text,
        pose_sequence=translation_result.to_pose_sequence(),
        facial_expressions=translation_result.to_facial_expressions(),
        duration_ms=translation_result.total_duration_ms,
        emotion_detected="neutral",  # TODO: Integrate with emotion analysis service
        confidence_score=translation_result.confidence_score,
        processing_time_ms=translation_result.processing_time_ms
    )
    
    return response


@router.post("/pipeline", response_model=TranslationResponse)
async def translate_with_integrated_pipeline(
    request: TranslationRequest,
    db: Session = Depends(get_db)
):
    """
    Translate text using the integrated speech-to-avatar pipeline.
    
    This endpoint demonstrates the complete pipeline integration by processing
    text through speech processing simulation, translation, and avatar rendering.
    """
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get translation service
    sign_language = SignLanguage.ASL  # Default to ASL, could be from request
    if hasattr(request, 'target_sign_language') and request.target_sign_language == "BSL":
        sign_language = SignLanguage.BSL
    
    # Use the integrated pipeline orchestrator for processing
    orchestrator = pipeline_manager.orchestrator
    
    # Simulate speech processing result for text input
    simulated_speech_result = {
        'transcript': request.text,
        'confidence': 0.95,  # High confidence for direct text input
        'processing_time_ms': 10,  # Minimal processing time for text
        'speaker_label': 'user',
        'active_speakers': ['user'],
        'speaker_count': 1,
        'is_speaker_change': False,
        'start_time': 0.0,
        'end_time': len(request.text) * 0.1  # Estimate based on text length
    }
    
    # Process through integrated pipeline
    pipeline_result = None
    
    async def capture_result(result):
        nonlocal pipeline_result
        pipeline_result = result
    
    # Create a simple audio generator for the text (simulation)
    async def text_audio_generator():
        # Simulate audio chunk for text processing
        yield b'simulated_audio_data'
    
    # Process through pipeline
    try:
        # Start a temporary session for this request
        temp_session_id = f"{session_id}_sync"
        await pipeline_manager.start_session(
            session_id=temp_session_id,
            audio_stream=text_audio_generator(),
            result_callback=capture_result,
            signing_speed=getattr(request, 'signing_speed', 1.0),
            target_sign_language=sign_language
        )
        
        # Wait a bit for processing (in real implementation, this would be event-driven)
        import asyncio
        await asyncio.sleep(0.1)
        
        # Stop the temporary session
        await pipeline_manager.stop_session(temp_session_id)
        
    except Exception as e:
        logger.error(f"Pipeline processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    
    # If no result captured, fall back to direct translation
    if not pipeline_result:
        translation_service = get_translation_service(sign_language)
        translation_result = translation_service.translate_text(
            request.text, 
            emotion_intensity=request.emotion_intensity
        )
        
        # Convert to API response format
        response = TranslationResponse(
            session_id=session_id,
            original_text=translation_result.original_text,
            normalized_text=translation_result.normalized_text,
            pose_sequence=translation_result.to_pose_sequence(),
            facial_expressions=translation_result.to_facial_expressions(),
            duration_ms=translation_result.total_duration_ms,
            emotion_detected="neutral",
            confidence_score=translation_result.confidence_score,
            processing_time_ms=translation_result.processing_time_ms
        )
    else:
        # Use pipeline result
        response = TranslationResponse(
            session_id=session_id,
            original_text=pipeline_result.transcript,
            normalized_text=pipeline_result.normalized_text,
            pose_sequence=pipeline_result.pose_sequence,
            facial_expressions=pipeline_result.facial_expressions,
            duration_ms=pipeline_result.animation_duration_ms,
            emotion_detected=pipeline_result.detected_emotion or "neutral",
            confidence_score=pipeline_result.translation_confidence,
            processing_time_ms=pipeline_result.total_processing_time_ms
        )
    
    return response


@router.get("/pipeline/stats")
async def get_pipeline_statistics():
    """Get comprehensive pipeline performance statistics."""
    
    stats = pipeline_manager.get_pipeline_stats()
    
    return {
        "pipeline_stats": stats,
        "service_health": {
            "status": "healthy" if stats.get('latency_compliance_rate', 0) > 0.8 else "degraded",
            "active_sessions": stats.get('active_sessions', 0),
            "total_translations": stats.get('successful_translations', 0),
            "average_latency_ms": stats.get('average_latency_ms', 0),
            "average_frame_rate": stats.get('average_frame_rate', 0),
            "real_time_compliance": {
                "latency_compliance_rate": stats.get('latency_compliance_rate', 0),
                "frame_rate_compliance_rate": stats.get('frame_rate_compliance_rate', 0)
            }
        }
    }
async def translate_text(
    request: TranslationRequest,
    db: Session = Depends(get_db)
):
    """
    Translate text to sign language animation.
    
    This endpoint processes spoken text and returns 3D avatar animation data
    for real-time sign language display.
    """
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get translation service
    sign_language = SignLanguage.ASL  # Default to ASL, could be from request
    if hasattr(request, 'target_sign_language') and request.target_sign_language == "BSL":
        sign_language = SignLanguage.BSL
    
    translation_service = get_translation_service(sign_language)
    
    # Perform translation
    translation_result = translation_service.translate_text(
        request.text, 
        emotion_intensity=request.emotion_intensity
    )
    
    # Convert to API response format
    response = TranslationResponse(
        session_id=session_id,
        original_text=translation_result.original_text,
        normalized_text=translation_result.normalized_text,
        pose_sequence=translation_result.to_pose_sequence(),
        facial_expressions=translation_result.to_facial_expressions(),
        duration_ms=translation_result.total_duration_ms,
        emotion_detected="neutral",  # TODO: Integrate with emotion analysis service
        confidence_score=translation_result.confidence_score,
        processing_time_ms=translation_result.processing_time_ms
    )
    
    return response


@router.websocket("/stream")
async def websocket_translation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming translation with integrated pipeline.
    
    Accepts WebRTC signaling messages and audio streams, returns real-time 
    avatar animation frames with complete speech-to-avatar processing.
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get('type', 'unknown')
            
            if message_type == 'webrtc-signaling':
                # Handle WebRTC signaling
                response = await webrtc_service.handle_websocket_message(
                    connection_id, data.get('payload', {})
                )
                
                await websocket.send_json({
                    'type': 'webrtc-response',
                    'payload': response
                })
                
                # If this is an offer, start integrated pipeline processing
                if data.get('payload', {}).get('type') == 'offer':
                    await _start_integrated_pipeline(
                        connection_id, websocket, data.get('config', {})
                    )
            
            elif message_type == 'audio-chunk':
                # Handle direct audio data (fallback for non-WebRTC clients)
                audio_data = data.get('audio_data', b'')
                if isinstance(audio_data, str):
                    import base64
                    audio_data = base64.b64decode(audio_data)
                
                # Process audio chunk through integrated pipeline
                await _process_audio_chunk_integrated(
                    connection_id, audio_data, websocket, data.get('config', {})
                )
            
            elif message_type == 'get-stats':
                # Return comprehensive pipeline statistics
                stats = {
                    'pipeline_stats': pipeline_manager.get_pipeline_stats(),
                    'webrtc_service': webrtc_service.get_service_stats(),
                    'connection_id': connection_id,
                    'active_sessions': pipeline_manager.get_active_sessions()
                }
                
                await websocket.send_json({
                    'type': 'stats',
                    'payload': stats
                })
            
            elif message_type == 'configure-pipeline':
                # Allow runtime pipeline configuration
                config = data.get('config', {})
                await _configure_pipeline_session(connection_id, config, websocket)
            
            else:
                # Handle legacy message types for backward compatibility
                response = {
                    'type': 'translation_frame',
                    'session_id': data.get('session_id', connection_id),
                    'timestamp': data.get('timestamp', 0),
                    'pose_data': {
                        'right_hand': {'x': 0.5, 'y': 0.8, 'z': 0.2},
                        'left_hand': {'x': -0.5, 'y': 0.8, 'z': 0.2},
                    },
                    'facial_expression': 'neutral',
                    'processing_time_ms': 50
                }
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
    finally:
        # Cleanup resources
        await pipeline_manager.stop_session(connection_id)
        webrtc_service.close_connection(connection_id)


async def _start_integrated_pipeline(connection_id: str, 
                                   websocket: WebSocket, 
                                   config: Dict[str, Any]):
    """Start integrated speech-to-avatar pipeline for WebRTC connection."""
    
    # Extract configuration
    signing_speed = config.get('signing_speed', 1.0)
    target_language = config.get('target_sign_language', 'ASL')
    sign_language = SignLanguage.ASL if target_language == 'ASL' else SignLanguage.BSL
    
    async def process_pipeline_result(result: PipelineResult):
        """Process and send complete pipeline results to client."""
        try:
            # Create comprehensive response with all pipeline data
            pipeline_response = {
                'type': 'pipeline_result',
                'session_id': result.session_id,
                'timestamp': result.timestamp,
                
                # Speech processing results
                'speech': {
                    'transcript': result.transcript,
                    'confidence': result.speech_confidence,
                    'speaker_info': result.speaker_info
                },
                
                # Translation results
                'translation': {
                    'original_text': result.transcript,
                    'normalized_text': result.normalized_text,
                    'confidence': result.translation_confidence
                },
                
                # Emotion analysis results
                'emotion': {
                    'detected_emotion': result.detected_emotion,
                    'intensity': result.emotion_intensity
                },
                
                # Avatar animation data
                'avatar': {
                    'pose_sequence': [
                        {
                            'timestamp': pose.timestamp,
                            'joints': {
                                joint_name: {
                                    'x': joint.x,
                                    'y': joint.y, 
                                    'z': joint.z
                                }
                                for joint_name, joint in pose.joints.items()
                            }
                        }
                        for pose in result.pose_sequence
                    ],
                    'facial_expressions': [
                        {
                            'timestamp': expr.timestamp,
                            'expression': expr.expression,
                            'intensity': expr.intensity
                        }
                        for expr in result.facial_expressions
                    ],
                    'duration_ms': result.animation_duration_ms,
                    'frame_rate': result.frame_rate
                },
                
                # Performance metrics
                'performance': {
                    'total_processing_time_ms': result.total_processing_time_ms,
                    'speech_processing_time_ms': result.speech_processing_time_ms,
                    'translation_time_ms': result.translation_time_ms,
                    'avatar_rendering_time_ms': result.avatar_rendering_time_ms,
                    'end_to_end_latency_ms': result.end_to_end_latency_ms,
                    'is_real_time_compliant': result.is_real_time_compliant
                }
            }
            
            await websocket.send_json(pipeline_response)
            
        except Exception as e:
            logger.error(f"Error sending pipeline result: {e}")
    
    # Get WebRTC connection and start pipeline
    webrtc_connection = webrtc_service.signaling_server.get_connection(connection_id)
    if webrtc_connection:
        # Start integrated pipeline with WebRTC audio stream
        await pipeline_manager.start_session(
            session_id=connection_id,
            audio_stream=webrtc_connection.get_audio_stream(),
            result_callback=process_pipeline_result,
            signing_speed=signing_speed,
            target_sign_language=sign_language
        )
    else:
        logger.error(f"WebRTC connection {connection_id} not found")


async def _process_audio_chunk_integrated(connection_id: str,
                                        audio_data: bytes, 
                                        websocket: WebSocket,
                                        config: Dict[str, Any]):
    """Process single audio chunk through integrated pipeline (non-WebRTC fallback)."""
    
    # Extract configuration
    signing_speed = config.get('signing_speed', 1.0)
    target_language = config.get('target_sign_language', 'ASL')
    sign_language = SignLanguage.ASL if target_language == 'ASL' else SignLanguage.BSL
    
    async def audio_generator():
        """Simple generator for single audio chunk."""
        yield audio_data
    
    async def process_result(result: PipelineResult):
        """Process single audio chunk result."""
        response = {
            'type': 'audio_chunk_result',
            'session_id': result.session_id,
            'transcript': result.transcript,
            'confidence': result.speech_confidence,
            'pose_count': len(result.pose_sequence),
            'processing_time_ms': result.total_processing_time_ms,
            'is_real_time_compliant': result.is_real_time_compliant
        }
        await websocket.send_json(response)
    
    # Process single audio chunk through pipeline
    await pipeline_manager.start_session(
        session_id=f"{connection_id}_chunk",
        audio_stream=audio_generator(),
        result_callback=process_result,
        signing_speed=signing_speed,
        target_sign_language=sign_language
    )


async def _configure_pipeline_session(connection_id: str,
                                    config: Dict[str, Any],
                                    websocket: WebSocket):
    """Configure pipeline session parameters."""
    
    # Validate configuration
    valid_config = {}
    
    if 'signing_speed' in config:
        speed = float(config['signing_speed'])
        if 0.5 <= speed <= 2.0:
            valid_config['signing_speed'] = speed
    
    if 'target_sign_language' in config:
        lang = config['target_sign_language']
        if lang in ['ASL', 'BSL']:
            valid_config['target_sign_language'] = lang
    
    # Send configuration confirmation
    response = {
        'type': 'pipeline_configured',
        'session_id': connection_id,
        'applied_config': valid_config,
        'timestamp': time.time()
    }
    
    await websocket.send_json(response)


@router.post("/sessions", response_model=TranslationSessionResponse)
async def create_session(
    session_data: TranslationSessionCreate,
    db: Session = Depends(get_db)
):
    """Create a new translation session."""
    
    session = TranslationSession(
        session_id=str(uuid.uuid4()),
        user_id=session_data.user_id,
        source_language=session_data.source_language,
        target_sign_language=session_data.target_sign_language,
        avatar_id=session_data.avatar_id,
        client_platform=session_data.client_platform,
        client_version=session_data.client_version,
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


@router.get("/sessions/{session_id}", response_model=TranslationSessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """Get translation session details."""
    
    session = db.query(TranslationSession).filter(
        TranslationSession.session_id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """Submit feedback on translation quality."""
    
    # Verify session exists
    session = db.query(TranslationSession).filter(
        TranslationSession.id == feedback.session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create feedback entry
    feedback_entry = TranslationFeedback(
        session_id=feedback.session_id,
        original_text=feedback.original_text,
        feedback_type=feedback.feedback_type,
        rating=feedback.rating,
        comment=feedback.comment,
        suggested_correction=feedback.suggested_correction,
        is_emergency_phrase=feedback.is_emergency_phrase,
        timestamp_in_session=feedback.timestamp_in_session,
        processing_priority=5 if feedback.is_emergency_phrase else 1,
    )
    
    db.add(feedback_entry)
    db.commit()
    db.refresh(feedback_entry)
    
    return feedback_entry


@router.get("/feedback/{session_id}", response_model=List[FeedbackResponse])
async def get_session_feedback(session_id: int, db: Session = Depends(get_db)):
    """Get all feedback for a translation session."""
    
    feedback_entries = db.query(TranslationFeedback).filter(
        TranslationFeedback.session_id == session_id
    ).all()
    
    return feedback_entries