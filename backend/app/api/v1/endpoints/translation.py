"""
Translation Endpoints

Real-time sign language translation API endpoints with speech processing integration.
"""

import uuid
import json
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
    
    # TODO: Implement actual translation logic
    # For now, return mock response structure
    
    mock_response = TranslationResponse(
        session_id=session_id,
        original_text=request.text,
        normalized_text=request.text.upper(),  # Simplified normalization
        pose_sequence=[
            {
                "timestamp": 0.0,
                "joints": {
                    "right_hand": {"x": 0.5, "y": 0.8, "z": 0.2},
                    "left_hand": {"x": -0.5, "y": 0.8, "z": 0.2},
                    # ... more joint positions
                }
            }
        ],
        facial_expressions=[
            {
                "timestamp": 0.0,
                "expression": "neutral",
                "intensity": 0.5
            }
        ],
        duration_ms=len(request.text) * 100,  # Rough estimate
        emotion_detected="neutral",
        confidence_score=0.85,
        processing_time_ms=150
    )
    
    return mock_response


@router.websocket("/stream")
async def websocket_translation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming translation with speech processing.
    
    Accepts WebRTC signaling messages and audio streams, returns real-time 
    avatar animation frames with speech-to-text processing.
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    speech_pipeline = SpeechProcessingPipeline()
    
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
                
                # If this is an offer, start audio processing
                if data.get('payload', {}).get('type') == 'offer':
                    await _start_audio_processing(
                        connection_id, websocket, speech_pipeline
                    )
            
            elif message_type == 'audio-chunk':
                # Handle direct audio data (fallback for non-WebRTC clients)
                audio_data = data.get('audio_data', b'')
                if isinstance(audio_data, str):
                    import base64
                    audio_data = base64.b64decode(audio_data)
                
                # Process audio chunk directly
                await _process_audio_chunk_direct(
                    audio_data, websocket, speech_pipeline
                )
            
            elif message_type == 'get-stats':
                # Return processing statistics
                stats = {
                    'speech_processing': speech_pipeline.get_processing_stats(),
                    'webrtc_service': webrtc_service.get_service_stats(),
                    'connection_id': connection_id
                }
                
                await websocket.send_json({
                    'type': 'stats',
                    'payload': stats
                })
            
            else:
                # Handle other message types (legacy support)
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
        webrtc_service.close_connection(connection_id)


async def _start_audio_processing(connection_id: str, 
                                websocket: WebSocket, 
                                speech_pipeline: SpeechProcessingPipeline):
    """Start audio processing for WebRTC connection."""
    
    async def process_transcription_result(result: Dict[str, Any]):
        """Process and send transcription results to client."""
        try:
            # Create translation response from speech processing result
            translation_response = {
                'type': 'translation_result',
                'session_id': connection_id,
                'timestamp': result.get('start_time', 0),
                'transcription': {
                    'text': result.get('transcript', ''),
                    'confidence': result.get('confidence', 0.0),
                    'is_partial': result.get('is_partial', True),
                    'speaker_label': result.get('speaker_label', 'unknown'),
                    'active_speakers': result.get('active_speakers', []),
                    'speaker_count': result.get('speaker_count', 1)
                },
                'pose_data': {
                    # TODO: Generate actual pose data from transcript
                    'right_hand': {'x': 0.5, 'y': 0.8, 'z': 0.2},
                    'left_hand': {'x': -0.5, 'y': 0.8, 'z': 0.2},
                },
                'facial_expression': 'neutral',
                'processing_time_ms': result.get('processing_time_ms', 0),
                'pipeline_stats': result.get('pipeline_stats', {})
            }
            
            await websocket.send_json(translation_response)
            
        except Exception as e:
            logger.error(f"Error sending transcription result: {e}")
    
    # Start audio stream processing
    await webrtc_service.start_audio_stream_processing(
        connection_id,
        lambda audio_stream: speech_pipeline.process_real_time_audio(
            audio_stream, process_transcription_result
        )
    )


async def _process_audio_chunk_direct(audio_data: bytes, 
                                    websocket: WebSocket,
                                    speech_pipeline: SpeechProcessingPipeline):
    """Process audio chunk directly (non-WebRTC fallback)."""
    
    async def audio_generator():
        """Simple generator for single audio chunk."""
        yield audio_data
    
    async def process_result(result: Dict[str, Any]):
        """Process single audio chunk result."""
        response = {
            'type': 'transcription_chunk',
            'transcript': result.get('transcript', ''),
            'confidence': result.get('confidence', 0.0),
            'processing_time_ms': result.get('processing_time_ms', 0)
        }
        await websocket.send_json(response)
    
    # Process single audio chunk
    await speech_pipeline.process_real_time_audio(
        audio_generator(), process_result
    )


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