"""
Pipeline Integration Service

Orchestrates the complete speech-to-avatar pipeline by connecting:
1. Speech processing (speech-to-text)
2. Sign language translation (text-to-gestures)
3. Avatar rendering (gestures-to-animation)

This service implements the real-time streaming pipeline for Requirements 1.1, 1.2, 1.3.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, AsyncGenerator, List
from dataclasses import dataclass
from datetime import datetime

from app.services.speech_processing import SpeechProcessingPipeline
from app.services.sign_language_translation import get_translation_service, SignLanguage
from app.services.avatar_rendering import avatar_rendering_service
from app.services.emotion_analysis import EmotionAnalysisService
from app.schemas.translation import PoseKeyframe, FacialExpressionKeyframe

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline processing result."""
    
    session_id: str
    timestamp: float
    
    # Speech processing results
    transcript: str
    speech_confidence: float
    speaker_info: Dict[str, Any]
    
    # Translation results
    normalized_text: str
    translation_confidence: float
    
    # Emotion analysis results
    detected_emotion: Optional[str]
    emotion_intensity: float
    
    # Avatar animation results
    pose_sequence: List[PoseKeyframe]
    facial_expressions: List[FacialExpressionKeyframe]
    animation_duration_ms: int
    
    # Performance metrics
    total_processing_time_ms: int
    speech_processing_time_ms: int
    translation_time_ms: int
    avatar_rendering_time_ms: int
    
    # Quality metrics
    end_to_end_latency_ms: int
    frame_rate: float
    is_real_time_compliant: bool


class RealTimePipelineOrchestrator:
    """Orchestrates the real-time speech-to-avatar pipeline."""
    
    def __init__(self):
        self.speech_pipeline = SpeechProcessingPipeline()
        self.emotion_service = EmotionAnalysisService()
        self.translation_service = get_translation_service(SignLanguage.ASL)
        self.avatar_service = avatar_rendering_service
        
        # Pipeline configuration
        self.max_latency_ms = 300  # Requirement: <300ms total latency
        self.min_frame_rate = 30   # Requirement: minimum 30 FPS
        
        # Performance tracking
        self.pipeline_stats = {
            'total_sessions': 0,
            'successful_translations': 0,
            'latency_violations': 0,
            'frame_rate_violations': 0,
            'average_latency_ms': 0.0,
            'average_frame_rate': 0.0
        }
    
    async def start_real_time_session(
        self,
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None],
        result_callback: Callable[[PipelineResult], None],
        signing_speed: float = 1.0,
        target_sign_language: SignLanguage = SignLanguage.ASL
    ) -> None:
        """
        Start a real-time speech-to-avatar translation session.
        
        Args:
            session_id: Unique session identifier
            audio_stream: Real-time audio stream from client
            result_callback: Callback for pipeline results
            signing_speed: Avatar signing speed multiplier
            target_sign_language: Target sign language
        """
        self.pipeline_stats['total_sessions'] += 1
        logger.info(f"Starting real-time pipeline session: {session_id}")
        
        # Update translation service if needed
        if self.translation_service.sign_language != target_sign_language:
            self.translation_service = get_translation_service(target_sign_language)
        
        # Create pipeline callback that processes speech results
        async def process_speech_result(speech_result: Dict[str, Any]):
            """Process speech-to-text result through the complete pipeline."""
            pipeline_start_time = time.time()
            
            try:
                # Extract speech processing results
                transcript = speech_result.get('transcript', '').strip()
                speech_confidence = speech_result.get('confidence', 0.0)
                speech_processing_time = speech_result.get('processing_time_ms', 0)
                speaker_info = {
                    'speaker_label': speech_result.get('speaker_label', 'unknown'),
                    'active_speakers': speech_result.get('active_speakers', []),
                    'speaker_count': speech_result.get('speaker_count', 1),
                    'is_speaker_change': speech_result.get('is_speaker_change', False)
                }
                
                # Skip processing if no meaningful transcript
                if not transcript or len(transcript.strip()) < 2:
                    return
                
                # Step 1: Parallel emotion analysis and translation
                emotion_start_time = time.time()
                translation_start_time = time.time()
                
                # Run emotion analysis and translation in parallel
                emotion_task = asyncio.create_task(
                    self._analyze_emotion_async(transcript)
                )
                translation_task = asyncio.create_task(
                    self._translate_text_async(transcript, signing_speed)
                )
                
                # Wait for both to complete
                emotion_result, translation_result = await asyncio.gather(
                    emotion_task, translation_task
                )
                
                emotion_time = (time.time() - emotion_start_time) * 1000
                translation_time = (time.time() - translation_start_time) * 1000
                
                # Step 2: Generate avatar animation with emotion modulation
                avatar_start_time = time.time()
                
                pose_sequence = self.avatar_service.generate_pose_sequence(
                    text=translation_result.normalized_text,
                    emotion=emotion_result.get('dominant_emotion'),
                    emotion_intensity=emotion_result.get('intensity', 0.5),
                    signing_speed=signing_speed
                )
                
                facial_expressions = self.avatar_service.generate_facial_expressions(
                    text=translation_result.normalized_text,
                    emotion=emotion_result.get('dominant_emotion'),
                    emotion_intensity=emotion_result.get('intensity', 0.5)
                )
                
                avatar_time = (time.time() - avatar_start_time) * 1000
                
                # Step 3: Calculate performance metrics
                total_processing_time = (time.time() - pipeline_start_time) * 1000
                
                # Calculate frame rate
                if pose_sequence:
                    total_duration_ms = pose_sequence[-1].timestamp if pose_sequence else 1000
                    frame_rate = len(pose_sequence) / (total_duration_ms / 1000.0)
                else:
                    frame_rate = 0.0
                
                # Check real-time compliance
                is_real_time_compliant = (
                    total_processing_time <= self.max_latency_ms and
                    frame_rate >= self.min_frame_rate
                )
                
                # Update statistics
                self._update_pipeline_stats(total_processing_time, frame_rate, is_real_time_compliant)
                
                # Step 4: Create complete pipeline result
                pipeline_result = PipelineResult(
                    session_id=session_id,
                    timestamp=time.time(),
                    
                    # Speech results
                    transcript=transcript,
                    speech_confidence=speech_confidence,
                    speaker_info=speaker_info,
                    
                    # Translation results
                    normalized_text=translation_result.normalized_text,
                    translation_confidence=translation_result.confidence_score,
                    
                    # Emotion results
                    detected_emotion=emotion_result.get('dominant_emotion'),
                    emotion_intensity=emotion_result.get('intensity', 0.5),
                    
                    # Avatar results
                    pose_sequence=pose_sequence,
                    facial_expressions=facial_expressions,
                    animation_duration_ms=int(total_duration_ms) if pose_sequence else 0,
                    
                    # Performance metrics
                    total_processing_time_ms=int(total_processing_time),
                    speech_processing_time_ms=int(speech_processing_time),
                    translation_time_ms=int(translation_time),
                    avatar_rendering_time_ms=int(avatar_time),
                    
                    # Quality metrics
                    end_to_end_latency_ms=int(total_processing_time),
                    frame_rate=frame_rate,
                    is_real_time_compliant=is_real_time_compliant
                )
                
                # Send result to callback
                if asyncio.iscoroutinefunction(result_callback):
                    await result_callback(pipeline_result)
                else:
                    result_callback(pipeline_result)
                
                # Log performance
                logger.info(
                    f"Pipeline processed in {total_processing_time:.1f}ms "
                    f"(speech: {speech_processing_time:.1f}ms, "
                    f"translation: {translation_time:.1f}ms, "
                    f"avatar: {avatar_time:.1f}ms) "
                    f"Frame rate: {frame_rate:.1f} FPS, "
                    f"Compliant: {is_real_time_compliant}"
                )
                
            except Exception as e:
                logger.error(f"Pipeline processing error: {e}")
                # Send error result
                error_result = PipelineResult(
                    session_id=session_id,
                    timestamp=time.time(),
                    transcript="",
                    speech_confidence=0.0,
                    speaker_info={},
                    normalized_text="",
                    translation_confidence=0.0,
                    detected_emotion=None,
                    emotion_intensity=0.0,
                    pose_sequence=[],
                    facial_expressions=[],
                    animation_duration_ms=0,
                    total_processing_time_ms=int((time.time() - pipeline_start_time) * 1000),
                    speech_processing_time_ms=0,
                    translation_time_ms=0,
                    avatar_rendering_time_ms=0,
                    end_to_end_latency_ms=int((time.time() - pipeline_start_time) * 1000),
                    frame_rate=0.0,
                    is_real_time_compliant=False
                )
                
                if asyncio.iscoroutinefunction(result_callback):
                    await result_callback(error_result)
                else:
                    result_callback(error_result)
        
        # Start speech processing with our pipeline callback
        await self.speech_pipeline.process_real_time_audio(
            audio_stream, process_speech_result
        )
    
    async def _analyze_emotion_async(self, text: str) -> Dict[str, Any]:
        """Analyze emotion from text asynchronously."""
        try:
            # Run emotion analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            emotion_result = await loop.run_in_executor(
                None, self.emotion_service.analyze_text_emotion, text
            )
            return emotion_result
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {'dominant_emotion': None, 'intensity': 0.5}
    
    async def _translate_text_async(self, text: str, signing_speed: float):
        """Translate text to sign language asynchronously."""
        try:
            # Calculate emotion intensity based on signing speed
            # Faster signing often indicates higher intensity
            emotion_intensity = min(200, max(50, int(100 * signing_speed)))
            
            # Run translation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            translation_result = await loop.run_in_executor(
                None, self.translation_service.translate_text, text, emotion_intensity
            )
            return translation_result
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return minimal result on error
            from app.services.sign_language_translation import TranslationResult
            return TranslationResult(
                original_text=text,
                normalized_text=text.upper(),
                gestures=[],
                total_duration_ms=1000,
                confidence_score=0.0,
                processing_time_ms=0
            )
    
    def _update_pipeline_stats(self, processing_time_ms: float, frame_rate: float, is_compliant: bool):
        """Update pipeline performance statistics."""
        self.pipeline_stats['successful_translations'] += 1
        
        if processing_time_ms > self.max_latency_ms:
            self.pipeline_stats['latency_violations'] += 1
        
        if frame_rate < self.min_frame_rate:
            self.pipeline_stats['frame_rate_violations'] += 1
        
        # Update running averages
        total_translations = self.pipeline_stats['successful_translations']
        
        self.pipeline_stats['average_latency_ms'] = (
            (self.pipeline_stats['average_latency_ms'] * (total_translations - 1) + processing_time_ms) /
            total_translations
        )
        
        self.pipeline_stats['average_frame_rate'] = (
            (self.pipeline_stats['average_frame_rate'] * (total_translations - 1) + frame_rate) /
            total_translations
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.pipeline_stats.copy()
        
        # Calculate compliance rates
        if stats['successful_translations'] > 0:
            stats['latency_compliance_rate'] = (
                1.0 - (stats['latency_violations'] / stats['successful_translations'])
            )
            stats['frame_rate_compliance_rate'] = (
                1.0 - (stats['frame_rate_violations'] / stats['successful_translations'])
            )
        else:
            stats['latency_compliance_rate'] = 1.0
            stats['frame_rate_compliance_rate'] = 1.0
        
        # Add configuration
        stats['configuration'] = {
            'max_latency_ms': self.max_latency_ms,
            'min_frame_rate': self.min_frame_rate,
            'target_sign_language': self.translation_service.sign_language.value
        }
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self.pipeline_stats = {
            'total_sessions': 0,
            'successful_translations': 0,
            'latency_violations': 0,
            'frame_rate_violations': 0,
            'average_latency_ms': 0.0,
            'average_frame_rate': 0.0
        }


class StreamingPipelineManager:
    """Manages multiple concurrent streaming pipeline sessions."""
    
    def __init__(self):
        self.orchestrator = RealTimePipelineOrchestrator()
        self.active_sessions: Dict[str, asyncio.Task] = {}
        self.session_callbacks: Dict[str, Callable] = {}
    
    async def start_session(
        self,
        session_id: str,
        audio_stream: AsyncGenerator[bytes, None],
        result_callback: Callable[[PipelineResult], None],
        signing_speed: float = 1.0,
        target_sign_language: SignLanguage = SignLanguage.ASL
    ) -> None:
        """Start a new streaming pipeline session."""
        
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} already active, stopping previous session")
            await self.stop_session(session_id)
        
        # Store callback for session
        self.session_callbacks[session_id] = result_callback
        
        # Create session task
        session_task = asyncio.create_task(
            self.orchestrator.start_real_time_session(
                session_id=session_id,
                audio_stream=audio_stream,
                result_callback=result_callback,
                signing_speed=signing_speed,
                target_sign_language=target_sign_language
            )
        )
        
        self.active_sessions[session_id] = session_task
        logger.info(f"Started streaming pipeline session: {session_id}")
    
    async def stop_session(self, session_id: str) -> None:
        """Stop a streaming pipeline session."""
        
        if session_id in self.active_sessions:
            task = self.active_sessions[session_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.active_sessions[session_id]
            
            if session_id in self.session_callbacks:
                del self.session_callbacks[session_id]
            
            logger.info(f"Stopped streaming pipeline session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.orchestrator.get_pipeline_stats()
        stats['active_sessions'] = self.get_session_count()
        stats['session_ids'] = self.get_active_sessions()
        return stats
    
    async def cleanup_all_sessions(self):
        """Stop all active sessions."""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.stop_session(session_id)


# Global pipeline manager instance
pipeline_manager = StreamingPipelineManager()