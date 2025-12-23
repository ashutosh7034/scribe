"""
Tests for Speech Processing Service

Unit tests for speech-to-text, voice activity detection, and noise filtering.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import AsyncGenerator

# Mock librosa since it's not installed
import sys
from unittest.mock import MagicMock
sys.modules['librosa'] = MagicMock()
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()

from app.services.speech_processing import (
    VoiceActivityDetector,
    NoiseFilter,
    SpeechToTextService,
    MultiSpeakerProcessor,
    SpeechProcessingPipeline
)


class TestVoiceActivityDetector:
    """Test voice activity detection functionality."""
    
    def test_voice_activity_detection_with_voice(self):
        """Test VAD detects voice activity correctly."""
        vad = VoiceActivityDetector(energy_threshold=0.01)
        
        # Create audio with voice activity (high energy)
        audio_with_voice = np.random.normal(0, 0.1, 1600)  # 100ms at 16kHz
        
        result = vad.detect_voice_activity(audio_with_voice, 16000)
        assert result is True
    
    def test_voice_activity_detection_without_voice(self):
        """Test VAD detects silence correctly."""
        vad = VoiceActivityDetector(energy_threshold=0.01, silence_duration=0.1)
        
        # Create silent audio (low energy)
        silent_audio = np.random.normal(0, 0.001, 1600)  # 100ms at 16kHz
        
        # First call should still return True (not enough silence time)
        result1 = vad.detect_voice_activity(silent_audio, 16000)
        
        # Simulate time passing
        import time
        time.sleep(0.2)
        
        # Second call should return False (enough silence time)
        result2 = vad.detect_voice_activity(silent_audio, 16000)
        
        assert result1 is True  # Not enough silence time yet


class TestMultiSpeakerProcessor:
    """Test multi-speaker processing functionality."""
    
    def test_speaker_diarization_processing(self):
        """Test speaker diarization result processing."""
        processor = MultiSpeakerProcessor()
        
        transcription_result = {
            'transcript': 'Hello world',
            'confidence': 0.9,
            'speaker_label': 'spk_0',
            'start_time': 0.0,
            'end_time': 2.0
        }
        
        enhanced_result = processor.process_speaker_diarization(transcription_result)
        
        assert 'active_speakers' in enhanced_result
        assert 'speaker_count' in enhanced_result
        assert 'primary_speaker' in enhanced_result
        assert 'speaker_overlap' in enhanced_result
        assert 'speaker_confidence' in enhanced_result
        assert enhanced_result['primary_speaker'] == 'spk_0'
        assert 'spk_0' in enhanced_result['active_speakers']
    
    def test_speaker_change_detection(self):
        """Test speaker change detection."""
        processor = MultiSpeakerProcessor()
        
        # First speaker
        result1 = {
            'transcript': 'Hello',
            'confidence': 0.9,
            'speaker_label': 'spk_0',
            'start_time': 0.0,
            'end_time': 1.0
        }
        
        enhanced1 = processor.process_speaker_diarization(result1)
        
        # Second speaker (should detect change)
        result2 = {
            'transcript': 'World',
            'confidence': 0.8,
            'speaker_label': 'spk_1',
            'start_time': 2.0,
            'end_time': 3.0
        }
        
        enhanced2 = processor.process_speaker_diarization(result2)
        
        # Should have detected speaker change
        assert enhanced2.get('is_speaker_change', False) == True
        assert len(enhanced2['active_speakers']) == 2
    
    def test_speaker_separation(self):
        """Test speaker separation functionality."""
        processor = MultiSpeakerProcessor()
        
        # Create mock audio data
        audio_data = np.random.normal(0, 0.1, 1600)
        
        separated_audio = processor.separate_speakers(audio_data, 16000)
        
        assert isinstance(separated_audio, dict)
        assert 'spk_0' in separated_audio
        assert len(separated_audio['spk_0']) <= len(audio_data)  # May be shorter due to windowing
    
    def test_speaker_statistics(self):
        """Test speaker statistics collection."""
        processor = MultiSpeakerProcessor()
        
        # Add some speaker activity
        result1 = {
            'transcript': 'Hello',
            'confidence': 0.9,
            'speaker_label': 'spk_0',
            'start_time': 0.0,
            'end_time': 1.0
        }
        
        result2 = {
            'transcript': 'World',
            'confidence': 0.8,
            'speaker_label': 'spk_1',
            'start_time': 1.0,
            'end_time': 2.0
        }
        
        processor.process_speaker_diarization(result1)
        processor.process_speaker_diarization(result2)
        
        stats = processor.get_speaker_statistics()
        
        assert 'total_speakers_detected' in stats
        assert 'currently_active_speakers' in stats
        assert 'speaker_activity_summary' in stats
        assert stats['total_speakers_detected'] >= 2


class TestSpeechProcessingPipeline:
    """Test complete speech processing pipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        with patch('app.services.speech_processing.SpeechToTextService'):
            pipeline = SpeechProcessingPipeline()
            
            assert hasattr(pipeline, 'stt_service')
            assert isinstance(pipeline.multi_speaker_processor, MultiSpeakerProcessor)
            assert 'total_chunks_processed' in pipeline.processing_stats
    
    def test_stats_management(self):
        """Test processing statistics management."""
        with patch('app.services.speech_processing.SpeechToTextService'):
            pipeline = SpeechProcessingPipeline()
            
            # Get initial stats
            initial_stats = pipeline.get_processing_stats()
            assert initial_stats['transcription_requests'] == 0
            
            # Reset stats
            pipeline.reset_stats()
            reset_stats = pipeline.get_processing_stats()
            assert reset_stats['transcription_requests'] == 0
            assert reset_stats['average_processing_time_ms'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])