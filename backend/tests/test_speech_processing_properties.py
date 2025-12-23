"""
Property-Based Tests for Speech Processing

Property tests for speech processing latency and performance characteristics.
"""

import pytest
import asyncio
import time
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, patch, AsyncMock
from typing import AsyncGenerator

# Mock external dependencies
import sys
from unittest.mock import MagicMock
sys.modules['librosa'] = MagicMock()
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()

from app.services.speech_processing import SpeechProcessingPipeline


# Test data strategies
@st.composite
def audio_chunk_strategy(draw):
    """Generate realistic audio chunks for testing."""
    # Audio chunk size (10ms to 100ms at 16kHz) - much smaller for testing
    chunk_size = draw(st.integers(min_value=160, max_value=1600))
    
    # Generate audio data as int16 PCM using binary data instead of lists
    # This is more efficient and avoids Hypothesis size limits
    audio_bytes = draw(st.binary(min_size=chunk_size*2, max_size=chunk_size*2))
    
    return audio_bytes


@st.composite
def audio_stream_strategy(draw):
    """Generate audio streams with multiple chunks."""
    num_chunks = draw(st.integers(min_value=1, max_value=5))
    chunks = [draw(audio_chunk_strategy()) for _ in range(num_chunks)]
    return chunks


class TestSpeechProcessingLatencyProperties:
    """Property-based tests for speech processing latency."""
    
    @given(audio_chunks=audio_stream_strategy())
    @settings(
        max_examples=100,
        deadline=5000,  # 5 second timeout per test
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_end_to_end_latency_performance(self, audio_chunks):
        """
        Feature: real-time-sign-language-translation, Property 1: End-to-End Latency Performance
        
        For any spoken input, the total time from audio capture to avatar display 
        should be less than 300ms under normal operating conditions.
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            # Setup mock STT service
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            # Mock the process_audio_stream method to simulate realistic processing
            async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                async for chunk in audio_stream:
                    # Simulate processing time (should be much less than 300ms)
                    await asyncio.sleep(0.05)  # 50ms processing time
                    
                    # Call callback with mock result
                    result = {
                        'transcript': 'test transcript',
                        'confidence': 0.9,
                        'speaker_label': 'spk_0',
                        'start_time': 0.0,
                        'end_time': 1.0
                    }
                    callback(result)
            
            mock_stt_instance.process_audio_stream = mock_process_audio_stream
            
            # Create pipeline
            pipeline = SpeechProcessingPipeline()
            
            # Create audio stream generator
            async def audio_stream():
                for chunk in audio_chunks:
                    yield chunk
            
            # Track processing times
            processing_times = []
            
            async def result_callback(result):
                processing_time = result.get('processing_time_ms', 0)
                processing_times.append(processing_time)
            
            # Measure end-to-end latency
            start_time = time.time()
            
            await pipeline.process_real_time_audio(audio_stream(), result_callback)
            
            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            
            # Property: Total latency should be less than 300ms per audio chunk
            # For multiple chunks, we allow proportional increase but with efficiency
            max_allowed_latency = 300 * len(audio_chunks)
            
            assert total_latency_ms < max_allowed_latency, (
                f"End-to-end latency {total_latency_ms:.2f}ms exceeds "
                f"maximum allowed {max_allowed_latency}ms for {len(audio_chunks)} chunks"
            )
            
            # Additional property: Each individual processing step should be reasonable
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
                assert avg_processing_time < 200, (
                    f"Average processing time {avg_processing_time:.2f}ms exceeds 200ms threshold"
                )
    
    @given(
        audio_chunk=audio_chunk_strategy(),
        noise_level=st.floats(min_value=0.0, max_value=60.0)
    )
    @settings(
        max_examples=50,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_latency_consistency_under_noise(self, audio_chunk, noise_level):
        """
        Test that latency remains consistent even with varying noise levels.
        
        This supports Property 1 by ensuring noise doesn't significantly impact processing time.
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            # Setup mock
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                async for chunk in audio_stream:
                    # Simulate slight increase in processing time with noise
                    noise_penalty = min(noise_level / 1000, 0.02)  # Max 20ms penalty
                    await asyncio.sleep(0.05 + noise_penalty)
                    
                    result = {
                        'transcript': 'test transcript',
                        'confidence': max(0.5, 0.9 - noise_level / 100),
                        'speaker_label': 'spk_0'
                    }
                    callback(result)
            
            mock_stt_instance.process_audio_stream = mock_process_audio_stream
            
            pipeline = SpeechProcessingPipeline()
            
            async def audio_stream():
                yield audio_chunk
            
            processing_times = []
            
            async def result_callback(result):
                processing_times.append(result.get('processing_time_ms', 0))
            
            # Measure latency
            start_time = time.time()
            await pipeline.process_real_time_audio(audio_stream(), result_callback)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            
            # Property: Even with noise, latency should remain under 300ms
            assert latency_ms < 300, (
                f"Latency {latency_ms:.2f}ms with noise level {noise_level}dB exceeds 300ms"
            )
    
    @given(
        num_concurrent_streams=st.integers(min_value=1, max_value=5),
        audio_chunk=audio_chunk_strategy()
    )
    @settings(
        max_examples=30,
        deadline=10000,  # Longer timeout for concurrent tests
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_latency_under_concurrent_load(self, num_concurrent_streams, audio_chunk):
        """
        Test that latency remains acceptable under concurrent processing load.
        
        This supports Property 1 by ensuring the system can handle multiple streams
        while maintaining latency requirements.
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            # Setup mock
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                async for chunk in audio_stream:
                    # Simulate slight increase in processing time under load
                    load_penalty = min(num_concurrent_streams * 0.01, 0.05)  # Max 50ms penalty
                    await asyncio.sleep(0.05 + load_penalty)
                    
                    result = {
                        'transcript': f'stream transcript',
                        'confidence': 0.85,
                        'speaker_label': 'spk_0'
                    }
                    callback(result)
            
            mock_stt_instance.process_audio_stream = mock_process_audio_stream
            
            # Create multiple concurrent processing tasks
            async def process_single_stream(stream_id):
                pipeline = SpeechProcessingPipeline()
                
                async def audio_stream():
                    yield audio_chunk
                
                latencies = []
                
                async def result_callback(result):
                    latencies.append(result.get('processing_time_ms', 0))
                
                start_time = time.time()
                await pipeline.process_real_time_audio(audio_stream(), result_callback)
                end_time = time.time()
                
                return (end_time - start_time) * 1000
            
            # Run concurrent streams
            tasks = [
                process_single_stream(i) 
                for i in range(num_concurrent_streams)
            ]
            
            latencies = await asyncio.gather(*tasks)
            
            # Property: All streams should complete within acceptable time
            max_latency = max(latencies)
            
            # Allow some increase for concurrent processing, but not excessive
            max_allowed = 300 + (num_concurrent_streams - 1) * 50  # 50ms penalty per additional stream
            
            assert max_latency < max_allowed, (
                f"Maximum latency {max_latency:.2f}ms under {num_concurrent_streams} "
                f"concurrent streams exceeds {max_allowed}ms"
            )
            
            # Property: Average latency should still be reasonable
            avg_latency = sum(latencies) / len(latencies)
            assert avg_latency < 400, (
                f"Average latency {avg_latency:.2f}ms under concurrent load exceeds 400ms"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])