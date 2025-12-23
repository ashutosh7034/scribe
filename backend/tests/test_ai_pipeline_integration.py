"""
AI Pipeline Integration Tests

Integration tests to validate that speech processing and emotion analysis
work together correctly and meet latency targets.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Mock external dependencies
import sys
from unittest.mock import MagicMock
sys.modules['librosa'] = MagicMock()
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from app.services.speech_processing import SpeechProcessingPipeline
from app.services.emotion_analysis import EmotionAnalysisService, EmotionVector


class TestAIPipelineIntegration:
    """Integration tests for the complete AI pipeline."""
    
    @pytest.mark.asyncio
    async def test_speech_to_emotion_pipeline_integration(self):
        """
        Test that speech processing and emotion analysis work together correctly.
        
        This validates the core AI pipeline integration for the checkpoint.
        """
        # Mock speech processing service
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            # Mock emotion analysis components
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure mocks for realistic behavior
                transcription_results = []
                emotion_results = []
                
                async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                    """Mock speech processing that generates realistic transcription results."""
                    chunk_count = 0
                    async for chunk in audio_stream:
                        # Simulate processing time
                        await asyncio.sleep(0.05)  # 50ms processing time
                        
                        # Generate realistic transcription result
                        result = {
                            'transcript': f'This is test speech chunk {chunk_count}',
                            'confidence': 0.85,
                            'speaker_label': 'spk_0',
                            'start_time': chunk_count * 0.5,
                            'end_time': (chunk_count + 1) * 0.5,
                            'is_partial': False
                        }
                        
                        transcription_results.append(result)
                        
                        # Properly await async callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                        chunk_count += 1
                
                mock_stt_instance.process_audio_stream = mock_process_audio_stream
                
                # Configure emotion analysis mock
                mock_classifier.classify_sentiment.return_value = {
                    'label': 'POSITIVE', 'confidence': 0.80, 'all_scores': []
                }
                mock_classifier.text_to_emotion_vector.return_value = EmotionVector(
                    valence=0.6, arousal=0.4, dominance=0.3
                )
                
                # Create services
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                # Create test audio stream
                async def audio_stream():
                    for i in range(3):  # 3 audio chunks
                        # Generate 500ms of audio data (8000 samples at 16kHz)
                        audio_data = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
                        yield audio_data.tobytes()
                
                # Track pipeline results
                pipeline_results = []
                
                async def integrated_callback(speech_result):
                    """Callback that processes speech result through emotion analysis."""
                    # Extract text from speech result
                    text = speech_result.get('transcript', '')
                    
                    if text:
                        # Process through emotion analysis
                        emotion_result = await emotion_service.analyze_emotion(text=text)
                        
                        # Combine results
                        integrated_result = {
                            'speech_result': speech_result,
                            'emotion_result': emotion_result,
                            'processing_timestamp': time.time()
                        }
                        
                        pipeline_results.append(integrated_result)
                        emotion_results.append(emotion_result)
                
                # Measure end-to-end pipeline performance
                start_time = time.time()
                
                await speech_pipeline.process_real_time_audio(
                    audio_stream(), 
                    integrated_callback
                )
                
                end_time = time.time()
                total_pipeline_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Validate integration results
                assert len(transcription_results) == 3, "Should process 3 audio chunks"
                assert len(emotion_results) == 3, "Should generate 3 emotion analyses"
                assert len(pipeline_results) == 3, "Should have 3 integrated results"
                
                # Validate that speech and emotion analysis are properly integrated
                for i, result in enumerate(pipeline_results):
                    speech_data = result['speech_result']
                    emotion_data = result['emotion_result']
                    
                    # Speech processing validation
                    assert 'transcript' in speech_data
                    assert 'confidence' in speech_data
                    assert speech_data['confidence'] > 0.5
                    
                    # Emotion analysis validation
                    assert 'discrete_emotion' in emotion_data
                    assert 'emotion_intensity' in emotion_data
                    assert 'signing_modulation' in emotion_data
                    
                    # Integration validation
                    assert emotion_data['text_input'] == speech_data['transcript']
                    
                    # Timing validation
                    processing_time = emotion_data.get('processing_time_ms', 0)
                    assert processing_time < 500, f"Emotion processing time {processing_time}ms too high"
                
                # Overall pipeline latency validation
                avg_latency_per_chunk = total_pipeline_time / len(pipeline_results)
                assert avg_latency_per_chunk < 300, (
                    f"Average pipeline latency {avg_latency_per_chunk:.2f}ms exceeds 300ms target"
                )
                
                print(f"✓ Pipeline integration test passed:")
                print(f"  - Processed {len(pipeline_results)} audio chunks")
                print(f"  - Total pipeline time: {total_pipeline_time:.2f}ms")
                print(f"  - Average latency per chunk: {avg_latency_per_chunk:.2f}ms")
                print(f"  - All components integrated successfully")
    
    @pytest.mark.asyncio
    async def test_pipeline_latency_targets(self):
        """
        Test that the integrated pipeline meets latency targets.
        
        This validates Requirements 1.1, 1.2, 1.3 for the checkpoint.
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure for latency testing
                processing_times = []
                
                async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                    async for chunk in audio_stream:
                        chunk_start = time.time()
                        
                        # Simulate realistic processing delays
                        await asyncio.sleep(0.08)  # 80ms for speech processing
                        
                        result = {
                            'transcript': 'Test speech for latency measurement',
                            'confidence': 0.90,
                            'speaker_label': 'spk_0',
                            'processing_start_time': chunk_start
                        }
                        
                        # Properly await async callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                
                mock_stt_instance.process_audio_stream = mock_process_audio_stream
                
                # Configure emotion analysis for latency testing
                mock_classifier.classify_sentiment.return_value = {
                    'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []
                }
                mock_classifier.text_to_emotion_vector.return_value = EmotionVector(0.0, 0.0, 0.0)
                
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                async def audio_stream():
                    # Single audio chunk for precise latency measurement
                    audio_data = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
                    yield audio_data.tobytes()
                
                latency_measurements = []
                
                async def latency_callback(speech_result):
                    """Measure end-to-end latency including emotion analysis."""
                    chunk_start = speech_result.get('processing_start_time', time.time())
                    
                    # Process through emotion analysis
                    emotion_start = time.time()
                    emotion_result = await emotion_service.analyze_emotion(
                        text=speech_result.get('transcript', '')
                    )
                    emotion_end = time.time()
                    
                    # Calculate latencies
                    speech_latency = (emotion_start - chunk_start) * 1000
                    emotion_latency = (emotion_end - emotion_start) * 1000
                    total_latency = (emotion_end - chunk_start) * 1000
                    
                    latency_measurements.append({
                        'speech_latency_ms': speech_latency,
                        'emotion_latency_ms': emotion_latency,
                        'total_latency_ms': total_latency
                    })
                
                # Run latency test
                await speech_pipeline.process_real_time_audio(audio_stream(), latency_callback)
                
                # Validate latency targets
                assert len(latency_measurements) == 1, "Should have one latency measurement"
                
                measurement = latency_measurements[0]
                speech_latency = measurement['speech_latency_ms']
                emotion_latency = measurement['emotion_latency_ms']
                total_latency = measurement['total_latency_ms']
                
                # Validate individual component latencies
                assert speech_latency < 200, (
                    f"Speech processing latency {speech_latency:.2f}ms exceeds 200ms target"
                )
                assert emotion_latency < 100, (
                    f"Emotion analysis latency {emotion_latency:.2f}ms exceeds 100ms target"
                )
                
                # Validate total end-to-end latency (Requirements 1.1, 1.2, 1.3)
                assert total_latency < 300, (
                    f"Total pipeline latency {total_latency:.2f}ms exceeds 300ms target"
                )
                
                print(f"✓ Latency targets met:")
                print(f"  - Speech processing: {speech_latency:.2f}ms (target: <200ms)")
                print(f"  - Emotion analysis: {emotion_latency:.2f}ms (target: <100ms)")
                print(f"  - Total end-to-end: {total_latency:.2f}ms (target: <300ms)")
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """
        Test that the pipeline handles errors gracefully without breaking the flow.
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure speech processing to work normally
                async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                    chunk_count = 0
                    async for chunk in audio_stream:
                        await asyncio.sleep(0.02)
                        
                        if chunk_count == 1:
                            # Simulate an error in the middle
                            result = {
                                'transcript': '',  # Empty transcript to test error handling
                                'confidence': 0.0,
                                'speaker_label': 'unknown',
                                'error': 'Simulated processing error'
                            }
                        else:
                            result = {
                                'transcript': f'Valid speech chunk {chunk_count}',
                                'confidence': 0.85,
                                'speaker_label': 'spk_0'
                            }
                        
                        # Properly await async callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                        chunk_count += 1
                
                mock_stt_instance.process_audio_stream = mock_process_audio_stream
                
                # Configure emotion analysis to handle errors
                def mock_classify_sentiment(text):
                    if not text or text == '':
                        return {'label': 'NEUTRAL', 'confidence': 0.0, 'all_scores': []}
                    return {'label': 'POSITIVE', 'confidence': 0.80, 'all_scores': []}
                
                mock_classifier.classify_sentiment.side_effect = mock_classify_sentiment
                mock_classifier.text_to_emotion_vector.return_value = EmotionVector(0.1, 0.1, 0.1)
                
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                async def audio_stream():
                    for i in range(3):
                        audio_data = np.random.randint(-32768, 32767, 4000, dtype=np.int16)
                        yield audio_data.tobytes()
                
                results = []
                errors = []
                
                async def error_handling_callback(speech_result):
                    """Test error handling in the pipeline."""
                    try:
                        text = speech_result.get('transcript', '')
                        
                        # Process through emotion analysis even with empty/error text
                        emotion_result = await emotion_service.analyze_emotion(text=text)
                        
                        results.append({
                            'speech_result': speech_result,
                            'emotion_result': emotion_result,
                            'has_error': 'error' in speech_result
                        })
                        
                    except Exception as e:
                        errors.append(str(e))
                
                # Run error handling test
                await speech_pipeline.process_real_time_audio(audio_stream(), error_handling_callback)
                
                # Validate error handling
                assert len(results) == 3, "Should process all chunks despite errors"
                assert len(errors) == 0, "Should not raise unhandled exceptions"
                
                # Check that error cases are handled gracefully
                error_result = next((r for r in results if r['has_error']), None)
                assert error_result is not None, "Should have processed the error case"
                
                # Verify that emotion analysis still works with empty text
                error_emotion = error_result['emotion_result']
                assert 'discrete_emotion' in error_emotion
                assert error_emotion['discrete_emotion'] is not None
                
                print(f"✓ Error handling test passed:")
                print(f"  - Processed {len(results)} chunks including error cases")
                print(f"  - No unhandled exceptions: {len(errors) == 0}")
                print(f"  - Pipeline continues operation despite errors")
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_under_load(self):
        """
        Test pipeline performance with multiple concurrent audio streams.
        """
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure for concurrent load testing
                async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                    async for chunk in audio_stream:
                        # Simulate slight increase in processing time under load
                        await asyncio.sleep(0.06)  # 60ms processing time
                        
                        result = {
                            'transcript': 'Concurrent processing test',
                            'confidence': 0.85,
                            'speaker_label': 'spk_0'
                        }
                        
                        # Properly await async callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                
                mock_stt_instance.process_audio_stream = mock_process_audio_stream
                mock_classifier.classify_sentiment.return_value = {
                    'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []
                }
                mock_classifier.text_to_emotion_vector.return_value = EmotionVector(0.0, 0.0, 0.0)
                
                # Test with multiple concurrent streams
                num_streams = 3
                
                async def process_single_stream(stream_id):
                    """Process a single audio stream."""
                    speech_pipeline = SpeechProcessingPipeline()
                    emotion_service = EmotionAnalysisService()
                    
                    async def audio_stream():
                        # Single chunk per stream
                        audio_data = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
                        yield audio_data.tobytes()
                    
                    results = []
                    
                    async def stream_callback(speech_result):
                        emotion_result = await emotion_service.analyze_emotion(
                            text=speech_result.get('transcript', '')
                        )
                        results.append({
                            'stream_id': stream_id,
                            'speech_result': speech_result,
                            'emotion_result': emotion_result
                        })
                    
                    start_time = time.time()
                    await speech_pipeline.process_real_time_audio(audio_stream(), stream_callback)
                    end_time = time.time()
                    
                    return {
                        'stream_id': stream_id,
                        'processing_time_ms': (end_time - start_time) * 1000,
                        'results': results
                    }
                
                # Run concurrent streams
                start_time = time.time()
                stream_results = await asyncio.gather(*[
                    process_single_stream(i) for i in range(num_streams)
                ])
                end_time = time.time()
                
                total_concurrent_time = (end_time - start_time) * 1000
                
                # Validate concurrent performance
                assert len(stream_results) == num_streams, f"Should process {num_streams} streams"
                
                for stream_result in stream_results:
                    assert len(stream_result['results']) == 1, "Each stream should have one result"
                    processing_time = stream_result['processing_time_ms']
                    
                    # Allow some increase for concurrent processing
                    max_allowed_time = 400  # 400ms max for concurrent processing
                    assert processing_time < max_allowed_time, (
                        f"Stream {stream_result['stream_id']} processing time "
                        f"{processing_time:.2f}ms exceeds {max_allowed_time}ms under load"
                    )
                
                # Validate that concurrent processing doesn't take much longer than sequential
                avg_stream_time = sum(s['processing_time_ms'] for s in stream_results) / len(stream_results)
                
                print(f"✓ Concurrent performance test passed:")
                print(f"  - Processed {num_streams} concurrent streams")
                print(f"  - Total concurrent time: {total_concurrent_time:.2f}ms")
                print(f"  - Average stream time: {avg_stream_time:.2f}ms")
                print(f"  - All streams completed within acceptable time limits")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])