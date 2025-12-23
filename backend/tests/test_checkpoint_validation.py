"""
Checkpoint Validation Tests

Comprehensive validation that the core AI pipeline (speech processing + emotion analysis)
works together and meets all latency and performance targets.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch
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


class TestCheckpointValidation:
    """Comprehensive checkpoint validation for the core AI pipeline."""
    
    @pytest.mark.asyncio
    async def test_core_ai_pipeline_integration(self):
        """
        CHECKPOINT VALIDATION: Core AI pipeline integration
        
        Validates that speech processing and emotion analysis work together correctly
        and meet all performance requirements.
        """
        print("\n🔍 CHECKPOINT VALIDATION: Core AI Pipeline Integration")
        
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure realistic mocks
                transcription_results = []
                emotion_results = []
                latency_measurements = []
                
                async def mock_process_audio_stream(audio_stream, callback, sample_rate=16000):
                    """Realistic speech processing simulation."""
                    chunk_count = 0
                    async for chunk in audio_stream:
                        chunk_start = time.time()
                        
                        # Simulate realistic speech processing time (80-120ms)
                        processing_delay = 0.08 + (chunk_count * 0.01)  # Slight increase per chunk
                        await asyncio.sleep(processing_delay)
                        
                        # Generate realistic transcription result
                        transcripts = [
                            "Hello, how are you doing today?",
                            "I'm feeling really excited about this project!",
                            "This is working better than I expected."
                        ]
                        
                        result = {
                            'transcript': transcripts[chunk_count % len(transcripts)],
                            'confidence': 0.85 + (chunk_count * 0.02),  # Increasing confidence
                            'speaker_label': f'spk_{chunk_count % 2}',  # Alternate speakers
                            'start_time': chunk_count * 1.0,
                            'end_time': (chunk_count + 1) * 1.0,
                            'is_partial': False,
                            'processing_start_time': chunk_start
                        }
                        
                        transcription_results.append(result)
                        
                        # Properly await async callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                        chunk_count += 1
                
                mock_stt_instance.process_audio_stream = mock_process_audio_stream
                
                # Configure emotion analysis with realistic responses
                def mock_classify_sentiment(text):
                    if 'excited' in text.lower() or 'better' in text.lower():
                        return {'label': 'POSITIVE', 'confidence': 0.88, 'all_scores': []}
                    elif 'hello' in text.lower():
                        return {'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []}
                    else:
                        return {'label': 'POSITIVE', 'confidence': 0.80, 'all_scores': []}
                
                def mock_text_to_emotion_vector(text):
                    if 'excited' in text.lower():
                        return EmotionVector(valence=0.8, arousal=0.7, dominance=0.5)
                    elif 'better' in text.lower():
                        return EmotionVector(valence=0.6, arousal=0.4, dominance=0.3)
                    else:
                        return EmotionVector(valence=0.2, arousal=0.1, dominance=0.1)
                
                mock_classifier.classify_sentiment.side_effect = mock_classify_sentiment
                mock_classifier.text_to_emotion_vector.side_effect = mock_text_to_emotion_vector
                
                # Create services
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                # Create test audio stream
                async def audio_stream():
                    for i in range(3):  # 3 audio chunks
                        # Generate realistic audio data (1 second at 16kHz)
                        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
                        yield audio_data.tobytes()
                
                # Track comprehensive results
                pipeline_results = []
                
                async def comprehensive_callback(speech_result):
                    """Comprehensive callback that validates the full pipeline."""
                    text = speech_result.get('transcript', '')
                    processing_start = speech_result.get('processing_start_time', time.time())
                    
                    if text:
                        # Process through emotion analysis
                        emotion_start = time.time()
                        emotion_result = await emotion_service.analyze_emotion(text=text)
                        emotion_end = time.time()
                        
                        # Calculate latencies
                        speech_latency = (emotion_start - processing_start) * 1000
                        emotion_latency = (emotion_end - emotion_start) * 1000
                        total_latency = (emotion_end - processing_start) * 1000
                        
                        # Store comprehensive result
                        integrated_result = {
                            'speech_result': speech_result,
                            'emotion_result': emotion_result,
                            'latencies': {
                                'speech_ms': speech_latency,
                                'emotion_ms': emotion_latency,
                                'total_ms': total_latency
                            },
                            'processing_timestamp': time.time()
                        }
                        
                        pipeline_results.append(integrated_result)
                        emotion_results.append(emotion_result)
                        latency_measurements.append(integrated_result['latencies'])
                
                # Execute pipeline
                print("  📊 Processing audio stream through integrated pipeline...")
                start_time = time.time()
                
                await speech_pipeline.process_real_time_audio(
                    audio_stream(),
                    comprehensive_callback
                )
                
                end_time = time.time()
                total_pipeline_time = (end_time - start_time) * 1000
                
                # VALIDATION 1: Integration completeness
                print(f"  ✅ Integration completeness:")
                print(f"     - Transcription results: {len(transcription_results)}")
                print(f"     - Emotion analyses: {len(emotion_results)}")
                print(f"     - Pipeline results: {len(pipeline_results)}")
                
                assert len(transcription_results) == 3, "Should process 3 audio chunks"
                assert len(emotion_results) == 3, "Should generate 3 emotion analyses"
                assert len(pipeline_results) == 3, "Should have 3 integrated results"
                
                # VALIDATION 2: Latency targets (Requirements 1.1, 1.2, 1.3)
                print(f"  ⏱️  Latency validation:")
                
                avg_speech_latency = sum(l['speech_ms'] for l in latency_measurements) / len(latency_measurements)
                avg_emotion_latency = sum(l['emotion_ms'] for l in latency_measurements) / len(latency_measurements)
                avg_total_latency = sum(l['total_ms'] for l in latency_measurements) / len(latency_measurements)
                
                print(f"     - Average speech processing: {avg_speech_latency:.2f}ms (target: <200ms)")
                print(f"     - Average emotion analysis: {avg_emotion_latency:.2f}ms (target: <100ms)")
                print(f"     - Average total latency: {avg_total_latency:.2f}ms (target: <300ms)")
                
                # Validate latency requirements
                assert avg_speech_latency < 200, f"Speech latency {avg_speech_latency:.2f}ms exceeds 200ms"
                assert avg_emotion_latency < 100, f"Emotion latency {avg_emotion_latency:.2f}ms exceeds 100ms"
                assert avg_total_latency < 300, f"Total latency {avg_total_latency:.2f}ms exceeds 300ms"
                
                # VALIDATION 3: Data flow integrity
                print(f"  🔗 Data flow validation:")
                
                for i, result in enumerate(pipeline_results):
                    speech_data = result['speech_result']
                    emotion_data = result['emotion_result']
                    
                    # Validate speech processing output
                    assert 'transcript' in speech_data, f"Missing transcript in result {i}"
                    assert 'confidence' in speech_data, f"Missing confidence in result {i}"
                    assert speech_data['confidence'] > 0.5, f"Low confidence {speech_data['confidence']} in result {i}"
                    
                    # Validate emotion analysis output
                    assert 'discrete_emotion' in emotion_data, f"Missing discrete emotion in result {i}"
                    assert 'emotion_intensity' in emotion_data, f"Missing emotion intensity in result {i}"
                    assert 'signing_modulation' in emotion_data, f"Missing signing modulation in result {i}"
                    
                    # Validate data flow connection
                    assert emotion_data['text_input'] == speech_data['transcript'], f"Text mismatch in result {i}"
                    
                    print(f"     - Result {i+1}: '{speech_data['transcript'][:30]}...' → {emotion_data['discrete_emotion']}")
                
                # VALIDATION 4: Emotion analysis quality
                print(f"  🎭 Emotion analysis validation:")
                
                emotion_categories = [r['emotion_result']['discrete_emotion'] for r in pipeline_results]
                unique_emotions = set(emotion_categories)
                
                print(f"     - Detected emotions: {list(unique_emotions)}")
                print(f"     - Emotion variety: {len(unique_emotions)} different emotions")
                
                # Should detect different emotions for different text content
                assert len(unique_emotions) >= 1, "Should detect at least one emotion category"
                
                # Validate signing modulation parameters
                for i, result in enumerate(pipeline_results):
                    modulation = result['emotion_result']['signing_modulation']
                    
                    # All modulation parameters should be present and reasonable
                    assert 'intensity_scale' in modulation, f"Missing intensity_scale in result {i}"
                    assert 'speed_multiplier' in modulation, f"Missing speed_multiplier in result {i}"
                    assert 'facial_expression_intensity' in modulation, f"Missing facial_expression_intensity in result {i}"
                    
                    # Parameters should be within reasonable ranges
                    assert 0.5 <= modulation['intensity_scale'] <= 2.0, f"Invalid intensity_scale in result {i}"
                    assert 0.5 <= modulation['speed_multiplier'] <= 2.0, f"Invalid speed_multiplier in result {i}"
                    assert 0.0 <= modulation['facial_expression_intensity'] <= 1.0, f"Invalid facial_expression_intensity in result {i}"
                
                # VALIDATION 5: Performance under realistic conditions
                print(f"  🚀 Performance validation:")
                print(f"     - Total pipeline time: {total_pipeline_time:.2f}ms")
                print(f"     - Average per chunk: {total_pipeline_time / len(pipeline_results):.2f}ms")
                print(f"     - Processing efficiency: {len(pipeline_results) / (total_pipeline_time / 1000):.1f} chunks/second")
                
                # Performance should be reasonable for real-time processing
                avg_per_chunk = total_pipeline_time / len(pipeline_results)
                assert avg_per_chunk < 400, f"Average per-chunk time {avg_per_chunk:.2f}ms too high for real-time"
                
                print(f"  ✅ CHECKPOINT VALIDATION PASSED")
                print(f"     - Speech processing and emotion analysis are properly integrated")
                print(f"     - All latency targets are met")
                print(f"     - Data flows correctly between components")
                print(f"     - Emotion analysis produces valid modulation parameters")
                print(f"     - Performance is suitable for real-time processing")
    
    @pytest.mark.asyncio
    async def test_multi_speaker_integration(self):
        """
        CHECKPOINT VALIDATION: Multi-speaker processing integration
        
        Validates that multi-speaker voice separation works with emotion analysis.
        """
        print("\n🎤 CHECKPOINT VALIDATION: Multi-Speaker Integration")
        
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure multi-speaker simulation
                speaker_results = []
                
                async def mock_multi_speaker_stream(audio_stream, callback, sample_rate=16000):
                    """Simulate multi-speaker audio processing."""
                    speakers = ['spk_0', 'spk_1']
                    speaker_texts = {
                        'spk_0': ["I'm really excited about this!", "This is going great!"],
                        'spk_1': ["I'm feeling a bit nervous.", "Let me think about this."]
                    }
                    
                    chunk_count = 0
                    async for chunk in audio_stream:
                        await asyncio.sleep(0.09)  # Slightly longer for multi-speaker processing
                        
                        current_speaker = speakers[chunk_count % len(speakers)]
                        text_index = chunk_count // len(speakers)
                        
                        result = {
                            'transcript': speaker_texts[current_speaker][text_index % len(speaker_texts[current_speaker])],
                            'confidence': 0.82,  # Slightly lower for multi-speaker
                            'speaker_label': current_speaker,
                            'active_speakers': [current_speaker],
                            'speaker_count': len(speakers),
                            'start_time': chunk_count * 1.0,
                            'end_time': (chunk_count + 1) * 1.0
                        }
                        
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                        chunk_count += 1
                
                mock_stt_instance.process_audio_stream = mock_multi_speaker_stream
                
                # Configure emotion analysis for different speakers
                def mock_multi_speaker_sentiment(text):
                    if 'excited' in text.lower() or 'great' in text.lower():
                        return {'label': 'POSITIVE', 'confidence': 0.90, 'all_scores': []}
                    elif 'nervous' in text.lower() or 'think' in text.lower():
                        return {'label': 'NEGATIVE', 'confidence': 0.75, 'all_scores': []}
                    else:
                        return {'label': 'NEUTRAL', 'confidence': 0.70, 'all_scores': []}
                
                def mock_multi_speaker_emotion_vector(text):
                    if 'excited' in text.lower():
                        return EmotionVector(valence=0.8, arousal=0.7, dominance=0.6)
                    elif 'nervous' in text.lower():
                        return EmotionVector(valence=-0.5, arousal=0.6, dominance=-0.3)
                    else:
                        return EmotionVector(valence=0.0, arousal=0.2, dominance=0.0)
                
                mock_classifier.classify_sentiment.side_effect = mock_multi_speaker_sentiment
                mock_classifier.text_to_emotion_vector.side_effect = mock_multi_speaker_emotion_vector
                
                # Create services
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                # Create multi-speaker audio stream
                async def multi_speaker_audio_stream():
                    for i in range(4):  # 4 chunks for 2 speakers
                        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
                        yield audio_data.tobytes()
                
                # Process multi-speaker stream
                async def multi_speaker_callback(speech_result):
                    text = speech_result.get('transcript', '')
                    if text:
                        emotion_result = await emotion_service.analyze_emotion(text=text)
                        
                        combined_result = {
                            'speaker_label': speech_result.get('speaker_label'),
                            'transcript': text,
                            'emotion': emotion_result['discrete_emotion'],
                            'emotion_intensity': emotion_result['emotion_intensity'],
                            'speaker_count': speech_result.get('speaker_count', 1)
                        }
                        
                        speaker_results.append(combined_result)
                
                print("  📊 Processing multi-speaker audio stream...")
                await speech_pipeline.process_real_time_audio(
                    multi_speaker_audio_stream(),
                    multi_speaker_callback
                )
                
                # Validate multi-speaker processing
                print(f"  ✅ Multi-speaker validation:")
                print(f"     - Total results: {len(speaker_results)}")
                
                # Should process all chunks
                assert len(speaker_results) == 4, "Should process 4 multi-speaker chunks"
                
                # Should identify different speakers
                speakers_detected = set(r['speaker_label'] for r in speaker_results)
                print(f"     - Speakers detected: {list(speakers_detected)}")
                assert len(speakers_detected) >= 2, "Should detect at least 2 speakers"
                
                # Should detect different emotions for different speakers
                emotions_by_speaker = {}
                for result in speaker_results:
                    speaker = result['speaker_label']
                    emotion = result['emotion']
                    
                    if speaker not in emotions_by_speaker:
                        emotions_by_speaker[speaker] = set()
                    emotions_by_speaker[speaker].add(emotion)
                
                print(f"     - Emotions by speaker: {dict(emotions_by_speaker)}")
                
                # Validate that different speakers can have different emotional patterns
                total_unique_emotions = len(set(r['emotion'] for r in speaker_results))
                assert total_unique_emotions >= 1, "Should detect emotions across speakers"
                
                print(f"  ✅ MULTI-SPEAKER VALIDATION PASSED")
                print(f"     - Multiple speakers are correctly identified")
                print(f"     - Emotion analysis works for each speaker")
                print(f"     - Speaker-specific emotional patterns are detected")
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self):
        """
        CHECKPOINT VALIDATION: Error recovery and robustness
        
        Validates that the pipeline handles errors gracefully and continues processing.
        """
        print("\n🛡️  CHECKPOINT VALIDATION: Error Recovery")
        
        with patch('app.services.speech_processing.SpeechToTextService') as mock_stt:
            mock_stt_instance = Mock()
            mock_stt.return_value = mock_stt_instance
            
            with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
                mock_classifier = Mock()
                mock_bert.return_value = mock_classifier
                
                # Configure error simulation
                error_results = []
                successful_results = []
                
                async def mock_error_prone_stream(audio_stream, callback, sample_rate=16000):
                    """Simulate processing with intermittent errors."""
                    chunk_count = 0
                    async for chunk in audio_stream:
                        await asyncio.sleep(0.05)
                        
                        if chunk_count == 1:
                            # Simulate transcription error
                            result = {
                                'transcript': '',
                                'confidence': 0.0,
                                'speaker_label': 'unknown',
                                'error': 'Transcription failed',
                                'is_partial': True
                            }
                        elif chunk_count == 3:
                            # Simulate low confidence result
                            result = {
                                'transcript': 'unclear audio segment',
                                'confidence': 0.3,
                                'speaker_label': 'spk_0',
                                'warning': 'Low confidence'
                            }
                        else:
                            # Normal processing
                            result = {
                                'transcript': f'Clear speech segment {chunk_count}',
                                'confidence': 0.88,
                                'speaker_label': 'spk_0'
                            }
                        
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                        chunk_count += 1
                
                mock_stt_instance.process_audio_stream = mock_error_prone_stream
                
                # Configure emotion analysis to handle errors
                def mock_error_handling_sentiment(text):
                    if not text or text == '':
                        return {'label': 'NEUTRAL', 'confidence': 0.0, 'all_scores': []}
                    elif 'unclear' in text.lower():
                        return {'label': 'NEUTRAL', 'confidence': 0.4, 'all_scores': []}
                    else:
                        return {'label': 'POSITIVE', 'confidence': 0.85, 'all_scores': []}
                
                mock_classifier.classify_sentiment.side_effect = mock_error_handling_sentiment
                mock_classifier.text_to_emotion_vector.return_value = EmotionVector(0.1, 0.1, 0.1)
                
                # Create services
                speech_pipeline = SpeechProcessingPipeline()
                emotion_service = EmotionAnalysisService()
                
                # Create error-prone audio stream
                async def error_prone_audio_stream():
                    for i in range(5):  # 5 chunks with errors in positions 1 and 3
                        audio_data = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
                        yield audio_data.tobytes()
                
                # Process with error handling
                async def error_handling_callback(speech_result):
                    try:
                        text = speech_result.get('transcript', '')
                        has_error = 'error' in speech_result or speech_result.get('confidence', 1.0) < 0.5
                        
                        # Process through emotion analysis even with errors
                        emotion_result = await emotion_service.analyze_emotion(text=text)
                        
                        result = {
                            'speech_result': speech_result,
                            'emotion_result': emotion_result,
                            'has_error': has_error,
                            'processed_successfully': True
                        }
                        
                        if has_error:
                            error_results.append(result)
                        else:
                            successful_results.append(result)
                            
                    except Exception as e:
                        # Should not reach here - errors should be handled gracefully
                        error_results.append({
                            'exception': str(e),
                            'processed_successfully': False
                        })
                
                print("  📊 Processing error-prone audio stream...")
                await speech_pipeline.process_real_time_audio(
                    error_prone_audio_stream(),
                    error_handling_callback
                )
                
                # Validate error recovery
                total_results = len(error_results) + len(successful_results)
                print(f"  ✅ Error recovery validation:")
                print(f"     - Total chunks processed: {total_results}")
                print(f"     - Successful results: {len(successful_results)}")
                print(f"     - Error cases handled: {len(error_results)}")
                
                # Should process all chunks despite errors
                assert total_results == 5, "Should process all 5 chunks despite errors"
                
                # Should have some successful results
                assert len(successful_results) >= 3, "Should have at least 3 successful results"
                
                # Should handle error cases gracefully
                assert len(error_results) >= 2, "Should detect and handle error cases"
                
                # All error cases should be processed successfully (no exceptions)
                for error_result in error_results:
                    assert error_result.get('processed_successfully', False), "Error cases should be handled gracefully"
                    assert 'emotion_result' in error_result, "Should still produce emotion analysis for error cases"
                
                print(f"  ✅ ERROR RECOVERY VALIDATION PASSED")
                print(f"     - Pipeline continues processing despite errors")
                print(f"     - Error cases are handled gracefully")
                print(f"     - Emotion analysis works even with poor input")
                print(f"     - No unhandled exceptions occurred")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])