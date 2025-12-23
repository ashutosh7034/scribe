"""
Property-Based Tests for Emotion Analysis

Property tests for emotion detection accuracy and consistency.
"""

import pytest
import asyncio
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Mock external dependencies with realistic behavior
import sys
from unittest.mock import MagicMock

# Create a more sophisticated librosa mock
librosa_mock = MagicMock()

def mock_pyin(y, fmin, fmax, sr):
    """Mock librosa.pyin to return realistic pitch data based on input audio."""
    # Calculate expected number of frames based on audio length
    hop_length = 512  # Default hop length
    n_frames = len(y) // hop_length + 1
    
    # Generate realistic pitch values (fundamental frequency)
    f0 = np.full(n_frames, 150.0)  # Base frequency around 150 Hz
    voiced_flag = np.ones(n_frames, dtype=bool)  # All frames voiced
    voiced_probs = np.ones(n_frames)  # High confidence
    
    return f0, voiced_flag, voiced_probs

def mock_rms(y, frame_length=2048, hop_length=512):
    """Mock librosa.feature.rms to return realistic energy values."""
    n_frames = len(y) // hop_length + 1
    rms_values = np.full((1, n_frames), 0.1)  # Moderate energy level
    return rms_values

def mock_zero_crossing_rate(y, frame_length=2048, hop_length=512):
    """Mock librosa.feature.zero_crossing_rate."""
    n_frames = len(y) // hop_length + 1
    zcr_values = np.full((1, n_frames), 0.05)  # Typical ZCR value
    return zcr_values

def mock_spectral_centroid(y, sr):
    """Mock librosa.feature.spectral_centroid."""
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    centroid_values = np.full((1, n_frames), 2000.0)  # Typical spectral centroid
    return centroid_values

def mock_spectral_rolloff(y, sr):
    """Mock librosa.feature.spectral_rolloff."""
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    rolloff_values = np.full((1, n_frames), 4000.0)  # Typical rolloff
    return rolloff_values

def mock_spectral_bandwidth(y, sr):
    """Mock librosa.feature.spectral_bandwidth."""
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    bandwidth_values = np.full((1, n_frames), 1500.0)  # Typical bandwidth
    return bandwidth_values

def mock_mfcc(y, sr, n_mfcc=13):
    """Mock librosa.feature.mfcc."""
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    mfcc_values = np.random.randn(n_mfcc, n_frames) * 0.1  # Realistic MFCC values
    return mfcc_values

def mock_onset_detect(y, sr):
    """Mock librosa.onset.onset_detect."""
    # Return some onset frames based on audio length
    duration = len(y) / sr
    n_onsets = max(1, int(duration * 2))  # ~2 onsets per second
    onset_frames = np.linspace(0, len(y) // 512, n_onsets, dtype=int)
    return onset_frames

def mock_stft(y):
    """Mock librosa.stft."""
    # Return a simple STFT-like array
    n_fft = 2048
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    n_freqs = n_fft // 2 + 1
    stft_matrix = np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)
    return stft_matrix * 0.1

def mock_spectral_flatness(y):
    """Mock librosa.feature.spectral_flatness."""
    hop_length = 512
    n_frames = len(y) // hop_length + 1
    flatness_values = np.full((1, n_frames), 0.1)  # Typical flatness value
    return flatness_values

def mock_temporal_features_extract(self, audio_data: np.ndarray) -> Dict[str, float]:
    """Mock _extract_temporal_features to use the correct sample rate."""
    # Get the sample rate from the current test context
    # This is a bit of a hack, but necessary for the mock
    sample_rate = getattr(self, 'sample_rate', 16000)
    
    # Speaking rate estimation
    onset_frames = librosa_mock.onset.onset_detect(y=audio_data, sr=sample_rate)
    speaking_rate = len(onset_frames) / (len(audio_data) / sample_rate) if len(audio_data) > 0 else 0.0
    
    # Pause detection
    rms = librosa_mock.feature.rms(y=audio_data)[0]
    silence_threshold = np.mean(rms) * 0.1
    silence_frames = np.sum(rms < silence_threshold)
    pause_ratio = silence_frames / len(rms) if len(rms) > 0 else 0.0
    
    return {
        'speaking_rate': float(speaking_rate),
        'pause_ratio': float(pause_ratio),
        'duration': float(len(audio_data) / sample_rate)  # Use the correct sample rate
    }

# Configure the librosa mock with realistic functions
librosa_mock.pyin = mock_pyin
librosa_mock.feature.rms = mock_rms
librosa_mock.feature.zero_crossing_rate = mock_zero_crossing_rate
librosa_mock.feature.spectral_centroid = mock_spectral_centroid
librosa_mock.feature.spectral_rolloff = mock_spectral_rolloff
librosa_mock.feature.spectral_bandwidth = mock_spectral_bandwidth
librosa_mock.feature.mfcc = mock_mfcc
librosa_mock.feature.spectral_flatness = mock_spectral_flatness
librosa_mock.onset.onset_detect = mock_onset_detect
librosa_mock.stft = mock_stft
librosa_mock.note_to_hz = mock_note_to_hz

sys.modules['librosa'] = librosa_mock
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()

from app.services.emotion_analysis import EmotionAnalysisService, EmotionVector


# Test data strategies
@st.composite
def emotion_text_strategy(draw):
    """Generate text with known emotional content for testing."""
    emotion_texts = {
        'positive': [
            "I'm so happy and excited about this!",
            "This is wonderful and amazing!",
            "I love this so much, it's fantastic!",
            "Great job, this is excellent work!",
            "I'm thrilled and delighted with the results!"
        ],
        'negative': [
            "I'm really upset and angry about this.",
            "This is terrible and disappointing.",
            "I hate this, it's awful and frustrating.",
            "This makes me sad and depressed.",
            "I'm furious and outraged by this situation."
        ],
        'neutral': [
            "The meeting is scheduled for tomorrow.",
            "Please review the document and provide feedback.",
            "The weather forecast shows partly cloudy skies.",
            "The report contains statistical information.",
            "We need to update the system configuration."
        ]
    }
    
    emotion_category = draw(st.sampled_from(['positive', 'negative', 'neutral']))
    text = draw(st.sampled_from(emotion_texts[emotion_category]))
    
    return text, emotion_category


@st.composite
def audio_data_strategy(draw):
    """Generate synthetic audio data for testing."""
    # Generate audio parameters
    duration = draw(st.floats(min_value=0.5, max_value=3.0))  # 0.5 to 3 seconds
    sample_rate = draw(st.integers(min_value=8000, max_value=48000))  # Match test sample rate
    num_samples = int(duration * sample_rate)
    
    # Generate synthetic audio with emotional characteristics
    emotion_type = draw(st.sampled_from(['calm', 'excited', 'sad', 'angry']))
    
    # Base frequency and amplitude based on emotion
    if emotion_type == 'excited':
        base_freq = draw(st.floats(min_value=200, max_value=400))
        amplitude = draw(st.floats(min_value=0.3, max_value=0.8))
        freq_variation = draw(st.floats(min_value=0.2, max_value=0.5))
    elif emotion_type == 'angry':
        base_freq = draw(st.floats(min_value=150, max_value=300))
        amplitude = draw(st.floats(min_value=0.4, max_value=0.9))
        freq_variation = draw(st.floats(min_value=0.3, max_value=0.6))
    elif emotion_type == 'sad':
        base_freq = draw(st.floats(min_value=80, max_value=180))
        amplitude = draw(st.floats(min_value=0.1, max_value=0.4))
        freq_variation = draw(st.floats(min_value=0.05, max_value=0.2))
    else:  # calm
        base_freq = draw(st.floats(min_value=120, max_value=250))
        amplitude = draw(st.floats(min_value=0.2, max_value=0.5))
        freq_variation = draw(st.floats(min_value=0.1, max_value=0.3))
    
    # Generate synthetic audio signal
    t = np.linspace(0, duration, num_samples)
    
    # Add frequency modulation for emotional variation
    freq_mod = base_freq * (1 + freq_variation * np.sin(2 * np.pi * 2 * t))
    audio_signal = amplitude * np.sin(2 * np.pi * freq_mod * t)
    
    # Add some noise for realism
    noise_level = draw(st.floats(min_value=0.01, max_value=0.1))
    noise = noise_level * np.random.randn(num_samples)
    audio_signal += noise
    
    # Normalize to [-1, 1] range
    audio_signal = np.clip(audio_signal, -1.0, 1.0)
    
    return audio_signal.astype(np.float32), emotion_type, sample_rate


@st.composite
def emotion_intensity_strategy(draw):
    """Generate emotion intensity values for testing."""
    return draw(st.floats(min_value=0.0, max_value=1.0))


class TestEmotionDetectionAccuracyProperties:
    """Property-based tests for emotion detection accuracy."""
    
    @given(
        text_and_emotion=emotion_text_strategy(),
        audio_and_emotion=audio_data_strategy()
    )
    @settings(
        max_examples=100,
        deadline=10000,  # 10 second timeout per test
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_emotional_response_consistency(self, text_and_emotion, audio_and_emotion):
        """
        Feature: real-time-sign-language-translation, Property 5: Emotional Response Consistency
        
        For any speech input with detectable emotional content, the avatar's signing intensity 
        and facial expressions should consistently reflect the detected emotion type and intensity.
        
        **Validates: Requirements 2.4, 3.1, 3.2**
        """
        text, expected_text_emotion = text_and_emotion
        audio_data, expected_audio_emotion, sample_rate = audio_and_emotion
        
        # Mock the BERT sentiment classifier to return predictable results
        with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
            mock_classifier = Mock()
            mock_bert.return_value = mock_classifier
            
            # Configure mock to return expected sentiment based on text emotion
            if expected_text_emotion == 'positive':
                mock_sentiment = {'label': 'POSITIVE', 'confidence': 0.85, 'all_scores': []}
                mock_emotion_vector = EmotionVector(valence=0.7, arousal=0.3, dominance=0.5)
            elif expected_text_emotion == 'negative':
                mock_sentiment = {'label': 'NEGATIVE', 'confidence': 0.80, 'all_scores': []}
                mock_emotion_vector = EmotionVector(valence=-0.7, arousal=0.2, dominance=-0.3)
            else:  # neutral
                mock_sentiment = {'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []}
                mock_emotion_vector = EmotionVector(valence=0.0, arousal=0.0, dominance=0.0)
            
            mock_classifier.classify_sentiment.return_value = mock_sentiment
            mock_classifier.text_to_emotion_vector.return_value = mock_emotion_vector
            
            # Create emotion analysis service
            emotion_service = EmotionAnalysisService()
            
            # Perform emotion analysis
            result = await emotion_service.analyze_emotion(
                text=text,
                audio_data=audio_data,
                sample_rate=sample_rate
            )
            
            # Verify emotional response consistency properties
            
            # Property 1: Emotion detection should be consistent with input
            detected_emotion = result['discrete_emotion']
            emotion_intensity = result['emotion_intensity']
            signing_modulation = result['signing_modulation']
            
            # The detected emotion should be reasonable given the inputs
            assert detected_emotion is not None, "No emotion was detected"
            assert isinstance(detected_emotion, str), "Emotion should be a string category"
            
            # Property 2: Emotion intensity should be within valid range
            assert 0.0 <= emotion_intensity <= 2.0, (
                f"Emotion intensity {emotion_intensity} outside valid range [0.0, 2.0]"
            )
            
            # Property 3: Signing modulation parameters should reflect emotion appropriately
            intensity_scale = signing_modulation['intensity_scale']
            speed_multiplier = signing_modulation['speed_multiplier']
            gesture_size_scale = signing_modulation['gesture_size_scale']
            facial_expression_intensity = signing_modulation['facial_expression_intensity']
            
            # All modulation parameters should be positive and reasonable
            assert intensity_scale > 0, f"Intensity scale {intensity_scale} should be positive"
            assert speed_multiplier > 0, f"Speed multiplier {speed_multiplier} should be positive"
            assert gesture_size_scale > 0, f"Gesture size scale {gesture_size_scale} should be positive"
            assert 0.0 <= facial_expression_intensity <= 1.0, (
                f"Facial expression intensity {facial_expression_intensity} outside [0.0, 1.0]"
            )
            
            # Property 4: High-intensity emotions should have stronger modulation
            if emotion_intensity > 0.5:
                # For high-intensity emotions, at least one modulation parameter should be enhanced
                enhanced_params = sum([
                    intensity_scale > 1.1,
                    speed_multiplier > 1.1 or speed_multiplier < 0.9,
                    gesture_size_scale > 1.1 or gesture_size_scale < 0.9,
                    facial_expression_intensity > 0.6
                ])
                
                assert enhanced_params >= 1, (
                    f"High-intensity emotion ({emotion_intensity:.2f}) should enhance at least one "
                    f"modulation parameter, but all are near baseline"
                )
            
            # Property 5: Emotion vectors should be within valid VAD space
            combined_vector = result['combined_emotion_vector']
            assert -1.0 <= combined_vector['valence'] <= 1.0, (
                f"Valence {combined_vector['valence']} outside [-1.0, 1.0]"
            )
            assert -1.0 <= combined_vector['arousal'] <= 1.0, (
                f"Arousal {combined_vector['arousal']} outside [-1.0, 1.0]"
            )
            assert -1.0 <= combined_vector['dominance'] <= 1.0, (
                f"Dominance {combined_vector['dominance']} outside [-1.0, 1.0]"
            )
            
            # Property 6: Processing should complete in reasonable time
            processing_time = result['processing_time_ms']
            assert processing_time < 1000, (
                f"Processing time {processing_time}ms exceeds 1000ms threshold"
            )
    
    @given(
        emotion_texts=st.lists(emotion_text_strategy(), min_size=2, max_size=5)
    )
    @settings(
        max_examples=50,
        deadline=15000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_emotion_consistency_across_similar_inputs(self, emotion_texts):
        """
        Test that similar emotional inputs produce consistent emotion detection results.
        
        This supports Property 5 by ensuring the system responds consistently to 
        similar emotional content.
        """
        with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
            mock_classifier = Mock()
            mock_bert.return_value = mock_classifier
            
            # Group texts by emotion category
            emotion_groups = {}
            for text, emotion_category in emotion_texts:
                if emotion_category not in emotion_groups:
                    emotion_groups[emotion_category] = []
                emotion_groups[emotion_category].append(text)
            
            emotion_service = EmotionAnalysisService()
            results_by_category = {}
            
            # Analyze each text
            for text, emotion_category in emotion_texts:
                # Configure mock based on expected emotion
                if emotion_category == 'positive':
                    mock_sentiment = {'label': 'POSITIVE', 'confidence': 0.85, 'all_scores': []}
                    mock_emotion_vector = EmotionVector(valence=0.7, arousal=0.3, dominance=0.5)
                elif emotion_category == 'negative':
                    mock_sentiment = {'label': 'NEGATIVE', 'confidence': 0.80, 'all_scores': []}
                    mock_emotion_vector = EmotionVector(valence=-0.7, arousal=0.2, dominance=-0.3)
                else:  # neutral
                    mock_sentiment = {'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []}
                    mock_emotion_vector = EmotionVector(valence=0.0, arousal=0.0, dominance=0.0)
                
                mock_classifier.classify_sentiment.return_value = mock_sentiment
                mock_classifier.text_to_emotion_vector.return_value = mock_emotion_vector
                
                result = await emotion_service.analyze_emotion(text=text)
                
                if emotion_category not in results_by_category:
                    results_by_category[emotion_category] = []
                results_by_category[emotion_category].append(result)
            
            # Property: Similar emotional inputs should produce consistent results
            for emotion_category, results in results_by_category.items():
                if len(results) >= 2:
                    # Check consistency of discrete emotions
                    discrete_emotions = [r['discrete_emotion'] for r in results]
                    
                    # At least 70% should have the same discrete emotion
                    most_common_emotion = max(set(discrete_emotions), key=discrete_emotions.count)
                    consistency_ratio = discrete_emotions.count(most_common_emotion) / len(discrete_emotions)
                    
                    assert consistency_ratio >= 0.7, (
                        f"Emotion consistency {consistency_ratio:.2f} below 70% for {emotion_category} texts"
                    )
                    
                    # Check consistency of emotion intensities
                    intensities = [r['emotion_intensity'] for r in results]
                    intensity_std = np.std(intensities)
                    
                    # Standard deviation should be reasonable (not too high)
                    assert intensity_std < 0.5, (
                        f"Emotion intensity variation {intensity_std:.2f} too high for {emotion_category} texts"
                    )
    
    @given(
        base_text_emotion=emotion_text_strategy(),
        intensity_modifier=emotion_intensity_strategy()
    )
    @settings(
        max_examples=50,
        deadline=8000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_emotion_intensity_scaling(self, base_text_emotion, intensity_modifier):
        """
        Test that emotion intensity appropriately scales signing modulation parameters.
        
        This supports Property 5 by ensuring signing intensity reflects emotion intensity.
        """
        text, emotion_category = base_text_emotion
        
        with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
            mock_classifier = Mock()
            mock_bert.return_value = mock_classifier
            
            # Create emotion vector with scaled intensity
            if emotion_category == 'positive':
                base_vector = EmotionVector(valence=0.7, arousal=0.3, dominance=0.5)
                mock_sentiment = {'label': 'POSITIVE', 'confidence': 0.85, 'all_scores': []}
            elif emotion_category == 'negative':
                base_vector = EmotionVector(valence=-0.7, arousal=0.2, dominance=-0.3)
                mock_sentiment = {'label': 'NEGATIVE', 'confidence': 0.80, 'all_scores': []}
            else:  # neutral
                base_vector = EmotionVector(valence=0.0, arousal=0.0, dominance=0.0)
                mock_sentiment = {'label': 'NEUTRAL', 'confidence': 0.75, 'all_scores': []}
            
            # Scale the emotion vector by intensity modifier
            scaled_vector = EmotionVector(
                valence=base_vector.valence * intensity_modifier,
                arousal=base_vector.arousal * intensity_modifier,
                dominance=base_vector.dominance * intensity_modifier
            )
            
            mock_classifier.classify_sentiment.return_value = mock_sentiment
            mock_classifier.text_to_emotion_vector.return_value = scaled_vector
            
            emotion_service = EmotionAnalysisService()
            result = await emotion_service.analyze_emotion(text=text)
            
            # Property: Signing modulation should scale with emotion intensity
            emotion_intensity = result['emotion_intensity']
            signing_modulation = result['signing_modulation']
            
            # For non-neutral emotions, intensity should correlate with modulation
            if emotion_category != 'neutral' and emotion_intensity > 0.1:
                intensity_scale = signing_modulation['intensity_scale']
                facial_expression_intensity = signing_modulation['facial_expression_intensity']
                
                # Higher emotion intensity should generally lead to more pronounced modulation
                if emotion_intensity > 0.5:
                    # At least one parameter should be enhanced for high intensity
                    enhanced = (
                        intensity_scale > 1.05 or 
                        facial_expression_intensity > 0.55 or
                        signing_modulation['speed_multiplier'] != 1.0 or
                        signing_modulation['gesture_size_scale'] != 1.0
                    )
                    
                    assert enhanced, (
                        f"High emotion intensity {emotion_intensity:.2f} should enhance modulation parameters"
                    )
                
                # All modulation parameters should remain within reasonable bounds
                assert 0.5 <= intensity_scale <= 2.0, (
                    f"Intensity scale {intensity_scale} outside reasonable range [0.5, 2.0]"
                )
                assert 0.5 <= signing_modulation['speed_multiplier'] <= 2.0, (
                    f"Speed multiplier outside reasonable range"
                )
                assert 0.5 <= signing_modulation['gesture_size_scale'] <= 2.0, (
                    f"Gesture size scale outside reasonable range"
                )
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.asyncio
    async def test_prosodic_feature_extraction_robustness(self, audio_data):
        """
        Test that prosodic feature extraction is robust across different audio conditions.
        
        This supports Property 5 by ensuring audio-based emotion detection works reliably.
        """
        audio_signal, expected_emotion_type, sample_rate = audio_data
        
        with patch('app.services.emotion_analysis.BERTSentimentClassifier') as mock_bert:
            mock_classifier = Mock()
            mock_bert.return_value = mock_classifier
            
            # Mock text analysis to focus on audio analysis
            mock_classifier.classify_sentiment.return_value = {
                'label': 'NEUTRAL', 'confidence': 0.5, 'all_scores': []
            }
            mock_classifier.text_to_emotion_vector.return_value = EmotionVector(0.0, 0.0, 0.0)
            
            emotion_service = EmotionAnalysisService()
            
            # Analyze with audio data
            result = await emotion_service.analyze_emotion(
                text="neutral text",
                audio_data=audio_signal,
                sample_rate=sample_rate
            )
            
            # Property: Prosodic features should be extracted successfully
            prosodic_features = result['prosodic_features']
            
            # All expected prosodic features should be present
            expected_features = [
                'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope', 'voiced_ratio',
                'energy_mean', 'energy_std', 'energy_max', 'zcr_mean', 'zcr_std',
                'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
                'spectral_bandwidth_mean', 'mfcc_mean', 'mfcc_std', 'speaking_rate',
                'pause_ratio', 'duration', 'hnr_estimate', 'spectral_flatness'
            ]
            
            for feature in expected_features:
                assert feature in prosodic_features, f"Missing prosodic feature: {feature}"
                assert isinstance(prosodic_features[feature], (int, float)), (
                    f"Prosodic feature {feature} should be numeric"
                )
                assert not np.isnan(prosodic_features[feature]), (
                    f"Prosodic feature {feature} should not be NaN"
                )
            
            # Property: Audio emotion vector should be reasonable
            audio_emotion_vector = result['audio_emotion_vector']
            assert -1.0 <= audio_emotion_vector['valence'] <= 1.0
            assert -1.0 <= audio_emotion_vector['arousal'] <= 1.0
            assert -1.0 <= audio_emotion_vector['dominance'] <= 1.0
            
            # Property: Duration should match input audio approximately
            expected_duration = len(audio_signal) / sample_rate
            actual_duration = prosodic_features['duration']
            duration_error = abs(actual_duration - expected_duration) / expected_duration
            
            assert duration_error < 0.1, (
                f"Duration error {duration_error:.2%} exceeds 10% tolerance"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])