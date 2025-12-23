"""
Emotion Analysis Service

BERT-based sentiment classification with prosodic feature extraction
for emotion-aware sign language synthesis.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import librosa
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json

from app.core.config import settings

# Import signing modulation service (avoiding circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.signing_modulation import SigningModulationService

logger = logging.getLogger(__name__)


class EmotionVector:
    """Represents emotion in valence-arousal-dominance space."""
    
    def __init__(self, valence: float, arousal: float, dominance: float):
        """
        Initialize emotion vector.
        
        Args:
            valence: Pleasure/displeasure (-1 to 1)
            arousal: Activation/deactivation (-1 to 1) 
            dominance: Control/lack of control (-1 to 1)
        """
        self.valence = max(-1.0, min(1.0, valence))
        self.arousal = max(-1.0, min(1.0, arousal))
        self.dominance = max(-1.0, min(1.0, dominance))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance
        }
    
    def magnitude(self) -> float:
        """Calculate emotion magnitude/intensity."""
        return np.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2)
    
    def __repr__(self) -> str:
        return f"EmotionVector(v={self.valence:.2f}, a={self.arousal:.2f}, d={self.dominance:.2f})"


class ProsodicFeatureExtractor:
    """Extract prosodic features from audio for emotion analysis."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.window_size = int(0.025 * sample_rate)  # 25ms windows
        self.hop_size = int(0.010 * sample_rate)     # 10ms hop
        
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive prosodic features from audio.
        
        Args:
            audio_data: Audio signal as numpy array
            
        Returns:
            Dictionary of prosodic features
        """
        if len(audio_data) == 0:
            return self._get_default_features()
        
        features = {}
        
        try:
            # Pitch features
            pitch_features = self._extract_pitch_features(audio_data)
            features.update(pitch_features)
            
            # Energy features
            energy_features = self._extract_energy_features(audio_data)
            features.update(energy_features)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio_data)
            features.update(spectral_features)
            
            # Temporal features
            temporal_features = self._extract_temporal_features(audio_data)
            features.update(temporal_features)
            
            # Voice quality features
            voice_quality_features = self._extract_voice_quality_features(audio_data)
            features.update(voice_quality_features)
            
        except Exception as e:
            logger.warning(f"Error extracting prosodic features: {e}")
            return self._get_default_features()
        
        return features
    
    def _extract_pitch_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract pitch-related features."""
        # Extract fundamental frequency using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0,
                'pitch_slope': 0.0,
                'voiced_ratio': 0.0
            }
        
        # Calculate pitch statistics
        pitch_mean = np.mean(f0_clean)
        pitch_std = np.std(f0_clean)
        pitch_range = np.max(f0_clean) - np.min(f0_clean)
        
        # Calculate pitch slope (trend over time)
        if len(f0_clean) > 1:
            x = np.arange(len(f0_clean))
            pitch_slope = np.polyfit(x, f0_clean, 1)[0]
        else:
            pitch_slope = 0.0
        
        # Voiced ratio
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0.0
        
        return {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_range': float(pitch_range),
            'pitch_slope': float(pitch_slope),
            'voiced_ratio': float(voiced_ratio)
        }
    
    def _extract_energy_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract energy-related features."""
        # RMS energy
        rms = librosa.feature.rms(y=audio_data, frame_length=self.window_size, hop_length=self.hop_size)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=self.window_size, hop_length=self.hop_size)[0]
        
        return {
            'energy_mean': float(np.mean(rms)),
            'energy_std': float(np.std(rms)),
            'energy_max': float(np.max(rms)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract spectral features."""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'mfcc_mean': float(np.mean(mfccs)),
            'mfcc_std': float(np.std(mfccs))
        }
    
    def _extract_temporal_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract temporal features."""
        # Speaking rate estimation
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
        speaking_rate = len(onset_frames) / (len(audio_data) / self.sample_rate) if len(audio_data) > 0 else 0.0
        
        # Pause detection
        rms = librosa.feature.rms(y=audio_data)[0]
        silence_threshold = np.mean(rms) * 0.1
        silence_frames = np.sum(rms < silence_threshold)
        pause_ratio = silence_frames / len(rms) if len(rms) > 0 else 0.0
        
        return {
            'speaking_rate': float(speaking_rate),
            'pause_ratio': float(pause_ratio),
            'duration': float(len(audio_data) / self.sample_rate)
        }
    
    def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract voice quality features."""
        # Jitter and shimmer approximation
        # These are simplified versions - production would use more sophisticated algorithms
        
        # Harmonic-to-noise ratio approximation
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Simple HNR estimation
        harmonic_energy = np.sum(magnitude[:magnitude.shape[0]//4, :])  # Lower frequencies
        noise_energy = np.sum(magnitude[magnitude.shape[0]//2:, :])     # Higher frequencies
        
        hnr = harmonic_energy / (noise_energy + 1e-8)
        
        return {
            'hnr_estimate': float(np.log10(hnr + 1e-8)),
            'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(y=audio_data)[0]))
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when extraction fails."""
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_range': 0.0,
            'pitch_slope': 0.0,
            'voiced_ratio': 0.0,
            'energy_mean': 0.0,
            'energy_std': 0.0,
            'energy_max': 0.0,
            'zcr_mean': 0.0,
            'zcr_std': 0.0,
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'mfcc_mean': 0.0,
            'mfcc_std': 0.0,
            'speaking_rate': 0.0,
            'pause_ratio': 0.0,
            'duration': 0.0,
            'hnr_estimate': 0.0,
            'spectral_flatness': 0.0
        }


class BERTSentimentClassifier:
    """BERT-based sentiment classification for emotion detection."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self._initialize_model()
        
        # Emotion mapping from sentiment to VAD space
        self.emotion_mappings = {
            'POSITIVE': EmotionVector(valence=0.7, arousal=0.3, dominance=0.5),
            'NEGATIVE': EmotionVector(valence=-0.7, arousal=0.2, dominance=-0.3),
            'NEUTRAL': EmotionVector(valence=0.0, arousal=0.0, dominance=0.0),
            'joy': EmotionVector(valence=0.8, arousal=0.6, dominance=0.4),
            'anger': EmotionVector(valence=-0.6, arousal=0.8, dominance=0.7),
            'sadness': EmotionVector(valence=-0.8, arousal=-0.4, dominance=-0.5),
            'fear': EmotionVector(valence=-0.7, arousal=0.5, dominance=-0.8),
            'surprise': EmotionVector(valence=0.2, arousal=0.7, dominance=0.0),
            'disgust': EmotionVector(valence=-0.8, arousal=0.3, dominance=0.2)
        }
    
    def _initialize_model(self):
        """Initialize BERT model and tokenizer."""
        try:
            # Use Hugging Face pipeline for simplicity
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"BERT sentiment model initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            # Fallback to a simpler model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU only for fallback
                )
                logger.info("Fallback sentiment model initialized")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {fallback_error}")
                self.sentiment_pipeline = None
    
    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Classify sentiment of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment classification results
        """
        if not self.sentiment_pipeline or not text.strip():
            return {
                'label': 'NEUTRAL',
                'confidence': 0.0,
                'all_scores': []
            }
        
        try:
            # Get sentiment prediction
            results = self.sentiment_pipeline(text)
            
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
            else:
                result = results
            
            # Normalize label names
            label = result['label'].upper()
            if label in ['LABEL_0', 'NEGATIVE']:
                label = 'NEGATIVE'
            elif label in ['LABEL_1', 'POSITIVE']:
                label = 'POSITIVE'
            elif label in ['LABEL_2', 'NEUTRAL']:
                label = 'NEUTRAL'
            
            return {
                'label': label,
                'confidence': float(result['score']),
                'all_scores': [result] if not isinstance(results, list) else results
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment classification: {e}")
            return {
                'label': 'NEUTRAL',
                'confidence': 0.0,
                'all_scores': []
            }
    
    def text_to_emotion_vector(self, text: str) -> EmotionVector:
        """
        Convert text to emotion vector in VAD space.
        
        Args:
            text: Input text
            
        Returns:
            EmotionVector representing the emotion
        """
        sentiment_result = self.classify_sentiment(text)
        label = sentiment_result['label']
        confidence = sentiment_result['confidence']
        
        # Get base emotion vector
        base_emotion = self.emotion_mappings.get(label, self.emotion_mappings['NEUTRAL'])
        
        # Scale by confidence
        scaled_emotion = EmotionVector(
            valence=base_emotion.valence * confidence,
            arousal=base_emotion.arousal * confidence,
            dominance=base_emotion.dominance * confidence
        )
        
        return scaled_emotion


class EmotionAnalysisService:
    """Main emotion analysis service combining text and audio analysis."""
    
    def __init__(self):
        self.prosodic_extractor = ProsodicFeatureExtractor()
        self.sentiment_classifier = BERTSentimentClassifier()
        self.emotion_history = []
        self.max_history_length = 10
        
        # Initialize signing modulation service (lazy import to avoid circular dependency)
        self._signing_modulation_service = None
    
    @property
    def signing_modulation_service(self):
        """Lazy initialization of signing modulation service."""
        if self._signing_modulation_service is None:
            from app.services.signing_modulation import SigningModulationService
            self._signing_modulation_service = SigningModulationService()
        return self._signing_modulation_service
        
    async def analyze_emotion(self, 
                            text: str, 
                            audio_data: Optional[np.ndarray] = None,
                            sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Comprehensive emotion analysis from text and audio.
        
        Args:
            text: Transcribed text
            audio_data: Optional audio signal
            sample_rate: Audio sample rate
            
        Returns:
            Complete emotion analysis result
        """
        analysis_start = datetime.now()
        
        # Text-based sentiment analysis
        text_sentiment = self.sentiment_classifier.classify_sentiment(text)
        text_emotion_vector = self.sentiment_classifier.text_to_emotion_vector(text)
        
        # Audio-based prosodic analysis
        prosodic_features = {}
        audio_emotion_vector = EmotionVector(0.0, 0.0, 0.0)
        
        if audio_data is not None and len(audio_data) > 0:
            prosodic_features = self.prosodic_extractor.extract_features(audio_data)
            audio_emotion_vector = self._prosodic_to_emotion_vector(prosodic_features)
        
        # Combine text and audio emotions
        combined_emotion = self._combine_emotion_vectors(
            text_emotion_vector, 
            audio_emotion_vector,
            text_weight=0.7,  # Text is more reliable for sentiment
            audio_weight=0.3  # Audio provides arousal/intensity info
        )
        
        # Determine discrete emotion category
        discrete_emotion = self._vector_to_discrete_emotion(combined_emotion)
        
        # Calculate emotion intensity
        emotion_intensity = combined_emotion.magnitude()
        
        # Create comprehensive result
        result = {
            'timestamp': analysis_start.isoformat(),
            'text_input': text,
            'has_audio': audio_data is not None,
            
            # Text analysis
            'text_sentiment': text_sentiment,
            'text_emotion_vector': text_emotion_vector.to_dict(),
            
            # Audio analysis
            'prosodic_features': prosodic_features,
            'audio_emotion_vector': audio_emotion_vector.to_dict(),
            
            # Combined analysis
            'combined_emotion_vector': combined_emotion.to_dict(),
            'discrete_emotion': discrete_emotion,
            'emotion_intensity': float(emotion_intensity),
            
            # Signing modulation parameters
            'signing_modulation': self._generate_signing_modulation(combined_emotion, discrete_emotion),
            
            # Processing metadata
            'processing_time_ms': (datetime.now() - analysis_start).total_seconds() * 1000
        }
        
        # Update emotion history
        self._update_emotion_history(result)
        
        return result
    
    def _prosodic_to_emotion_vector(self, prosodic_features: Dict[str, float]) -> EmotionVector:
        """Convert prosodic features to emotion vector."""
        # This is a simplified mapping - production would use trained models
        
        # Arousal from energy and pitch variation
        arousal = 0.0
        if prosodic_features.get('energy_mean', 0) > 0.1:
            arousal += 0.3
        if prosodic_features.get('pitch_std', 0) > 50:
            arousal += 0.4
        if prosodic_features.get('speaking_rate', 0) > 3:
            arousal += 0.3
        
        # Valence from pitch and voice quality
        valence = 0.0
        if prosodic_features.get('pitch_mean', 0) > 200:  # Higher pitch often positive
            valence += 0.2
        if prosodic_features.get('hnr_estimate', 0) > 0:  # Clear voice often positive
            valence += 0.2
        
        # Dominance from energy and pitch range
        dominance = 0.0
        if prosodic_features.get('energy_max', 0) > 0.5:
            dominance += 0.3
        if prosodic_features.get('pitch_range', 0) > 100:
            dominance += 0.2
        
        # Normalize to [-1, 1] range
        arousal = max(-1.0, min(1.0, arousal - 0.5))
        valence = max(-1.0, min(1.0, valence - 0.2))
        dominance = max(-1.0, min(1.0, dominance - 0.25))
        
        return EmotionVector(valence, arousal, dominance)
    
    def _combine_emotion_vectors(self, 
                               text_emotion: EmotionVector,
                               audio_emotion: EmotionVector,
                               text_weight: float = 0.7,
                               audio_weight: float = 0.3) -> EmotionVector:
        """Combine text and audio emotion vectors."""
        combined_valence = (text_emotion.valence * text_weight + 
                          audio_emotion.valence * audio_weight)
        combined_arousal = (text_emotion.arousal * text_weight + 
                          audio_emotion.arousal * audio_weight)
        combined_dominance = (text_emotion.dominance * text_weight + 
                            audio_emotion.dominance * audio_weight)
        
        return EmotionVector(combined_valence, combined_arousal, combined_dominance)
    
    def _vector_to_discrete_emotion(self, emotion_vector: EmotionVector) -> str:
        """Map emotion vector to discrete emotion category."""
        v, a, d = emotion_vector.valence, emotion_vector.arousal, emotion_vector.dominance
        
        # More sensitive rule-based mapping to detect emotions at lower thresholds
        if abs(v) < 0.1 and abs(a) < 0.1:
            return 'neutral'
        elif v > 0.2 and a > 0.1:  # Lowered thresholds
            return 'joy'
        elif v < -0.2 and a > 0.2 and d > 0.1:  # Lowered thresholds
            return 'anger'
        elif v < -0.2 and a < 0.1:  # Lowered thresholds
            return 'sadness'
        elif v < -0.1 and a > 0.1 and d < -0.2:  # Lowered thresholds
            return 'fear'
        elif v > 0.1 and a > 0.3:  # Lowered thresholds
            return 'excitement'
        elif v < -0.3:  # Lowered threshold
            return 'disgust'
        elif v > 0.1:  # Catch positive emotions
            return 'joy'
        elif v < -0.1:  # Catch negative emotions
            return 'sadness'
        else:
            return 'neutral'
    
    def _generate_signing_modulation(self, 
                                   emotion_vector: EmotionVector, 
                                   discrete_emotion: str) -> Dict[str, Any]:
        """Generate signing modulation parameters based on emotion using the signing modulation service."""
        intensity = emotion_vector.magnitude()
        
        # Use the signing modulation service for comprehensive modulation
        modulation_params = self.signing_modulation_service.generate_signing_modulation(
            emotion_vector=emotion_vector,
            discrete_emotion=discrete_emotion,
            emotion_intensity=intensity
        )
        
        # Convert to dictionary format for backward compatibility
        modulation_dict = {
            'intensity_scale': modulation_params.intensity_scale,
            'speed_multiplier': modulation_params.speed_multiplier,
            'gesture_size_scale': modulation_params.gesture_size_scale,
            'facial_expression_intensity': modulation_params.facial_expression_intensity,
            'signing_space_scale': modulation_params.signing_space_scale,
            
            # Additional parameters from the comprehensive modulation
            'gesture_amplitude_scale': modulation_params.gesture_amplitude_scale,
            'pause_duration_scale': modulation_params.pause_duration_scale,
            'transition_smoothness': modulation_params.transition_smoothness,
            'hand_separation_scale': modulation_params.hand_separation_scale,
            'posture_tension': modulation_params.posture_tension,
            'shoulder_elevation': modulation_params.shoulder_elevation,
            'torso_lean': modulation_params.torso_lean,
            'rhythm_regularity': modulation_params.rhythm_regularity,
            'dynamic_range': modulation_params.dynamic_range,
            'accent_strength': modulation_params.accent_strength,
            
            # FACS Action Units for facial expression
            'facs_action_units': [
                {
                    'au_number': au.au_number,
                    'name': au.name,
                    'intensity': au.intensity,
                    'description': au.description
                }
                for au in modulation_params.facs_action_units
            ]
        }
        
        return modulation_dict
    
    def _update_emotion_history(self, emotion_result: Dict[str, Any]):
        """Update emotion history for temporal analysis."""
        self.emotion_history.append({
            'timestamp': emotion_result['timestamp'],
            'discrete_emotion': emotion_result['discrete_emotion'],
            'emotion_vector': emotion_result['combined_emotion_vector'],
            'intensity': emotion_result['emotion_intensity']
        })
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history_length:
            self.emotion_history = self.emotion_history[-self.max_history_length:]
    
    def get_emotion_trends(self) -> Dict[str, Any]:
        """Analyze emotion trends over recent history."""
        if not self.emotion_history:
            return {'trend': 'stable', 'dominant_emotion': 'neutral'}
        
        recent_emotions = [entry['discrete_emotion'] for entry in self.emotion_history[-5:]]
        recent_intensities = [entry['intensity'] for entry in self.emotion_history[-5:]]
        
        # Find dominant emotion
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Analyze intensity trend
        if len(recent_intensities) >= 3:
            if recent_intensities[-1] > recent_intensities[-3] * 1.2:
                trend = 'increasing'
            elif recent_intensities[-1] < recent_intensities[-3] * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'dominant_emotion': dominant_emotion,
            'average_intensity': np.mean(recent_intensities),
            'emotion_distribution': emotion_counts
        }
    
    def reset_history(self):
        """Reset emotion analysis history."""
        self.emotion_history.clear()