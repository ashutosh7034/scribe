"""
Signing Modulation Service

Emotion-to-signing modulation with intensity scaling, temporal modulation,
and facial expression mapping using FACS (Facial Action Coding System).
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from app.services.emotion_analysis import EmotionVector

logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Discrete emotion categories for signing modulation."""
    NEUTRAL = "neutral"
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    EXCITEMENT = "excitement"


@dataclass
class FACSActionUnit:
    """Facial Action Coding System (FACS) Action Unit."""
    au_number: int
    name: str
    intensity: float  # 0.0 to 1.0
    description: str


@dataclass
class SigningModulationParameters:
    """Complete set of signing modulation parameters."""
    # Intensity and amplitude
    intensity_scale: float = 1.0
    gesture_amplitude_scale: float = 1.0
    
    # Temporal modulation
    speed_multiplier: float = 1.0
    pause_duration_scale: float = 1.0
    transition_smoothness: float = 1.0
    
    # Spatial modulation
    gesture_size_scale: float = 1.0
    signing_space_scale: float = 1.0
    hand_separation_scale: float = 1.0
    
    # Facial expression
    facial_expression_intensity: float = 0.5
    facs_action_units: List[FACSActionUnit] = None
    
    # Body posture
    posture_tension: float = 0.5
    shoulder_elevation: float = 0.0
    torso_lean: float = 0.0
    
    # Rhythm and dynamics
    rhythm_regularity: float = 1.0
    dynamic_range: float = 1.0
    accent_strength: float = 1.0
    
    def __post_init__(self):
        if self.facs_action_units is None:
            self.facs_action_units = []


class FACSExpressionMapper:
    """Maps emotions to FACS Action Units for facial expression control."""
    
    def __init__(self):
        # FACS Action Unit definitions
        self.action_units = {
            # Upper face
            1: FACSActionUnit(1, "Inner Brow Raiser", 0.0, "Raises inner portion of eyebrows"),
            2: FACSActionUnit(2, "Outer Brow Raiser", 0.0, "Raises outer portion of eyebrows"),
            4: FACSActionUnit(4, "Brow Lowerer", 0.0, "Lowers and draws eyebrows together"),
            5: FACSActionUnit(5, "Upper Lid Raiser", 0.0, "Raises upper eyelid"),
            6: FACSActionUnit(6, "Cheek Raiser", 0.0, "Raises cheeks and narrows eyes"),
            7: FACSActionUnit(7, "Lid Tightener", 0.0, "Tightens eyelids"),
            
            # Lower face
            9: FACSActionUnit(9, "Nose Wrinkler", 0.0, "Wrinkles nose"),
            10: FACSActionUnit(10, "Upper Lip Raiser", 0.0, "Raises upper lip"),
            12: FACSActionUnit(12, "Lip Corner Puller", 0.0, "Pulls lip corners up (smile)"),
            15: FACSActionUnit(15, "Lip Corner Depressor", 0.0, "Pulls lip corners down"),
            16: FACSActionUnit(16, "Lower Lip Depressor", 0.0, "Depresses lower lip"),
            17: FACSActionUnit(17, "Chin Raiser", 0.0, "Raises and protrudes chin"),
            20: FACSActionUnit(20, "Lip Stretcher", 0.0, "Stretches lips horizontally"),
            22: FACSActionUnit(22, "Lip Funneler", 0.0, "Funnels and protrudes lips"),
            23: FACSActionUnit(23, "Lip Tightener", 0.0, "Tightens and narrows lips"),
            24: FACSActionUnit(24, "Lip Pressor", 0.0, "Presses lips together"),
            25: FACSActionUnit(25, "Lips Part", 0.0, "Parts lips slightly"),
            26: FACSActionUnit(26, "Jaw Drop", 0.0, "Drops jaw/opens mouth"),
            27: FACSActionUnit(27, "Mouth Stretch", 0.0, "Stretches mouth open"),
        }
        
        # Emotion to FACS mappings
        self.emotion_facs_mappings = {
            EmotionCategory.JOY: [
                (6, 0.7),   # Cheek Raiser (Duchenne marker)
                (12, 0.8),  # Lip Corner Puller (smile)
                (25, 0.3),  # Lips Part (slight)
            ],
            EmotionCategory.ANGER: [
                (4, 0.8),   # Brow Lowerer
                (5, 0.6),   # Upper Lid Raiser
                (7, 0.7),   # Lid Tightener
                (23, 0.6),  # Lip Tightener
                (24, 0.5),  # Lip Pressor
            ],
            EmotionCategory.SADNESS: [
                (1, 0.6),   # Inner Brow Raiser
                (4, 0.4),   # Brow Lowerer (slight)
                (15, 0.7),  # Lip Corner Depressor
                (17, 0.5),  # Chin Raiser
            ],
            EmotionCategory.FEAR: [
                (1, 0.8),   # Inner Brow Raiser
                (2, 0.7),   # Outer Brow Raiser
                (5, 0.8),   # Upper Lid Raiser
                (20, 0.6),  # Lip Stretcher
                (25, 0.4),  # Lips Part
                (26, 0.3),  # Jaw Drop (slight)
            ],
            EmotionCategory.SURPRISE: [
                (1, 0.9),   # Inner Brow Raiser
                (2, 0.9),   # Outer Brow Raiser
                (5, 0.8),   # Upper Lid Raiser
                (26, 0.6),  # Jaw Drop
                (27, 0.4),  # Mouth Stretch
            ],
            EmotionCategory.DISGUST: [
                (4, 0.5),   # Brow Lowerer
                (9, 0.8),   # Nose Wrinkler
                (10, 0.7),  # Upper Lip Raiser
                (16, 0.6),  # Lower Lip Depressor
            ],
            EmotionCategory.EXCITEMENT: [
                (2, 0.6),   # Outer Brow Raiser
                (5, 0.7),   # Upper Lid Raiser
                (6, 0.8),   # Cheek Raiser
                (12, 0.9),  # Lip Corner Puller
                (25, 0.5),  # Lips Part
                (26, 0.4),  # Jaw Drop (slight)
            ],
            EmotionCategory.NEUTRAL: [
                # Minimal activation for neutral expression
                (25, 0.1),  # Lips Part (very slight)
            ]
        }
    
    def map_emotion_to_facs(self, emotion_category: EmotionCategory, intensity: float) -> List[FACSActionUnit]:
        """
        Map emotion category and intensity to FACS Action Units.
        
        Args:
            emotion_category: The discrete emotion category
            intensity: Emotion intensity (0.0 to 1.0)
            
        Returns:
            List of activated FACS Action Units
        """
        if emotion_category not in self.emotion_facs_mappings:
            emotion_category = EmotionCategory.NEUTRAL
        
        facs_mapping = self.emotion_facs_mappings[emotion_category]
        activated_aus = []
        
        for au_number, base_intensity in facs_mapping:
            # Scale base intensity by emotion intensity
            scaled_intensity = min(1.0, base_intensity * intensity)
            
            # Ensure high-intensity emotions have enhanced facial expressions
            if intensity > 0.5 and emotion_category != EmotionCategory.NEUTRAL:
                enhancement_factor = (intensity - 0.5) * 0.4  # 0.0 to 0.2 enhancement
                scaled_intensity = min(1.0, scaled_intensity + enhancement_factor)
            
            if scaled_intensity > 0.1:  # Only include if significant activation
                au = FACSActionUnit(
                    au_number=au_number,
                    name=self.action_units[au_number].name,
                    intensity=scaled_intensity,
                    description=self.action_units[au_number].description
                )
                activated_aus.append(au)
        
        return activated_aus
    
    def blend_facs_expressions(self, primary_aus: List[FACSActionUnit], 
                             secondary_aus: List[FACSActionUnit], 
                             blend_ratio: float = 0.7) -> List[FACSActionUnit]:
        """
        Blend two sets of FACS Action Units for mixed emotions.
        
        Args:
            primary_aus: Primary emotion FACS AUs
            secondary_aus: Secondary emotion FACS AUs
            blend_ratio: Ratio of primary to secondary (0.0 to 1.0)
            
        Returns:
            Blended FACS Action Units
        """
        blended_aus = {}
        
        # Add primary AUs
        for au in primary_aus:
            blended_aus[au.au_number] = FACSActionUnit(
                au_number=au.au_number,
                name=au.name,
                intensity=au.intensity * blend_ratio,
                description=au.description
            )
        
        # Add secondary AUs
        for au in secondary_aus:
            if au.au_number in blended_aus:
                # Blend intensities
                existing_intensity = blended_aus[au.au_number].intensity
                secondary_intensity = au.intensity * (1.0 - blend_ratio)
                blended_aus[au.au_number].intensity = min(1.0, existing_intensity + secondary_intensity)
            else:
                blended_aus[au.au_number] = FACSActionUnit(
                    au_number=au.au_number,
                    name=au.name,
                    intensity=au.intensity * (1.0 - blend_ratio),
                    description=au.description
                )
        
        return list(blended_aus.values())


class TemporalModulator:
    """Handles temporal modulation of signing based on emotion."""
    
    def __init__(self):
        # Emotion-specific temporal characteristics
        self.temporal_profiles = {
            EmotionCategory.JOY: {
                'speed_multiplier': 1.1,
                'pause_duration_scale': 0.8,
                'transition_smoothness': 1.2,
                'rhythm_regularity': 1.1,
                'dynamic_range': 1.3,
                'accent_strength': 1.2
            },
            EmotionCategory.ANGER: {
                'speed_multiplier': 1.3,
                'pause_duration_scale': 0.6,
                'transition_smoothness': 0.7,
                'rhythm_regularity': 0.8,
                'dynamic_range': 1.5,
                'accent_strength': 1.6
            },
            EmotionCategory.SADNESS: {
                'speed_multiplier': 0.7,
                'pause_duration_scale': 1.4,
                'transition_smoothness': 1.3,
                'rhythm_regularity': 0.9,
                'dynamic_range': 0.7,
                'accent_strength': 0.6
            },
            EmotionCategory.FEAR: {
                'speed_multiplier': 1.4,
                'pause_duration_scale': 0.5,
                'transition_smoothness': 0.6,
                'rhythm_regularity': 0.7,
                'dynamic_range': 1.2,
                'accent_strength': 1.1
            },
            EmotionCategory.SURPRISE: {
                'speed_multiplier': 1.5,
                'pause_duration_scale': 0.4,
                'transition_smoothness': 0.5,
                'rhythm_regularity': 0.6,
                'dynamic_range': 1.4,
                'accent_strength': 1.3
            },
            EmotionCategory.DISGUST: {
                'speed_multiplier': 0.9,
                'pause_duration_scale': 1.2,
                'transition_smoothness': 0.8,
                'rhythm_regularity': 0.9,
                'dynamic_range': 1.1,
                'accent_strength': 1.0
            },
            EmotionCategory.EXCITEMENT: {
                'speed_multiplier': 1.4,
                'pause_duration_scale': 0.5,
                'transition_smoothness': 1.1,
                'rhythm_regularity': 1.2,
                'dynamic_range': 1.6,
                'accent_strength': 1.4
            },
            EmotionCategory.NEUTRAL: {
                'speed_multiplier': 1.0,
                'pause_duration_scale': 1.0,
                'transition_smoothness': 1.0,
                'rhythm_regularity': 1.0,
                'dynamic_range': 1.0,
                'accent_strength': 1.0
            }
        }
    
    def get_temporal_modulation(self, emotion_category: EmotionCategory, 
                              intensity: float) -> Dict[str, float]:
        """
        Get temporal modulation parameters for an emotion.
        
        Args:
            emotion_category: The discrete emotion category
            intensity: Emotion intensity (0.0 to 1.0)
            
        Returns:
            Dictionary of temporal modulation parameters
        """
        if emotion_category not in self.temporal_profiles:
            emotion_category = EmotionCategory.NEUTRAL
        
        base_profile = self.temporal_profiles[emotion_category]
        neutral_profile = self.temporal_profiles[EmotionCategory.NEUTRAL]
        
        # Scale modulation by intensity
        modulated_profile = {}
        for param, base_value in base_profile.items():
            neutral_value = neutral_profile[param]
            # Interpolate between neutral and emotional value based on intensity
            modulated_value = neutral_value + (base_value - neutral_value) * intensity
            modulated_profile[param] = modulated_value
        
        # Ensure high-intensity emotions have enhanced temporal parameters
        if intensity > 0.5 and emotion_category != EmotionCategory.NEUTRAL:
            enhancement_factor = (intensity - 0.5) * 0.4  # 0.0 to 0.2 enhancement
            
            # Enhance speed for high-arousal emotions
            if emotion_category in [EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.EXCITEMENT, EmotionCategory.SURPRISE]:
                modulated_profile['speed_multiplier'] = max(modulated_profile['speed_multiplier'], 1.1 + enhancement_factor)
            # Slow down for low-arousal emotions
            elif emotion_category == EmotionCategory.SADNESS:
                modulated_profile['speed_multiplier'] = min(modulated_profile['speed_multiplier'], 0.9 - enhancement_factor)
        
        return modulated_profile
    
    def apply_micro_timing_variations(self, base_timing: List[float], 
                                    emotion_category: EmotionCategory,
                                    intensity: float) -> List[float]:
        """
        Apply micro-timing variations based on emotion.
        
        Args:
            base_timing: Base timing sequence
            emotion_category: Emotion category
            intensity: Emotion intensity
            
        Returns:
            Modified timing sequence with emotional variations
        """
        if not base_timing:
            return base_timing
        
        temporal_params = self.get_temporal_modulation(emotion_category, intensity)
        rhythm_regularity = temporal_params['rhythm_regularity']
        
        # Apply rhythm variations
        modified_timing = []
        for i, timing in enumerate(base_timing):
            # Add micro-variations based on emotion
            if emotion_category == EmotionCategory.ANGER:
                # More abrupt, irregular timing
                variation = np.random.normal(0, 0.1 * intensity) * (1 - rhythm_regularity)
            elif emotion_category == EmotionCategory.SADNESS:
                # Slower, more drawn out
                variation = np.random.normal(0.05 * intensity, 0.05 * intensity)
            elif emotion_category == EmotionCategory.FEAR:
                # Rapid, jittery variations
                variation = np.random.normal(0, 0.15 * intensity) * (1 - rhythm_regularity)
            else:
                # General emotional variation
                variation = np.random.normal(0, 0.05 * intensity) * (1 - rhythm_regularity)
            
            modified_timing.append(max(0.01, timing + variation))
        
        return modified_timing


class IntensityScaler:
    """Handles intensity scaling of signing gestures based on emotion."""
    
    def __init__(self):
        # Emotion-specific intensity profiles
        self.intensity_profiles = {
            EmotionCategory.JOY: {
                'intensity_scale': 1.2,
                'gesture_amplitude_scale': 1.1,
                'gesture_size_scale': 1.1,
                'signing_space_scale': 1.0,
                'hand_separation_scale': 1.0
            },
            EmotionCategory.ANGER: {
                'intensity_scale': 1.5,
                'gesture_amplitude_scale': 1.4,
                'gesture_size_scale': 1.3,
                'signing_space_scale': 1.2,
                'hand_separation_scale': 1.1
            },
            EmotionCategory.SADNESS: {
                'intensity_scale': 0.7,
                'gesture_amplitude_scale': 0.8,
                'gesture_size_scale': 0.9,
                'signing_space_scale': 0.9,
                'hand_separation_scale': 0.95
            },
            EmotionCategory.FEAR: {
                'intensity_scale': 1.1,
                'gesture_amplitude_scale': 0.9,
                'gesture_size_scale': 0.8,
                'signing_space_scale': 0.8,
                'hand_separation_scale': 0.9
            },
            EmotionCategory.SURPRISE: {
                'intensity_scale': 1.3,
                'gesture_amplitude_scale': 1.2,
                'gesture_size_scale': 1.1,
                'signing_space_scale': 1.0,
                'hand_separation_scale': 1.0
            },
            EmotionCategory.DISGUST: {
                'intensity_scale': 1.1,
                'gesture_amplitude_scale': 1.0,
                'gesture_size_scale': 0.95,
                'signing_space_scale': 0.95,
                'hand_separation_scale': 1.0
            },
            EmotionCategory.EXCITEMENT: {
                'intensity_scale': 1.4,
                'gesture_amplitude_scale': 1.3,
                'gesture_size_scale': 1.2,
                'signing_space_scale': 1.1,
                'hand_separation_scale': 1.05
            },
            EmotionCategory.NEUTRAL: {
                'intensity_scale': 1.0,
                'gesture_amplitude_scale': 1.0,
                'gesture_size_scale': 1.0,
                'signing_space_scale': 1.0,
                'hand_separation_scale': 1.0
            }
        }
    
    def get_intensity_scaling(self, emotion_category: EmotionCategory, 
                            intensity: float) -> Dict[str, float]:
        """
        Get intensity scaling parameters for an emotion.
        
        Args:
            emotion_category: The discrete emotion category
            intensity: Emotion intensity (0.0 to 1.0)
            
        Returns:
            Dictionary of intensity scaling parameters
        """
        if emotion_category not in self.intensity_profiles:
            emotion_category = EmotionCategory.NEUTRAL
        
        base_profile = self.intensity_profiles[emotion_category]
        neutral_profile = self.intensity_profiles[EmotionCategory.NEUTRAL]
        
        # Scale modulation by intensity
        scaled_profile = {}
        for param, base_value in base_profile.items():
            neutral_value = neutral_profile[param]
            # Interpolate between neutral and emotional value based on intensity
            scaled_value = neutral_value + (base_value - neutral_value) * intensity
            scaled_profile[param] = scaled_value
        
        # Ensure high-intensity emotions have enhanced parameters
        if intensity > 0.5 and emotion_category != EmotionCategory.NEUTRAL:
            # Guarantee at least one parameter is significantly enhanced
            enhancement_factor = 0.1 + (intensity - 0.5) * 0.2  # 0.1 to 0.3 enhancement
            
            # Choose which parameter to enhance based on emotion type
            if emotion_category in [EmotionCategory.ANGER, EmotionCategory.EXCITEMENT]:
                scaled_profile['intensity_scale'] = max(scaled_profile['intensity_scale'], 1.1 + enhancement_factor)
            elif emotion_category == EmotionCategory.JOY:
                scaled_profile['gesture_size_scale'] = max(scaled_profile['gesture_size_scale'], 1.1 + enhancement_factor)
            elif emotion_category == EmotionCategory.FEAR:
                scaled_profile['gesture_size_scale'] = min(scaled_profile['gesture_size_scale'], 0.9 - enhancement_factor)
            elif emotion_category == EmotionCategory.SADNESS:
                scaled_profile['intensity_scale'] = min(scaled_profile['intensity_scale'], 0.9 - enhancement_factor)
            else:
                # Default enhancement for other emotions
                scaled_profile['intensity_scale'] = max(scaled_profile['intensity_scale'], 1.1 + enhancement_factor)
        
        return scaled_profile


class PostureModulator:
    """Handles body posture modulation based on emotion."""
    
    def __init__(self):
        # Emotion-specific posture profiles
        self.posture_profiles = {
            EmotionCategory.JOY: {
                'posture_tension': 0.4,
                'shoulder_elevation': 0.1,
                'torso_lean': 0.05  # Slight forward lean
            },
            EmotionCategory.ANGER: {
                'posture_tension': 0.8,
                'shoulder_elevation': 0.3,
                'torso_lean': 0.1  # Forward lean
            },
            EmotionCategory.SADNESS: {
                'posture_tension': 0.2,
                'shoulder_elevation': -0.1,  # Dropped shoulders
                'torso_lean': -0.05  # Slight backward lean
            },
            EmotionCategory.FEAR: {
                'posture_tension': 0.7,
                'shoulder_elevation': 0.4,
                'torso_lean': -0.1  # Backward lean
            },
            EmotionCategory.SURPRISE: {
                'posture_tension': 0.6,
                'shoulder_elevation': 0.2,
                'torso_lean': 0.0
            },
            EmotionCategory.DISGUST: {
                'posture_tension': 0.6,
                'shoulder_elevation': 0.1,
                'torso_lean': -0.05  # Slight backward lean
            },
            EmotionCategory.EXCITEMENT: {
                'posture_tension': 0.7,
                'shoulder_elevation': 0.2,
                'torso_lean': 0.1  # Forward lean
            },
            EmotionCategory.NEUTRAL: {
                'posture_tension': 0.5,
                'shoulder_elevation': 0.0,
                'torso_lean': 0.0
            }
        }
    
    def get_posture_modulation(self, emotion_category: EmotionCategory, 
                             intensity: float) -> Dict[str, float]:
        """
        Get posture modulation parameters for an emotion.
        
        Args:
            emotion_category: The discrete emotion category
            intensity: Emotion intensity (0.0 to 1.0)
            
        Returns:
            Dictionary of posture modulation parameters
        """
        if emotion_category not in self.posture_profiles:
            emotion_category = EmotionCategory.NEUTRAL
        
        base_profile = self.posture_profiles[emotion_category]
        neutral_profile = self.posture_profiles[EmotionCategory.NEUTRAL]
        
        # Scale modulation by intensity
        modulated_profile = {}
        for param, base_value in base_profile.items():
            neutral_value = neutral_profile[param]
            # Interpolate between neutral and emotional value based on intensity
            modulated_value = neutral_value + (base_value - neutral_value) * intensity
            modulated_profile[param] = modulated_value
        
        return modulated_profile


class SigningModulationService:
    """Main service for emotion-to-signing modulation."""
    
    def __init__(self):
        self.facs_mapper = FACSExpressionMapper()
        self.temporal_modulator = TemporalModulator()
        self.intensity_scaler = IntensityScaler()
        self.posture_modulator = PostureModulator()
    
    def generate_signing_modulation(self, 
                                  emotion_vector: EmotionVector,
                                  discrete_emotion: str,
                                  emotion_intensity: float,
                                  context: Optional[Dict[str, Any]] = None) -> SigningModulationParameters:
        """
        Generate comprehensive signing modulation parameters from emotion analysis.
        
        Args:
            emotion_vector: Emotion in VAD space
            discrete_emotion: Discrete emotion category
            emotion_intensity: Overall emotion intensity
            context: Optional context information
            
        Returns:
            Complete signing modulation parameters
        """
        # Convert discrete emotion to enum
        try:
            emotion_category = EmotionCategory(discrete_emotion.lower())
        except ValueError:
            emotion_category = EmotionCategory.NEUTRAL
        
        # Get modulation parameters from each component
        temporal_params = self.temporal_modulator.get_temporal_modulation(
            emotion_category, emotion_intensity
        )
        
        intensity_params = self.intensity_scaler.get_intensity_scaling(
            emotion_category, emotion_intensity
        )
        
        posture_params = self.posture_modulator.get_posture_modulation(
            emotion_category, emotion_intensity
        )
        
        # Generate FACS Action Units
        facs_aus = self.facs_mapper.map_emotion_to_facs(
            emotion_category, emotion_intensity
        )
        
        # Calculate facial expression intensity from FACS
        if facs_aus:
            facial_intensity = np.mean([au.intensity for au in facs_aus])
        else:
            facial_intensity = 0.5
        
        # Create comprehensive modulation parameters
        modulation = SigningModulationParameters(
            # Intensity and amplitude
            intensity_scale=intensity_params['intensity_scale'],
            gesture_amplitude_scale=intensity_params['gesture_amplitude_scale'],
            
            # Temporal modulation
            speed_multiplier=temporal_params['speed_multiplier'],
            pause_duration_scale=temporal_params['pause_duration_scale'],
            transition_smoothness=temporal_params['transition_smoothness'],
            
            # Spatial modulation
            gesture_size_scale=intensity_params['gesture_size_scale'],
            signing_space_scale=intensity_params['signing_space_scale'],
            hand_separation_scale=intensity_params['hand_separation_scale'],
            
            # Facial expression
            facial_expression_intensity=facial_intensity,
            facs_action_units=facs_aus,
            
            # Body posture
            posture_tension=posture_params['posture_tension'],
            shoulder_elevation=posture_params['shoulder_elevation'],
            torso_lean=posture_params['torso_lean'],
            
            # Rhythm and dynamics
            rhythm_regularity=temporal_params['rhythm_regularity'],
            dynamic_range=temporal_params['dynamic_range'],
            accent_strength=temporal_params['accent_strength']
        )
        
        return modulation
    
    def apply_contextual_adjustments(self, 
                                   modulation: SigningModulationParameters,
                                   context: Dict[str, Any]) -> SigningModulationParameters:
        """
        Apply contextual adjustments to modulation parameters.
        
        Args:
            modulation: Base modulation parameters
            context: Context information (urgency, formality, etc.)
            
        Returns:
            Adjusted modulation parameters
        """
        adjusted = modulation
        
        # Handle urgency context
        urgency = context.get('urgency', 0.0)  # 0.0 to 1.0
        if urgency > 0.5:
            adjusted.speed_multiplier *= (1.0 + urgency * 0.3)
            adjusted.intensity_scale *= (1.0 + urgency * 0.2)
            adjusted.accent_strength *= (1.0 + urgency * 0.4)
        
        # Handle formality context
        formality = context.get('formality', 0.5)  # 0.0 (casual) to 1.0 (formal)
        if formality > 0.7:
            # More formal signing
            adjusted.gesture_size_scale *= 0.9
            adjusted.speed_multiplier *= 0.95
            adjusted.facial_expression_intensity *= 0.8
        elif formality < 0.3:
            # More casual signing
            adjusted.gesture_size_scale *= 1.1
            adjusted.dynamic_range *= 1.2
        
        # Handle audience size context
        audience_size = context.get('audience_size', 1)
        if audience_size > 10:
            # Larger gestures for bigger audience
            adjusted.gesture_size_scale *= 1.2
            adjusted.signing_space_scale *= 1.1
            adjusted.intensity_scale *= 1.1
        
        return adjusted
    
    def blend_emotions(self, 
                      primary_emotion: Tuple[str, float],
                      secondary_emotion: Tuple[str, float],
                      blend_ratio: float = 0.7) -> SigningModulationParameters:
        """
        Blend modulation parameters for mixed emotions.
        
        Args:
            primary_emotion: (emotion_name, intensity) tuple
            secondary_emotion: (emotion_name, intensity) tuple
            blend_ratio: Ratio of primary to secondary emotion
            
        Returns:
            Blended modulation parameters
        """
        # Generate modulation for each emotion
        primary_modulation = self.generate_signing_modulation(
            EmotionVector(0, 0, 0),  # Placeholder
            primary_emotion[0],
            primary_emotion[1]
        )
        
        secondary_modulation = self.generate_signing_modulation(
            EmotionVector(0, 0, 0),  # Placeholder
            secondary_emotion[0],
            secondary_emotion[1]
        )
        
        # Blend parameters
        blended = SigningModulationParameters()
        
        # Blend scalar parameters
        scalar_params = [
            'intensity_scale', 'gesture_amplitude_scale', 'speed_multiplier',
            'pause_duration_scale', 'transition_smoothness', 'gesture_size_scale',
            'signing_space_scale', 'hand_separation_scale', 'facial_expression_intensity',
            'posture_tension', 'shoulder_elevation', 'torso_lean',
            'rhythm_regularity', 'dynamic_range', 'accent_strength'
        ]
        
        for param in scalar_params:
            primary_val = getattr(primary_modulation, param)
            secondary_val = getattr(secondary_modulation, param)
            blended_val = primary_val * blend_ratio + secondary_val * (1 - blend_ratio)
            setattr(blended, param, blended_val)
        
        # Blend FACS Action Units
        blended.facs_action_units = self.facs_mapper.blend_facs_expressions(
            primary_modulation.facs_action_units,
            secondary_modulation.facs_action_units,
            blend_ratio
        )
        
        return blended
    
    def get_modulation_summary(self, modulation: SigningModulationParameters) -> Dict[str, Any]:
        """
        Get a summary of modulation parameters for debugging/monitoring.
        
        Args:
            modulation: Modulation parameters
            
        Returns:
            Summary dictionary
        """
        return {
            'intensity_scale': modulation.intensity_scale,
            'speed_multiplier': modulation.speed_multiplier,
            'gesture_size_scale': modulation.gesture_size_scale,
            'facial_expression_intensity': modulation.facial_expression_intensity,
            'signing_space_scale': modulation.signing_space_scale,
            'active_facs_aus': len(modulation.facs_action_units),
            'facs_au_numbers': [au.au_number for au in modulation.facs_action_units],
            'posture_tension': modulation.posture_tension,
            'dynamic_range': modulation.dynamic_range
        }