"""
Sign Language Translation Service

Implements basic dictionary-based sign language translation with grammar rules
and gesture sequence generation for real-time avatar display.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignLanguage(Enum):
    """Supported sign languages."""
    ASL = "ASL"  # American Sign Language
    BSL = "BSL"  # British Sign Language


class GestureType(Enum):
    """Types of sign language gestures."""
    MANUAL = "manual"  # Hand/arm movements
    FACIAL = "facial"  # Facial expressions
    SPATIAL = "spatial"  # Spatial references
    FINGERSPELLING = "fingerspelling"  # Letter-by-letter spelling


@dataclass
class SignGesture:
    """Represents a single sign language gesture."""
    
    sign_id: str
    word: str
    gesture_type: GestureType
    handshape: str
    location: str
    movement: str
    orientation: str
    duration_ms: int
    spatial_reference: Optional[str] = None
    facial_expression: Optional[str] = None
    
    def to_pose_data(self) -> Dict[str, Any]:
        """Convert gesture to 3D pose data for avatar rendering."""
        return {
            "sign_id": self.sign_id,
            "word": self.word,
            "gesture_type": self.gesture_type.value,
            "pose_keyframes": self._generate_pose_keyframes(),
            "duration_ms": self.duration_ms,
            "spatial_reference": self.spatial_reference,
            "facial_expression": self.facial_expression
        }
    
    def _generate_pose_keyframes(self) -> List[Dict[str, Any]]:
        """Generate basic pose keyframes for the gesture."""
        # Basic pose generation - in a real implementation, this would use
        # the diffusion model or pre-computed pose sequences
        
        base_poses = {
            "neutral": {
                "right_hand": {"x": 0.3, "y": 0.5, "z": 0.0},
                "left_hand": {"x": -0.3, "y": 0.5, "z": 0.0},
                "head": {"x": 0.0, "y": 0.0, "z": 0.0, "rotation": 0.0}
            },
            "chest_level": {
                "right_hand": {"x": 0.2, "y": 0.8, "z": 0.1},
                "left_hand": {"x": -0.2, "y": 0.8, "z": 0.1},
                "head": {"x": 0.0, "y": 0.0, "z": 0.0, "rotation": 0.0}
            },
            "head_level": {
                "right_hand": {"x": 0.15, "y": 1.0, "z": 0.2},
                "left_hand": {"x": -0.15, "y": 1.0, "z": 0.2},
                "head": {"x": 0.0, "y": 0.0, "z": 0.0, "rotation": 0.0}
            }
        }
        
        # Select pose based on location
        if self.location in base_poses:
            pose = base_poses[self.location]
        else:
            pose = base_poses["neutral"]
        
        # Generate keyframes with basic animation
        keyframes = []
        num_frames = max(1, self.duration_ms // 50)  # 20 FPS
        
        for i in range(num_frames):
            progress = i / max(1, num_frames - 1)
            keyframe = {
                "timestamp": i * 50,  # 50ms intervals
                "joints": pose.copy(),
                "progress": progress
            }
            keyframes.append(keyframe)
        
        return keyframes


@dataclass
class TranslationResult:
    """Result of sign language translation."""
    
    original_text: str
    normalized_text: str
    gestures: List[SignGesture]
    total_duration_ms: int
    confidence_score: float
    processing_time_ms: int
    
    def to_pose_sequence(self) -> List[Dict[str, Any]]:
        """Convert all gestures to a complete pose sequence."""
        pose_sequence = []
        current_time = 0
        
        for gesture in self.gestures:
            gesture_poses = gesture.to_pose_data()
            
            # Adjust timestamps to be sequential
            for keyframe in gesture_poses["pose_keyframes"]:
                keyframe["timestamp"] += current_time
                pose_sequence.append(keyframe)
            
            current_time += gesture.duration_ms
        
        return pose_sequence
    
    def to_facial_expressions(self) -> List[Dict[str, Any]]:
        """Extract facial expression sequence."""
        expressions = []
        current_time = 0
        
        for gesture in self.gestures:
            if gesture.facial_expression:
                expressions.append({
                    "timestamp": current_time,
                    "expression": gesture.facial_expression,
                    "intensity": 0.7,
                    "duration_ms": gesture.duration_ms
                })
            current_time += gesture.duration_ms
        
        return expressions


class SignLanguageTranslationService:
    """Service for translating text to sign language gestures."""
    
    def __init__(self, sign_language: SignLanguage = SignLanguage.ASL):
        self.sign_language = sign_language
        self.sign_dictionary = self._load_sign_dictionary()
        self.grammar_rules = self._load_grammar_rules()
        
    def _load_sign_dictionary(self) -> Dict[str, SignGesture]:
        """Load basic sign language dictionary."""
        # Basic ASL dictionary - in production, this would be loaded from a database
        # or comprehensive sign language corpus
        
        basic_signs = {
            # Common words
            "hello": SignGesture(
                sign_id="hello_001",
                word="hello",
                gesture_type=GestureType.MANUAL,
                handshape="open_palm",
                location="head_level",
                movement="wave",
                orientation="palm_out",
                duration_ms=800,
                facial_expression="smile"
            ),
            "goodbye": SignGesture(
                sign_id="goodbye_001",
                word="goodbye",
                gesture_type=GestureType.MANUAL,
                handshape="open_palm",
                location="head_level",
                movement="wave",
                orientation="palm_out",
                duration_ms=1000,
                facial_expression="neutral"
            ),
            "thank": SignGesture(
                sign_id="thank_001",
                word="thank",
                gesture_type=GestureType.MANUAL,
                handshape="flat_hand",
                location="chest_level",
                movement="forward",
                orientation="palm_up",
                duration_ms=600,
                facial_expression="smile"
            ),
            "you": SignGesture(
                sign_id="you_001",
                word="you",
                gesture_type=GestureType.SPATIAL,
                handshape="index_point",
                location="chest_level",
                movement="point",
                orientation="forward",
                duration_ms=400,
                spatial_reference="addressee"
            ),
            "me": SignGesture(
                sign_id="me_001",
                word="me",
                gesture_type=GestureType.SPATIAL,
                handshape="index_point",
                location="chest_level",
                movement="point",
                orientation="toward_self",
                duration_ms=400,
                spatial_reference="self"
            ),
            "meet": SignGesture(
                sign_id="meet_001",
                word="meet",
                gesture_type=GestureType.MANUAL,
                handshape="index_hands",
                location="chest_level",
                movement="approach",
                orientation="toward_each_other",
                duration_ms=800
            ),
            "restaurant": SignGesture(
                sign_id="restaurant_001",
                word="restaurant",
                gesture_type=GestureType.MANUAL,
                handshape="r_handshape",
                location="chest_level",
                movement="circular",
                orientation="palm_down",
                duration_ms=1000
            ),
            "tomorrow": SignGesture(
                sign_id="tomorrow_001",
                word="tomorrow",
                gesture_type=GestureType.SPATIAL,
                handshape="a_handshape",
                location="head_level",
                movement="forward",
                orientation="thumb_forward",
                duration_ms=600,
                spatial_reference="future"
            ),
            "today": SignGesture(
                sign_id="today_001",
                word="today",
                gesture_type=GestureType.MANUAL,
                handshape="y_handshape",
                location="chest_level",
                movement="downward",
                orientation="palm_down",
                duration_ms=500
            ),
            "please": SignGesture(
                sign_id="please_001",
                word="please",
                gesture_type=GestureType.MANUAL,
                handshape="flat_hand",
                location="chest_level",
                movement="circular",
                orientation="palm_on_chest",
                duration_ms=700,
                facial_expression="polite"
            ),
            "sorry": SignGesture(
                sign_id="sorry_001",
                word="sorry",
                gesture_type=GestureType.MANUAL,
                handshape="s_handshape",
                location="chest_level",
                movement="circular",
                orientation="palm_on_chest",
                duration_ms=800,
                facial_expression="apologetic"
            ),
            "help": SignGesture(
                sign_id="help_001",
                word="help",
                gesture_type=GestureType.MANUAL,
                handshape="a_handshape",
                location="chest_level",
                movement="upward",
                orientation="thumb_up",
                duration_ms=600
            ),
            "understand": SignGesture(
                sign_id="understand_001",
                word="understand",
                gesture_type=GestureType.MANUAL,
                handshape="index_point",
                location="head_level",
                movement="flick",
                orientation="upward",
                duration_ms=500,
                facial_expression="comprehension"
            ),
            "good": SignGesture(
                sign_id="good_001",
                word="good",
                gesture_type=GestureType.MANUAL,
                handshape="flat_hand",
                location="chest_level",
                movement="forward",
                orientation="palm_up",
                duration_ms=500,
                facial_expression="positive"
            ),
            "bad": SignGesture(
                sign_id="bad_001",
                word="bad",
                gesture_type=GestureType.MANUAL,
                handshape="flat_hand",
                location="chest_level",
                movement="downward_flip",
                orientation="palm_down",
                duration_ms=600,
                facial_expression="negative"
            )
        }
        
        return basic_signs
    
    def _load_grammar_rules(self) -> Dict[str, Any]:
        """Load sign language grammar transformation rules."""
        return {
            "word_order_rules": {
                # Convert English SVO to ASL topic-comment structure
                "svo_to_topic_comment": True,
                "time_first": True,  # Temporal markers come first
                "location_second": True,  # Location/spatial info comes second
            },
            "spatial_grammar": {
                "pronouns": {
                    "i": "self",
                    "me": "self", 
                    "you": "addressee",
                    "he": "third_person_1",
                    "she": "third_person_1",
                    "they": "third_person_multiple"
                },
                "locations": {
                    "here": "present_location",
                    "there": "distant_location",
                    "restaurant": "established_location"
                }
            },
            "temporal_markers": {
                "future": ["tomorrow", "next", "will", "going to"],
                "past": ["yesterday", "last", "was", "were", "did"],
                "present": ["now", "today", "currently", "am", "is", "are"]
            },
            "classifiers": {
                "person": "person_classifier",
                "vehicle": "vehicle_classifier", 
                "building": "building_classifier",
                "animal": "animal_classifier"
            }
        }
    
    def translate_text(self, text: str, emotion_intensity: int = 100) -> TranslationResult:
        """
        Translate text to sign language gestures.
        
        Args:
            text: Input text to translate
            emotion_intensity: Emotion intensity percentage (0-200)
            
        Returns:
            TranslationResult with gesture sequence and metadata
        """
        import time
        start_time = time.time()
        
        # Step 1: Normalize text for sign language grammar
        normalized_text = self._normalize_text_for_sign_grammar(text)
        
        # Step 2: Generate gesture sequence
        gestures = self._generate_gesture_sequence(normalized_text, emotion_intensity)
        
        # Step 3: Calculate metrics
        total_duration = sum(gesture.duration_ms for gesture in gestures)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Step 4: Calculate confidence score based on dictionary coverage
        words = normalized_text.lower().split()
        known_words = sum(1 for word in words if word in self.sign_dictionary)
        confidence = known_words / len(words) if words else 0.0
        
        return TranslationResult(
            original_text=text,
            normalized_text=normalized_text,
            gestures=gestures,
            total_duration_ms=total_duration,
            confidence_score=confidence,
            processing_time_ms=processing_time
        )
    
    def _normalize_text_for_sign_grammar(self, text: str) -> str:
        """
        Normalize English text to follow sign language grammar rules.
        
        Converts English SVO structure to ASL topic-comment structure.
        """
        # Basic text preprocessing
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        
        if not words:
            return ""
        
        # Apply grammar transformation rules
        normalized_words = []
        temporal_words = []
        spatial_words = []
        main_words = []
        
        # Separate words by type
        for word in words:
            if self._is_temporal_marker(word):
                temporal_words.append(word)
            elif self._is_spatial_marker(word):
                spatial_words.append(word)
            else:
                main_words.append(word)
        
        # Reconstruct in ASL order: TIME + LOCATION + TOPIC + COMMENT
        normalized_words.extend(temporal_words)
        normalized_words.extend(spatial_words)
        
        # Apply topic-comment structure to remaining words
        if len(main_words) >= 3:
            # Try to identify SVO pattern and convert
            subject, verb, obj = main_words[0], main_words[1], main_words[2:]
            
            # ASL: TOPIC (object/location) + COMMENT (subject + verb)
            if obj:
                normalized_words.extend(obj)  # Object becomes topic
            normalized_words.append(subject)  # Subject
            normalized_words.append(verb)    # Verb
            
            # Add remaining words
            if len(main_words) > 3:
                normalized_words.extend(main_words[3:])
        else:
            # For shorter phrases, maintain order but apply spatial grammar
            normalized_words.extend(main_words)
        
        return " ".join(normalized_words).upper()
    
    def _is_temporal_marker(self, word: str) -> bool:
        """Check if word is a temporal marker."""
        temporal_words = []
        for markers in self.grammar_rules["temporal_markers"].values():
            temporal_words.extend(markers)
        return word.lower() in temporal_words
    
    def _is_spatial_marker(self, word: str) -> bool:
        """Check if word is a spatial marker."""
        spatial_words = list(self.grammar_rules["spatial_grammar"]["locations"].keys())
        return word.lower() in spatial_words
    
    def _generate_gesture_sequence(self, normalized_text: str, emotion_intensity: int) -> List[SignGesture]:
        """
        Generate sequence of sign language gestures from normalized text.
        """
        words = normalized_text.lower().split()
        gestures = []
        
        for word in words:
            if word in self.sign_dictionary:
                # Use dictionary sign
                gesture = self.sign_dictionary[word]
                
                # Apply emotion intensity modulation
                modulated_gesture = self._apply_emotion_modulation(gesture, emotion_intensity)
                gestures.append(modulated_gesture)
                
            else:
                # Use fingerspelling for unknown words
                fingerspelled_gestures = self._generate_fingerspelling(word)
                gestures.extend(fingerspelled_gestures)
        
        return gestures
    
    def _apply_emotion_modulation(self, gesture: SignGesture, emotion_intensity: int) -> SignGesture:
        """
        Apply emotion intensity modulation to gesture.
        
        Args:
            gesture: Base gesture to modulate
            emotion_intensity: Intensity percentage (0-200)
            
        Returns:
            Modulated gesture with adjusted timing and expression
        """
        # Create a copy of the gesture
        modulated = SignGesture(
            sign_id=gesture.sign_id,
            word=gesture.word,
            gesture_type=gesture.gesture_type,
            handshape=gesture.handshape,
            location=gesture.location,
            movement=gesture.movement,
            orientation=gesture.orientation,
            duration_ms=gesture.duration_ms,
            spatial_reference=gesture.spatial_reference,
            facial_expression=gesture.facial_expression
        )
        
        # Apply intensity modulation
        intensity_factor = emotion_intensity / 100.0
        
        # Adjust duration based on intensity
        if intensity_factor > 1.2:
            # High intensity: faster, more emphatic
            modulated.duration_ms = int(gesture.duration_ms * 0.8)
            if modulated.facial_expression:
                modulated.facial_expression = f"intense_{modulated.facial_expression}"
        elif intensity_factor < 0.8:
            # Low intensity: slower, more subdued
            modulated.duration_ms = int(gesture.duration_ms * 1.2)
            if modulated.facial_expression:
                modulated.facial_expression = f"subdued_{modulated.facial_expression}"
        
        return modulated
    
    def _generate_fingerspelling(self, word: str) -> List[SignGesture]:
        """
        Generate fingerspelling gestures for unknown words.
        """
        gestures = []
        
        for i, letter in enumerate(word):
            if letter.isalpha():
                gesture = SignGesture(
                    sign_id=f"fingerspell_{letter}",
                    word=letter,
                    gesture_type=GestureType.FINGERSPELLING,
                    handshape=f"{letter.lower()}_handshape",
                    location="chest_level",
                    movement="static",
                    orientation="palm_out",
                    duration_ms=300,
                    spatial_reference=None,
                    facial_expression=None
                )
                gestures.append(gesture)
        
        return gestures
    
    def get_supported_words(self) -> List[str]:
        """Get list of words supported by the sign dictionary."""
        return list(self.sign_dictionary.keys())
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation service statistics."""
        return {
            "sign_language": self.sign_language.value,
            "dictionary_size": len(self.sign_dictionary),
            "supported_gesture_types": [gt.value for gt in GestureType],
            "grammar_rules_loaded": len(self.grammar_rules)
        }


# Global service instance
_translation_service = None


def get_translation_service(sign_language: SignLanguage = SignLanguage.ASL) -> SignLanguageTranslationService:
    """Get or create the global translation service instance."""
    global _translation_service
    
    if _translation_service is None or _translation_service.sign_language != sign_language:
        _translation_service = SignLanguageTranslationService(sign_language)
    
    return _translation_service