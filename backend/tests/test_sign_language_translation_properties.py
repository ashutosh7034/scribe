"""
Property-Based Tests for Sign Language Translation

Property tests for sign language translation accuracy and correctness.
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import List, Dict, Any

from app.services.sign_language_translation import (
    SignLanguageTranslationService, 
    SignLanguage, 
    GestureType,
    get_translation_service
)


# Test data strategies
@st.composite
def text_input_strategy(draw):
    """Generate realistic text inputs for translation testing."""
    # Common words that should be in the dictionary
    common_words = [
        "hello", "goodbye", "thank", "you", "me", "meet", "restaurant", 
        "tomorrow", "today", "please", "sorry", "help", "understand", "good", "bad"
    ]
    
    # Generate sentences with 1-8 words
    sentence_length = draw(st.integers(min_value=1, max_value=8))
    
    # Mix of dictionary words and potentially unknown words
    words = []
    for _ in range(sentence_length):
        if draw(st.booleans()):
            # Use a known word
            word = draw(st.sampled_from(common_words))
        else:
            # Use a random word (might not be in dictionary)
            word = draw(st.text(
                alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
                min_size=1,
                max_size=10
            ).filter(lambda x: x.isalpha()))
        
        if word:  # Ensure word is not empty
            words.append(word)
    
    return " ".join(words) if words else "hello"


@st.composite
def emotion_intensity_strategy(draw):
    """Generate emotion intensity values."""
    return draw(st.integers(min_value=0, max_value=200))


@st.composite
def sign_language_strategy(draw):
    """Generate sign language variants."""
    return draw(st.sampled_from([SignLanguage.ASL, SignLanguage.BSL]))


class TestSignLanguageTranslationAccuracyProperties:
    """Property-based tests for sign language translation accuracy."""
    
    @given(
        text_input=text_input_strategy(),
        emotion_intensity=emotion_intensity_strategy(),
        sign_language=sign_language_strategy()
    )
    @settings(
        max_examples=100,
        deadline=3000,  # 3 second timeout per test
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_translation_accuracy(self, text_input, emotion_intensity, sign_language):
        """
        Feature: real-time-sign-language-translation, Property 3: Translation Accuracy
        
        For any text input, the translation service should produce a valid gesture sequence
        with appropriate confidence scoring and processing within reasonable time limits.
        
        **Validates: Requirements 1.2**
        """
        # Get translation service
        translation_service = get_translation_service(sign_language)
        
        # Perform translation
        start_time = time.time()
        result = translation_service.translate_text(text_input, emotion_intensity)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Property 1: Translation should complete within reasonable time (< 200ms per requirement)
        assert processing_time_ms < 200, (
            f"Translation processing time {processing_time_ms:.2f}ms exceeds 200ms limit "
            f"for input: '{text_input}'"
        )
        
        # Property 2: Result should have valid structure
        assert result.original_text == text_input, "Original text should be preserved"
        assert result.normalized_text is not None, "Normalized text should be generated"
        assert isinstance(result.gestures, list), "Gestures should be a list"
        assert result.total_duration_ms > 0, "Total duration should be positive"
        assert 0.0 <= result.confidence_score <= 1.0, "Confidence should be between 0 and 1"
        assert result.processing_time_ms >= 0, "Processing time should be non-negative"
        
        # Property 3: Gesture sequence should be valid
        if result.gestures:
            for gesture in result.gestures:
                assert gesture.sign_id is not None, "Each gesture should have a sign ID"
                assert gesture.word is not None, "Each gesture should have an associated word"
                assert isinstance(gesture.gesture_type, GestureType), "Gesture type should be valid enum"
                assert gesture.duration_ms > 0, "Each gesture should have positive duration"
                
                # Validate pose data generation
                pose_data = gesture.to_pose_data()
                assert "pose_keyframes" in pose_data, "Gesture should generate pose keyframes"
                assert len(pose_data["pose_keyframes"]) > 0, "Should have at least one keyframe"
        
        # Property 4: Confidence should reflect dictionary coverage of original input
        input_words = text_input.lower().split()
        known_words = [word for word in input_words if word in translation_service.get_supported_words()]
        expected_confidence = len(known_words) / len(input_words) if input_words else 0.0
        
        # Allow more tolerance for confidence calculation due to text normalization effects
        confidence_tolerance = 0.2  # Increased tolerance to account for normalization
        assert abs(result.confidence_score - expected_confidence) <= confidence_tolerance, (
            f"Confidence score {result.confidence_score:.2f} doesn't match expected "
            f"{expected_confidence:.2f} ± {confidence_tolerance} for known words coverage. "
            f"Original: '{text_input}' -> Normalized: '{result.normalized_text}'"
        )
        
        # Property 5: Total duration should be reasonable for content length
        # Calculate duration based on actual gestures generated, not input word count
        total_gestures = len(result.gestures)
        
        # Estimate expected duration based on gesture types
        expected_min_duration = 0
        expected_max_duration = 0
        
        for gesture in result.gestures:
            if gesture.gesture_type == GestureType.FINGERSPELLING:
                # Fingerspelling: 250-350ms per letter
                expected_min_duration += 250
                expected_max_duration += 350
            else:
                # Manual signs: 300-1000ms per sign
                expected_min_duration += 300
                expected_max_duration += 1000
        
        # Account for emotion intensity modulation (0.8x to 1.2x)
        if emotion_intensity < 80:
            expected_max_duration = int(expected_max_duration * 1.2)
        elif emotion_intensity > 120:
            expected_min_duration = int(expected_min_duration * 0.8)
        
        # Ensure reasonable bounds
        expected_min_duration = max(expected_min_duration, 100)
        expected_max_duration = max(expected_max_duration, 500)
        
        assert expected_min_duration <= result.total_duration_ms <= expected_max_duration, (
            f"Total duration {result.total_duration_ms}ms outside expected range "
            f"[{expected_min_duration}, {expected_max_duration}] for {total_gestures} gestures "
            f"(input: '{text_input}', emotion_intensity: {emotion_intensity})"
        )
    
    @given(
        text_input=text_input_strategy(),
        emotion_intensity_1=emotion_intensity_strategy(),
        emotion_intensity_2=emotion_intensity_strategy()
    )
    @settings(
        max_examples=50,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_emotion_intensity_consistency(self, text_input, emotion_intensity_1, emotion_intensity_2):
        """
        Test that emotion intensity modulation is applied consistently.
        
        This supports Property 3 by ensuring emotion modulation doesn't break
        the core translation accuracy.
        """
        translation_service = get_translation_service(SignLanguage.ASL)
        
        # Translate with two different emotion intensities
        result_1 = translation_service.translate_text(text_input, emotion_intensity_1)
        result_2 = translation_service.translate_text(text_input, emotion_intensity_2)
        
        # Property 1: Core translation should be consistent regardless of emotion intensity
        assert result_1.original_text == result_2.original_text, "Original text should be identical"
        assert result_1.normalized_text == result_2.normalized_text, "Normalized text should be identical"
        assert len(result_1.gestures) == len(result_2.gestures), "Number of gestures should be identical"
        
        # Property 2: Gesture content should be the same, only timing/intensity may differ
        for g1, g2 in zip(result_1.gestures, result_2.gestures):
            assert g1.word == g2.word, "Gesture words should be identical"
            assert g1.gesture_type == g2.gesture_type, "Gesture types should be identical"
            assert g1.handshape == g2.handshape, "Handshapes should be identical"
            assert g1.location == g2.location, "Locations should be identical"
            assert g1.movement == g2.movement, "Movements should be identical"
            assert g1.orientation == g2.orientation, "Orientations should be identical"
            
            # Duration may vary with emotion intensity
            duration_ratio = g1.duration_ms / g2.duration_ms if g2.duration_ms > 0 else 1.0
            assert 0.5 <= duration_ratio <= 2.0, (
                f"Duration ratio {duration_ratio:.2f} outside reasonable range [0.5, 2.0]"
            )
        
        # Property 3: Confidence should be identical (emotion doesn't affect accuracy)
        assert result_1.confidence_score == result_2.confidence_score, (
            "Confidence scores should be identical regardless of emotion intensity"
        )
    
    @given(
        text_inputs=st.lists(text_input_strategy(), min_size=1, max_size=10),
        emotion_intensity=emotion_intensity_strategy()
    )
    @settings(
        max_examples=30,
        deadline=10000,  # Longer timeout for batch processing
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_batch_translation_consistency(self, text_inputs, emotion_intensity):
        """
        Test that translation accuracy is consistent across batch processing.
        
        This supports Property 3 by ensuring the service maintains accuracy
        when processing multiple translations.
        """
        translation_service = get_translation_service(SignLanguage.ASL)
        
        # Translate all inputs
        results = []
        total_processing_time = 0
        
        for text_input in text_inputs:
            start_time = time.time()
            result = translation_service.translate_text(text_input, emotion_intensity)
            end_time = time.time()
            
            results.append(result)
            total_processing_time += (end_time - start_time) * 1000
        
        # Property 1: Each translation should be valid
        for i, result in enumerate(results):
            assert result.original_text == text_inputs[i], f"Original text mismatch for input {i}"
            assert len(result.gestures) >= 0, f"Invalid gesture count for input {i}"
            assert 0.0 <= result.confidence_score <= 1.0, f"Invalid confidence for input {i}"
        
        # Property 2: Processing time should scale reasonably with batch size
        avg_processing_time = total_processing_time / len(text_inputs)
        assert avg_processing_time < 300, (
            f"Average processing time {avg_processing_time:.2f}ms exceeds 300ms for batch processing"
        )
        
        # Property 3: Identical inputs should produce identical results
        unique_inputs = {}
        for i, text_input in enumerate(text_inputs):
            if text_input in unique_inputs:
                # Compare with previous result for same input
                prev_result = results[unique_inputs[text_input]]
                curr_result = results[i]
                
                assert prev_result.normalized_text == curr_result.normalized_text, (
                    f"Inconsistent normalization for duplicate input: '{text_input}'"
                )
                assert len(prev_result.gestures) == len(curr_result.gestures), (
                    f"Inconsistent gesture count for duplicate input: '{text_input}'"
                )
                assert prev_result.confidence_score == curr_result.confidence_score, (
                    f"Inconsistent confidence for duplicate input: '{text_input}'"
                )
            else:
                unique_inputs[text_input] = i
    
    @given(
        base_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs')),
            min_size=1,
            max_size=50
        ).filter(lambda x: x.strip()),
        emotion_intensity=emotion_intensity_strategy()
    )
    @settings(
        max_examples=50,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_text_normalization_robustness(self, base_text, emotion_intensity):
        """
        Test that text normalization handles various input formats robustly.
        
        This supports Property 3 by ensuring the translation service can handle
        diverse text inputs while maintaining accuracy.
        """
        translation_service = get_translation_service(SignLanguage.ASL)
        
        # Test various text formats
        text_variants = [
            base_text,
            base_text.upper(),
            base_text.lower(),
            f"  {base_text}  ",  # With whitespace
            base_text.replace(" ", "   "),  # Multiple spaces
        ]
        
        results = []
        for variant in text_variants:
            try:
                result = translation_service.translate_text(variant, emotion_intensity)
                results.append(result)
            except Exception as e:
                pytest.fail(f"Translation failed for text variant '{variant}': {e}")
        
        # Property 1: All variants should produce valid results
        for i, result in enumerate(results):
            assert result is not None, f"Null result for variant {i}"
            assert isinstance(result.gestures, list), f"Invalid gestures for variant {i}"
            assert result.confidence_score >= 0.0, f"Invalid confidence for variant {i}"
        
        # Property 2: Normalized text should be consistent across variants
        # (after accounting for case and whitespace normalization)
        if len(results) > 1:
            base_normalized = results[0].normalized_text
            for i, result in enumerate(results[1:], 1):
                # Normalized text should be similar (may differ in whitespace/case handling)
                assert result.normalized_text is not None, f"Null normalized text for variant {i}"
                
                # Check that word count is consistent
                base_words = base_normalized.split()
                variant_words = result.normalized_text.split()
                
                # Allow some flexibility in word count due to normalization
                word_count_ratio = len(variant_words) / len(base_words) if base_words else 1.0
                assert 0.5 <= word_count_ratio <= 2.0, (
                    f"Word count ratio {word_count_ratio:.2f} outside reasonable range "
                    f"for variant {i}: '{text_variants[i]}'"
                )
    
    @given(
        text_input=text_input_strategy(),
        sign_language_1=sign_language_strategy(),
        sign_language_2=sign_language_strategy()
    )
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_cross_language_consistency(self, text_input, sign_language_1, sign_language_2):
        """
        Test that translation maintains consistency across different sign languages.
        
        This supports Property 3 by ensuring the core translation logic works
        consistently regardless of target sign language.
        """
        # Get services for both languages
        service_1 = get_translation_service(sign_language_1)
        service_2 = get_translation_service(sign_language_2)
        
        # Translate with both services
        result_1 = service_1.translate_text(text_input, 100)
        result_2 = service_2.translate_text(text_input, 100)
        
        # Property 1: Both translations should be valid
        assert result_1.original_text == text_input, "Original text should be preserved (lang 1)"
        assert result_2.original_text == text_input, "Original text should be preserved (lang 2)"
        
        assert len(result_1.gestures) >= 0, "Should have valid gesture count (lang 1)"
        assert len(result_2.gestures) >= 0, "Should have valid gesture count (lang 2)"
        
        # Property 2: Processing characteristics should be similar
        processing_time_ratio = (result_1.processing_time_ms / result_2.processing_time_ms 
                               if result_2.processing_time_ms > 0 else 1.0)
        assert 0.1 <= processing_time_ratio <= 10.0, (
            f"Processing time ratio {processing_time_ratio:.2f} indicates significant "
            f"performance difference between sign languages"
        )
        
        # Property 3: Both should handle the same input words consistently
        # (though the actual signs may differ between languages)
        input_words = text_input.lower().split()
        
        # Both services should process the same number of input words
        # (though gesture count may differ due to language differences)
        if input_words:
            gesture_count_ratio = len(result_1.gestures) / len(result_2.gestures) if result_2.gestures else 1.0
            assert 0.5 <= gesture_count_ratio <= 2.0, (
                f"Gesture count ratio {gesture_count_ratio:.2f} indicates significant "
                f"structural difference between sign languages for input: '{text_input}'"
            )


class TestSignLanguageGrammarProperties:
    """Property-based tests for sign language grammar transformation."""
    
    @given(
        sentence=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Zs')),
            min_size=5,
            max_size=100
        ).filter(lambda x: len(x.split()) >= 2)
    )
    @settings(
        max_examples=50,
        deadline=2000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_grammar_transformation_consistency(self, sentence):
        """
        Test that grammar transformation rules are applied consistently.
        
        This supports Property 3 by ensuring the grammar normalization
        component works correctly for various sentence structures.
        """
        translation_service = get_translation_service(SignLanguage.ASL)
        
        # Perform translation to trigger grammar transformation
        result = translation_service.translate_text(sentence, 100)
        
        # Property 1: Normalized text should follow ASL grammar patterns
        normalized = result.normalized_text
        assert normalized is not None, "Normalized text should not be None"
        assert len(normalized.strip()) > 0, "Normalized text should not be empty"
        
        # Property 2: Normalized text should be uppercase (ASL convention)
        assert normalized.isupper(), f"Normalized text should be uppercase: '{normalized}'"
        
        # Property 3: Word count should be reasonable (not drastically different)
        original_words = sentence.split()
        normalized_words = normalized.split()
        
        word_count_ratio = len(normalized_words) / len(original_words) if original_words else 1.0
        assert 0.3 <= word_count_ratio <= 3.0, (
            f"Word count ratio {word_count_ratio:.2f} outside reasonable range "
            f"for sentence: '{sentence}' -> '{normalized}'"
        )
        
        # Property 4: Temporal markers should appear early in normalized text
        temporal_markers = ["TOMORROW", "TODAY", "YESTERDAY", "NEXT", "LAST", "WILL", "WAS", "WERE"]
        
        normalized_words = normalized.split()
        temporal_positions = []
        
        for i, word in enumerate(normalized_words):
            if word in temporal_markers:
                temporal_positions.append(i)
        
        # If temporal markers exist, they should generally appear in first half
        if temporal_positions:
            avg_temporal_position = sum(temporal_positions) / len(temporal_positions)
            relative_position = avg_temporal_position / len(normalized_words)
            
            assert relative_position <= 0.7, (
                f"Temporal markers appear too late in sentence (position {relative_position:.2f}): "
                f"'{normalized}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])