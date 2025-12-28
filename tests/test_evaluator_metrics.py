"""
Unit tests for new evaluator metrics: ContradictionDetection, InformationPreservation, HallucinationDetection
"""

import pytest
from unittest.mock import patch, MagicMock

from src.evaluator import (
    ContradictionDetection,
    InformationPreservation,
    HallucinationDetection,
    EvaluationScore,
    EvaluationError,
)


class TestContradictionDetection:
    """Test ContradictionDetection metric"""

    @pytest.fixture
    def contradiction_detector(self):
        """Create ContradictionDetection instance for testing"""
        return ContradictionDetection()

    def test_initialization(self, contradiction_detector):
        """Test proper initialization of ContradictionDetection"""
        assert hasattr(contradiction_detector, "medical_knowledge_base")
        assert hasattr(contradiction_detector, "contradiction_patterns")
        assert isinstance(contradiction_detector.medical_knowledge_base, dict)
        assert isinstance(contradiction_detector.contradiction_patterns, list)

    def test_no_contradictions_returns_high_score(self, contradiction_detector):
        """Test that text without contradictions returns high score"""
        clean_text = "Hypertension is treated with lifestyle changes and medication as prescribed by a doctor."
        score = contradiction_detector.calculate(clean_text, "patient")
        assert score == 1.0

    def test_pattern_based_contradiction_detection(self, contradiction_detector):
        """Test detection of pattern-based contradictions"""
        contradictory_text = "You can treat viral infections with antibiotics effectively."
        score = contradiction_detector.calculate(contradictory_text, "patient")
        assert score < 1.0  # Should detect contradiction about antibiotics treating viruses

    def test_knowledge_base_contradiction_detection(self, contradiction_detector):
        """Test detection of contradictions against knowledge base"""
        contradictory_text = "High blood pressure is not related to heart disease or stroke."
        score = contradiction_detector.calculate(contradictory_text, "patient")
        assert score < 1.0  # Should detect contradiction about hypertension consequences

    def test_empty_text_returns_neutral_score(self, contradiction_detector):
        """Test that empty text returns neutral score"""
        score = contradiction_detector.calculate("", "patient")
        assert score == 0.5

        score = contradiction_detector.calculate("   ", "patient")
        assert score == 0.5

    def test_multiple_contradictions_lower_score(self, contradiction_detector):
        """Test that multiple contradictions result in lower scores"""
        single_contradiction = "Antibiotics treat viral infections."
        multiple_contradictions = (
            "Antibiotics treat viral infections. "
            "You should stop medication immediately when you feel better. "
            "Aspirin is safe for everyone to use daily."
        )

        single_score = contradiction_detector.calculate(single_contradiction, "patient")
        multiple_score = contradiction_detector.calculate(multiple_contradictions, "patient")

        assert multiple_score < single_score

    def test_case_insensitive_detection(self, contradiction_detector):
        """Test that contradiction detection is case-insensitive"""
        upper_text = "ANTIBIOTICS TREAT VIRUS INFECTIONS"
        lower_text = "antibiotics treat virus infections"

        upper_score = contradiction_detector.calculate(upper_text, "patient")
        lower_score = contradiction_detector.calculate(lower_text, "patient")

        assert upper_score == lower_score
        assert upper_score < 1.0


class TestInformationPreservation:
    """Test InformationPreservation metric"""

    @pytest.fixture
    def info_preservation(self):
        """Create InformationPreservation instance for testing"""
        return InformationPreservation()

    def test_initialization(self, info_preservation):
        """Test proper initialization of InformationPreservation"""
        assert hasattr(info_preservation, "critical_info_patterns")
        assert isinstance(info_preservation.critical_info_patterns, dict)
        assert "dosages" in info_preservation.critical_info_patterns
        assert "warnings" in info_preservation.critical_info_patterns

    def test_perfect_preservation_returns_high_score(self, info_preservation):
        """Test that perfect information preservation returns high score"""
        original = "Take 10 mg twice daily with food. Do not drink alcohol while taking this medication."
        generated = "You should take 10 mg two times per day with food. Avoid alcohol while on this medication."

        score = info_preservation.calculate(generated, "patient", original=original)
        assert score > 0.5  # Should preserve most critical information

    def test_dosage_preservation(self, info_preservation):
        """Test preservation of dosage information"""
        original = "Take 20 mg once daily before breakfast."
        generated_good = "Take 20 mg one time each day before breakfast."
        generated_bad = "Take medication before breakfast."

        good_score = info_preservation.calculate(generated_good, "patient", original=original)
        bad_score = info_preservation.calculate(generated_bad, "patient", original=original)

        assert good_score > bad_score

    def test_warning_preservation(self, info_preservation):
        """Test preservation of warning information"""
        original = "Do not take with alcohol. Avoid driving. Side effects may include dizziness."
        generated_good = "Don't drink alcohol. Be careful driving. May cause dizziness."
        generated_bad = "Take as directed."

        good_score = info_preservation.calculate(generated_good, "patient", original=original)
        bad_score = info_preservation.calculate(generated_bad, "patient", original=original)

        assert good_score > bad_score

    def test_timing_preservation(self, info_preservation):
        """Test preservation of timing information"""
        original = "Take before meals with water on empty stomach."
        generated_good = "Take before eating with water when stomach is empty."
        generated_bad = "Take with water."

        good_score = info_preservation.calculate(generated_good, "patient", original=original)
        bad_score = info_preservation.calculate(generated_bad, "patient", original=original)

        assert good_score > bad_score

    def test_empty_texts_return_zero(self, info_preservation):
        """Test that empty texts return zero score"""
        score = info_preservation.calculate("", "patient", original="Some content")
        assert score == 0.0

        score = info_preservation.calculate("Some content", "patient", original="")
        assert score == 0.0

    def test_no_critical_info_returns_high_score(self, info_preservation):
        """Test that texts without critical info return high score"""
        original = "This is general medical information about health."
        generated = "This discusses general health topics."

        score = info_preservation.calculate(generated, "patient", original=original)
        assert score == 1.0  # No critical info to preserve

    def test_paraphrased_preservation_detection(self, info_preservation):
        """Test detection of paraphrased information preservation"""
        original = "Take 5 mg tablets twice daily"
        generated = "Take the dose two times each day"  # Paraphrased but preserves key info

        score = info_preservation.calculate(generated, "patient", original=original)
        assert score > 0.0  # Should detect some preservation through paraphrasing


class TestHallucinationDetection:
    """Test HallucinationDetection metric"""

    @pytest.fixture
    def hallucination_detector(self):
        """Create HallucinationDetection instance for testing"""
        return HallucinationDetection()

    def test_initialization(self, hallucination_detector):
        """Test proper initialization of HallucinationDetection"""
        assert hasattr(hallucination_detector, "medical_entities")
        assert isinstance(hallucination_detector.medical_entities, dict)
        assert "medications" in hallucination_detector.medical_entities
        assert "conditions" in hallucination_detector.medical_entities

    def test_no_hallucinations_returns_high_score(self, hallucination_detector):
        """Test that text without hallucinations returns high score"""
        original = "Patient has hypertension and takes aspirin daily."
        generated = "The patient has high blood pressure and takes aspirin every day."

        score = hallucination_detector.calculate(generated, "patient", original=original)
        assert score >= 0.8  # Should be high since no new entities are introduced

    def test_hallucinated_medication_detection(self, hallucination_detector):
        """Test detection of hallucinated medications"""
        original = "Patient has headache."
        generated = "Patient has headache and should take ibuprofen and metformin."  # Metformin not related to headache

        score = hallucination_detector.calculate(generated, "patient", original=original)
        assert score < 1.0  # Should detect hallucinated medications

    def test_hallucinated_condition_detection(self, hallucination_detector):
        """Test detection of hallucinated medical conditions"""
        original = "Patient reports fatigue."
        generated = "Patient has diabetes and hypertension causing fatigue."  # New conditions not in original

        score = hallucination_detector.calculate(generated, "patient", original=original)
        assert score < 1.0  # Should detect hallucinated conditions

    def test_empty_texts_return_neutral_score(self, hallucination_detector):
        """Test that empty texts return neutral score"""
        score = hallucination_detector.calculate("", "patient", original="Some content")
        assert score == 0.5

        score = hallucination_detector.calculate("Some content", "patient", original="")
        assert score == 0.5

    def test_no_entities_returns_high_score(self, hallucination_detector):
        """Test that text without medical entities returns high score"""
        original = "This is general health advice."
        generated = "This provides general health guidance."

        score = hallucination_detector.calculate(generated, "patient", original=original)
        assert score == 1.0  # No entities, so no hallucinations

    def test_entity_extraction_methods(self, hallucination_detector):
        """Test different entity extraction methods"""
        text = "Patient has diabetes and takes insulin and aspirin."
        entities = hallucination_detector._extract_medical_entities(text)

        # Should extract known medical entities
        assert "diabetes" in entities
        assert "insulin" in entities
        assert "aspirin" in entities

    @patch("src.evaluator.spacy")
    def test_spacy_ner_integration(self, mock_spacy, hallucination_detector):
        """Test spaCy NER integration when available"""
        # Mock spaCy NLP pipeline
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "custom_medication"
        mock_ent.label_ = "PRODUCT"
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        # Create detector with mocked spaCy
        detector = HallucinationDetection()
        detector.nlp = mock_nlp

        text = "Patient takes custom_medication"
        entities = detector._extract_medical_entities(text)

        # Should include spaCy-detected entities if they match medical terms
        mock_nlp.assert_called_once_with(text)

    def test_case_insensitive_entity_extraction(self, hallucination_detector):
        """Test that entity extraction is case-insensitive"""
        text_upper = "PATIENT HAS DIABETES AND TAKES INSULIN"
        text_lower = "patient has diabetes and takes insulin"

        entities_upper = hallucination_detector._extract_medical_entities(text_upper)
        entities_lower = hallucination_detector._extract_medical_entities(text_lower)

        assert entities_upper == entities_lower
        assert "diabetes" in entities_upper
        assert "insulin" in entities_upper


class TestEvaluatorIntegration:
    """Integration tests for new metrics with MedExplainEvaluator"""

    def test_evaluation_score_with_new_metrics(self):
        """Test that EvaluationScore includes new metrics"""
        score = EvaluationScore(
            readability=0.8,
            terminology=0.7,
            safety=0.9,
            coverage=0.8,
            quality=0.7,
            contradiction=0.9,
            information_preservation=0.8,
            hallucination=0.8,
            overall=0.8,
        )

        assert score.contradiction == 0.9
        assert score.information_preservation == 0.8
        assert score.hallucination == 0.8

    def test_evaluation_score_to_dict_includes_new_metrics(self):
        """Test that to_dict method includes new metrics"""
        score = EvaluationScore(
            readability=0.8,
            terminology=0.7,
            safety=0.9,
            coverage=0.8,
            quality=0.7,
            contradiction=0.9,
            information_preservation=0.8,
            hallucination=0.8,
            overall=0.8,
        )

        score_dict = score.to_dict()
        assert "contradiction" in score_dict
        assert "information_preservation" in score_dict
        assert "hallucination" in score_dict
        assert score_dict["contradiction"] == 0.9
        assert score_dict["information_preservation"] == 0.8
        assert score_dict["hallucination"] == 0.8

    @patch("src.evaluator.config")
    def test_new_metrics_error_handling(self, mock_config):
        """Test error handling in new metrics"""
        # Mock config to avoid configuration issues
        mock_config.get_evaluation_config.return_value = {"safety": {"danger_words": [], "safety_words": []}}
        mock_config.get_scoring_config.return_value = {
            "weights": {"readability": 0.2, "terminology": 0.2, "safety": 0.2, "coverage": 0.2, "quality": 0.2},
            "parameters": {"safety_multiplier": 0.5},
        }
        mock_config.get_audiences.return_value = ["physician", "nurse", "patient", "caregiver"]

        # Create evaluator instance
        from src.evaluator import MedExplainEvaluator

        evaluator = MedExplainEvaluator()

        # Test that new metrics are properly initialized
        assert hasattr(evaluator, "contradiction_detector")
        assert hasattr(evaluator, "information_preservation")
        assert hasattr(evaluator, "hallucination_detector")

    def test_metric_calculation_robustness(self):
        """Test that metrics handle edge cases robustly"""
        # Test all metrics with edge cases
        contradiction_detector = ContradictionDetection()
        info_preservation = InformationPreservation()
        hallucination_detector = HallucinationDetection()

        # Edge case: very short text
        short_text = "Hi"
        original_short = "Hello"

        # All metrics should handle short text without crashing
        try:
            contradiction_score = contradiction_detector.calculate(short_text, "patient")
            info_score = info_preservation.calculate(short_text, "patient", original=original_short)
            hallucination_score = hallucination_detector.calculate(short_text, "patient", original=original_short)

            # Scores should be valid floats between 0 and 1
            assert 0 <= contradiction_score <= 1
            assert 0 <= info_score <= 1
            assert 0 <= hallucination_score <= 1

        except Exception as e:
            pytest.fail(f"Metrics should handle short text without crashing: {e}")

        # Edge case: text with special characters
        special_text = "Take 10mg @#$%^&*() twice daily!!!"
        original_special = "Medication: 10mg dosage instructions."

        try:
            contradiction_score = contradiction_detector.calculate(special_text, "patient")
            info_score = info_preservation.calculate(special_text, "patient", original=original_special)
            hallucination_score = hallucination_detector.calculate(special_text, "patient", original=original_special)

            assert 0 <= contradiction_score <= 1
            assert 0 <= info_score <= 1
            assert 0 <= hallucination_score <= 1

        except Exception as e:
            pytest.fail(f"Metrics should handle special characters without crashing: {e}")


class TestMetricPerformance:
    """Test performance characteristics of new metrics"""

    def test_metrics_complete_quickly(self):
        """Test that metrics complete within reasonable time"""
        import time

        # Create large text for performance testing
        large_text = (
            "This is a medical explanation about hypertension and diabetes. " * 100
            + "Patient should take medication as prescribed. " * 50
            + "Avoid alcohol and maintain healthy diet. " * 50
        )

        original_text = "Patient has hypertension and diabetes requiring medication management."

        # Initialize metrics
        contradiction_detector = ContradictionDetection()
        info_preservation = InformationPreservation()
        hallucination_detector = HallucinationDetection()

        # Time each metric
        start_time = time.time()
        contradiction_detector.calculate(large_text, "patient")
        contradiction_time = time.time() - start_time

        start_time = time.time()
        info_preservation.calculate(large_text, "patient", original=original_text)
        info_time = time.time() - start_time

        start_time = time.time()
        hallucination_detector.calculate(large_text, "patient", original=original_text)
        hallucination_time = time.time() - start_time

        # Each metric should complete within 5 seconds for large text
        assert contradiction_time < 5.0, f"ContradictionDetection took {contradiction_time:.2f}s"
        assert info_time < 5.0, f"InformationPreservation took {info_time:.2f}s"
        assert hallucination_time < 5.0, f"HallucinationDetection took {hallucination_time:.2f}s"
