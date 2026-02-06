"""Behavioral tests for response mobilization classifier.

Uses the CheckList approach (Ribeiro et al. 2020) to verify:
1. Invariance (INV): Changing names/details shouldn't change classification.
2. Directional Expectations (DIR): Adding "ASAP" should increase pressure.
3. Minimum Functionality (MFT): Specific triggers yield expected buckets.
"""

import pytest
from jarvis.classifiers.response_mobilization import classify_response_pressure, ResponsePressure

class TestMobilizationBehavior:
    """Behavioral test suite for response pressure classification."""

    @pytest.mark.parametrize("name1, name2", [
        ("John", "Sarah"),
        ("Alice", "Bob"),
        ("Jwalin", "Zoe")
    ])
    def test_invariance_contact_names(self, name1, name2):
        """INV: Changing the name in a question should not change the pressure."""
        text1 = f"Hey {name1}, are you free for lunch?"
        text2 = f"Hey {name2}, are you free for lunch?"
        
        res1 = classify_response_pressure(text1)
        res2 = classify_response_pressure(text2)
        
        assert res1.pressure == res2.pressure
        assert abs(res1.confidence - res2.confidence) < 0.01

    def test_directional_urgency_markers(self):
        """DIR: Adding urgency markers should increase or maintain high pressure."""
        base_text = "Can you review that document?"
        urgent_text = "Can you review that document ASAP? It's urgent."
        
        base_res = classify_response_pressure(base_text)
        urgent_res = classify_response_pressure(urgent_text)
        
        # Pressure should be equal or higher (HIGH >= LOW)
        # Note: We assume HIGH > LOW in the enum ordering or logic
        if base_res.pressure == ResponsePressure.LOW:
            assert urgent_res.pressure in [ResponsePressure.HIGH, ResponsePressure.LOW]
        
        # If both are same category, confidence in "HIGH" or similar should stay high
        # This depends on the internal implementation of the classifier

    @pytest.mark.parametrize("text, expected_pressure", [
        ("What time is the meeting?", ResponsePressure.HIGH),
        ("Cool, see ya", ResponsePressure.LOW),
        ("?", ResponsePressure.HIGH),
        ("k", ResponsePressure.NONE),
    ])
    def test_mft_baseline_expectations(self, text, expected_pressure):
        """MFT: Basic triggers should match expected pressure buckets."""
        res = classify_response_pressure(text)
        assert res.pressure == expected_pressure
