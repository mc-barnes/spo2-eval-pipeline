"""Tests for the urgent false-negative safety check and emergency tier in Tier 1.

These tests verify the critical safety invariant: if the raw SpO2 signal
contains a sustained desaturation below the GA-adjusted threshold, the trace
CANNOT be labeled anything other than 'urgent' or 'emergency'.
"""
import numpy as np
import pytest

from src.data_gen.synthetic import NightTrace, BabyProfile
from src.rules.tier1_engine import apply_rules, _urgent_safety_check


def _make_trace(
    spo2_array: np.ndarray,
    accel_magnitude: np.ndarray,
    ga_weeks: int = 38,
    ga_category: str = "term",
    ground_truth: str = "urgent",
) -> NightTrace:
    """Helper: create a minimal NightTrace for testing."""
    baby = BabyProfile(
        baby_id="test-baby",
        gestational_age_weeks=ga_weeks,
        ga_category=ga_category,
        birth_weight_grams=3000,
        days_since_birth=5,
        known_conditions=[],
        spo2_baseline=98.0,
        spo2_variability=0.8,
    )
    n = len(spo2_array)
    return NightTrace(
        baby=baby,
        night_id="test-night",
        night_number=1,
        timestamp_start="2026-01-01T00:00:00",
        spo2=spo2_array,
        accelerometer=np.zeros((n, 3)),
        accel_magnitude=accel_magnitude,
        ground_truth_label=ground_truth,
        events=[],
    )


def test_artifact_does_not_mask_genuine_urgent():
    """A trace with both artifact and genuine desat must be labeled urgent/emergency."""
    n = 28800  # 8 hours at 1 Hz
    spo2 = np.full(n, 97.0)
    accel = np.full(n, 0.5)

    # Inject artifact: high accel + SpO2 drop at t=1000
    accel[1000:1010] = 4.0
    spo2[995:1020] = 80.0

    # Inject GENUINE desat: no accel spike, sustained <90% for 20s at t=5000
    spo2[5000:5020] = 85.0

    trace = _make_trace(spo2, accel)
    result = apply_rules(trace)
    assert result.label in ("urgent", "emergency"), (
        f"Expected urgent/emergency but got {result.label}"
    )


def test_pure_artifact_still_labeled_artifact():
    """A trace with only brief artifact drops (no sustained desat) stays artifact.

    Artifact drops are <10s so they don't trigger the sustained desat safety check.
    Real motion artifacts are typically 1-5 second spikes, not sustained events.
    """
    n = 28800
    spo2 = np.full(n, 97.0)
    accel = np.full(n, 0.5)

    # Inject two brief artifact events: high accel + short SpO2 drop (<10s each)
    accel[1000:1010] = 4.0
    spo2[1000:1008] = 80.0   # 8s — too brief for sustained desat rule
    accel[2000:2010] = 3.5
    spo2[2000:2007] = 82.0   # 7s — too brief

    trace = _make_trace(spo2, accel, ground_truth="artifact")
    result = apply_rules(trace)
    assert result.label == "artifact", f"Expected artifact but got {result.label}"


def test_emergency_threshold():
    """SpO2 <80% sustained should trigger emergency label."""
    n = 28800
    spo2 = np.full(n, 97.0)
    accel = np.full(n, 0.5)
    spo2[3000:3020] = 72.0  # 20s below 80%

    trace = _make_trace(spo2, accel, ground_truth="urgent")
    result = apply_rules(trace)
    assert result.label == "emergency", f"Expected emergency but got {result.label}"


def test_ga_adjusted_threshold_preterm():
    """Extremely preterm baby: 87% should NOT be urgent (threshold is 85%)."""
    n = 28800
    spo2 = np.full(n, 91.0)  # Normal baseline for extremely preterm
    accel = np.full(n, 0.5)
    spo2[3000:3015] = 87.0  # Below 90 but above 85

    trace = _make_trace(
        spo2, accel,
        ga_weeks=26, ga_category="extremely_preterm",
        ground_truth="normal",
    )
    result = apply_rules(trace)
    assert result.label not in ("urgent", "emergency"), (
        f"87% should not be urgent for extremely preterm (threshold=85%), got {result.label}"
    )


def test_safety_check_raw_signal():
    """_urgent_safety_check should find desats in raw signal regardless of artifacts."""
    n = 28800
    spo2 = np.full(n, 97.0)
    spo2[5000:5015] = 85.0  # 15s sustained below 90%

    events = _urgent_safety_check(spo2, ga_category="term")
    assert len(events) >= 1, "Safety check should find the desat in raw signal"
    assert events[0]["min_spo2"] == 85.0


def test_urgent_not_emergency():
    """SpO2 between 80-90% sustained should be urgent, not emergency."""
    n = 28800
    spo2 = np.full(n, 97.0)
    accel = np.full(n, 0.5)
    spo2[3000:3020] = 85.0  # Below 90 but above 80

    trace = _make_trace(spo2, accel, ground_truth="urgent")
    result = apply_rules(trace)
    assert result.label == "urgent", f"Expected urgent but got {result.label}"
