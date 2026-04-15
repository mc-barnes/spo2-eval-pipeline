"""Synthetic neonatal SpO2 trace generator.

Generates realistic nightly SpO2 traces for four pattern types:
Normal, Urgent, Borderline, and Artifact. Varies by gestational age.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.config import (
    GA_CATEGORIES,
    SPO2_BASELINES,
    BIRTH_WEIGHT_RANGES,
    NIGHT_DURATION_S,
    SAMPLING_RATE_HZ,
    BREATHING_RATE_BPM_RANGE,
    PATTERN_WEIGHTS,
    DEFAULT_N_BABIES,
    DEFAULT_NIGHTS_PER_BABY,
    DEFAULT_SEED,
    OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BabyProfile:
    """Metadata for a single baby."""
    baby_id: str
    gestational_age_weeks: int
    ga_category: str
    birth_weight_grams: int
    days_since_birth: int
    known_conditions: list[str]
    spo2_baseline: float
    spo2_variability: float


@dataclass
class NightTrace:
    """One night of SpO2 + accelerometer data for a baby."""
    baby: BabyProfile
    night_id: str
    night_number: int
    timestamp_start: str  # ISO format
    spo2: np.ndarray             # shape (N,)
    accelerometer: np.ndarray    # shape (N, 3)
    accel_magnitude: np.ndarray  # shape (N,)
    ground_truth_label: str      # "normal", "urgent", "borderline", "artifact"
    events: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Baby cohort generation
# ---------------------------------------------------------------------------

def _classify_ga(ga_weeks: int) -> str:
    """Return GA category string for a given gestational age."""
    for cat, (lo, hi) in GA_CATEGORIES.items():
        if lo <= ga_weeks < hi:
            return cat
    return "term"


def generate_baby_cohort(n: int, rng: np.random.Generator) -> list[BabyProfile]:
    """Generate a cohort of baby profiles with realistic distributions."""
    babies = []
    # Distribution skewed toward preterm (clinically interesting)
    cat_weights = [0.20, 0.25, 0.25, 0.30]
    cat_names = list(GA_CATEGORIES.keys())

    for _ in range(n):
        cat = rng.choice(cat_names, p=cat_weights)
        lo, hi = GA_CATEGORIES[cat]
        ga = rng.integers(lo, hi)

        wt_lo, wt_hi = BIRTH_WEIGHT_RANGES[cat]
        weight = int(rng.integers(wt_lo, wt_hi))

        days = int(rng.integers(1, 91))

        # Known conditions
        conditions = []
        if cat == "extremely_preterm":
            if rng.random() < 0.40:
                conditions.append("apnea_of_prematurity")
            if rng.random() < 0.20:
                conditions.append("bpd")
        elif cat == "very_preterm":
            if rng.random() < 0.20:
                conditions.append("apnea_of_prematurity")
            if rng.random() < 0.20:
                conditions.append("bpd")
        elif cat == "moderate_preterm":
            if rng.random() < 0.05:
                conditions.append("apnea_of_prematurity")
        if not conditions:
            conditions.append("none")

        # SpO2 baseline with individual variation
        base_mean, base_std = SPO2_BASELINES[cat]
        spo2_base = base_mean + rng.normal(0, 1.0)
        spo2_var = base_std + rng.normal(0, 0.3)
        spo2_var = max(0.3, spo2_var)

        babies.append(BabyProfile(
            baby_id=str(uuid.uuid4())[:8],
            gestational_age_weeks=int(ga),
            ga_category=cat,
            birth_weight_grams=weight,
            days_since_birth=days,
            known_conditions=conditions,
            spo2_baseline=round(spo2_base, 1),
            spo2_variability=round(spo2_var, 2),
        ))
    return babies


# ---------------------------------------------------------------------------
# Signal component primitives
# ---------------------------------------------------------------------------

def _generate_baseline(n_samples: int, baseline: float, rng: np.random.Generator) -> np.ndarray:
    """Slow sinusoidal drift around baseline SpO2."""
    t = np.arange(n_samples, dtype=np.float64)
    drift_period = (2.0 + rng.uniform(-0.5, 0.5)) * 3600  # ~2hr period
    drift_amp = rng.uniform(0.2, 0.5)
    phase = rng.uniform(0, 2 * np.pi)
    return baseline + drift_amp * np.sin(2 * np.pi * t / drift_period + phase)


def _generate_breathing(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Respiratory-rate oscillation in SpO2 signal."""
    t = np.arange(n_samples, dtype=np.float64)
    rate_bpm = rng.integers(*BREATHING_RATE_BPM_RANGE)
    amplitude = rng.uniform(0.2, 0.5)
    return amplitude * np.sin(2 * np.pi * rate_bpm / 60.0 * t)


def _generate_noise(n_samples: int, std: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian sensor noise."""
    return rng.normal(0, std, size=n_samples)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _generate_desat_event(
    n_samples: int,
    onset_s: int,
    depth: float,
    duration_s: int,
    recovery_s: int,
) -> np.ndarray:
    """Generate a single desaturation dip (subtracted from baseline).

    Returns an array of shape (n_samples,) with negative values at the dip.
    Uses sigmoid transitions for smooth, physiologically plausible shapes.
    """
    t = np.arange(n_samples, dtype=np.float64)

    # Descent: sigmoid centered at onset
    k_down = 6.0 / max(duration_s * 0.3, 1)  # steepness of descent
    descent = -depth * _sigmoid(k_down * (t - onset_s))

    # Recovery: sigmoid centered at onset + duration
    recovery_start = onset_s + duration_s
    k_up = 6.0 / max(recovery_s, 1)
    recovery = depth * _sigmoid(k_up * (t - recovery_start))

    signal = descent + recovery
    # Only keep the negative (dip) portion
    signal = np.minimum(signal, 0)

    return signal


def _generate_accelerometer(
    n_samples: int,
    movement_events: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """3-axis accelerometer data. Events are (start_s, duration_s) tuples."""
    accel = rng.normal(0, 0.05, size=(n_samples, 3))
    for start, dur in movement_events:
        end = min(start + dur, n_samples)
        intensity = rng.uniform(1.0, 3.0)
        accel[start:end] += rng.normal(0, intensity, size=(end - start, 3))
    return accel


# ---------------------------------------------------------------------------
# Pattern-specific trace generators
# ---------------------------------------------------------------------------

def _generate_normal(baby: BabyProfile, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Normal trace: stable SpO2 >95%, minimal events."""
    spo2 = _generate_baseline(n_samples, baby.spo2_baseline, rng)
    spo2 += _generate_breathing(n_samples, rng)
    spo2 += _generate_noise(n_samples, 0.2, rng)

    # Maybe 1-2 very mild, brief dips (clinically insignificant)
    events = []
    n_mild = rng.integers(0, 3)
    for _ in range(n_mild):
        onset = int(rng.integers(1800, n_samples - 1800))
        depth = rng.uniform(1.0, 2.5)
        dur = int(rng.integers(3, 8))
        spo2 += _generate_desat_event(n_samples, onset, depth, dur, int(rng.integers(5, 15)))
        events.append({"type": "mild_dip", "onset_s": onset, "depth_pct": round(depth, 1), "duration_s": dur})

    # Light movement events (routine care)
    n_moves = int(rng.integers(0, 3))
    move_events = [(int(rng.integers(0, n_samples - 60)), int(rng.integers(5, 30))) for _ in range(n_moves)]
    accel = _generate_accelerometer(n_samples, move_events, rng)

    # Clamp to realistic range
    spo2 = np.clip(spo2, 70, 100)
    return spo2, accel, events


def _generate_urgent(baby: BabyProfile, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Urgent trace: sustained SpO2 <90%, clustered desats.

    Depth is calculated to ensure SpO2 reliably reaches below 90%
    regardless of the baby's baseline.
    """
    spo2 = _generate_baseline(n_samples, baby.spo2_baseline, rng)
    spo2 += _generate_breathing(n_samples, rng)
    spo2 += _generate_noise(n_samples, 0.15, rng)

    events = []
    n_desats = int(rng.integers(2, 9))

    # Cluster some events in a 30-minute window
    cluster_start = int(rng.integers(3600, n_samples - 7200))
    for i in range(n_desats):
        if i < n_desats // 2:
            onset = cluster_start + int(rng.integers(0, 1800))
        else:
            onset = int(rng.integers(1800, n_samples - 1800))

        # Target nadir clearly below 90% (72-88%)
        target_nadir = rng.uniform(72, 88)
        depth = max(3.0, baby.spo2_baseline - target_nadir)
        dur = int(rng.integers(12, 120))  # sustained >10s (12+ ensures it)
        recovery = int(rng.integers(15, 60))
        spo2 += _generate_desat_event(n_samples, onset, depth, dur, recovery)
        events.append({
            "type": "urgent_desat",
            "onset_s": onset,
            "depth_pct": round(depth, 1),
            "target_nadir": round(target_nadir, 1),
            "duration_s": dur,
            "recovery_s": recovery,
        })

    # Minimal movement (genuine desat, not artifact)
    accel = _generate_accelerometer(n_samples, [], rng)

    spo2 = np.clip(spo2, 50, 100)
    return spo2, accel, events


def _generate_borderline(baby: BabyProfile, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Borderline trace: SpO2 dips into 90-94% range, ambiguous events.

    Key: depth is calculated to target the 90-94% SpO2 range specifically,
    not a fixed depth below baseline. This prevents preterm traces from
    accidentally triggering the urgent rule (<90% >10s).
    """
    spo2 = _generate_baseline(n_samples, baby.spo2_baseline, rng)
    spo2 += _generate_breathing(n_samples, rng)
    spo2 += _generate_noise(n_samples, 0.15, rng)

    events = []
    n_desats = int(rng.integers(3, 11))

    # 30% chance of "2am dip" cluster (seeded pattern for Phase 3 discovery)
    has_2am_dip = rng.random() < 0.30
    two_am_offset = 2 * 3600  # 2 hours into the night

    for i in range(n_desats):
        if has_2am_dip and i < 3:
            onset = two_am_offset + int(rng.integers(0, 3600))
        else:
            onset = int(rng.integers(1800, n_samples - 600))

        # Target SpO2 nadir between 90-94% (borderline range)
        target_nadir = rng.uniform(90.5, 94.0)
        depth = max(1.0, baby.spo2_baseline - target_nadir)
        # Mix of brief and sustained desats. Brief ones are ambiguous for rules.
        # Some longer ones (40-60s) can trigger R2 for term babies (median >96).
        dur = int(rng.integers(5, 60))
        recovery = int(rng.integers(10, 40))
        spo2 += _generate_desat_event(n_samples, onset, depth, dur, recovery)
        events.append({
            "type": "borderline_desat",
            "onset_s": onset,
            "depth_pct": round(depth, 1),
            "target_nadir": round(target_nadir, 1),
            "duration_s": dur,
            "has_2am_pattern": has_2am_dip and i < 3,
        })

    # Some mild movement
    n_moves = int(rng.integers(0, 3))
    move_events = [(int(rng.integers(0, n_samples - 60)), int(rng.integers(5, 20))) for _ in range(n_moves)]
    accel = _generate_accelerometer(n_samples, move_events, rng)

    spo2 = np.clip(spo2, 85, 100)  # floor at 85 to prevent accidental urgent-level dips
    return spo2, accel, events


def _generate_artifact(baby: BabyProfile, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Artifact trace: implausible SpO2 drops + accelerometer spikes."""
    spo2 = _generate_baseline(n_samples, baby.spo2_baseline, rng)
    spo2 += _generate_breathing(n_samples, rng)
    spo2 += _generate_noise(n_samples, 0.2, rng)

    events = []
    move_events = []
    n_artifacts = int(rng.integers(1, 6))

    for _ in range(n_artifacts):
        onset = int(rng.integers(1800, n_samples - 1800))
        # Implausibly fast SpO2 drop (20-50% in 1-3 seconds)
        drop_depth = rng.uniform(20, 50)
        drop_dur = int(rng.integers(1, 4))
        # Sharp drop then sharp recovery
        recovery = int(rng.integers(2, 5))

        # Apply artifact to SpO2
        end = min(onset + drop_dur, n_samples)
        spo2[onset:end] -= drop_depth
        rec_end = min(end + recovery, n_samples)
        # Gradual recovery
        if rec_end > end:
            rec_steps = rec_end - end
            spo2[end:rec_end] -= drop_depth * np.linspace(1, 0, rec_steps)

        # Simultaneous accelerometer spike
        accel_dur = drop_dur + recovery + int(rng.integers(2, 10))
        move_events.append((onset, accel_dur))

        events.append({
            "type": "artifact",
            "onset_s": onset,
            "drop_pct": round(drop_depth, 1),
            "drop_duration_s": drop_dur,
            "accel_duration_s": accel_dur,
        })

    # May also have 1-2 genuine mild desats mixed in
    n_genuine = int(rng.integers(0, 3))
    for _ in range(n_genuine):
        onset = int(rng.integers(1800, n_samples - 1800))
        depth = rng.uniform(2, 5)
        dur = int(rng.integers(5, 20))
        spo2 += _generate_desat_event(n_samples, onset, depth, dur, int(rng.integers(10, 30)))
        events.append({"type": "mild_genuine_desat", "onset_s": onset, "depth_pct": round(depth, 1), "duration_s": dur})

    accel = _generate_accelerometer(n_samples, move_events, rng)
    spo2 = np.clip(spo2, 30, 100)
    return spo2, accel, events


_GENERATORS = {
    "normal": _generate_normal,
    "urgent": _generate_urgent,
    "borderline": _generate_borderline,
    "artifact": _generate_artifact,
}


# ---------------------------------------------------------------------------
# Pattern assignment logic
# ---------------------------------------------------------------------------

def _assign_pattern(baby: BabyProfile, rng: np.random.Generator) -> str:
    """Assign pattern type weighted by GA category.

    Preterm babies get more borderline/urgent. Term babies mostly normal.
    """
    cat = baby.ga_category
    # Weights: [normal, urgent, borderline, artifact]
    if cat in ("extremely_preterm", "very_preterm"):
        weights = [0.15, 0.20, 0.50, 0.15]
    elif cat == "moderate_preterm":
        weights = [0.30, 0.10, 0.40, 0.20]
    else:  # term
        weights = [0.55, 0.05, 0.20, 0.20]

    patterns = ["normal", "urgent", "borderline", "artifact"]
    return rng.choice(patterns, p=weights)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_trace(
    baby: BabyProfile,
    pattern_type: str,
    night_number: int,
    rng: np.random.Generator,
) -> NightTrace:
    """Generate a single night trace for a baby with the given pattern type."""
    n_samples = NIGHT_DURATION_S * SAMPLING_RATE_HZ

    generator = _GENERATORS[pattern_type]
    spo2, accel, events = generator(baby, n_samples, rng)

    accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))

    # Timestamp: arbitrary start date, 9pm
    start = datetime(2025, 1, 1, 21, 0) + timedelta(days=night_number - 1)

    # Refine urgent → emergency when the generated signal has sustained SpO2 <80%
    label = pattern_type
    if pattern_type == "urgent" and np.min(spo2) < 80:
        label = "emergency"

    return NightTrace(
        baby=baby,
        night_id=str(uuid.uuid4())[:8],
        night_number=night_number,
        timestamp_start=start.isoformat(),
        spo2=spo2,
        accelerometer=accel,
        accel_magnitude=accel_mag,
        ground_truth_label=label,
        events=events,
    )


def generate_dataset(
    n_babies: int = DEFAULT_N_BABIES,
    nights_per_baby: int = DEFAULT_NIGHTS_PER_BABY,
    seed: int = DEFAULT_SEED,
) -> list[NightTrace]:
    """Generate a full synthetic dataset.

    Returns a list of NightTrace objects (n_babies * nights_per_baby).
    """
    rng = np.random.default_rng(seed)
    babies = generate_baby_cohort(n_babies, rng)
    traces = []

    for baby in babies:
        for night in range(1, nights_per_baby + 1):
            pattern = _assign_pattern(baby, rng)
            trace = generate_trace(baby, pattern, night, rng)
            traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trace(trace: NightTrace, output_dir: Path) -> Path:
    """Save a single trace to disk (npz + JSON sidecar)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{trace.baby.baby_id}_night{trace.night_number}_{trace.night_id}"

    # Save signal arrays
    npz_path = output_dir / f"{prefix}.npz"
    np.savez_compressed(
        npz_path,
        spo2=trace.spo2,
        accelerometer=trace.accelerometer,
        accel_magnitude=trace.accel_magnitude,
    )

    # Save metadata as JSON sidecar
    meta = {
        "baby": asdict(trace.baby),
        "night_id": trace.night_id,
        "night_number": trace.night_number,
        "timestamp_start": trace.timestamp_start,
        "ground_truth_label": trace.ground_truth_label,
        "events": trace.events,
        "n_samples": len(trace.spo2),
    }
    json_path = output_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(meta, indent=2))

    return npz_path


def save_dataset(traces: list[NightTrace], output_dir: Path | None = None) -> Path:
    """Save all traces and a manifest CSV."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for trace in traces:
        save_trace(trace, output_dir)
        manifest_rows.append({
            "baby_id": trace.baby.baby_id,
            "night_id": trace.night_id,
            "night_number": trace.night_number,
            "ga_weeks": trace.baby.gestational_age_weeks,
            "ga_category": trace.baby.ga_category,
            "ground_truth": trace.ground_truth_label,
            "spo2_mean": round(float(np.mean(trace.spo2)), 1),
            "spo2_min": round(float(np.min(trace.spo2)), 1),
            "n_events": len(trace.events),
        })

    import pandas as pd
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return output_dir


# ---------------------------------------------------------------------------
# Quick summary for debugging
# ---------------------------------------------------------------------------

def summarize_dataset(traces: list[NightTrace]) -> dict:
    """Print and return summary stats for a generated dataset."""
    from collections import Counter
    labels = Counter(t.ground_truth_label for t in traces)
    ga_cats = Counter(t.baby.ga_category for t in traces)

    summary = {
        "total_traces": len(traces),
        "label_distribution": dict(labels),
        "ga_distribution": dict(ga_cats),
        "spo2_range": (
            round(float(min(t.spo2.min() for t in traces)), 1),
            round(float(max(t.spo2.max() for t in traces)), 1),
        ),
    }

    print(f"\n{'='*50}")
    print(f"Dataset Summary: {summary['total_traces']} traces")
    print(f"{'='*50}")
    print(f"Labels: {dict(labels)}")
    print(f"GA cats: {dict(ga_cats)}")
    print(f"SpO2 range: {summary['spo2_range']}")
    print(f"{'='*50}\n")

    return summary


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    traces = generate_dataset()
    summary = summarize_dataset(traces)
    output_path = save_dataset(traces)
    print(f"Saved to: {output_path}")
