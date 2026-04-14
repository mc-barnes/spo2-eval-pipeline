"""Feature engineering for pattern mining.

Extracts clinically meaningful features from SpO2 traces for use by
the pattern mining layer and Tier 2 classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    SPO2_URGENT_THRESHOLD,
    SPO2_BORDERLINE_HIGH,
    NIGHT_DURATION_S,
    ACCEL_ARTIFACT_THRESHOLD_G,
    GA_URGENT_THRESHOLDS,
)
from src.data_gen.synthetic import NightTrace
from src.rules.tier1_engine import RuleResult


def _count_desat_events(spo2: np.ndarray, threshold: float, min_dur: int = 5) -> list[tuple[int, int]]:
    """Find contiguous runs where SpO2 < threshold for >= min_dur seconds."""
    below = spo2 < threshold
    events = []
    in_event = False
    start = 0
    for i in range(len(below)):
        if below[i] and not in_event:
            in_event = True
            start = i
        elif not below[i] and in_event:
            in_event = False
            if i - start >= min_dur:
                events.append((start, i))
    if in_event and len(below) - start >= min_dur:
        events.append((start, len(below)))
    return events


def _desat_depth(spo2: np.ndarray, start: int, end: int, baseline: float) -> float:
    """Max depth of a desaturation event below baseline."""
    return float(baseline - np.min(spo2[start:end]))


def extract_features(trace: NightTrace, consecutive_borderline_nights: int = 0) -> dict:
    """Extract features from a single trace for pattern mining.

    Args:
        trace: the NightTrace to analyze
        consecutive_borderline_nights: how many prior consecutive borderline
            nights this baby has had (for multi-night trend feature)
    """
    spo2 = trace.spo2
    accel_mag = trace.accel_magnitude
    baby = trace.baby
    n = len(spo2)

    # Baseline estimate: median of the trace
    baseline = float(np.median(spo2))

    # Statistical features
    mean_spo2 = float(np.mean(spo2))
    min_spo2 = float(np.min(spo2))
    std_spo2 = float(np.std(spo2))
    pct_below_94 = float(np.mean(spo2 < SPO2_BORDERLINE_HIGH))
    pct_below_90 = float(np.mean(spo2 < SPO2_URGENT_THRESHOLD))

    # Desaturation events (SpO2 drops >3% from baseline for >5s)
    desat_threshold = baseline - 3.0
    desat_events = _count_desat_events(spo2, desat_threshold, min_dur=5)
    n_desat_events = len(desat_events)

    if desat_events:
        depths = [_desat_depth(spo2, s, e, baseline) for s, e in desat_events]
        durations = [e - s for s, e in desat_events]
        max_desat_depth = max(depths)
        max_desat_duration = max(durations)
        mean_desat_duration = float(np.mean(durations))
    else:
        max_desat_depth = 0.0
        max_desat_duration = 0
        mean_desat_duration = 0.0

    # Temporal features: cluster score (max events in any 30-min window)
    window_30min = 1800  # seconds
    desat_cluster_score = 0
    if desat_events:
        starts = [s for s, _ in desat_events]
        for s in starts:
            count_in_window = sum(1 for s2 in starts if s <= s2 < s + window_30min)
            desat_cluster_score = max(desat_cluster_score, count_in_window)

    # 2am-3am window analysis (hours 2-3 of the night = samples 7200-10800)
    hour_2_start = 2 * 3600
    hour_3_end = 3 * 3600
    hour_2_3_desats = [
        (s, e) for s, e in desat_events
        if hour_2_start <= s < hour_3_end
    ]
    hour_2_3_desat_count = len(hour_2_3_desats)

    # Ratio of 2-3am desat rate to rest-of-night rate
    rest_desats = len(desat_events) - hour_2_3_desat_count
    night_hours_ex_2_3 = (NIGHT_DURATION_S - 3600) / 3600  # 7 hours
    hour_2_3_rate = hour_2_3_desat_count / 1.0  # per 1 hour
    rest_rate = rest_desats / max(night_hours_ex_2_3, 1.0)
    hour_2_3_vs_rest_ratio = hour_2_3_rate / max(rest_rate, 0.01)

    # Accelerometer features
    accel_mean_magnitude = float(np.mean(accel_mag))
    accel_spike_count = int(np.sum(
        np.diff((accel_mag > ACCEL_ARTIFACT_THRESHOLD_G).astype(int)) == 1
    ))

    # SatSeconds: integral of (threshold - SpO2) for sub-threshold samples.
    # Higher = more severe hypoxemic burden. GA-adjusted threshold.
    ga_threshold = GA_URGENT_THRESHOLDS.get(baby.ga_category, SPO2_URGENT_THRESHOLD)
    sat_seconds = float(np.sum(np.maximum(0, ga_threshold - spo2)))

    # Correlation between SpO2 drops and accel spikes
    if n > 100:
        # Downsample to 1-minute windows for correlation
        n_windows = n // 60
        spo2_windowed = np.array([np.min(spo2[i*60:(i+1)*60]) for i in range(n_windows)])
        accel_windowed = np.array([np.max(accel_mag[i*60:(i+1)*60]) for i in range(n_windows)])
        if np.std(spo2_windowed) > 0 and np.std(accel_windowed) > 0:
            desat_accel_corr = float(np.corrcoef(
                -spo2_windowed, accel_windowed  # negative SpO2 ~ positive accel = artifact
            )[0, 1])
        else:
            desat_accel_corr = 0.0
    else:
        desat_accel_corr = 0.0

    return {
        # Baby context
        "baby_id": baby.baby_id,
        "ga_weeks": baby.gestational_age_weeks,
        "ga_category": baby.ga_category,
        "days_since_birth": baby.days_since_birth,
        "has_apnea_condition": "apnea_of_prematurity" in baby.known_conditions,
        "spo2_baseline_expected": baby.spo2_baseline,
        # Statistical
        "mean_spo2": round(mean_spo2, 2),
        "min_spo2": round(min_spo2, 2),
        "std_spo2": round(std_spo2, 3),
        "pct_below_94": round(pct_below_94, 4),
        "pct_below_90": round(pct_below_90, 4),
        # Event-based
        "n_desat_events": n_desat_events,
        "max_desat_depth": round(max_desat_depth, 2),
        "max_desat_duration": max_desat_duration,
        "mean_desat_duration": round(mean_desat_duration, 1),
        # Temporal
        "desat_cluster_score": desat_cluster_score,
        "hour_2_3_desat_count": hour_2_3_desat_count,
        "hour_2_3_vs_rest_ratio": round(hour_2_3_vs_rest_ratio, 2),
        # Multi-night
        "consecutive_borderline_nights": consecutive_borderline_nights,
        # Artifact indicators
        "accel_mean_magnitude": round(accel_mean_magnitude, 4),
        "accel_spike_count": accel_spike_count,
        "desat_accel_correlation": round(desat_accel_corr, 3),
        # Severity
        "sat_seconds": round(sat_seconds, 1),
        # Trace metadata
        "night_id": trace.night_id,
        "ground_truth": trace.ground_truth_label,
    }


def build_feature_matrix(
    traces: list[NightTrace],
    rule_results: list[RuleResult] | None = None,
) -> pd.DataFrame:
    """Build a feature matrix from a list of traces.

    Computes consecutive_borderline_nights for multi-night trend analysis
    by tracking each baby's sequence of borderline labels.
    """
    # Build lookup for consecutive borderline nights per baby
    baby_nights: dict[str, list[tuple[int, str]]] = {}
    for trace in traces:
        bid = trace.baby.baby_id
        if bid not in baby_nights:
            baby_nights[bid] = []
        baby_nights[bid].append((trace.night_number, trace.ground_truth_label))

    # Sort by night number and compute consecutive borderline count
    consecutive_map: dict[str, dict[int, int]] = {}
    for bid, nights in baby_nights.items():
        nights.sort(key=lambda x: x[0])
        consecutive_map[bid] = {}
        streak = 0
        for night_num, label in nights:
            if label == "borderline":
                streak += 1
            else:
                streak = 0
            consecutive_map[bid][night_num] = streak

    rows = []
    for trace in traces:
        bid = trace.baby.baby_id
        consec = consecutive_map.get(bid, {}).get(trace.night_number, 0)
        features = extract_features(trace, consecutive_borderline_nights=consec)
        rows.append(features)

    df = pd.DataFrame(rows)

    # Add rule engine label if available
    if rule_results is not None:
        rule_labels = {r.trace_id: r.label for r in rule_results}
        df["rule_label"] = df["night_id"].map(rule_labels)

    return df


if __name__ == "__main__":
    from src.data_gen.synthetic import generate_dataset

    traces = generate_dataset(n_babies=10, nights_per_baby=3, seed=42)
    df = build_feature_matrix(traces)
    print(f"Feature matrix: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample row:\n{df.iloc[0].to_dict()}")
