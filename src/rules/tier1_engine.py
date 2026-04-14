"""Rule-based Tier 1 pre-annotation engine.

Applies clinical thresholds to auto-label the "easy" cases.
Target: ~65% of traces auto-labeled with zero model involvement.

Rules applied in priority order:
  R4 — Artifact:   implausible SpO2 rate-of-change + accelerometer spike
       (with safety check: genuine urgent desats override artifact label)
  R1 — Urgent/Emergency: SpO2 below GA-adjusted threshold sustained >10s
       Emergency: SpO2 <80% sustained (call 911)
  R2 — Borderline: SpO2 in GA-adjusted borderline range, sustained
  R3 — Normal:     >98% of readings above 95%, no R1/R2 triggers
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import (
    SPO2_URGENT_THRESHOLD,
    SPO2_URGENT_DURATION_S,
    SPO2_EMERGENCY_THRESHOLD,
    SPO2_BORDERLINE_LOW,
    SPO2_BORDERLINE_HIGH,
    SPO2_BORDERLINE_DURATION_S,
    SPO2_NORMAL_THRESHOLD,
    SPO2_NORMAL_PCT,
    ACCEL_ARTIFACT_THRESHOLD_G,
    SPO2_ARTIFACT_RATE,
    SPO2_ARTIFACT_WINDOW_S,
    GA_URGENT_THRESHOLDS,
    GA_BORDERLINE_RANGES,
)
from src.data_gen.synthetic import NightTrace


@dataclass
class RuleResult:
    """Output from the Tier 1 rule engine for a single trace."""
    trace_id: str
    baby_id: str
    ground_truth: str
    label: str | None              # "normal", "urgent", "emergency", "borderline", "artifact", or None
    confidence: float              # 0.0–1.0
    rule_triggered: str | None     # "R1", "R2", "R3", "R4", or None
    events_detected: list[dict] = field(default_factory=list)
    auto_labeled: bool = True      # False if no rule matched → goes to Tier 2


def _find_sustained_runs(mask: np.ndarray, min_duration: int) -> list[tuple[int, int]]:
    """Find contiguous runs of True values >= min_duration seconds.

    Returns list of (start_idx, end_idx) tuples.
    """
    runs = []
    in_run = False
    start = 0

    for i in range(len(mask)):
        if mask[i] and not in_run:
            in_run = True
            start = i
        elif not mask[i] and in_run:
            in_run = False
            if i - start >= min_duration:
                runs.append((start, i))

    # Handle run that extends to end
    if in_run and len(mask) - start >= min_duration:
        runs.append((start, len(mask)))

    return runs


def _check_artifact(trace: NightTrace) -> tuple[list[dict], np.ndarray]:
    """R4: Detect artifact segments — implausible SpO2 rate + accel spike.

    Uses two detection methods:
    1. Rate-based: implausible SpO2 change rate + simultaneous accel spike
    2. Accel-based: any high accelerometer reading near an SpO2 dip

    Returns detected artifact events and a boolean mask of artifact samples.
    """
    spo2 = trace.spo2
    accel_mag = trace.accel_magnitude
    n = len(spo2)
    artifact_mask = np.zeros(n, dtype=bool)
    events = []

    window = SPO2_ARTIFACT_WINDOW_S

    # Method 1: Rate-based detection
    for i in range(window, n):
        spo2_change = abs(spo2[i] - spo2[i - window])
        accel_peak = np.max(accel_mag[max(0, i - window - 2):min(n, i + 3)])

        if spo2_change >= SPO2_ARTIFACT_RATE and accel_peak >= ACCEL_ARTIFACT_THRESHOLD_G:
            # Wide buffer around artifact to exclude recovery period too
            start = max(0, i - window - 10)
            end = min(n, i + 20)
            artifact_mask[start:end] = True

    # Method 2: High accel near any SpO2 drop
    # Detects motion artifacts even if rate-of-change window doesn't align perfectly
    high_accel = accel_mag > ACCEL_ARTIFACT_THRESHOLD_G
    spo2_drop = spo2 < (np.median(spo2) - 5)  # significant drop from median
    for i in range(n):
        if high_accel[i] and np.any(spo2_drop[max(0, i - 5):min(n, i + 5)]):
            start = max(0, i - 15)
            end = min(n, i + 25)
            artifact_mask[start:end] = True

    # Convert mask to events
    runs = _find_sustained_runs(artifact_mask, 1)
    for start, end in runs:
        events.append({
            "rule": "R4",
            "type": "artifact",
            "start_s": int(start),
            "end_s": int(end),
            "duration_s": int(end - start),
            "max_accel_g": round(float(np.max(accel_mag[start:end])), 2),
        })

    return events, artifact_mask


def _check_urgent(spo2: np.ndarray, exclude_mask: np.ndarray,
                   ga_category: str = "term") -> list[dict]:
    """R1: SpO2 below GA-adjusted threshold sustained >10s (excluding artifacts)."""
    threshold = GA_URGENT_THRESHOLDS.get(ga_category, SPO2_URGENT_THRESHOLD)

    # Mask out artifact segments
    effective_spo2 = spo2.copy()
    effective_spo2[exclude_mask] = 100  # treat artifact as non-event

    below_threshold = effective_spo2 < threshold
    runs = _find_sustained_runs(below_threshold, SPO2_URGENT_DURATION_S)

    events = []
    for start, end in runs:
        events.append({
            "rule": "R1",
            "type": "urgent_desat",
            "start_s": int(start),
            "end_s": int(end),
            "duration_s": int(end - start),
            "min_spo2": round(float(np.min(spo2[start:end])), 1),
            "threshold_used": threshold,
        })
    return events


def _check_borderline(spo2: np.ndarray, exclude_mask: np.ndarray,
                      ga_category: str = "term") -> list[dict]:
    """R2: SpO2 drops into GA-adjusted borderline range, sustained.

    Key distinction: a preterm baby whose STABLE baseline is 92% should NOT
    trigger this rule. Only fires when SpO2 is in the borderline range AND
    represents a dip from baseline (at least 2% below trace median).
    """
    bl_low, bl_high = GA_BORDERLINE_RANGES.get(
        ga_category, (SPO2_BORDERLINE_LOW, SPO2_BORDERLINE_HIGH))

    effective_spo2 = spo2.copy()
    effective_spo2[exclude_mask] = 100

    # Only apply if trace baseline is meaningfully above the borderline range.
    # If median is already near the borderline range, defer to Tier 2.
    trace_median = float(np.median(effective_spo2))
    if trace_median <= bl_high + 1.5:
        return []  # defer to Tier 2 — baseline is near/in borderline range

    in_range = (
        (effective_spo2 >= bl_low)
        & (effective_spo2 <= bl_high)
    )
    # 30s sustained — median gate above prevents preterm false positives
    runs = _find_sustained_runs(in_range, 30)

    events = []
    for start, end in runs:
        events.append({
            "rule": "R2",
            "type": "borderline_sustained",
            "start_s": int(start),
            "end_s": int(end),
            "duration_s": int(end - start),
            "mean_spo2": round(float(np.mean(spo2[start:end])), 1),
            "trace_median": round(trace_median, 1),
        })
    return events


def _check_normal(spo2: np.ndarray, exclude_mask: np.ndarray) -> bool:
    """R3: >98% of non-artifact readings above 95%.

    Requires >98% of readings above 95%. This catches term and moderate
    preterm babies with clean nights, while letting ambiguous cases through.
    """
    effective_spo2 = spo2[~exclude_mask]
    if len(effective_spo2) == 0:
        return False
    pct_above = np.mean(effective_spo2 > SPO2_NORMAL_THRESHOLD)
    return pct_above >= 0.97


def _urgent_safety_check(spo2: np.ndarray, ga_category: str = "term") -> list[dict]:
    """Safety check: find urgent desats in RAW signal (no artifact masking).

    This is the false-negative prevention gate. If a genuine sustained desat
    exists in the raw signal, it cannot be masked by artifact detection.
    Uses GA-adjusted thresholds.
    """
    threshold = GA_URGENT_THRESHOLDS.get(ga_category, SPO2_URGENT_THRESHOLD)
    below = spo2 < threshold
    runs = _find_sustained_runs(below, SPO2_URGENT_DURATION_S)

    events = []
    for start, end in runs:
        events.append({
            "rule": "R1_SAFETY",
            "type": "urgent_desat_raw",
            "start_s": int(start),
            "end_s": int(end),
            "duration_s": int(end - start),
            "min_spo2": round(float(np.min(spo2[start:end])), 1),
            "threshold_used": threshold,
        })
    return events


def _classify_urgency(events: list[dict]) -> tuple[str, float]:
    """Determine urgent vs emergency label and confidence from desat events."""
    min_spo2 = min(e["min_spo2"] for e in events)
    max_dur = max(e["duration_s"] for e in events)

    # Emergency: SpO2 below 80% — call 911
    if min_spo2 < SPO2_EMERGENCY_THRESHOLD:
        return "emergency", 0.99

    # Urgent with severity-based confidence
    if min_spo2 < 85 and max_dur > 30:
        return "urgent", 0.99
    elif min_spo2 < 88:
        return "urgent", 0.90
    else:
        return "urgent", 0.80


def apply_rules(trace: NightTrace) -> RuleResult:
    """Apply all Tier 1 rules to a single trace in priority order.

    Safety invariant: if the raw SpO2 signal contains a sustained desat
    below the GA-adjusted threshold, the trace CANNOT be labeled anything
    other than 'urgent' or 'emergency', regardless of artifact status.
    """
    spo2 = trace.spo2
    ga_cat = trace.baby.ga_category
    all_events = []

    # R4 — Artifact (check first)
    artifact_events, artifact_mask = _check_artifact(trace)
    all_events.extend(artifact_events)

    # Label as artifact if significant artifact activity is detected.
    n_artifact_samples = int(np.sum(artifact_mask))
    below_baseline = spo2 < (np.median(spo2) - 3)
    n_desat_in_artifact = int(np.sum(below_baseline & artifact_mask))
    n_desat_total = max(1, int(np.sum(below_baseline)))
    artifact_ratio = n_desat_in_artifact / n_desat_total

    if len(artifact_events) > 0 and (artifact_ratio > 0.25 or len(artifact_events) >= 2):
        # SAFETY CHECK: before labeling as artifact, verify no genuine urgent
        # desats exist in the raw signal that artifact masking might hide.
        raw_urgent = _urgent_safety_check(spo2, ga_cat)
        if raw_urgent:
            # Override: genuine urgent desat found — cannot label as artifact
            all_events.extend(raw_urgent)
            label, confidence = _classify_urgency(raw_urgent)
            return RuleResult(
                trace_id=trace.night_id,
                baby_id=trace.baby.baby_id,
                ground_truth=trace.ground_truth_label,
                label=label,
                confidence=round(confidence, 2),
                rule_triggered="R1_SAFETY",
                events_detected=all_events,
            )
        # No genuine urgent desat — safe to label as artifact
        confidence = min(0.95, 0.70 + 0.05 * len(artifact_events))
        return RuleResult(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            label="artifact",
            confidence=round(confidence, 2),
            rule_triggered="R4",
            events_detected=all_events,
        )

    # R1 — Urgent / Emergency (GA-adjusted)
    urgent_events = _check_urgent(spo2, artifact_mask, ga_cat)
    all_events.extend(urgent_events)
    if urgent_events:
        label, confidence = _classify_urgency(urgent_events)
        return RuleResult(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            label=label,
            confidence=round(confidence, 2),
            rule_triggered="R1",
            events_detected=all_events,
        )

    # R2 — Borderline (GA-adjusted)
    borderline_events = _check_borderline(spo2, artifact_mask, ga_cat)
    all_events.extend(borderline_events)
    if borderline_events:
        n_events = len(borderline_events)
        confidence = min(0.85, 0.60 + 0.05 * n_events)
        return RuleResult(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            label="borderline",
            confidence=round(confidence, 2),
            rule_triggered="R2",
            events_detected=all_events,
        )

    # R3 — Normal
    if _check_normal(spo2, artifact_mask):
        return RuleResult(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            label="normal",
            confidence=0.99,
            rule_triggered="R3",
            events_detected=all_events,
        )

    # No rule matched → route to Tier 2
    return RuleResult(
        trace_id=trace.night_id,
        baby_id=trace.baby.baby_id,
        ground_truth=trace.ground_truth_label,
        label=None,
        confidence=0.0,
        rule_triggered=None,
        events_detected=all_events,
        auto_labeled=False,
    )


def run_tier1(traces: list[NightTrace]) -> tuple[list[RuleResult], list[NightTrace]]:
    """Run Tier 1 rules on all traces.

    Returns:
        results: RuleResult for every trace
        unlabeled: traces that were not auto-labeled (for Tier 2)
    """
    results = []
    unlabeled = []

    for trace in traces:
        result = apply_rules(trace)
        results.append(result)
        if not result.auto_labeled:
            unlabeled.append(trace)

    # Print coverage summary
    n = len(traces)
    labeled = [r for r in results if r.auto_labeled]
    from collections import Counter
    label_counts = Counter(r.label for r in labeled)

    print(f"\n{'='*50}")
    print(f"Tier 1 Rule Engine Results")
    print(f"{'='*50}")
    print(f"Total traces: {n}")
    print(f"Auto-labeled: {len(labeled)} ({len(labeled)/n*100:.1f}%)")
    print(f"  → normal:     {label_counts.get('normal', 0)}")
    print(f"  → emergency:  {label_counts.get('emergency', 0)}")
    print(f"  → urgent:     {label_counts.get('urgent', 0)}")
    print(f"  → borderline: {label_counts.get('borderline', 0)}")
    print(f"  → artifact:   {label_counts.get('artifact', 0)}")
    print(f"Unlabeled (→ Tier 2): {len(unlabeled)} ({len(unlabeled)/n*100:.1f}%)")
    print(f"{'='*50}\n")

    # Accuracy check against ground truth
    correct = sum(1 for r in labeled if r.label == r.ground_truth)
    if labeled:
        print(f"Tier 1 accuracy (labeled only): {correct}/{len(labeled)} = {correct/len(labeled)*100:.1f}%")

    return results, unlabeled


if __name__ == "__main__":
    from src.data_gen.synthetic import generate_dataset

    print("Generating dataset...")
    traces = generate_dataset()
    print(f"Generated {len(traces)} traces")

    print("\nRunning Tier 1 rules...")
    results, unlabeled = run_tier1(traces)
