"""Tier 2 pre-annotation classifier.

Logistic regression trained on Tier 1 auto-labeled data.
High-confidence predictions pass through; low-confidence → expert queue.
Target: handles ~25% of total traces, ~10% go to expert queue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.config import (
    TIER2_CONFIDENCE_NORMAL,
    TIER2_CONFIDENCE_URGENT,
    TIER2_CONFIDENCE_DEFAULT,
)
from src.data_gen.synthetic import NightTrace
from src.rules.tier1_engine import RuleResult
from src.patterns.feature_eng import build_feature_matrix
from src.patterns.miner import FEATURE_COLS


@dataclass
class Tier2Result:
    """Classification result for a single trace from the Tier 2 classifier."""
    trace_id: str
    baby_id: str
    ground_truth: str
    predicted_label: str
    confidence: float
    routed_to: str  # "auto" or "expert_queue"


# Feature columns for the classifier (same as pattern mining)
CLASSIFIER_FEATURES = [
    "ga_weeks", "days_since_birth", "has_apnea_condition",
    "mean_spo2", "min_spo2", "std_spo2", "pct_below_94", "pct_below_90",
    "n_desat_events", "max_desat_depth", "max_desat_duration",
    "mean_desat_duration", "desat_cluster_score",
    "hour_2_3_desat_count", "hour_2_3_vs_rest_ratio",
    "consecutive_borderline_nights",
    "accel_mean_magnitude", "accel_spike_count", "desat_accel_correlation",
    "sat_seconds",
]


def train_tier2(
    tier1_results: list[RuleResult],
    all_traces: list[NightTrace],
) -> tuple[LogisticRegression, LabelEncoder, dict]:
    """Train the Tier 2 classifier on Tier 1 auto-labeled data.

    Returns the trained model, label encoder, and metrics dict.
    """
    # Get auto-labeled traces for training
    labeled_ids = {r.trace_id for r in tier1_results if r.auto_labeled}
    labeled_traces = [t for t in all_traces if t.night_id in labeled_ids]
    labeled_results = [r for r in tier1_results if r.auto_labeled]

    if len(labeled_traces) == 0:
        raise ValueError("No auto-labeled traces from Tier 1 to train on")

    # Build feature matrix
    df = build_feature_matrix(labeled_traces)

    # Add rule labels — merge emergency into urgent for training
    # (Tier 2 only handles ambiguous cases; emergency/urgent are clear cases)
    rule_label_map = {r.trace_id: ("urgent" if r.label == "emergency" else r.label)
                      for r in labeled_results}
    df["rule_label"] = df["night_id"].map(rule_label_map)
    df = df.dropna(subset=["rule_label"])

    X = df[CLASSIFIER_FEATURES].copy()
    X["has_apnea_condition"] = X["has_apnea_condition"].astype(int)

    le = LabelEncoder()
    y = le.fit_transform(df["rule_label"])

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    model = LogisticRegression(
        max_iter=5000,
        random_state=42,
        C=1.0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)

    val_accuracy = float(np.mean(y_pred == y_val))
    print(f"\nTier 2 Classifier Validation:")
    print(f"  Accuracy: {val_accuracy:.1%}")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    metrics = {
        "val_accuracy": val_accuracy,
        "n_training_samples": len(X_train),
        "n_val_samples": len(X_val),
        "report": report,
    }

    return model, le, metrics


def predict_tier2(
    model: LogisticRegression,
    le: LabelEncoder,
    unlabeled_traces: list[NightTrace],
) -> list[Tier2Result]:
    """Run Tier 2 classifier on unlabeled traces from Tier 1.

    Routes high-confidence predictions to auto-label, low-confidence to expert queue.
    """
    if not unlabeled_traces:
        return []

    df = build_feature_matrix(unlabeled_traces)
    X = df[CLASSIFIER_FEATURES].copy()
    X["has_apnea_condition"] = X["has_apnea_condition"].astype(int)

    probas = model.predict_proba(X)
    predictions = model.predict(X)
    labels = le.inverse_transform(predictions)

    results = []
    for i, trace in enumerate(unlabeled_traces):
        confidence = float(np.max(probas[i]))
        label = labels[i]

        # Determine confidence threshold based on predicted label
        if label == "normal":
            threshold = TIER2_CONFIDENCE_NORMAL
        elif label == "urgent":
            threshold = TIER2_CONFIDENCE_URGENT
        else:
            threshold = TIER2_CONFIDENCE_DEFAULT

        routed_to = "auto" if confidence >= threshold else "expert_queue"

        results.append(Tier2Result(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            predicted_label=label,
            confidence=round(confidence, 3),
            routed_to=routed_to,
        ))

    # Summary
    auto_count = sum(1 for r in results if r.routed_to == "auto")
    expert_count = sum(1 for r in results if r.routed_to == "expert_queue")
    print(f"\nTier 2 Predictions:")
    print(f"  Auto-labeled: {auto_count}")
    print(f"  Expert queue: {expert_count}")

    return results


if __name__ == "__main__":
    from src.data_gen.synthetic import generate_dataset
    from src.rules.tier1_engine import run_tier1

    print("Generating dataset...")
    traces = generate_dataset()

    print("Running Tier 1...")
    tier1_results, unlabeled = run_tier1(traces)

    print(f"\nTraining Tier 2 on {len(tier1_results) - len(unlabeled)} labeled traces...")
    model, le, metrics = train_tier2(tier1_results, traces)

    print(f"\nPredicting on {len(unlabeled)} unlabeled traces...")
    tier2_results = predict_tier2(model, le, unlabeled)
