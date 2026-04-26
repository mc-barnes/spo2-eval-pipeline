"""Export pipeline results to static JSON for the React dashboard.

Runs the full mock pipeline (seed=42, 100 babies, 3 nights) and writes
pre-computed results to data/export/. No API calls, deterministic, ~5s.

Usage:
    cd spo2-eval-pipeline
    source venv/bin/activate
    python scripts/export_dashboard_data.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_gen.synthetic import NightTrace, generate_dataset
from src.rules.tier1_engine import run_tier1, apply_rules
from src.patterns.feature_eng import build_feature_matrix
from src.patterns.miner import run_pattern_mining, FEATURE_COLS
from src.classifier.tier2 import train_tier2, predict_tier2
from src.classifier.expert_sim import run_expert_queue
from src.handoff.generator import generate_handoff
from src.interop.hl7_messages import build_adt_a01, build_ack_a01, build_oru_r01, parse_adt_a01
from src.pipeline.orchestrator import FinalTriage
from src.evals.clinical_accuracy import evaluate_clinical_accuracy
from src.evals.handoff_quality import evaluate_handoff_quality
from src.evals.artifact_handling import evaluate_artifact_handling

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_BABIES = 100
NIGHTS = 3
SEED = 42
DOWNSAMPLE_STEP = 30  # 30s intervals → 960 points per 8h trace
EXPORT_DIR = PROJECT_ROOT / "data" / "export"


def run_pipeline():
    """Run the full mock pipeline and return all intermediate results."""
    print("Phase 1: Generating synthetic data...")
    traces = generate_dataset(N_BABIES, NIGHTS, SEED)

    print("Phase 2: Running Tier 1 rules...")
    tier1_results, unlabeled = run_tier1(traces)

    print("Phase 3: Pattern mining...")
    all_df = build_feature_matrix(traces, tier1_results)
    candidate_rules, tree = run_pattern_mining(all_df)

    print("Phase 4: Training Tier 2 classifier...")
    model, le, metrics = train_tier2(tier1_results, traces)
    tier2_results = predict_tier2(model, le, unlabeled)

    print("Phase 4b: Expert queue...")
    expert_traces = [
        t for t in unlabeled
        for r in tier2_results
        if r.trace_id == t.night_id and r.routed_to == "expert_queue"
    ]
    expert_results = run_expert_queue(expert_traces, tier2_results, seed=SEED)

    # Build final label map
    final_labels = {}
    final_sources = {}
    final_confidence = {}
    for r in tier1_results:
        if r.auto_labeled:
            final_labels[r.trace_id] = r.label
            final_sources[r.trace_id] = "tier1_rules"
            final_confidence[r.trace_id] = r.confidence
    for r in tier2_results:
        if r.routed_to == "auto":
            final_labels[r.trace_id] = r.predicted_label
            final_sources[r.trace_id] = "tier2_classifier"
            final_confidence[r.trace_id] = r.confidence
    for r in expert_results:
        final_labels[r.trace_id] = r.expert_label
        final_sources[r.trace_id] = "expert_review"
        final_confidence[r.trace_id] = r.expert_confidence

    # Generate mock handoffs
    print("Phase 6: Generating handoffs...")
    handoffs_map = {}
    for trace in traces:
        label = final_labels.get(trace.night_id, "normal")
        handoffs_map[trace.night_id] = generate_handoff(trace, label, use_llm=False)

    # Run mock evals
    print("Phase 5: Running evaluators...")
    rng = np.random.default_rng(SEED)
    eval_results = []
    for trace in traces:
        label = final_labels.get(trace.night_id, "normal")
        s = int(rng.integers(0, 2**31))
        eval_results.append(evaluate_clinical_accuracy(trace, label, use_llm=False, seed=s))
        handoff = handoffs_map.get(trace.night_id)
        if handoff:
            eval_results.append(evaluate_handoff_quality(trace, handoff, label, use_llm=False, seed=s + 1))
        eval_results.append(evaluate_artifact_handling(trace, label, use_llm=False, seed=s + 2))

    return {
        "traces": traces,
        "tier1_results": tier1_results,
        "tier2_results": tier2_results,
        "expert_results": expert_results,
        "candidate_rules": candidate_rules,
        "tree": tree,
        "final_labels": final_labels,
        "final_sources": final_sources,
        "final_confidence": final_confidence,
        "handoffs_map": handoffs_map,
        "eval_results": eval_results,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_pipeline_summary(data: dict) -> dict:
    """Export top-level pipeline metrics."""
    traces = data["traces"]
    tier1 = data["tier1_results"]
    tier2 = data["tier2_results"]
    expert = data["expert_results"]
    final_labels = data["final_labels"]

    total = len(traces)
    t1 = sum(1 for r in tier1 if r.auto_labeled)
    t2 = sum(1 for r in tier2 if r.routed_to == "auto")
    eq = len(expert)

    # Per-tier accuracy
    t1_correct = sum(1 for r in tier1 if r.auto_labeled and r.label == r.ground_truth)
    t2_correct = sum(1 for r in tier2 if r.routed_to == "auto" and r.predicted_label == r.ground_truth)
    ex_correct = sum(1 for r in expert if r.expert_label == r.ground_truth)

    # Overall accuracy
    all_correct = t1_correct + t2_correct + ex_correct
    overall_acc = all_correct / total * 100 if total else 0

    # Emergency count
    emergency_count = sum(1 for t in traces if t.ground_truth_label == "emergency")

    # Urgent false negatives (urgent/emergency ground truth labeled as something else)
    urgent_fn = sum(
        1 for t in traces
        if t.ground_truth_label in ("urgent", "emergency")
        and final_labels.get(t.night_id) not in ("urgent", "emergency")
    )

    return {
        "total_traces": total,
        "tier1_auto": t1,
        "tier2_auto": t2,
        "expert_queue": eq,
        "tier1_pct": round(t1 / total * 100, 1),
        "tier2_pct": round(t2 / total * 100, 1),
        "expert_pct": round(eq / total * 100, 1),
        "overall_accuracy": round(overall_acc, 1),
        "tier1_accuracy": round(t1_correct / t1 * 100, 1) if t1 else 0,
        "tier2_accuracy": round(t2_correct / t2 * 100, 1) if t2 else 0,
        "expert_accuracy": round(ex_correct / eq * 100, 1) if eq else 0,
        "emergency_count": emergency_count,
        "urgent_false_negatives": urgent_fn,
    }


def export_traces_meta(data: dict) -> list[dict]:
    """Export metadata for all 300 traces."""
    traces = data["traces"]
    final_labels = data["final_labels"]
    final_sources = data["final_sources"]
    final_confidence = data["final_confidence"]

    records = []
    for t in traces:
        spo2 = t.spo2
        records.append({
            "night_id": t.night_id,
            "baby": {
                "baby_id": t.baby.baby_id,
                "gestational_age_weeks": t.baby.gestational_age_weeks,
                "ga_category": t.baby.ga_category,
                "birth_weight_grams": t.baby.birth_weight_grams,
                "days_since_birth": t.baby.days_since_birth,
                "known_conditions": t.baby.known_conditions,
                "spo2_baseline": round(t.baby.spo2_baseline, 1),
                "spo2_variability": round(t.baby.spo2_variability, 2),
            },
            "night_number": t.night_number,
            "ground_truth_label": t.ground_truth_label,
            "final_label": final_labels.get(t.night_id, "unknown"),
            "source": final_sources.get(t.night_id, "unknown"),
            "confidence": round(final_confidence.get(t.night_id, 0.0), 3),
            "stats": {
                "mean_spo2": round(float(np.mean(spo2)), 1),
                "min_spo2": round(float(np.min(spo2)), 1),
                "std_spo2": round(float(np.std(spo2)), 2),
                "n_events": len(t.events),
            },
        })
    return records


def export_coverage_breakdown(data: dict) -> dict:
    """Export coverage crosstab and pre-computed per-label metrics."""
    tier1 = data["tier1_results"]
    tier2 = data["tier2_results"]
    expert = data["expert_results"]

    # Build predictions list
    all_preds = []
    for r in tier1:
        if r.auto_labeled:
            all_preds.append({"true": r.ground_truth, "pred": r.label, "tier": "Tier 1"})
    for r in tier2:
        if r.routed_to == "auto":
            all_preds.append({"true": r.ground_truth, "pred": r.predicted_label, "tier": "Tier 2"})
    for r in expert:
        all_preds.append({"true": r.ground_truth, "pred": r.expert_label, "tier": "Expert"})

    # Crosstab: ground_truth × tier
    crosstab = {}
    for p in all_preds:
        gt = p["true"]
        tier = p["tier"]
        if gt not in crosstab:
            crosstab[gt] = {"Tier 1": 0, "Tier 2": 0, "Expert": 0}
        crosstab[gt][tier] += 1

    crosstab_rows = [
        {"ground_truth": gt, **counts}
        for gt, counts in sorted(crosstab.items())
    ]

    # Per-label metrics (sensitivity, PPV, F1, support)
    all_labels = ["normal", "borderline", "urgent", "emergency", "artifact"]
    true_list = [p["true"] for p in all_preds]
    pred_list = [p["pred"] for p in all_preds]
    labels_present = [l for l in all_labels if l in true_list or l in pred_list]

    precision, recall, f1, support = precision_recall_fscore_support(
        true_list, pred_list, labels=labels_present, zero_division=0
    )

    per_label_metrics = {}
    for i, label in enumerate(labels_present):
        per_label_metrics[label] = {
            "sensitivity": round(float(recall[i]) * 100, 1),
            "ppv": round(float(precision[i]) * 100, 1),
            "f1": round(float(f1[i]) * 100, 1),
            "support": int(support[i]),
        }

    # Per-tier accuracy
    tier_accuracy = {}
    for tier_name in ["Tier 1", "Tier 2", "Expert"]:
        tier_preds = [p for p in all_preds if p["tier"] == tier_name]
        if tier_preds:
            correct = sum(1 for p in tier_preds if p["true"] == p["pred"])
            tier_accuracy[tier_name] = round(correct / len(tier_preds) * 100, 1)

    return {
        "crosstab": crosstab_rows,
        "per_label_metrics": per_label_metrics,
        "tier_accuracy": tier_accuracy,
    }


def export_rules_discovered(data: dict) -> dict:
    """Export candidate rules and normalized feature importance."""
    candidate_rules = data["candidate_rules"]
    tree = data["tree"]

    rules = [
        {
            "rule_id": r.rule_id,
            "source": r.source,
            "description": r.description,
            "confidence": round(r.confidence, 3),
            "support": r.support,
            "consequent": r.consequent,
        }
        for r in candidate_rules[:30]
    ]

    # Feature importance (normalized to max)
    feature_importance = []
    if tree is not None:
        importances = tree.feature_importances_
        max_imp = max(importances) if max(importances) > 0 else 1
        for feat, imp in zip(FEATURE_COLS, importances):
            if imp > 0:
                feature_importance.append({
                    "feature": feat,
                    "importance": round(float(imp / max_imp * 100), 1),
                })
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    tree_count = sum(1 for r in candidate_rules if r.source == "decision_tree")
    apriori_count = sum(1 for r in candidate_rules if r.source == "apriori")

    return {
        "total_rules": len(candidate_rules),
        "tree_rules_count": tree_count,
        "apriori_rules_count": apriori_count,
        "rules": rules,
        "feature_importance": feature_importance,
    }


def export_eval_scores(data: dict) -> dict:
    """Export eval results with pre-computed pass rates."""
    eval_results = data["eval_results"]

    # Per-evaluator summary
    by_evaluator: dict[str, list] = {}
    for r in eval_results:
        by_evaluator.setdefault(r.evaluator, []).append(r)

    summary = {}
    for evaluator, results in by_evaluator.items():
        pass_count = sum(1 for r in results if r.answer == "Pass")
        summary[evaluator] = {
            "pass_rate": round(pass_count / len(results) * 100, 1),
            "total": len(results),
            "pass_count": pass_count,
        }

    overall_pass = sum(1 for r in eval_results if r.answer == "Pass")
    overall_rate = round(overall_pass / len(eval_results) * 100, 1) if eval_results else 0

    # Detail rows (first 50)
    details = [
        {
            "trace_id": r.trace_id,
            "evaluator": r.evaluator,
            "answer": r.answer,
            "source": r.source,
        }
        for r in eval_results[:50]
    ]

    return {
        "overall_pass_rate": overall_rate,
        "overall_total": len(eval_results),
        "summary": summary,
        "details": details,
    }


def export_handoffs_samples(data: dict) -> dict:
    """Export handoff for every trace, keyed by night_id."""
    traces = data["traces"]
    final_labels = data["final_labels"]
    handoffs_map = data["handoffs_map"]

    samples = {}
    for trace in traces:
        handoff = handoffs_map.get(trace.night_id)
        if not handoff:
            continue

        final_label = final_labels.get(trace.night_id, trace.ground_truth_label)
        spo2 = trace.spo2

        samples[trace.night_id] = {
            "trace_id": trace.night_id,
            "urgency_level": handoff.urgency_level,
            "summary_text": handoff.summary_text,
            "source": handoff.source,
            "final_label": final_label,
            "baby": {
                "baby_id": trace.baby.baby_id,
                "gestational_age_weeks": trace.baby.gestational_age_weeks,
                "ga_category": trace.baby.ga_category,
                "birth_weight_grams": trace.baby.birth_weight_grams,
                "days_since_birth": trace.baby.days_since_birth,
                "known_conditions": trace.baby.known_conditions,
            },
            "stats": {
                "mean_spo2": round(float(np.mean(spo2)), 1),
                "min_spo2": round(float(np.min(spo2)), 0),
                "n_events": len(trace.events),
            },
        }

    return samples


def export_hl7_messages(data: dict) -> dict:
    """Export pre-built HL7 messages for one trace per label type."""
    traces = data["traces"]
    final_labels = data["final_labels"]
    final_sources = data["final_sources"]
    handoffs_map = data["handoffs_map"]
    tier1_results = data["tier1_results"]

    label_samples = {}
    for t in traces:
        gt = t.ground_truth_label
        if gt not in label_samples:
            label_samples[gt] = t

    result = {}
    for label, trace in label_samples.items():
        # Build messages
        adt_msg = build_adt_a01(trace.baby)
        ack_msg = build_ack_a01(adt_msg)
        parsed_baby = parse_adt_a01(adt_msg)

        final_label = final_labels.get(trace.night_id, label)
        source = final_sources.get(trace.night_id, "pipeline")
        triage = FinalTriage(
            trace_id=trace.night_id,
            baby_id=trace.baby.baby_id,
            ground_truth=trace.ground_truth_label,
            final_label=final_label,
            source=source,
            confidence=0.95,
        )
        handoff = handoffs_map.get(trace.night_id)
        if not handoff:
            handoff = generate_handoff(trace, final_label, use_llm=False)

        tier1_events = [r.events_detected for r in tier1_results if r.trace_id == trace.night_id]
        rule_events = tier1_events[0] if tier1_events else []
        oru_msg = build_oru_r01(trace, triage, handoff, rule_events=rule_events)

        # Round-trip validation
        rt_fields = [
            {"field": "baby_id", "original": trace.baby.baby_id, "parsed": parsed_baby.baby_id},
            {"field": "ga_weeks", "original": str(trace.baby.gestational_age_weeks), "parsed": str(parsed_baby.gestational_age_weeks)},
            {"field": "birth_weight", "original": str(trace.baby.birth_weight_grams), "parsed": str(parsed_baby.birth_weight_grams)},
            {"field": "spo2_baseline", "original": f"{trace.baby.spo2_baseline:.1f}", "parsed": f"{parsed_baby.spo2_baseline:.1f}"},
            {"field": "ga_category", "original": trace.baby.ga_category, "parsed": parsed_baby.ga_category},
        ]
        for f in rt_fields:
            f["match"] = f["original"] == f["parsed"]

        # Segment counts
        adt_seg_count = len([s for s in adt_msg.split("\r") if s.strip()])
        oru_obx_count = sum(1 for s in oru_msg.split("\r") if s.startswith("OBX"))
        round_trip_match = all(f["match"] for f in rt_fields)

        result[label] = {
            "trace_id": trace.night_id,
            "adt_a01": adt_msg,
            "ack_a01": ack_msg,
            "oru_r01": oru_msg,
            "round_trip_fields": rt_fields,
            "adt_segment_count": adt_seg_count,
            "oru_obx_count": oru_obx_count,
            "round_trip_match": round_trip_match,
            "urgency_level": handoff.urgency_level,
            "final_label": final_label,
            "baby": {
                "baby_id": trace.baby.baby_id,
                "gestational_age_weeks": trace.baby.gestational_age_weeks,
                "ga_category": trace.baby.ga_category,
                "birth_weight_grams": trace.baby.birth_weight_grams,
                "days_since_birth": trace.baby.days_since_birth,
                "known_conditions": trace.baby.known_conditions,
                "spo2_baseline": round(trace.baby.spo2_baseline, 1),
            },
            "stats": {
                "mean_spo2": round(float(np.mean(trace.spo2)), 1),
                "min_spo2": round(float(np.min(trace.spo2)), 0),
            },
        }

    # Segment mapping reference (static)
    result["_segment_mapping"] = [
        {"field": "PID-3", "name": "Patient ID", "source": "baby.baby_id", "description": "Medical record number"},
        {"field": "PID-7", "name": "Date of Birth", "source": "computed from days_since_birth", "description": "HL7 format YYYYMMDD"},
        {"field": "OBX-3", "name": "Observation ID", "source": "LOINC 59408-5", "description": "Pulse oximetry observation"},
        {"field": "OBX-5", "name": "Mean SpO2", "source": "np.mean(trace.spo2)", "description": "Overnight mean saturation"},
        {"field": "OBX-8", "name": "Abnormal Flags", "source": "urgency → AA/A/H/N", "description": "Drives clinical alerting rules"},
        {"field": "OBX-14", "name": "Observation Time", "source": "trace.timestamp_start", "description": "Audit trail timestamp"},
        {"field": "OBR-4", "name": "Service ID", "source": "LOINC 59408-5", "description": "Overnight SpO2 monitoring order"},
        {"field": "NTE-3", "name": "Comment", "source": "handoff.summary_text", "description": "Nurse handoff narrative"},
        {"field": "DG1-3", "name": "Diagnosis", "source": "ICD-10 (P28.4, P27.1)", "description": "Known neonatal conditions"},
        {"field": "MSA-1", "name": "ACK Code", "source": "AA (accepted)", "description": "Rhapsody handshake confirmation"},
    ]

    # Production considerations (static)
    result["_production_notes"] = [
        {"title": "ACK/NAK Handling", "description": "Rhapsody retries on NAK, routes to error queue after N failures. Configurable retry with exponential backoff."},
        {"title": "IHE PCD-01 Profile", "description": "Patient Care Device observation — IHE standard for pulse oximetry and physiological monitoring data."},
        {"title": "PHI Encryption", "description": "TLS 1.2+ required for HL7 messages in transit per HIPAA Security Rule. Message-level encryption at rest."},
        {"title": "Error Queue Routing", "description": "Failed messages route to dead letter queue. Rhapsody provides error visualization, replay, and depth alerting."},
        {"title": "Conformance Validation", "description": "HL7 message structure validation against hospital-specific conformance profiles before routing."},
        {"title": "Batch vs Real-Time", "description": "Overnight SpO2 = batch ORU. Rhapsody also supports real-time streaming with waveform data in OBX segments."},
    ]

    return result


def export_waveforms(data: dict, waveforms_dir: Path):
    """Export downsampled waveforms as individual trace files."""
    traces = data["traces"]
    waveforms_dir.mkdir(parents=True, exist_ok=True)

    for trace in traces:
        spo2 = trace.spo2[::DOWNSAMPLE_STEP]
        accel = trace.accel_magnitude[::DOWNSAMPLE_STEP]
        n_points = len(spo2)
        hours = [round(i * DOWNSAMPLE_STEP / 3600, 4) for i in range(n_points)]

        waveform = {
            "spo2": [round(float(v), 1) for v in spo2],
            "accel": [round(float(v), 3) for v in accel],
            "hours": hours,
        }

        filepath = waveforms_dir / f"{trace.night_id}.json"
        with open(filepath, "w") as f:
            json.dump(waveform, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_json(filepath: Path, data):
    """Write JSON with compact formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size_kb = filepath.stat().st_size / 1024
    print(f"  {filepath.name}: {size_kb:.1f} KB")


def main():
    print("=" * 60)
    print("SpO2 Eval Pipeline — Dashboard Data Export")
    print("=" * 60)
    print(f"Config: {N_BABIES} babies, {NIGHTS} nights, seed={SEED}")
    print()

    data = run_pipeline()
    print()
    print("Exporting JSON files...")

    # Clean export dir
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    write_json(EXPORT_DIR / "pipeline-summary.json", export_pipeline_summary(data))
    write_json(EXPORT_DIR / "traces-meta.json", export_traces_meta(data))
    write_json(EXPORT_DIR / "coverage-breakdown.json", export_coverage_breakdown(data))
    write_json(EXPORT_DIR / "rules-discovered.json", export_rules_discovered(data))
    write_json(EXPORT_DIR / "eval-scores.json", export_eval_scores(data))
    write_json(EXPORT_DIR / "handoffs-samples.json", export_handoffs_samples(data))
    write_json(EXPORT_DIR / "hl7-messages.json", export_hl7_messages(data))

    print("\nExporting waveforms (individual trace files)...")
    waveforms_dir = EXPORT_DIR / "waveforms"
    export_waveforms(data, waveforms_dir)
    wf_count = len(list(waveforms_dir.glob("*.json")))
    total_kb = sum(f.stat().st_size for f in waveforms_dir.glob("*.json")) / 1024
    print(f"  {wf_count} waveform files, {total_kb:.0f} KB total")

    # Summary
    total_size = sum(f.stat().st_size for f in EXPORT_DIR.rglob("*.json")) / 1024
    print(f"\nDone! Total export: {total_size:.0f} KB in {EXPORT_DIR}")


if __name__ == "__main__":
    main()
