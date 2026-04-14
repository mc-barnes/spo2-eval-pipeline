"""End-to-end pipeline orchestrator.

Connects all 7 phases: synthetic data → Tier 1 rules → pattern mining →
Tier 2 classifier → expert queue → handoff generation → LLM evals.

Mock mode (default): no API calls, $0. Pass use_llm=True for real Claude calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

from src.data_gen.synthetic import (
    NightTrace,
    generate_dataset,
    summarize_dataset,
)
from src.rules.tier1_engine import RuleResult, run_tier1
from src.patterns.feature_eng import build_feature_matrix
from src.patterns.miner import CandidateRule, run_pattern_mining
from src.classifier.tier2 import Tier2Result, train_tier2, predict_tier2
from src.classifier.expert_sim import ExpertResult, run_expert_queue


@dataclass
class CoverageReport:
    """Pipeline coverage statistics."""
    total_traces: int
    tier1_auto: int
    tier2_auto: int
    expert_queue: int
    tier1_pct: float
    tier2_pct: float
    expert_pct: float


@dataclass
class FinalTriage:
    """Merged triage result for a single trace."""
    trace_id: str
    baby_id: str
    ground_truth: str
    final_label: str
    source: str  # "tier1_rules", "tier2_classifier", "expert_review"
    confidence: float


@dataclass
class PipelineResults:
    """All outputs from a full pipeline run."""
    traces: list[NightTrace]
    tier1_results: list[RuleResult]
    tier2_results: list[Tier2Result]
    expert_results: list[ExpertResult]
    coverage: CoverageReport
    candidate_rules: list[CandidateRule]
    final_triage: list[FinalTriage]
    tree: object = None  # fitted DecisionTreeClassifier for visualization


def merge_triage(
    tier1_results: list[RuleResult],
    tier2_results: list[Tier2Result],
    expert_results: list[ExpertResult],
) -> list[FinalTriage]:
    """Merge results from all tiers into a single final triage list."""
    final = []

    # Tier 1 labeled traces
    for r in tier1_results:
        if r.auto_labeled:
            final.append(FinalTriage(
                trace_id=r.trace_id,
                baby_id=r.baby_id,
                ground_truth=r.ground_truth,
                final_label=r.label,
                source="tier1_rules",
                confidence=r.confidence,
            ))

    # Tier 2 auto-labeled traces
    for r in tier2_results:
        if r.routed_to == "auto":
            final.append(FinalTriage(
                trace_id=r.trace_id,
                baby_id=r.baby_id,
                ground_truth=r.ground_truth,
                final_label=r.predicted_label,
                source="tier2_classifier",
                confidence=r.confidence,
            ))

    # Expert-reviewed traces
    for r in expert_results:
        final.append(FinalTriage(
            trace_id=r.trace_id,
            baby_id=r.baby_id,
            ground_truth=r.ground_truth,
            final_label=r.expert_label,
            source="expert_review",
            confidence=r.expert_confidence,
        ))

    return final


def compute_coverage(
    tier1_results: list[RuleResult],
    tier2_results: list[Tier2Result],
    expert_results: list[ExpertResult],
    total_traces: int,
) -> CoverageReport:
    """Compute coverage statistics across all tiers."""
    tier1_auto = sum(1 for r in tier1_results if r.auto_labeled)
    tier2_auto = sum(1 for r in tier2_results if r.routed_to == "auto")
    expert = len(expert_results)

    return CoverageReport(
        total_traces=total_traces,
        tier1_auto=tier1_auto,
        tier2_auto=tier2_auto,
        expert_queue=expert,
        tier1_pct=round(tier1_auto / total_traces * 100, 1),
        tier2_pct=round(tier2_auto / total_traces * 100, 1),
        expert_pct=round(expert / total_traces * 100, 1),
    )


def run_pipeline(
    n_babies: int = 100,
    nights_per_baby: int = 3,
    seed: int = 42,
    use_llm: bool = False,
    llm_sample_size: int = 15,
    model: str | None = None,
) -> PipelineResults:
    """Run the full pipeline end-to-end.

    Args:
        use_llm: If True, uses Claude API for handoffs + evals (costs money).
                 If False (default), uses mock mode ($0).
        llm_sample_size: When use_llm=True, only run LLM on this many traces.
        model: Override model (e.g., "claude-haiku-4-5-20251001" for dev).
    """
    print("=" * 60)
    print(f"SpO2 AI Eval Pipeline — {'LIVE MODE' if use_llm else 'MOCK MODE'}")
    print("=" * 60)

    # Phase 1: Generate synthetic data
    print("\n[Phase 1] Generating synthetic data...")
    traces = generate_dataset(n_babies, nights_per_baby, seed)
    summarize_dataset(traces)

    # Phase 2: Tier 1 rule-based pre-annotation
    print("[Phase 2] Running Tier 1 rules...")
    tier1_results, unlabeled_traces = run_tier1(traces)

    # Phase 3: Pattern mining
    print("[Phase 3] Mining patterns...")
    all_df = build_feature_matrix(traces, tier1_results)
    candidate_rules, tree = run_pattern_mining(all_df)

    # Phase 4: Tier 2 classifier + expert queue
    print("[Phase 4] Training Tier 2 classifier...")
    model, le, metrics = train_tier2(tier1_results, traces)

    print(f"\n[Phase 4] Predicting on {len(unlabeled_traces)} unlabeled traces...")
    tier2_results = predict_tier2(model, le, unlabeled_traces)

    # Route low-confidence traces to expert queue
    expert_traces = [
        t for t in unlabeled_traces
        for r in tier2_results
        if r.trace_id == t.night_id and r.routed_to == "expert_queue"
    ]
    print(f"\n[Phase 4] Processing {len(expert_traces)} traces in expert queue...")
    expert_results = run_expert_queue(expert_traces, tier2_results, seed=seed)

    # Merge and compute coverage
    final_triage = merge_triage(tier1_results, tier2_results, expert_results)
    coverage = compute_coverage(tier1_results, tier2_results, expert_results, len(traces))

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline Coverage Summary")
    print(f"{'=' * 60}")
    print(f"Total traces:       {coverage.total_traces}")
    print(f"Tier 1 (rules):     {coverage.tier1_auto:4d} ({coverage.tier1_pct}%)")
    print(f"Tier 2 (classifier):{coverage.tier2_auto:4d} ({coverage.tier2_pct}%)")
    print(f"Expert queue:       {coverage.expert_queue:4d} ({coverage.expert_pct}%)")
    print(f"{'=' * 60}")

    # Accuracy
    correct = sum(1 for t in final_triage if t.final_label == t.ground_truth)
    print(f"\nOverall accuracy: {correct}/{len(final_triage)} = {correct/len(final_triage)*100:.1f}%")

    by_source = Counter(t.source for t in final_triage)
    for source, count in sorted(by_source.items()):
        source_correct = sum(1 for t in final_triage if t.source == source and t.final_label == t.ground_truth)
        print(f"  {source:20s}: {source_correct}/{count} = {source_correct/count*100:.1f}%")

    # Build final label map for Phases 5-6
    final_label_map = {t.trace_id: t.final_label for t in final_triage}
    final_source_map = {t.trace_id: t.source for t in final_triage}

    # Phase 6: Handoff generation (before Phase 5 — evals need the handoff text)
    print(f"\n[Phase 6] Generating handoffs ({'live' if use_llm else 'mock'})...")
    from src.handoff.generator import generate_handoff
    handoffs = {}
    trace_subset = traces[:llm_sample_size] if use_llm else traces
    for trace in trace_subset:
        label = final_label_map.get(trace.night_id, trace.ground_truth_label)
        source = final_source_map.get(trace.night_id, "pipeline")
        handoffs[trace.night_id] = generate_handoff(
            trace, label, classified_by=source,
            use_llm=use_llm, model=model,
        )
    print(f"  Generated {len(handoffs)} handoffs")

    # Phase 5: LLM evals
    print(f"\n[Phase 5] Running evaluators ({'live' if use_llm else 'mock'})...")
    from src.evals.clinical_accuracy import evaluate_clinical_accuracy
    from src.evals.handoff_quality import evaluate_handoff_quality
    from src.evals.artifact_handling import evaluate_artifact_handling

    import numpy as np
    eval_rng = np.random.default_rng(seed)
    eval_results = []
    for trace in trace_subset:
        label = final_label_map.get(trace.night_id, trace.ground_truth_label)
        s = int(eval_rng.integers(0, 2**31))

        eval_results.append(evaluate_clinical_accuracy(
            trace, label, use_llm=use_llm, model=model, seed=s))

        handoff = handoffs.get(trace.night_id)
        if handoff:
            eval_results.append(evaluate_handoff_quality(
                trace, handoff, label, use_llm=use_llm, model=model, seed=s+1))

        eval_results.append(evaluate_artifact_handling(
            trace, label, use_llm=use_llm, model=model, seed=s+2))

    # Eval summary
    from collections import Counter as Ctr
    eval_by_type = {}
    for r in eval_results:
        if r.evaluator not in eval_by_type:
            eval_by_type[r.evaluator] = {"Pass": 0, "Fail": 0}
        eval_by_type[r.evaluator][r.answer] += 1

    print(f"\n  Eval results ({len(eval_results)} total):")
    for name, counts in eval_by_type.items():
        total = counts["Pass"] + counts["Fail"]
        pct = counts["Pass"] / total * 100 if total else 0
        print(f"    {name:25s}: {counts['Pass']}/{total} Pass ({pct:.1f}%)")

    if use_llm:
        from src.llm_utils import get_tracker
        print(f"\n  {get_tracker().summary()}")

    return PipelineResults(
        traces=traces,
        tier1_results=tier1_results,
        tier2_results=tier2_results,
        expert_results=expert_results,
        coverage=coverage,
        candidate_rules=candidate_rules,
        final_triage=final_triage,
        tree=tree,
    )


if __name__ == "__main__":
    results = run_pipeline()
