"""SpO2 AI Eval Pipeline — Streamlit Dashboard.

Four views + interactive pipeline runner.
Run: streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_gen.synthetic import NightTrace, generate_dataset, generate_baby_cohort, generate_trace
from src.rules.tier1_engine import run_tier1, apply_rules
from src.patterns.feature_eng import build_feature_matrix
from src.patterns.miner import run_pattern_mining, FEATURE_COLS
from src.classifier.tier2 import train_tier2, predict_tier2
from src.classifier.expert_sim import run_expert_queue
from src.handoff.generator import generate_handoff
from src.evals.clinical_accuracy import evaluate_clinical_accuracy
from src.evals.handoff_quality import evaluate_handoff_quality
from src.evals.artifact_handling import evaluate_artifact_handling
from src.evals.base import EvalResult
from src.llm_utils import get_tracker, reset_tracker
from app.components.trace_viewer import plot_trace


st.set_page_config(
    page_title="SpO2 AI Eval Pipeline",
    page_icon="🫁",
    layout="wide",
)

st.title("Neonatal SpO2 AI Eval Pipeline")
st.caption("Portfolio demo — synthetic data, multi-tier triage, LLM evals")


# ---------------------------------------------------------------------------
# Session state: run the pipeline once, cache results
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running pipeline (Phases 1-4)...")
def _run_phases_1_4(n_babies: int, nights: int, seed: int):
    """Run Phases 1-4 and return serializable results."""
    traces = generate_dataset(n_babies, nights, seed)
    tier1_results, unlabeled = run_tier1(traces)

    all_df = build_feature_matrix(traces, tier1_results)
    candidate_rules, tree = run_pattern_mining(all_df)

    model, le, metrics = train_tier2(tier1_results, traces)
    tier2_results = predict_tier2(model, le, unlabeled)

    expert_traces = [
        t for t in unlabeled
        for r in tier2_results
        if r.trace_id == t.night_id and r.routed_to == "expert_queue"
    ]
    expert_results = run_expert_queue(expert_traces, tier2_results, seed=seed)

    # Build final label map
    final_labels = {}
    final_sources = {}
    for r in tier1_results:
        if r.auto_labeled:
            final_labels[r.trace_id] = r.label
            final_sources[r.trace_id] = "Tier 1 (rules)"
    for r in tier2_results:
        if r.routed_to == "auto":
            final_labels[r.trace_id] = r.predicted_label
            final_sources[r.trace_id] = "Tier 2 (classifier)"
    for r in expert_results:
        final_labels[r.trace_id] = r.expert_label
        final_sources[r.trace_id] = "Expert review"

    return (traces, tier1_results, tier2_results, expert_results,
            candidate_rules, tree, metrics, final_labels, final_sources, all_df)


def _run_evals_mock(traces, final_labels, handoffs_map, seed=42):
    """Run mock evals on all traces."""
    rng = np.random.default_rng(seed)
    results = []
    for trace in traces:
        label = final_labels.get(trace.night_id, "normal")
        s = int(rng.integers(0, 2**31))

        results.append(evaluate_clinical_accuracy(
            trace, label, use_llm=False, seed=s))

        handoff = handoffs_map.get(trace.night_id)
        if handoff:
            results.append(evaluate_handoff_quality(
                trace, handoff, label, use_llm=False, seed=s+1))

        results.append(evaluate_artifact_handling(
            trace, label, use_llm=False, seed=s+2))

    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("View", [
    "Pipeline Overview",
    "Pre-Annotation Coverage",
    "Rule Discovery",
    "Eval Scores",
    "Sample Handoffs",
    "Run Single Trace",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Settings")
n_babies = st.sidebar.slider("Babies", 10, 200, 100)
nights = st.sidebar.slider("Nights per baby", 1, 5, 3)
seed = st.sidebar.number_input("Random seed", value=42)

# Run pipeline
data = _run_phases_1_4(n_babies, nights, int(seed))
(traces, tier1_results, tier2_results, expert_results,
 candidate_rules, tree, classifier_metrics, final_labels, final_sources, features_df) = data

# Generate mock handoffs
handoffs_map = {}
for trace in traces:
    label = final_labels.get(trace.night_id, "normal")
    handoffs_map[trace.night_id] = generate_handoff(trace, label, use_llm=False)


# ---------------------------------------------------------------------------
# Page: Pipeline Overview
# ---------------------------------------------------------------------------

if page == "Pipeline Overview":
    st.header("Pipeline Overview")

    total = len(traces)
    t1 = sum(1 for r in tier1_results if r.auto_labeled)
    t2 = sum(1 for r in tier2_results if r.routed_to == "auto")
    eq = len(expert_results)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Traces", total)
    col2.metric("Tier 1 (Rules)", f"{t1} ({t1/total*100:.0f}%)")
    col3.metric("Tier 2 (ML)", f"{t2} ({t2/total*100:.0f}%)")
    col4.metric("Expert Queue", f"{eq} ({eq/total*100:.0f}%)")

    # Accuracy by tier
    st.subheader("Accuracy by Tier")
    tier_data = []
    for r in tier1_results:
        if r.auto_labeled:
            tier_data.append({"Tier": "Tier 1", "Correct": r.label == r.ground_truth})
    for r in tier2_results:
        if r.routed_to == "auto":
            tier_data.append({"Tier": "Tier 2", "Correct": r.predicted_label == r.ground_truth})
    for r in expert_results:
        tier_data.append({"Tier": "Expert", "Correct": r.expert_label == r.ground_truth})

    tier_df = pd.DataFrame(tier_data)
    if not tier_df.empty:
        acc = tier_df.groupby("Tier")["Correct"].mean().reset_index()
        acc.columns = ["Tier", "Accuracy"]
        acc["Accuracy"] = (acc["Accuracy"] * 100).round(1)
        st.dataframe(acc, use_container_width=True, hide_index=True)

    # Ground truth distribution
    st.subheader("Ground Truth Distribution")
    gt_counts = Counter(t.ground_truth_label for t in traces)
    fig = px.pie(
        names=list(gt_counts.keys()),
        values=list(gt_counts.values()),
        color_discrete_sequence=["#16a34a", "#dc2626", "#f59e0b", "#6b7280"],
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Pre-Annotation Coverage
# ---------------------------------------------------------------------------

elif page == "Pre-Annotation Coverage":
    st.header("Pre-Annotation Coverage")

    total = len(traces)
    t1 = sum(1 for r in tier1_results if r.auto_labeled)
    t2 = sum(1 for r in tier2_results if r.routed_to == "auto")
    eq = len(expert_results)

    # Stacked bar
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Tier 1 (Rules)", x=["Coverage"], y=[t1],
                         marker_color="#16a34a"))
    fig.add_trace(go.Bar(name="Tier 2 (ML)", x=["Coverage"], y=[t2],
                         marker_color="#2563eb"))
    fig.add_trace(go.Bar(name="Expert Queue", x=["Coverage"], y=[eq],
                         marker_color="#f59e0b"))
    fig.update_layout(barmode="stack", height=350, yaxis_title="Traces")
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown by ground truth label × tier
    st.subheader("Breakdown by Pattern Type")
    rows = []
    for r in tier1_results:
        if r.auto_labeled:
            rows.append({"Ground Truth": r.ground_truth, "Tier": "Tier 1", "Label": r.label})
    for r in tier2_results:
        if r.routed_to == "auto":
            rows.append({"Ground Truth": r.ground_truth, "Tier": "Tier 2", "Label": r.predicted_label})
    for r in expert_results:
        rows.append({"Ground Truth": r.ground_truth, "Tier": "Expert", "Label": r.expert_label})

    if rows:
        breakdown = pd.DataFrame(rows)
        ct = pd.crosstab(breakdown["Ground Truth"], breakdown["Tier"])
        st.dataframe(ct, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Rule Discovery
# ---------------------------------------------------------------------------

elif page == "Rule Discovery":
    st.header("Discovered Rules")
    st.caption(f"{len(candidate_rules)} candidate rules from decision tree + Apriori mining")

    if candidate_rules:
        rules_df = pd.DataFrame([
            {
                "ID": r.rule_id,
                "Source": r.source,
                "Description": r.description,
                "Confidence": r.confidence,
                "Support": r.support,
            }
            for r in candidate_rules[:30]
        ])
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

    # Feature importance from decision tree
    if tree is not None:
        st.subheader("Decision Tree Feature Importance")
        importances = tree.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)
        feat_imp = feat_imp[feat_imp["Importance"] > 0]

        if not feat_imp.empty:
            fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h")
            fig.update_layout(height=max(300, len(feat_imp) * 30))
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Eval Scores
# ---------------------------------------------------------------------------

elif page == "Eval Scores":
    st.header("LLM Evaluator Scores (Mock Mode)")
    st.caption("Using mock evaluators — switch to live mode for real Claude judgments")

    eval_results = _run_evals_mock(traces, final_labels, handoffs_map, seed=int(seed))

    eval_df = pd.DataFrame([
        {"Evaluator": r.evaluator, "Answer": r.answer, "Source": r.source}
        for r in eval_results
    ])

    if not eval_df.empty:
        # Pass rates per evaluator
        pass_rates = eval_df.groupby("Evaluator")["Answer"].apply(
            lambda x: (x == "Pass").mean() * 100
        ).reset_index()
        pass_rates.columns = ["Evaluator", "Pass Rate (%)"]

        col1, col2, col3 = st.columns(3)
        for i, (_, row) in enumerate(pass_rates.iterrows()):
            col = [col1, col2, col3][i]
            col.metric(row["Evaluator"].replace("_", " ").title(),
                       f"{row['Pass Rate (%)']:.1f}%")

        # Bar chart
        fig = px.bar(pass_rates, x="Evaluator", y="Pass Rate (%)",
                     color="Evaluator", range_y=[0, 100])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed results
        st.subheader("Detailed Results")
        st.dataframe(eval_df.head(50), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Sample Handoffs
# ---------------------------------------------------------------------------

elif page == "Sample Handoffs":
    st.header("Sample Nurse Handoffs")

    # Pick sample traces (one per label)
    label_samples = {}
    for trace in traces:
        gt = trace.ground_truth_label
        if gt not in label_samples:
            label_samples[gt] = trace

    selected_label = st.selectbox(
        "Select pattern type",
        list(label_samples.keys()),
    )
    trace = label_samples[selected_label]

    # Show trace plot
    fig = plot_trace(trace, show_accel=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show handoff
    final_label = final_labels.get(trace.night_id, trace.ground_truth_label)
    source = final_sources.get(trace.night_id, "unknown")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline Handoff (Mock)")
        handoff = handoffs_map.get(trace.night_id)
        if handoff:
            st.info(f"**{handoff.urgency_level}**")
            st.write(handoff.summary_text)
            st.caption(f"Source: {handoff.source} | Triage: {final_label} via {source}")

    with col2:
        st.subheader("Trace Details")
        st.write(f"**Baby ID:** {trace.baby.baby_id}")
        st.write(f"**GA:** {trace.baby.gestational_age_weeks} weeks ({trace.baby.ga_category})")
        st.write(f"**Birth weight:** {trace.baby.birth_weight_grams}g")
        st.write(f"**Days since birth:** {trace.baby.days_since_birth}")
        st.write(f"**Conditions:** {', '.join(trace.baby.known_conditions)}")
        st.write(f"**Ground truth:** {trace.ground_truth_label}")
        st.write(f"**Pipeline label:** {final_label}")
        st.write(f"**Mean SpO2:** {np.mean(trace.spo2):.1f}%")
        st.write(f"**Min SpO2:** {np.min(trace.spo2):.0f}%")
        st.write(f"**Events:** {len(trace.events)}")


# ---------------------------------------------------------------------------
# Page: Run Single Trace
# ---------------------------------------------------------------------------

elif page == "Run Single Trace":
    st.header("Interactive Single-Trace Demo")

    col1, col2 = st.columns(2)
    with col1:
        ga = st.slider("Gestational age (weeks)", 24, 42, 34)
        pattern = st.selectbox("Pattern type", ["normal", "urgent", "borderline", "artifact"])
    with col2:
        trace_seed = st.number_input("Trace seed", value=123)

    if st.button("Generate & Analyze", type="primary"):
        rng = np.random.default_rng(int(trace_seed))
        babies = generate_baby_cohort(1, rng)
        baby = babies[0]
        # Override GA
        from src.data_gen.synthetic import _classify_ga
        baby.gestational_age_weeks = ga
        baby.ga_category = _classify_ga(ga)

        trace = generate_trace(baby, pattern, 1, rng)

        # Show trace
        fig = plot_trace(trace, show_accel=True)
        st.plotly_chart(fig, use_container_width=True)

        # Run rules
        rule_result = apply_rules(trace)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Tier 1 (Rules)")
            if rule_result.auto_labeled:
                st.success(f"Label: **{rule_result.label}**")
                st.write(f"Rule: {rule_result.rule_triggered}")
                st.write(f"Confidence: {rule_result.confidence}")
            else:
                st.warning("No rule matched → Tier 2")

        with col2:
            st.subheader("Classification")
            final_label = rule_result.label or pattern  # fallback
            st.write(f"**Final label:** {final_label}")
            st.write(f"**Ground truth:** {pattern}")
            if final_label == pattern:
                st.success("Correct!")
            else:
                st.error(f"Mismatch: predicted {final_label}, actual {pattern}")

        with col3:
            st.subheader("Handoff")
            handoff = generate_handoff(trace, final_label, use_llm=False)
            st.info(f"**{handoff.urgency_level}**")
            st.write(handoff.summary_text)

        # Eval results
        st.subheader("Evaluator Results (Mock)")
        s = int(trace_seed)
        evals = [
            evaluate_clinical_accuracy(trace, final_label, seed=s),
            evaluate_handoff_quality(trace, handoff, final_label, seed=s+1),
            evaluate_artifact_handling(trace, final_label, seed=s+2),
        ]
        eval_df = pd.DataFrame([
            {"Evaluator": e.evaluator, "Answer": e.answer, "Reasoning": e.reasoning}
            for e in evals
        ])
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
