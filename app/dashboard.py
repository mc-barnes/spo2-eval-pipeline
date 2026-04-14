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

# ---------------------------------------------------------------------------
# Owlet brand palette (matched from owletcare.com)
# ---------------------------------------------------------------------------
# Headings / dark text — dusty teal, NOT forest green
TEAL_DARK = "#2C5F5B"
# Primary accent — the soft sage-teal from Owlet's logo/buttons
TEAL_PRIMARY = "#5BA69E"
# Lighter accent — icon circle backgrounds
TEAL_LIGHT = "#6BACA4"
# Muted sage for secondary elements
SAGE = "#8CBDB7"
# Very light teal tint for subtle backgrounds
SAGE_BG = "#E8F1EF"
# Page background — warm cream (Owlet product page)
CREAM_BG = "#F7F0EA"
# Card / content background
WARM_WHITE = "#FEFCFA"
# Body text — neutral warm gray
BODY_TEXT = "#4A5568"
# Subtle borders
BORDER = "#E2DDD8"
# Clinical status colors (kept distinct but warmed)
URGENT_RED = "#C1565B"
AMBER = "#D4A054"
NEUTRAL_GRAY = "#9CA3AF"

# Shared Plotly layout
PLOTLY_LAYOUT = dict(
    font=dict(family="Georgia, 'Times New Roman', serif", color=TEAL_DARK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=WARM_WHITE,
    margin=dict(l=40, r=20, t=40, b=40),
)


st.set_page_config(
    page_title="SpO2 AI Eval Pipeline",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Owlet-inspired theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

/* --- Global --- */
.stApp {
    background-color: #F7F0EA;
}

/* --- Sidebar: warm cream, not dark --- */
section[data-testid="stSidebar"] {
    background-color: #FEFCFA;
    border-right: 1px solid #E2DDD8;
}
section[data-testid="stSidebar"] * {
    color: #2C5F5B !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #6BACA4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #E2DDD8 !important;
}

/* --- Typography: serif headings like Owlet --- */
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #2C5F5B !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}
h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #2C5F5B !important;
    font-weight: 500 !important;
}
p, span, div, label, li {
    font-family: 'DM Sans', system-ui, sans-serif;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #7A8B87 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* --- Metric cards: clean white with warm border --- */
[data-testid="stMetric"] {
    background: #FEFCFA;
    border: 1px solid #E2DDD8;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(44, 95, 91, 0.04);
}
[data-testid="stMetricLabel"] {
    color: #7A8B87 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #2C5F5B !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 600 !important;
}

/* --- Dataframes --- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E2DDD8;
}

/* --- Buttons: Owlet teal --- */
.stButton > button[kind="primary"] {
    background-color: #5BA69E !important;
    border-color: #5BA69E !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2C5F5B !important;
}
.stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* --- Alerts --- */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* --- Selectbox --- */
.stSelectbox > div > div {
    border-radius: 8px !important;
}

/* --- Expander --- */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif !important;
    color: #2C5F5B !important;
}

/* --- Slider: Owlet teal instead of red --- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #5BA69E !important;
    border-color: #5BA69E !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] {
    color: #7A8B87 !important;
}
/* Slider track fill */
div[data-baseweb="slider"] div[role="progressbar"] > div {
    background-color: #5BA69E !important;
}
/* Slider thumb value */
div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: #5BA69E !important;
}

/* --- Radio buttons: teal accent --- */
.stRadio [data-testid="stMarkdownContainer"] {
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="radio"] label span[data-testid] {
    color: #5BA69E !important;
}
/* Override the default Streamlit red primary color */
:root {
    --primary-color: #5BA69E;
}
.st-emotion-cache-1gulkj5 {
    background-color: #5BA69E !important;
}
/* Radio dot fill */
div[role="radiogroup"] label div[data-checked="true"] {
    background-color: #5BA69E !important;
    border-color: #5BA69E !important;
}
/* Generic primary color overrides */
[data-testid="stWidgetLabel"] {
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<h1 style="margin-bottom: 2px; font-size: 2rem;">Neonatal SpO2 AI Eval Pipeline</h1>',
    unsafe_allow_html=True,
)
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

st.sidebar.markdown(
    '<p style="font-family: Playfair Display, Georgia, serif; font-size:1.3rem; '
    'font-weight:600; color:#2C5F5B !important; margin-bottom:2px;">SpO2 Pipeline</p>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Neonatal monitoring demo")
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

    # Two-column layout: accuracy chart + pie chart
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Accuracy by Tier")
        tier_data = []
        for r in tier1_results:
            if r.auto_labeled:
                tier_data.append({"Tier": "Tier 1 (Rules)", "Correct": r.label == r.ground_truth})
        for r in tier2_results:
            if r.routed_to == "auto":
                tier_data.append({"Tier": "Tier 2 (ML)", "Correct": r.predicted_label == r.ground_truth})
        for r in expert_results:
            tier_data.append({"Tier": "Expert Review", "Correct": r.expert_label == r.ground_truth})

        tier_df = pd.DataFrame(tier_data)
        if not tier_df.empty:
            acc = tier_df.groupby("Tier")["Correct"].mean().reset_index()
            acc.columns = ["Tier", "Accuracy"]
            acc["Accuracy"] = (acc["Accuracy"] * 100).round(1)

            fig_acc = go.Figure(go.Bar(
                x=acc["Accuracy"],
                y=acc["Tier"],
                orientation="h",
                marker_color=[TEAL_PRIMARY, SAGE, AMBER],
                text=[f"{v:.1f}%" for v in acc["Accuracy"]],
                textposition="auto",
                textfont=dict(color="white", size=13, family="DM Sans, sans-serif"),
            ))
            fig_acc.update_layout(
                **PLOTLY_LAYOUT,
                height=250,
                xaxis=dict(range=[0, 105], title="Accuracy (%)", gridcolor=BORDER),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    with right_col:
        st.subheader("Ground Truth Distribution")
        gt_counts = Counter(t.ground_truth_label for t in traces)
        label_order = ["normal", "borderline", "urgent", "artifact"]
        ordered_labels = [l for l in label_order if l in gt_counts]
        label_colors = {
            "normal": TEAL_LIGHT, "borderline": AMBER,
            "urgent": URGENT_RED, "artifact": NEUTRAL_GRAY,
        }

        fig_pie = go.Figure(go.Pie(
            labels=ordered_labels,
            values=[gt_counts[l] for l in ordered_labels],
            marker=dict(
                colors=[label_colors[l] for l in ordered_labels],
                line=dict(color=WARM_WHITE, width=2),
            ),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=12, family="DM Sans, sans-serif", color=TEAL_DARK),
            hoverinfo="label+value+percent",
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Coverage funnel
    st.subheader("Triage Funnel")
    fig_funnel = go.Figure(go.Funnel(
        y=["All Traces", "Tier 1 Auto-labeled", "Tier 2 Auto-labeled", "Expert Queue"],
        x=[total, t1, t2, eq],
        marker=dict(color=[TEAL_DARK, TEAL_PRIMARY, SAGE, AMBER]),
        textinfo="value+percent initial",
        textfont=dict(family="DM Sans, sans-serif", size=13),
    ))
    fig_funnel.update_layout(**PLOTLY_LAYOUT, height=260)
    st.plotly_chart(fig_funnel, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Pre-Annotation Coverage
# ---------------------------------------------------------------------------

elif page == "Pre-Annotation Coverage":
    st.header("Pre-Annotation Coverage")

    total = len(traces)
    t1 = sum(1 for r in tier1_results if r.auto_labeled)
    t2 = sum(1 for r in tier2_results if r.routed_to == "auto")
    eq = len(expert_results)

    # Metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Tier 1 (Rules)", f"{t1} traces", f"{t1/total*100:.0f}% of total")
    col2.metric("Tier 2 (ML)", f"{t2} traces", f"{t2/total*100:.0f}% of total")
    col3.metric("Expert Queue", f"{eq} traces", f"{eq/total*100:.0f}% of total")

    # Stacked bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Tier 1 (Rules)", x=["Coverage"], y=[t1],
        marker_color=TEAL_MID, text=[t1], textposition="inside",
        textfont=dict(color="white", size=14),
    ))
    fig.add_trace(go.Bar(
        name="Tier 2 (ML)", x=["Coverage"], y=[t2],
        marker_color=SAGE, text=[t2], textposition="inside",
        textfont=dict(color=TEAL_DARK, size=14),
    ))
    fig.add_trace(go.Bar(
        name="Expert Queue", x=["Coverage"], y=[eq],
        marker_color=AMBER, text=[eq], textposition="inside",
        textfont=dict(color="white", size=14),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="stack", height=300,
        yaxis_title="Traces", xaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown by ground truth label x tier
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

        # Grouped bar chart instead of raw crosstab
        ct = pd.crosstab(breakdown["Ground Truth"], breakdown["Tier"])
        tier_order = ["Tier 1", "Tier 2", "Expert"]
        tier_colors_list = [TEAL_MID, SAGE, AMBER]

        fig_bt = go.Figure()
        for tier, color in zip(tier_order, tier_colors_list):
            if tier in ct.columns:
                fig_bt.add_trace(go.Bar(
                    name=tier, x=ct.index, y=ct[tier],
                    marker_color=color,
                ))
        fig_bt.update_layout(
            **PLOTLY_LAYOUT, barmode="group", height=350,
            yaxis_title="Traces", xaxis_title="Ground Truth Label",
            legend=dict(orientation="h", yanchor="bottom", y=1.05),
        )
        st.plotly_chart(fig_bt, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Rule Discovery
# ---------------------------------------------------------------------------

elif page == "Rule Discovery":
    st.header("Discovered Rules")
    st.caption(f"{len(candidate_rules)} candidate rules from decision tree + Apriori mining")

    # Split by source
    tree_rules = [r for r in candidate_rules if r.source == "decision_tree"]
    apriori_rules = [r for r in candidate_rules if r.source == "apriori"]

    col1, col2 = st.columns(2)
    col1.metric("Decision Tree Rules", len(tree_rules))
    col2.metric("Apriori Rules", len(apriori_rules))

    if candidate_rules:
        rules_df = pd.DataFrame([
            {
                "ID": r.rule_id,
                "Source": r.source,
                "Description": r.description,
                "Confidence": round(r.confidence, 3),
                "Support": round(r.support, 3),
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
            fig = go.Figure(go.Bar(
                x=feat_imp["Importance"],
                y=feat_imp["Feature"],
                orientation="h",
                marker_color=TEAL_MID,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=max(280, len(feat_imp) * 35),
                xaxis=dict(title="Importance", gridcolor="#E5E7EB"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Eval Scores
# ---------------------------------------------------------------------------

elif page == "Eval Scores":
    st.header("LLM Evaluator Scores")
    st.caption("Mock evaluators — switch to live mode for real Claude judgments")

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

        # Metric cards with nice labels
        pretty_names = {
            "artifact_handling": "Artifact Handling",
            "clinical_accuracy": "Clinical Accuracy",
            "handoff_quality": "Handoff Quality",
        }
        col1, col2, col3 = st.columns(3)
        for i, (_, row) in enumerate(pass_rates.iterrows()):
            col = [col1, col2, col3][i]
            name = pretty_names.get(row["Evaluator"], row["Evaluator"])
            pct = row["Pass Rate (%)"]
            col.metric(name, f"{pct:.1f}%")

        # Gauge-style bar chart
        eval_colors = [TEAL_MID, SAGE, AMBER]
        fig = go.Figure()
        for i, (_, row) in enumerate(pass_rates.iterrows()):
            name = pretty_names.get(row["Evaluator"], row["Evaluator"])
            fig.add_trace(go.Bar(
                x=[row["Pass Rate (%)"]], y=[name],
                orientation="h", name=name,
                marker_color=eval_colors[i % len(eval_colors)],
                text=[f"{row['Pass Rate (%)']:.1f}%"],
                textposition="auto",
                textfont=dict(color="white", size=13),
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=220,
            xaxis=dict(range=[0, 105], title="Pass Rate (%)", gridcolor="#E5E7EB"),
            showlegend=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed results
        with st.expander("Detailed Results", expanded=False):
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

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Pipeline Handoff")
        handoff = handoffs_map.get(trace.night_id)
        if handoff:
            # Color-coded urgency badge
            urgency_colors = {
                "URGENT": URGENT_RED, "MONITOR": AMBER,
                "ROUTINE": TEAL_MID, "ARTIFACT REVIEW": NEUTRAL_GRAY,
            }
            badge_color = urgency_colors.get(handoff.urgency_level, TEAL_LIGHT)
            st.markdown(
                f'<div style="display:inline-block; background:{badge_color}; '
                f'color:white; padding:6px 16px; border-radius:20px; '
                f'font-weight:600; font-size:0.85rem; letter-spacing:0.05em;">'
                f'{handoff.urgency_level}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(handoff.summary_text)
            st.caption(f"Source: {handoff.source} | Triage: {final_label} via {source}")

    with col2:
        st.subheader("Patient Details")
        details = {
            "Baby ID": trace.baby.baby_id,
            "GA": f"{trace.baby.gestational_age_weeks}w ({trace.baby.ga_category})",
            "Birth weight": f"{trace.baby.birth_weight_grams}g",
            "Days since birth": trace.baby.days_since_birth,
            "Conditions": ", ".join(trace.baby.known_conditions) or "None",
            "Ground truth": trace.ground_truth_label,
            "Pipeline label": final_label,
            "Mean SpO2": f"{np.mean(trace.spo2):.1f}%",
            "Min SpO2": f"{np.min(trace.spo2):.0f}%",
            "Events": len(trace.events),
        }
        for k, v in details.items():
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:4px 0; border-bottom:1px solid #E5E7EB;">'
                f'<span style="color:{TEAL_LIGHT}; font-size:0.85rem;">{k}</span>'
                f'<span style="color:{TEAL_DARK}; font-weight:500; font-size:0.85rem;">{v}</span>'
                f'</div>', unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page: Run Single Trace
# ---------------------------------------------------------------------------

elif page == "Run Single Trace":
    st.header("Interactive Single-Trace Demo")
    st.caption("Generate a trace, run it through rules, and see the handoff + evals")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ga = st.slider("Gestational age (weeks)", 24, 42, 34)
    with col2:
        pattern = st.selectbox("Pattern type", ["normal", "urgent", "borderline", "artifact"])
    with col3:
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
                st.warning("No rule matched — routes to Tier 2")

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
            urgency_colors = {
                "URGENT": URGENT_RED, "MONITOR": AMBER,
                "ROUTINE": TEAL_MID, "ARTIFACT REVIEW": NEUTRAL_GRAY,
            }
            badge_color = urgency_colors.get(handoff.urgency_level, TEAL_LIGHT)
            st.markdown(
                f'<div style="display:inline-block; background:{badge_color}; '
                f'color:white; padding:4px 14px; border-radius:20px; '
                f'font-weight:600; font-size:0.8rem;">'
                f'{handoff.urgency_level}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(handoff.summary_text)

        # Eval results
        st.subheader("Evaluator Results")
        s = int(trace_seed)
        evals = [
            evaluate_clinical_accuracy(trace, final_label, seed=s),
            evaluate_handoff_quality(trace, handoff, final_label, seed=s+1),
            evaluate_artifact_handling(trace, final_label, seed=s+2),
        ]
        eval_df = pd.DataFrame([
            {"Evaluator": e.evaluator.replace("_", " ").title(),
             "Result": e.answer, "Reasoning": e.reasoning}
            for e in evals
        ])
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
