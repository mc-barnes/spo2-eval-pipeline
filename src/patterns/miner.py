"""Pattern mining: discovers candidate rules from borderline/unlabeled traces.

Two complementary approaches:
1. Decision tree → interpretable if-then rules
2. Apriori association rules → frequent pattern discovery

Seeded patterns the system should discover:
- 2am dip cluster → elevated risk
- 3+ consecutive borderline nights → escalation signal
- GA <30 + desat_duration >20s → urgent despite SpO2 >90%
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


@dataclass
class CandidateRule:
    """A discovered pattern/rule with confidence score."""
    rule_id: str
    description: str
    antecedents: dict
    consequent: str
    confidence: float
    support: int
    source: str  # "decision_tree" or "apriori"


# ---------------------------------------------------------------------------
# Feature columns used for mining (exclude metadata/ID columns)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "ga_weeks", "days_since_birth", "has_apnea_condition",
    "mean_spo2", "min_spo2", "std_spo2", "pct_below_94", "pct_below_90",
    "n_desat_events", "max_desat_depth", "max_desat_duration",
    "mean_desat_duration", "desat_cluster_score",
    "hour_2_3_desat_count", "hour_2_3_vs_rest_ratio",
    "consecutive_borderline_nights",
    "accel_mean_magnitude", "accel_spike_count", "desat_accel_correlation",
    "sat_seconds",
]


# ---------------------------------------------------------------------------
# Approach 1: Decision Tree Rule Extraction
# ---------------------------------------------------------------------------

def _extract_tree_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
) -> list[CandidateRule]:
    """Walk the tree and extract if-then rules from leaf nodes."""
    tree_ = tree.tree_
    rules = []
    rule_counter = 0

    def _walk(node: int, conditions: list[str], antecedents: dict):
        nonlocal rule_counter

        if tree_.feature[node] == -2:  # leaf
            # Get the class with most samples
            class_idx = int(np.argmax(tree_.value[node][0]))
            total = int(np.sum(tree_.value[node][0]))
            class_count = int(tree_.value[node][0][class_idx])
            confidence = class_count / max(total, 1)

            if confidence >= 0.5:  # minimum confidence
                rule_counter += 1
                rules.append(CandidateRule(
                    rule_id=f"DT-{rule_counter:03d}",
                    description=" AND ".join(conditions) + f" → {class_names[class_idx]}",
                    antecedents=dict(antecedents),
                    consequent=class_names[class_idx],
                    confidence=round(confidence, 3),
                    support=total,
                    source="decision_tree",
                ))
            return

        feat_name = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]

        # Left branch: feature <= threshold
        left_cond = f"{feat_name} <= {threshold:.2f}"
        left_ante = {**antecedents, feat_name: f"<= {threshold:.2f}"}
        _walk(tree_.children_left[node], conditions + [left_cond], left_ante)

        # Right branch: feature > threshold
        right_cond = f"{feat_name} > {threshold:.2f}"
        right_ante = {**antecedents, feat_name: f"> {threshold:.2f}"}
        _walk(tree_.children_right[node], conditions + [right_cond], right_ante)

    _walk(0, [], {})
    return rules


def discover_rules_tree(
    df: pd.DataFrame,
    target_col: str = "ground_truth",
    max_depth: int = 4,
) -> tuple[list[CandidateRule], DecisionTreeClassifier]:
    """Train a decision tree and extract interpretable rules.

    Returns the discovered rules and the fitted tree (for visualization).
    """
    feature_df = df[FEATURE_COLS].copy()
    feature_df["has_apnea_condition"] = feature_df["has_apnea_condition"].astype(int)

    target = df[target_col]
    le = LabelEncoder()
    y = le.fit_transform(target)

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=5, random_state=42)
    tree.fit(feature_df, y)

    rules = _extract_tree_rules(tree, FEATURE_COLS, list(le.classes_))

    # Sort by confidence descending
    rules.sort(key=lambda r: (-r.confidence, -r.support))

    return rules, tree


# ---------------------------------------------------------------------------
# Approach 2: Apriori Association Rules
# ---------------------------------------------------------------------------

def _discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bin continuous features for association rule mining."""
    binned = pd.DataFrame()

    # GA categories
    binned["ga_very_preterm"] = df["ga_weeks"] < 32
    binned["ga_preterm"] = (df["ga_weeks"] >= 32) & (df["ga_weeks"] < 37)
    binned["ga_term"] = df["ga_weeks"] >= 37

    # SpO2 levels
    binned["low_mean_spo2"] = df["mean_spo2"] < 94
    binned["high_pct_below_94"] = df["pct_below_94"] > 0.10
    binned["any_below_90"] = df["pct_below_90"] > 0

    # Event counts
    binned["many_desats"] = df["n_desat_events"] >= 5
    binned["deep_desat"] = df["max_desat_depth"] > 5
    binned["long_desat"] = df["max_desat_duration"] > 20

    # Temporal patterns
    binned["high_cluster"] = df["desat_cluster_score"] >= 3
    binned["2am_dips"] = df["hour_2_3_desat_count"] >= 2
    binned["high_2am_ratio"] = df["hour_2_3_vs_rest_ratio"] > 2.0

    # Multi-night
    binned["consec_borderline_3plus"] = df["consecutive_borderline_nights"] >= 3

    # Conditions
    binned["has_apnea"] = df["has_apnea_condition"]

    # Accel
    binned["high_accel"] = df["accel_spike_count"] >= 2
    binned["high_desat_accel_corr"] = df["desat_accel_correlation"] > 0.3

    # Ground truth labels (as consequents)
    for label in ["normal", "urgent", "borderline", "artifact"]:
        binned[f"label_{label}"] = df["ground_truth"] == label

    return binned


def discover_rules_apriori(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.6,
) -> list[CandidateRule]:
    """Find association rules using Apriori algorithm."""
    if not HAS_MLXTEND:
        print("WARNING: mlxtend not installed, skipping Apriori mining")
        return []

    binned = _discretize_features(df)

    # Convert to transaction format: each row is a set of "true" items
    transactions = []
    for _, row in binned.iterrows():
        transaction = [col for col in binned.columns if row[col]]
        transactions.append(transaction)

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # Find frequent itemsets
    freq_items = apriori(te_df, min_support=min_support, use_colnames=True)
    if freq_items.empty:
        return []

    # Generate rules
    rules_df = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    if rules_df.empty:
        return []

    # Filter: consequent must be a label
    label_cols = {"label_normal", "label_urgent", "label_borderline", "label_artifact"}
    rules_df = rules_df[
        rules_df["consequents"].apply(lambda x: len(x & label_cols) > 0)
    ]

    candidate_rules = []
    for i, row in rules_df.iterrows():
        antecedents = {item: "True" for item in row["antecedents"]}
        consequent_labels = [c for c in row["consequents"] if c in label_cols]
        if not consequent_labels:
            continue
        consequent = consequent_labels[0].replace("label_", "")

        candidate_rules.append(CandidateRule(
            rule_id=f"AP-{len(candidate_rules)+1:03d}",
            description=" AND ".join(sorted(row["antecedents"])) + f" → {consequent}",
            antecedents=antecedents,
            consequent=consequent,
            confidence=round(float(row["confidence"]), 3),
            support=int(row["support"] * len(df)),
            source="apriori",
        ))

    candidate_rules.sort(key=lambda r: (-r.confidence, -r.support))

    # Deduplicate: if two rules have the same consequent and one's antecedents
    # are a subset of the other's, keep the simpler one (fewer antecedents)
    deduped = []
    seen_consequent_antecedent_pairs = set()
    for rule in candidate_rules:
        key = (frozenset(rule.antecedents.keys()), rule.consequent)
        if key not in seen_consequent_antecedent_pairs:
            seen_consequent_antecedent_pairs.add(key)
            deduped.append(rule)

    # Re-number
    for i, rule in enumerate(deduped):
        rule.rule_id = f"AP-{i+1:03d}"

    return deduped[:50]  # cap at top 50 rules


# ---------------------------------------------------------------------------
# Combined mining
# ---------------------------------------------------------------------------

def run_pattern_mining(
    df: pd.DataFrame,
) -> tuple[list[CandidateRule], DecisionTreeClassifier | None]:
    """Run both mining approaches and combine results.

    Returns all discovered rules and the decision tree (for visualization).
    """
    print(f"\nRunning pattern mining on {len(df)} traces...")

    # Decision tree
    tree_rules, tree = discover_rules_tree(df)
    print(f"Decision tree: {len(tree_rules)} rules discovered")

    # Apriori
    apriori_rules = discover_rules_apriori(df)
    print(f"Apriori: {len(apriori_rules)} rules discovered")

    all_rules = tree_rules + apriori_rules
    all_rules.sort(key=lambda r: (-r.confidence, -r.support))

    # Print top rules
    print(f"\n{'='*60}")
    print(f"Top Discovered Rules ({len(all_rules)} total)")
    print(f"{'='*60}")
    for rule in all_rules[:15]:
        print(f"  [{rule.source:14s}] conf={rule.confidence:.2f} "
              f"sup={rule.support:3d} | {rule.description}")
    print(f"{'='*60}\n")

    return all_rules, tree


if __name__ == "__main__":
    from src.data_gen.synthetic import generate_dataset
    from src.patterns.feature_eng import build_feature_matrix

    traces = generate_dataset()
    df = build_feature_matrix(traces)

    # Mine on all traces (for demo purposes)
    rules, tree = run_pattern_mining(df)
