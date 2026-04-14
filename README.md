# SpO2 AI Eval Pipeline

End-to-end AI evaluation pipeline for neonatal SpO2 monitoring — the kind of system needed to validate overnight triage algorithms at scale for consumer pulse oximetry devices.

Generates synthetic pulse oximetry data with gestational-age-adjusted baselines, classifies traces through a three-tier triage system, generates nurse handoff reports, and evaluates everything with LLM-as-judge evaluators.

## Architecture

```
Synthetic Data (300 traces)
    ↓
Tier 1: Rule Engine (58%) ── GA-adjusted thresholds, safety check, emergency tier
    ↓ unlabeled
Tier 2: Classifier (39.3%) ── Logistic regression on Tier 1 labels
    ↓ low confidence
Expert Queue (2.7%) ── Simulated clinical review
    ↓
Handoff Generator ── 5 templates (emergency/urgent/monitor/routine/artifact)
    ↓
LLM Evaluators ── Clinical accuracy, handoff quality, artifact handling
```

## Key Features

- **GA-adjusted thresholds** from published neonatal references (Castillo 2008, Hay 2002)
- **Safety constraint**: artifact detection cannot mask genuine urgent desaturations
- **Emergency tier**: SpO2 <80% sustained triggers "call 911" action
- **SatSeconds metric**: quantifies hypoxemic burden (integral of depth x duration below threshold)
- **LLM-as-judge evals**: 3 evaluators with few-shot prompts, mock/live modes
- **Streamlit dashboard**: 6 views with custom theming, per-label metrics

## Results

| Metric | Value |
|--------|-------|
| Overall accuracy | 76.3% |
| Urgent false negatives | **0** |
| Emergency cases detected | 36 |
| Clinical accuracy (live eval) | 80% |
| Handoff quality (live eval) | 90% |
| Artifact handling (live eval) | 100% |

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (mock mode, $0)
python -m src.pipeline.orchestrator

# Run dashboard
streamlit run app/dashboard.py

# Run safety tests
python -m pytest tests/test_safety_check.py -v
```

## Project Structure

```
src/
├── config.py                 # Thresholds, GA-adjusted configs
├── data_gen/synthetic.py     # 300-trace synthetic data generator
├── rules/tier1_engine.py     # Rule engine with safety check
├── patterns/
│   ├── feature_eng.py        # Feature extraction + SatSeconds
│   └── miner.py              # Decision tree + Apriori mining
├── classifier/
│   ├── tier2.py              # Logistic regression classifier
│   └── expert_sim.py         # Simulated expert review
├── handoff/generator.py      # Nurse handoff templates
├── evals/
│   ├── clinical_accuracy.py  # LLM evaluator: triage correctness
│   ├── handoff_quality.py    # LLM evaluator: handoff content
│   └── artifact_handling.py  # LLM evaluator: motion artifact
├── pipeline/orchestrator.py  # End-to-end orchestrator
└── llm_utils.py              # Claude API client + cost tracker
app/
├── dashboard.py              # Streamlit dashboard (6 views)
└── theme.py                  # Dashboard theme module
tests/
└── test_safety_check.py      # 6 safety-critical tests
```

## Live Eval Mode

Runs evaluators through the Claude API. Requires an `ANTHROPIC_API_KEY` in `.env`.

```bash
python -c "from src.pipeline.orchestrator import run_pipeline; run_pipeline(use_llm=True, llm_sample_size=10)"
```

Cost: ~$0.15-0.35 for 10 traces. Hard cap at $1.00 enforced by `CostTracker`.

## Documentation

- `STATUS.md` — Phase tracker, coverage metrics, known issues
- `LEARNINGS.md` — 18 engineering notes for data science/engineering handoff
- `PLAN-clinical-fixes.md` — V2 clinical review fix implementation plan

## Tech Stack

Python 3.12 | NumPy | Pandas | scikit-learn 1.8 | mlxtend | Streamlit | Plotly | Anthropic SDK
