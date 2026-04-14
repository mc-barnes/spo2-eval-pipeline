# SpO2 Eval Pipeline — Status

## Current State: All 7 Phases Built (Mock Mode)
**Last updated**: April 14, 2026

## Phase Tracker

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Synthetic Data Generator | Done | 300 traces (100 babies x 3 nights), 4 pattern types, GA-adjusted |
| 2 | Rule Engine (Tier 1) | Done | 61% auto-labeled, 84.7% accuracy |
| 3 | Pattern Mining Layer | Done | 54 rules discovered (4 tree + 50 Apriori) |
| 4 | Pre-Annotation Classifier (Tier 2) | Done | 31.3% coverage, expert queue 7.7% |
| 5 | LLM Evaluators | Done (mock) | 3 evaluators wired, mock pass rates 87-95% |
| 6 | Nurse Handoff Generator | Done (mock) | Template-based handoffs, live Claude path ready |
| 7 | Streamlit Dashboard | Done | 6 views, Owlet-inspired theming in progress |

## Pipeline Coverage
- Tier 1 (rules): 61.0%
- Tier 2 (classifier): 31.3%
- Expert queue: 7.7%
- Overall accuracy: 68.3%

## Mock Eval Results (900 total)
- Clinical accuracy: ~89% pass
- Handoff quality: ~95% pass
- Artifact handling: ~87% pass

## Known Issues
- Tier 2 accuracy on ambiguous cases: 28.4% (expected — domain shift, see LEARNINGS.md #5)
- Expert queue shows 100% accuracy (simulated oracle, see LEARNINGS.md #8)
- Mock evals don't exercise real Claude judgment
- Dashboard Owlet theming: Pipeline Overview updated, other pages pending

## Next Actions
- [ ] Run 10-15 traces through live Claude evals (confirm with user first, ~$0.15-0.30)
- [ ] Finish dashboard styling across all pages
- [ ] Write 60-second interview talk track
- [ ] Push to GitHub with README
- [ ] Consider Tier 2 accuracy improvements (see LEARNINGS.md #9)

## How To Run
```bash
cd /Users/Sterdb/pm-os/projects/spo2-eval-pipeline
source venv/bin/activate

# Full pipeline (mock mode, $0)
python -m src.pipeline.orchestrator

# Dashboard
streamlit run app/dashboard.py

# Live mode (costs money — confirm first)
# python -c "from src.pipeline.orchestrator import run_pipeline; run_pipeline(use_llm=True, llm_sample_size=15)"
```

## Commits
1. `da08465` — Phases 1-4 (data pipeline)
2. `2324f48` — LEARNINGS.md
3. `dcce022` — Phases 5-7 (evals, handoffs, dashboard)

## Interview Talk Track
_60 seconds — what this is, why, what you learned:_

> I built an end-to-end AI evaluation pipeline for neonatal SpO2 monitoring — the kind of system a company like Owlet would need to validate their overnight triage algorithms at scale.
>
> It generates synthetic pulse oximetry data with gestational-age-adjusted baselines, then runs it through a three-tier classification system: rules for clear cases, a classifier for ambiguous ones, and a simulated expert queue for the rest. On top of that, I built three LLM-as-judge evaluators that assess clinical accuracy, handoff quality, and artifact handling.
>
> The biggest engineering learning was the preterm baseline problem — fixed SpO2 thresholds don't work across gestational ages. A premature baby sitting at 92% is normal, but a term baby at 92% needs attention. I had to add GA-aware gating to the rule engine, which is the same problem real clinical systems face.
>
> With real data, I'd focus on GA-adjusted thresholds from published neonatal reference ranges, and I'd use the LLM evaluators to catch the edge cases that rules miss — that's where the eval pipeline adds the most value.
