# SpO2 Eval Pipeline

## Current State
All 7 phases built (mock mode). No real API calls yet ($4.98 balance — confirm before spending).
Dashboard has custom theming across all pages. V2 clinical fixes applied.
Next: live eval on v2 → GitHub push with README.

## Key Files
- `STATUS.md` — phase tracker, known issues, next actions, interview talk track
- `LEARNINGS.md` — engineering notes for handoff (preterm baseline problem, Tier 2 accuracy gap, etc.)
- `src/pipeline/orchestrator.py` — end-to-end pipeline entry point
- `app/dashboard.py` — Streamlit dashboard (6 views)
- `.streamlit/config.toml` — Dashboard theme colors

## Tech Stack
Python 3.12, numpy, pandas, scikit-learn 1.8, mlxtend, streamlit, plotly, anthropic SDK

## How to Run
```bash
source venv/bin/activate
streamlit run app/dashboard.py          # dashboard
python -m src.pipeline.orchestrator     # CLI pipeline (mock mode)
```

## Constraints
- Confirm approach before implementing changes
- Confirm before any API spending (use_llm=True)
- Mock mode is default everywhere — $0
