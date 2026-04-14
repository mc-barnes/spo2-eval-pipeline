# SpO2 Pipeline — Clinical Review Fixes (P1 + P2)

## Context

A clinical domain review (modeled on Dr. Bonafide, CHOP/UPenn) found 3 critical and 4 important issues. The user wants all P1 + P2 fixes. Tier 2 stays active but gets prominent warnings.

**Root cause of urgent false negatives:** R4 (artifact) evaluates before R1 (urgent) with early return. If a genuine desat overlaps artifact buffer (±30 samples), the trace exits as "artifact" and R1 never runs.

---

## Step 1: Config — New constants
**File:** `src/config.py`

- Add `SPO2_EMERGENCY_THRESHOLD = 80`
- Add `GA_URGENT_THRESHOLDS` dict (extremely_preterm: 85, very_preterm: 88, moderate_preterm: 89, term: 90)
- Add `GA_BORDERLINE_RANGES` dict with (low, high) tuples per GA category
- Fix config-vs-code mismatch comments (borderline duration 60→30, normal pct 0.95→0.97)

---

## Step 2: Rule Engine — Safety check + emergency + GA thresholds
**File:** `src/rules/tier1_engine.py`

**2A — Urgent FN safety constraint (P1-1):**
- Add `_urgent_safety_check(spo2, ga_category)` — runs R1 logic on RAW (unmasked) signal
- In `apply_rules()`: after R4 decides "artifact" but BEFORE early return, call safety check
- If genuine sustained desat found in raw signal → override label to "urgent" or "emergency"
- Principle: sustained desat <threshold >10s in raw signal = cannot be labeled anything but urgent/emergency

**2B — Emergency tier (P2-4):**
- If min SpO2 in urgent events < 80% → label = "emergency" instead of "urgent"
- Applies in both normal R1 path and safety check path

**2C — GA-adjusted thresholds (P2-5):**
- `_check_urgent()` accepts `ga_category`, uses `GA_URGENT_THRESHOLDS` instead of fixed 90%
- `_check_borderline()` accepts `ga_category`, uses `GA_BORDERLINE_RANGES` instead of fixed 90-94%
- `apply_rules()` passes `trace.baby.ga_category` to both

---

## Step 3: SatSeconds Metric
**Files:** `src/patterns/feature_eng.py`, `src/rules/tier1_engine.py`

- Add SatSeconds calculation: `sum(threshold - spo2)` for each sub-threshold sample, using GA-adjusted threshold
- Add to feature extraction in `feature_eng.py` (new feature for Tier 2)
- Add to urgent event dicts in rule engine (metadata for handoffs)
- Add `"sat_seconds"` to `CLASSIFIER_FEATURES` in `tier2.py`

---

## Step 4: Handoff Generator — Emergency template + SatSeconds
**Files:** `src/handoff/generator.py`, `app/theme.py`

- Add `"emergency": "EMERGENCY"` to `_URGENCY_MAP`
- Add emergency template: "Advise family to call 911 or go to nearest ED immediately. If unreachable within 15 minutes, escalate to on-call physician."
- Add `sat_seconds` to `_compute_trace_stats()` and urgent/emergency templates
- Add `"EMERGENCY": "#8B2020"` to theme colors

---

## Step 5: Dashboard — Warnings + per-tier metrics
**File:** `app/dashboard.py`

**5A — Tier 2 warning (P1-2):**
- Yellow callout below Tier 2 accuracy explaining domain shift: "This reflects performance on ambiguous cases that Tier 1 rules could not classify — by definition, the hardest traces."

**5B — Expert accuracy footnote (P1-3):**
- Rename to "Expert Review*"
- Footnote: "Expert accuracy is simulated (95% oracle). Real accuracy depends on reviewer fatigue, inter-rater variability, and staffing."

**5C — Per-label sensitivity/specificity/PPV (P2-7):**
- New "Per-Label Metrics" section card with table: Label | Sensitivity | PPV | F1 | N
- Uses `sklearn.metrics.precision_recall_fscore_support`
- Expandable confusion matrix per tier

---

## Step 6: Orchestrator, Evals, Tests, Docs

**Orchestrator** (`src/pipeline/orchestrator.py`):
- Handle "emergency" label in summary print
- No structural changes needed (labels are strings)

**Clinical accuracy eval** (`src/evals/clinical_accuracy.py`):
- Add emergency to eval criteria and few-shot examples

**Tests** (`tests/test_safety_check.py` — new):
- `test_artifact_does_not_mask_genuine_urgent` — both artifact + real desat → urgent
- `test_pure_artifact_still_labeled_artifact` — artifact-only → artifact
- `test_emergency_threshold` — SpO2 <80% sustained → emergency
- `test_ga_adjusted_threshold_preterm` — 87% not urgent for extremely preterm

**Docs** (`STATUS.md`, `CLAUDE.md`):
- Update pipeline state, known issues, coverage numbers

---

## Risks

1. **Coverage split will shift** — GA-adjusted thresholds will increase Tier 1 catch rate for preterm babies. The 61/31/8 split will change. Expected and desirable.
2. **"emergency" is new vocabulary** — every downstream label consumer needs to handle it (merge_triage, dashboard loops, eval mocks, handoff LLM prompt). Plan covers all of these.
3. **Safety check is intentionally conservative** — some artifact+urgent traces will now be labeled urgent. Correct bias for medical device context (favor sensitivity).

---

## Verification

1. `python -c "from src.config import GA_URGENT_THRESHOLDS, SPO2_EMERGENCY_THRESHOLD; print('OK')"`
2. `python -m src.pipeline.orchestrator` — full pipeline mock run, verify no urgent FNs, check coverage split
3. `pytest tests/test_safety_check.py` — all 4 safety tests pass
4. `streamlit run app/dashboard.py` — visual check: Tier 2 warning, expert footnote, per-label metrics, emergency badge
5. Rerun live eval (10 traces, ~$0.35) to confirm clinical accuracy improves

---

## Files Modified (10 total)

| File | Changes |
|------|---------|
| `src/config.py` | Emergency threshold, GA-adjusted threshold dicts |
| `src/rules/tier1_engine.py` | Safety check, emergency tier, GA thresholds |
| `src/patterns/feature_eng.py` | SatSeconds feature |
| `src/classifier/tier2.py` | Add sat_seconds to feature list |
| `src/handoff/generator.py` | Emergency template, SatSeconds in stats |
| `src/evals/clinical_accuracy.py` | Emergency in criteria + few-shot |
| `src/pipeline/orchestrator.py` | Emergency count in summary |
| `app/dashboard.py` | Tier 2 warning, expert footnote, per-label metrics |
| `app/theme.py` | Emergency color |
| `tests/test_safety_check.py` | 4 new safety tests |
