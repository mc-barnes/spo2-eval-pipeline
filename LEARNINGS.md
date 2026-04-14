# SpO2 Eval Pipeline — Engineering Learnings

Technical notes from building Phases 1-4 of the neonatal SpO2 AI eval pipeline. Written for a data scientist or engineer picking up this codebase or building a production version with real clinical data.

---

## 1. The Preterm Baseline Problem

**The single biggest design issue we hit.**

### What happened
The rule engine uses fixed SpO2 thresholds: <90% = urgent, 90-94% = borderline, >95% = normal. These thresholds work perfectly for term babies (baseline ~98%). But premature babies (<37 weeks gestational age) have physiological baselines of 91-95%. A 27-week preemie sitting at 92% all night is *normal for that baby* — but the fixed-threshold rule engine flags the entire 8-hour trace as "borderline."

### Impact
In the first iteration, the rule engine labeled 91.7% of traces (target was ~65%) with only 46% accuracy. Nearly every preterm normal trace was mislabeled as borderline, and many preterm borderline traces were mislabeled as urgent.

### How we solved it (synthetic version)
Added a **median gate** to the borderline rule (R2): only fire if the trace's median SpO2 is >95.5%. If the median is already in the 90-94 range, the trace defers to Tier 2, which has access to gestational age context.

This is a clinically meaningful distinction:
- Median 98%, readings at 92% → **dip from baseline** → rules can label it
- Median 92%, readings at 92% → **this IS the baseline** → needs GA-aware classification

### What a real system would need
- **GA-adjusted thresholds** in the rule engine. A preemie at 90% is a different clinical situation than a term baby at 90%. Published neonatal reference ranges exist by gestational age (e.g., Hay et al., 2002).
- Alternatively, keep fixed thresholds but **route all preterm traces to Tier 2 by default**. The rule engine's value is highest for term babies where thresholds are unambiguous.
- In a real NICU, the SpO2 alarm limits are already GA-adjusted by clinical staff. If your data comes from monitors, those alarm settings may be available as metadata.

**File:** `src/rules/tier1_engine.py`, `_check_borderline()` — the median gate logic.

---

## 2. Artifact Detection Is Harder Than It Looks

### What happened
First-pass artifact detection caught only 21% of artifact traces (15 out of 71). Most artifact traces were instead labeled as "urgent" or "borderline" because:

1. The artifact SpO2 drops (20-50% in 1-3 seconds) easily triggered the urgent rule (<90% for >10s) before the artifact check could flag them
2. The initial artifact detection only used one method (rate-of-change + simultaneous accel spike) with a narrow detection buffer

### How we solved it
Two changes:

**Wider detection buffers.** Artifact events in SpO2 data have tails — the signal doesn't snap back instantly. The recovery period after an artifact can still show sub-90% readings for 10-20 seconds, which triggers the urgent rule. We expanded the artifact mask buffer from ±5-10 samples to -15/+25 samples around each detection.

**Dual detection method.** Added a second artifact detector: any high accelerometer reading (>2.5g) near any significant SpO2 drop (>5% below trace median). This catches artifacts where the rate-of-change window doesn't align perfectly with the 3-second check window.

After these fixes: 59/66 artifact traces correctly detected (89%).

### What a real system would need
- **Signal quality indices (SQI).** Real pulse oximeters compute signal quality metrics from the raw photoplethysmographic waveform. Low SQI + SpO2 drop = almost certainly artifact. This pipeline doesn't have access to SQI because we're generating post-processed 1 Hz data.
- **Motion artifact has temporal structure.** Real motion artifacts from handling (diaper changes, repositioning, feeding) follow predictable patterns — they happen at care times, last 30-120 seconds, and have characteristic accelerometer signatures. A time-of-day feature or nursing schedule correlation could improve detection.
- **The artifact-mixed-with-real-desat case** is the hardest. A baby can have a genuine desaturation AND motion artifact in the same 10-minute window. The current system treats the entire window as artifact if it detects motion. A production system would need segment-level artifact masking rather than trace-level labeling.

**File:** `src/rules/tier1_engine.py`, `_check_artifact()` — dual detection with wide buffers.

---

## 3. Synthetic Data Design Decisions and Their Consequences

### Signal composition approach
Each trace is built by composing numpy arrays: baseline + breathing oscillation + noise + desaturation events + artifacts. This is additive — each component contributes independently.

**Limitation:** Real SpO2 signals have non-linear interactions. A desaturation event changes the baseline for subsequent readings (the baby doesn't instantly return to pre-desat levels). Our sigmoid-based desat events recover to baseline cleanly, which is more optimistic than reality.

### Pattern seeding
We deliberately injected three patterns for the pattern mining layer to discover:
1. **2am dip cluster** — 30% of borderline traces have desats concentrated at hours 2-3
2. **Consecutive borderline nights** — tracked per-baby across the 3-night series
3. **GA + desat duration** — preterm babies with long desats should escalate

**Result:** The pattern miner found artifact patterns overwhelmingly (accelerometer correlation is the strongest signal in the data by far). The seeded clinical patterns exist but don't surface as top rules. This is partly a feature engineering issue — the discretization bins in the Apriori miner may not be tuned to highlight these patterns — and partly because the artifact signal genuinely is the most predictive feature in this dataset.

**For a real dataset:** The seeded patterns are based on published clinical findings. If the pattern miner doesn't find them in real data, that's informative — it either means the data doesn't support those patterns at scale, or the feature engineering needs adjustment.

### GA distribution
We skewed the cohort toward preterm (70% preterm, 30% term) because that's the clinically interesting population. In reality, the distribution depends on the monitoring program's patient mix. A NICU dataset would be almost entirely preterm. A home monitoring program (like the one in the SPEC) would skew toward term/near-term babies with risk factors.

**File:** `src/data_gen/synthetic.py` — `_assign_pattern()` for GA-weighted pattern assignment, `_generate_borderline()` for the 2am dip seeding.

---

## 4. Rule Engine Threshold Tuning Is a Trap

### What happened
We spent several iterations adjusting thresholds to hit the 65% Tier 1 coverage target:
- R3 normal: tried 95%, 98%, 99%, 99.5% pct_above thresholds, with and without min_spo2 checks
- R2 borderline: tried 60s, 120s, 300s sustained duration, with and without median gates
- Each change improved one metric while degrading another

### The lesson
Chasing a specific coverage number by tweaking thresholds is a local optimization that doesn't generalize. The 65% target was a design aspiration, not a clinical requirement. What matters is:
1. **Accuracy on labeled cases** — when the rule engine labels something, is it right? (84.7% — decent)
2. **False negative rate for urgent** — does the rule engine ever label a truly urgent trace as normal? (2 cases — low but nonzero)
3. **Conservative on ambiguity** — does it defer uncertain cases to ML/expert? (yes, by design)

**For a real system:** Set thresholds based on clinical risk tolerance, not coverage targets. An urgent false negative (missed desat → no nurse callout) is far worse than an urgent false positive (unnecessary callout). Calibrate the urgent rule (R1) to maximize sensitivity even at the cost of specificity.

---

## 5. Tier 2 Classifier Accuracy on Ambiguous Cases

### The numbers
- Trained on: Tier 1 auto-labeled data (183 "easy" traces, 84.7% accurate labels)
- Predicted on: 117 unlabeled traces (the ones rules couldn't handle)
- Validation accuracy: 89.2% (on held-out Tier 1 data — easy cases)
- Production accuracy: 28.7% (on the actual ambiguous cases)

### Why the gap is expected
The classifier was trained on clear-cut cases and asked to predict ambiguous ones. This is a domain shift problem. The Tier 1 labeled data has:
- Normal traces with >97% readings above 95%
- Urgent traces with obvious <90% sustained desats
- Artifact traces with clear accel spikes

The unlabeled traces are borderline-by-definition: preterm normals hovering near 95%, borderline traces with brief dips that don't sustain long enough for R2, artifact traces with subtle motion signatures.

### Options for improvement
1. **Accept it and frame correctly** — "The classifier provides a best-guess label with a confidence score. Low-confidence cases go to expert review." This is the honest story.
2. **Semi-supervised learning** — Use the Tier 1 labels as seeds, then iteratively label the most confident Tier 2 predictions and retrain. This would improve accuracy but adds complexity.
3. **Train on ground truth** — For the demo, you could train on the synthetic ground truth labels instead of Tier 1 labels. This is cheating (you wouldn't have ground truth in production) but would show what the classifier *could* do with perfect training data.
4. **Better features for the ambiguous cases** — The current feature set was designed for all traces. Features specifically targeting the borderline/ambiguous cases (e.g., proximity to threshold boundaries, rate of SpO2 change near 90%, GA-adjusted z-scores) might help.

**File:** `src/classifier/tier2.py` — `train_tier2()` and `predict_tier2()`.

---

## 6. Multi-Night Trend Detection

### Current state
The feature engineering computes `consecutive_borderline_nights` per baby — how many consecutive nights have been labeled borderline. The decision tree found this as its top splitting feature (which is promising). But the 3-night series is very short for trend detection.

### What a real system would need
With 90 nights of data per baby (3 months), you could detect:
- **Deterioration trends** — SpO2 baseline drifting downward over weeks
- **Periodic patterns** — every 3rd night is worse (could indicate cyclical physiological processes)
- **Recovery trajectories** — preterm babies should improve as they mature; failure to improve is clinically significant
- **Event frequency trends** — increasing desat frequency even when individual events are mild

The multi-night feature is the highest-leverage improvement for a real version. A single borderline night is ambiguous. Three consecutive borderline nights is a trend. Seven consecutive borderline nights is clinically actionable.

---

## 7. Dependencies and Environment Notes

- **scikit-learn 1.8.0** removed the `multi_class` parameter from `LogisticRegression` (it's always multinomial now). The spec referenced this parameter; had to remove it.
- **mlxtend** is required for Apriori association rules but the code gracefully degrades if it's not installed (decision tree still runs).
- **numpy's random Generator** (`np.random.default_rng(seed)`) is used throughout for reproducibility. All results are deterministic given the same seed (42).
- Python 3.12.4 — no 3.13+ features used.

---

## 8. Open Questions for a Production Version

1. **What averaging window does the pulse oximeter use?** Clinical monitors average SpO2 over 3-16 heartbeats. Our 1 Hz synthetic data assumes post-averaging output. If you get raw PPG data, the signal processing pipeline is entirely different.

2. **How are alarm limits set per-patient?** If the clinical workflow already adjusts alarm thresholds by GA, the rule engine should inherit those settings rather than using fixed thresholds.

3. **What's the false negative tolerance?** If a truly urgent night gets labeled normal and no nurse calls the family, what's the clinical consequence? This drives the sensitivity/specificity tradeoff for every threshold in the system.

4. **Is the 2.5M baby / 225M trace scale real?** If so, the Tier 2 classifier needs to be extremely lightweight (logistic regression is fine) and the pattern mining needs to be done on samples, not the full dataset. The rule engine scales linearly and should handle it.

5. **What does "expert review" actually look like?** Our simulation uses ground truth + noise. In production, expert review means a clinician looks at the trace. How long does that take? What's the queue capacity? If expert review is the bottleneck, the system needs to minimize expert queue volume even at the cost of some Tier 2 accuracy.
