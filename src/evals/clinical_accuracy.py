"""Evaluator 1: Clinical Accuracy.

Did the pipeline correctly triage this baby's night?
"""
from __future__ import annotations

import numpy as np

from src.data_gen.synthetic import NightTrace
from src.evals.base import EvalResult, mock_eval, parse_eval_response


_PROMPT = """You are a neonatal clinical expert evaluating an AI triage system for overnight SpO2 monitoring.

TASK: Determine if the system's triage classification is clinically correct.

PATIENT CONTEXT:
- Gestational age: {ga_weeks} weeks ({ga_category})
- Birth weight: {birth_weight}g
- Days since birth: {days_since_birth}
- Known conditions: {known_conditions}

OVERNIGHT SpO2 SUMMARY:
- Mean SpO2: {mean_spo2:.1f}%
- Min SpO2: {min_spo2:.0f}%
- Number of desaturation events (SpO2 <90% for >10s): {n_urgent_events}
- Number of borderline events (SpO2 90-94% sustained): {n_borderline_events}
- Longest desaturation duration: {max_desat_duration}s

SYSTEM'S TRIAGE: {assigned_label}

EVALUATION CRITERIA:
- Pass: The triage label matches the clinical severity. Emergency (SpO2 <80% sustained) = emergency label. Urgent night = urgent label. Normal night = normal label. Borderline is for genuinely ambiguous cases.
- Fail: The triage label is incorrect. Examples: labeling an urgent night as normal, labeling artifact as urgent, labeling sustained <80% as merely urgent (should be emergency).

FEW-SHOT EXAMPLES:

Example 1 (Pass):
Patient: 29 weeks GA, mean SpO2 91%, min SpO2 78%, 4 events <90% for >10s. System triage: urgent.
{{"reasoning": "A 29-week preterm infant with 4 sustained desaturation events reaching 78% is clearly urgent. Triage is correct.", "answer": "Pass"}}

Example 2 (Fail):
Patient: 38 weeks GA, mean SpO2 98%, min SpO2 96%, 0 events. System triage: borderline.
{{"reasoning": "A term infant with consistently normal SpO2 and no desaturation events should be normal, not borderline. Over-triage error.", "answer": "Fail"}}

Example 3 (Pass):
Patient: 26 weeks GA, mean SpO2 88%, min SpO2 72%, 2 events <80% for >15s. System triage: emergency.
{{"reasoning": "An extremely preterm infant with sustained SpO2 below 80% reaching 72% warrants emergency classification requiring immediate 911 response. Triage is correct.", "answer": "Pass"}}

Respond ONLY with valid JSON: {{"reasoning": "<your reasoning>", "answer": "Pass" or "Fail"}}"""


def evaluate_clinical_accuracy(
    trace: NightTrace,
    assigned_label: str,
    use_llm: bool = False,
    model: str | None = None,
    seed: int | None = None,
) -> EvalResult:
    """Evaluate whether the triage classification is clinically correct."""
    if not use_llm:
        return mock_eval(
            trace.night_id, "clinical_accuracy",
            trace.ground_truth_label, assigned_label,
            accuracy=0.85, seed=seed,
        )

    from src.llm_utils import call_llm

    spo2 = trace.spo2
    baby = trace.baby

    # Count events
    n_urgent = sum(1 for e in trace.events if "urgent" in e.get("type", ""))
    n_borderline = sum(1 for e in trace.events if "borderline" in e.get("type", ""))
    durations = [e.get("duration_s", 0) for e in trace.events]

    prompt = _PROMPT.format(
        ga_weeks=baby.gestational_age_weeks,
        ga_category=baby.ga_category,
        birth_weight=baby.birth_weight_grams,
        days_since_birth=baby.days_since_birth,
        known_conditions=", ".join(baby.known_conditions),
        mean_spo2=float(np.mean(spo2)),
        min_spo2=float(np.min(spo2)),
        n_urgent_events=n_urgent,
        n_borderline_events=n_borderline,
        max_desat_duration=max(durations) if durations else 0,
        assigned_label=assigned_label,
    )

    result = call_llm(prompt, model=model, max_tokens=300)
    if result is None:
        return mock_eval(
            trace.night_id, "clinical_accuracy",
            trace.ground_truth_label, assigned_label,
            accuracy=0.85, seed=seed,
        )

    parsed = parse_eval_response(result["text"])
    return EvalResult(
        trace_id=trace.night_id,
        evaluator="clinical_accuracy",
        answer=parsed["answer"],
        reasoning=parsed["reasoning"],
        source="claude_api",
        latency_ms=result["latency_ms"],
    )
