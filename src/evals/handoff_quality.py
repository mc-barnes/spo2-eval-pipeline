"""Evaluator 2: Handoff Quality.

Is the nurse summary urgent-first, plain language, actionable?
"""
from __future__ import annotations

from src.data_gen.synthetic import NightTrace
from src.handoff.generator import HandoffSummary
from src.evals.base import EvalResult, mock_eval, parse_eval_response


_PROMPT = """You are a telehealth nursing supervisor evaluating AI-generated handoff summaries for neonatal overnight SpO2 monitoring.

TASK: Determine if this handoff summary meets quality standards for a telehealth nurse.

PATIENT TRIAGE LEVEL: {triage_label}

HANDOFF SUMMARY:
---
{handoff_text}
---

EVALUATION CRITERIA (ALL must be met for Pass):
1. URGENCY FIRST: Summary leads with urgency level and most critical finding.
2. PLAIN LANGUAGE: No jargon a telehealth nurse would need to look up. "Desaturation" and "SpO2" are acceptable nurse vocabulary. "V/Q mismatch" or "periodic breathing disorder" are not.
3. GESTATIONAL CONTEXT: Mentions gestational age and how it affects interpretation.
4. ACTIONABLE NEXT STEP: Includes a specific action (e.g., "Call family within 1 hour" or "Schedule follow-up within 48 hours").

FEW-SHOT EXAMPLES:

Example 1 (Pass):
{{"reasoning": "Summary leads with URGENT, uses nurse-appropriate language, notes gestational age context, and recommends calling family within 30 minutes. All four criteria met.", "answer": "Pass"}}

Example 2 (Fail):
{{"reasoning": "Summary buries urgency at end of paragraph, uses 'periodic breathing with apneic episodes secondary to immature respiratory drive' which is physician-level jargon, and provides no specific next step.", "answer": "Fail"}}

Respond ONLY with valid JSON: {{"reasoning": "<your reasoning>", "answer": "Pass" or "Fail"}}"""


def evaluate_handoff_quality(
    trace: NightTrace,
    handoff: HandoffSummary,
    assigned_label: str,
    use_llm: bool = False,
    model: str | None = None,
    seed: int | None = None,
) -> EvalResult:
    """Evaluate whether the handoff summary meets quality standards."""
    if not use_llm:
        # Mock: handoffs from templates always pass (they're designed to),
        # handoffs from LLM pass 90% of the time
        accuracy = 0.95 if handoff.source == "mock_template" else 0.90
        return mock_eval(
            trace.night_id, "handoff_quality",
            trace.ground_truth_label, assigned_label,
            accuracy=accuracy, seed=seed,
        )

    from src.llm_utils import call_llm

    prompt = _PROMPT.format(
        triage_label=assigned_label,
        handoff_text=handoff.summary_text,
    )

    result = call_llm(prompt, model=model, max_tokens=300)
    if result is None:
        return mock_eval(
            trace.night_id, "handoff_quality",
            trace.ground_truth_label, assigned_label,
            accuracy=0.90, seed=seed,
        )

    parsed = parse_eval_response(result["text"])
    return EvalResult(
        trace_id=trace.night_id,
        evaluator="handoff_quality",
        answer=parsed["answer"],
        reasoning=parsed["reasoning"],
        source="claude_api",
        latency_ms=result["latency_ms"],
    )
