"""Evaluator 3: Artifact Handling.

Did the system correctly exclude noise/artifact events?
"""
from __future__ import annotations

import numpy as np

from src.data_gen.synthetic import NightTrace
from src.evals.base import EvalResult, mock_eval, parse_eval_response


_PROMPT = """You are a biomedical signal quality expert evaluating an AI system's ability to distinguish motion artifact from genuine SpO2 desaturation in neonatal monitoring.

TASK: Determine if the system correctly handled artifact events.

TRACE SUMMARY:
- Total apparent desaturation events: {total_events}
- Events flagged as artifact: {n_artifact_flagged}
- Events kept as genuine: {n_genuine_kept}
- Accelerometer spikes detected: {n_accel_spikes}
- Ground truth artifacts: {n_true_artifacts}
- Ground truth genuine events: {n_true_genuine}

SYSTEM'S TRIAGE: {assigned_label}

EVALUATION CRITERIA:
- Pass: System correctly excluded artifact events (SpO2 drops with concurrent high accelerometer) AND correctly retained genuine events (SpO2 drops without motion).
- Fail: System either (a) included obvious artifacts causing false alerts, or (b) excluded genuine desaturations as artifacts, missing real clinical events.

FEW-SHOT EXAMPLES:

Example 1 (Pass):
{{"reasoning": "3 of 5 events correctly flagged as artifacts (all had accelerometer >3g). The 2 genuine events had normal accelerometer and plausible SpO2 decline rates.", "answer": "Pass"}}

Example 2 (Fail):
{{"reasoning": "System flagged a genuine 15-second desaturation to 84% as artifact despite normal accelerometer. This was a true clinical event that would be missed.", "answer": "Fail"}}

Respond ONLY with valid JSON: {{"reasoning": "<your reasoning>", "answer": "Pass" or "Fail"}}"""


def evaluate_artifact_handling(
    trace: NightTrace,
    assigned_label: str,
    artifact_events_detected: int = 0,
    use_llm: bool = False,
    model: str | None = None,
    seed: int | None = None,
) -> EvalResult:
    """Evaluate whether the system correctly handled artifacts."""
    # Count ground truth events by type
    n_true_artifacts = sum(1 for e in trace.events if e.get("type") == "artifact")
    n_true_genuine = sum(1 for e in trace.events if e.get("type") != "artifact")
    total_events = len(trace.events)

    if not use_llm:
        # Mock: if the trace is artifact and labeled artifact, pass.
        # Otherwise evaluate based on whether artifacts were properly handled.
        if trace.ground_truth_label == "artifact":
            is_correct = assigned_label == "artifact"
        else:
            # For non-artifact traces, pass if no artifacts were falsely flagged
            is_correct = assigned_label != "artifact"

        accuracy = 0.90 if is_correct else 0.30
        return mock_eval(
            trace.night_id, "artifact_handling",
            trace.ground_truth_label, assigned_label,
            accuracy=accuracy, seed=seed,
        )

    from src.llm_utils import call_llm

    # Estimate accel spikes
    accel_mag = trace.accel_magnitude
    n_accel_spikes = int(np.sum(np.diff((accel_mag > 2.5).astype(int)) == 1))

    prompt = _PROMPT.format(
        total_events=total_events,
        n_artifact_flagged=artifact_events_detected,
        n_genuine_kept=total_events - artifact_events_detected,
        n_accel_spikes=n_accel_spikes,
        n_true_artifacts=n_true_artifacts,
        n_true_genuine=n_true_genuine,
        assigned_label=assigned_label,
    )

    result = call_llm(prompt, model=model, max_tokens=300)
    if result is None:
        return mock_eval(
            trace.night_id, "artifact_handling",
            trace.ground_truth_label, assigned_label,
            accuracy=0.85, seed=seed,
        )

    parsed = parse_eval_response(result["text"])
    return EvalResult(
        trace_id=trace.night_id,
        evaluator="artifact_handling",
        answer=parsed["answer"],
        reasoning=parsed["reasoning"],
        source="claude_api",
        latency_ms=result["latency_ms"],
    )
