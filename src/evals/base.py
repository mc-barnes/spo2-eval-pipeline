"""Base evaluator infrastructure for the LLM eval pipeline.

Each evaluator is a separate Claude judge with Pass/Fail output.
Mock mode returns deterministic results from ground truth (no API calls).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    """Result from a single evaluator on a single trace."""
    trace_id: str
    evaluator: str       # "clinical_accuracy", "handoff_quality", "artifact_handling"
    answer: str          # "Pass" or "Fail"
    reasoning: str
    source: str          # "mock" or "claude_api"
    latency_ms: int = 0


def parse_eval_response(text: str) -> dict:
    """Parse JSON eval response from Claude. Handles markdown code fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", "").strip()
        reasoning = parsed.get("reasoning", "").strip()
        # Normalize answer
        if answer.lower() in ("pass", "yes", "true"):
            answer = "Pass"
        else:
            answer = "Fail"
        return {"answer": answer, "reasoning": reasoning}
    except json.JSONDecodeError:
        # Try to extract answer from free text
        text_lower = text.lower()
        if '"pass"' in text_lower or "answer: pass" in text_lower:
            return {"answer": "Pass", "reasoning": text[:200]}
        return {"answer": "Fail", "reasoning": f"Failed to parse response: {text[:200]}"}


def mock_eval(
    trace_id: str,
    evaluator_name: str,
    ground_truth: str,
    predicted_label: str,
    accuracy: float = 0.85,
    seed: int | None = None,
) -> EvalResult:
    """Mock evaluator: returns Pass/Fail based on ground truth match + noise."""
    rng = np.random.default_rng(seed)

    labels_match = ground_truth == predicted_label

    if rng.random() < accuracy:
        # Evaluator agrees with reality
        answer = "Pass" if labels_match else "Fail"
    else:
        # Evaluator makes an error (15% of the time)
        answer = "Fail" if labels_match else "Pass"

    if answer == "Pass":
        reasoning = f"The {predicted_label} classification appears clinically appropriate for this trace."
    else:
        reasoning = f"The {predicted_label} classification may not match the clinical presentation."

    return EvalResult(
        trace_id=trace_id,
        evaluator=evaluator_name,
        answer=answer,
        reasoning=reasoning,
        source="mock",
        latency_ms=0,
    )
