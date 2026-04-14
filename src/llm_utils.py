"""Shared LLM utilities: client management, cost tracking, and guardrails.

All LLM calls in the pipeline go through this module to enforce:
- Mock mode by default (no API calls unless explicitly enabled)
- Hard call limit per run (default 20)
- Cost tracking with per-run spending cap
- Model selection (Haiku for dev, Sonnet for demo)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL


# Approximate cost per 1K tokens (input/output) as of April 2026
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
}

# Default token estimates per call type
ESTIMATED_TOKENS = {
    "handoff": {"input": 800, "output": 300},
    "eval": {"input": 1200, "output": 200},
}


@dataclass
class CostTracker:
    """Tracks API usage and enforces spending limits."""
    max_calls: int = 20
    max_spend_usd: float = 1.00
    calls_made: int = 0
    estimated_spend: float = 0.0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    model: str = ""

    def estimate_run_cost(self, n_handoffs: int, n_evals: int, model: str) -> float:
        """Estimate cost before making any calls."""
        costs = MODEL_COSTS.get(model, MODEL_COSTS["claude-sonnet-4-6"])
        handoff_cost = n_handoffs * (
            ESTIMATED_TOKENS["handoff"]["input"] * costs["input"] / 1000
            + ESTIMATED_TOKENS["handoff"]["output"] * costs["output"] / 1000
        )
        eval_cost = n_evals * (
            ESTIMATED_TOKENS["eval"]["input"] * costs["input"] / 1000
            + ESTIMATED_TOKENS["eval"]["output"] * costs["output"] / 1000
        )
        return handoff_cost + eval_cost

    def check_budget(self) -> bool:
        """Returns True if we can make another call."""
        if self.calls_made >= self.max_calls:
            print(f"[COST GUARD] Hit call limit: {self.calls_made}/{self.max_calls}")
            return False
        if self.estimated_spend >= self.max_spend_usd:
            print(f"[COST GUARD] Hit spend limit: ${self.estimated_spend:.3f}/${self.max_spend_usd:.2f}")
            return False
        return True

    def record_call(self, input_tokens: int, output_tokens: int, model: str):
        """Record a completed API call."""
        self.calls_made += 1
        self.actual_input_tokens += input_tokens
        self.actual_output_tokens += output_tokens
        self.model = model
        costs = MODEL_COSTS.get(model, MODEL_COSTS["claude-sonnet-4-6"])
        call_cost = (
            input_tokens * costs["input"] / 1000
            + output_tokens * costs["output"] / 1000
        )
        self.estimated_spend += call_cost

    def summary(self) -> str:
        return (
            f"API calls: {self.calls_made}/{self.max_calls} | "
            f"Tokens: {self.actual_input_tokens} in / {self.actual_output_tokens} out | "
            f"Est. cost: ${self.estimated_spend:.4f} / ${self.max_spend_usd:.2f} cap"
        )


# Global tracker — reset per pipeline run
_tracker = CostTracker()


def get_tracker() -> CostTracker:
    return _tracker


def reset_tracker(max_calls: int = 20, max_spend_usd: float = 1.00):
    global _tracker
    _tracker = CostTracker(max_calls=max_calls, max_spend_usd=max_spend_usd)


def get_client():
    """Get an Anthropic client. Returns None if no API key."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        print("WARNING: anthropic package not installed")
        return None


def call_llm(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 500,
    system: str | None = None,
) -> dict | None:
    """Make a single LLM call with cost tracking and guardrails.

    Returns {"text": str, "input_tokens": int, "output_tokens": int, "latency_ms": int}
    or None if budget exceeded or no client available.
    """
    tracker = get_tracker()
    if not tracker.check_budget():
        return None

    client = get_client()
    if client is None:
        return None

    model = model or CLAUDE_MODEL

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    start = time.time()
    try:
        response = client.messages.create(**kwargs)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None
    latency_ms = int((time.time() - start) * 1000)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    text = response.content[0].text

    tracker.record_call(input_tokens, output_tokens, model)

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }
