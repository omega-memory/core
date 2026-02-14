"""CostTracker — token and cost accounting per phase."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.memorystress.adapters.base import CostSnapshot

# Approximate costs per 1M tokens (input) for common models
_COST_PER_1M_TOKENS = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4.1": 2.00,
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "claude-sonnet-4-5-20250929": 3.00,
    "claude-haiku-4-5-20251001": 0.80,
    "claude-opus-4-6": 15.00,
    "gemini-2.5-pro": 1.25,
    "gemini-2.5-flash": 0.15,
}


@dataclass
class PhaseMetrics:
    """Cost metrics for a single phase."""

    phase: int
    ingest_tokens: int = 0
    ingest_api_calls: int = 0
    query_tokens: int = 0
    query_api_calls: int = 0
    grading_tokens: int = 0
    grading_api_calls: int = 0
    questions_answered: int = 0
    questions_correct: int = 0


class CostTracker:
    """Track token usage and estimated cost across benchmark phases."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._phases: dict[int, PhaseMetrics] = {}
        self._cost_per_token = _COST_PER_1M_TOKENS.get(model, 2.50) / 1_000_000

    def _phase(self, phase: int) -> PhaseMetrics:
        if phase not in self._phases:
            self._phases[phase] = PhaseMetrics(phase=phase)
        return self._phases[phase]

    def record_ingest(self, phase: int, tokens: int, api_calls: int = 1) -> None:
        p = self._phase(phase)
        p.ingest_tokens += tokens
        p.ingest_api_calls += api_calls

    def record_query(self, phase: int, tokens: int, api_calls: int = 1) -> None:
        p = self._phase(phase)
        p.query_tokens += tokens
        p.query_api_calls += api_calls

    def record_grading(self, phase: int, tokens: int, api_calls: int = 1) -> None:
        p = self._phase(phase)
        p.grading_tokens += tokens
        p.grading_api_calls += api_calls

    def record_result(self, phase: int, correct: bool) -> None:
        p = self._phase(phase)
        p.questions_answered += 1
        if correct:
            p.questions_correct += 1

    def get_phase_cost(self, phase: int) -> float:
        """Estimated USD cost for a phase."""
        p = self._phases.get(phase)
        if not p:
            return 0.0
        total_tokens = p.ingest_tokens + p.query_tokens + p.grading_tokens
        return total_tokens * self._cost_per_token

    def get_cost_per_correct(self, phase: int) -> float:
        """USD per correct answer for a phase."""
        p = self._phases.get(phase)
        if not p or p.questions_correct == 0:
            return float("inf")
        return self.get_phase_cost(phase) / p.questions_correct

    def get_total_snapshot(self) -> CostSnapshot:
        """Aggregate cost snapshot across all phases."""
        total_ingest = sum(p.ingest_tokens for p in self._phases.values())
        total_query = sum(p.query_tokens for p in self._phases.values())
        total_grading = sum(p.grading_tokens for p in self._phases.values())
        total_tokens = total_ingest + total_query + total_grading
        total_calls = sum(
            p.ingest_api_calls + p.query_api_calls + p.grading_api_calls
            for p in self._phases.values()
        )
        return CostSnapshot(
            total_tokens=total_tokens,
            total_api_calls=total_calls,
            estimated_cost_usd=total_tokens * self._cost_per_token,
            ingest_tokens=total_ingest,
            query_tokens=total_query + total_grading,
        )

    def summary(self) -> dict:
        """Per-phase and total cost summary for reporting."""
        phases = {}
        for phase_num in sorted(self._phases):
            p = self._phases[phase_num]
            phases[phase_num] = {
                "ingest_tokens": p.ingest_tokens,
                "query_tokens": p.query_tokens,
                "grading_tokens": p.grading_tokens,
                "total_tokens": p.ingest_tokens + p.query_tokens + p.grading_tokens,
                "api_calls": p.ingest_api_calls + p.query_api_calls + p.grading_api_calls,
                "estimated_cost_usd": self.get_phase_cost(phase_num),
                "cost_per_correct": self.get_cost_per_correct(phase_num),
                "accuracy": (
                    p.questions_correct / p.questions_answered
                    if p.questions_answered > 0
                    else 0.0
                ),
            }
        snap = self.get_total_snapshot()
        return {
            "model": self.model,
            "phases": phases,
            "total": {
                "tokens": snap.total_tokens,
                "api_calls": snap.total_api_calls,
                "estimated_cost_usd": snap.estimated_cost_usd,
            },
        }
