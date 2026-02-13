"""MetricsEngine — compute 6 scoring dimensions for MemoryStress.

Dimensions:
1. Recall@age       — accuracy by age bucket (0-100, 100-500, 500-1000 sessions)
2. Degradation curve — accuracy at each phase checkpoint → slope = degradation rate
3. Contradiction resolution — accuracy on contradiction_resolution questions only
4. Cost efficiency  — $/correct_answer at each phase
5. Cross-agent recall — accuracy on cross_agent_recall questions
6. Cold start recovery — accuracy of fresh agent on existing store
"""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.memorystress.cost_tracker import CostTracker


@dataclass
class GradedQuestion:
    """A graded question result."""

    question_id: str
    question_type: str
    phase_asked: int
    age_sessions: int
    correct: bool
    hypothesis: str = ""
    cold_start: bool = False


@dataclass
class MetricsResult:
    """Full metrics output."""

    recall_at_age: dict[str, float] = field(default_factory=dict)
    degradation_curve: dict[int, float] = field(default_factory=dict)
    degradation_slope: float = 0.0
    contradiction_accuracy: float = 0.0
    contradiction_total: int = 0
    cost_per_correct: dict[int, float] = field(default_factory=dict)
    cross_agent_accuracy: float = 0.0
    cross_agent_total: int = 0
    cold_start_accuracy: float = 0.0
    cold_start_total: int = 0
    overall_accuracy: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    per_type: dict[str, dict] = field(default_factory=dict)


# Age bucket boundaries (in sessions)
_AGE_BUCKETS = [
    ("0-100", 0, 100),
    ("100-500", 100, 500),
    ("500-1000", 500, 1000),
]


class MetricsEngine:
    """Compute benchmark metrics from graded results."""

    def __init__(self, cost_tracker: CostTracker | None = None):
        self.cost_tracker = cost_tracker

    def compute(self, graded: list[GradedQuestion]) -> MetricsResult:
        result = MetricsResult()
        if not graded:
            return result

        result.total_questions = len(graded)
        result.total_correct = sum(1 for g in graded if g.correct)
        result.overall_accuracy = (
            result.total_correct / result.total_questions
            if result.total_questions > 0
            else 0.0
        )

        # 1. Recall@age — accuracy by age bucket
        for label, lo, hi in _AGE_BUCKETS:
            bucket = [g for g in graded if lo <= g.age_sessions < hi]
            if bucket:
                correct = sum(1 for g in bucket if g.correct)
                result.recall_at_age[label] = correct / len(bucket)
            else:
                result.recall_at_age[label] = 0.0

        # 2. Degradation curve — accuracy at each phase checkpoint
        phases = sorted(set(g.phase_asked for g in graded))
        for phase in phases:
            phase_qs = [g for g in graded if g.phase_asked == phase]
            if phase_qs:
                correct = sum(1 for g in phase_qs if g.correct)
                result.degradation_curve[phase] = correct / len(phase_qs)

        # Compute slope (linear regression over phase accuracies)
        if len(result.degradation_curve) >= 2:
            xs = sorted(result.degradation_curve.keys())
            ys = [result.degradation_curve[x] for x in xs]
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs)
            result.degradation_slope = num / den if den != 0 else 0.0

        # 3. Contradiction resolution
        contra_qs = [g for g in graded if g.question_type == "contradiction_resolution"]
        result.contradiction_total = len(contra_qs)
        if contra_qs:
            result.contradiction_accuracy = (
                sum(1 for g in contra_qs if g.correct) / len(contra_qs)
            )

        # 4. Cost efficiency ($/correct_answer per phase)
        if self.cost_tracker:
            for phase in phases:
                result.cost_per_correct[phase] = self.cost_tracker.get_cost_per_correct(phase)

        # 5. Cross-agent recall
        cross_qs = [g for g in graded if g.question_type == "cross_agent_recall"]
        result.cross_agent_total = len(cross_qs)
        if cross_qs:
            result.cross_agent_accuracy = (
                sum(1 for g in cross_qs if g.correct) / len(cross_qs)
            )

        # 6. Cold start recovery
        cold_qs = [g for g in graded if g.cold_start]
        result.cold_start_total = len(cold_qs)
        if cold_qs:
            result.cold_start_accuracy = (
                sum(1 for g in cold_qs if g.correct) / len(cold_qs)
            )

        # Per-type breakdown
        types = sorted(set(g.question_type for g in graded))
        for qt in types:
            type_qs = [g for g in graded if g.question_type == qt]
            correct = sum(1 for g in type_qs if g.correct)
            result.per_type[qt] = {
                "total": len(type_qs),
                "correct": correct,
                "accuracy": correct / len(type_qs) if type_qs else 0.0,
            }

        return result
