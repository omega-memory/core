"""Report generator — Markdown summary + degradation curve JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmarks.memorystress.cost_tracker import CostTracker
from benchmarks.memorystress.metrics import MetricsResult


def generate_report(
    metrics: MetricsResult,
    cost_tracker: CostTracker | None = None,
    adapter_name: str = "unknown",
    model: str = "gpt-4o",
    output_dir: str | Path = ".",
) -> Path:
    """Generate a Markdown report and degradation curve JSON.

    Returns the path to the generated summary.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write degradation curve JSON
    curve_path = output_dir / "degradation_curve.json"
    curve_data = {
        "adapter": adapter_name,
        "model": model,
        "phases": {str(k): v for k, v in metrics.degradation_curve.items()},
        "slope": metrics.degradation_slope,
        "recall_at_age": metrics.recall_at_age,
    }
    curve_path.write_text(json.dumps(curve_data, indent=2))

    # Write full metrics JSON
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2, default=str))

    # Write cost summary if available
    if cost_tracker:
        cost_path = output_dir / "cost.json"
        cost_path.write_text(json.dumps(cost_tracker.summary(), indent=2))

    # Generate Markdown summary
    summary_path = output_dir / "summary.md"
    lines = [
        f"# MemoryStress Benchmark Report",
        f"",
        f"- **Adapter**: {adapter_name}",
        f"- **Model**: {model}",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **Overall**: {metrics.total_correct}/{metrics.total_questions} "
        f"({metrics.overall_accuracy:.1%})",
        f"",
        f"## Degradation Curve",
        f"",
        f"| Phase | Accuracy |",
        f"|-------|----------|",
    ]
    for phase in sorted(metrics.degradation_curve):
        acc = metrics.degradation_curve[phase]
        bar = "#" * int(acc * 20)
        lines.append(f"| {phase} | {acc:.1%} {bar} |")

    lines.append(f"")
    lines.append(f"**Degradation slope**: {metrics.degradation_slope:+.4f} per phase")
    if metrics.degradation_slope < -0.05:
        lines.append(f"> Warning: Significant degradation detected.")
    elif metrics.degradation_slope > -0.01:
        lines.append(f"> Minimal degradation — memory system is resilient.")

    lines.extend([
        f"",
        f"## Recall by Age",
        f"",
        f"| Age Bucket | Accuracy |",
        f"|------------|----------|",
    ])
    for bucket, acc in sorted(metrics.recall_at_age.items()):
        lines.append(f"| {bucket} sessions | {acc:.1%} |")

    lines.extend([
        f"",
        f"## Scoring Dimensions",
        f"",
        f"| Dimension | Score |",
        f"|-----------|-------|",
        f"| Contradiction Resolution | {metrics.contradiction_accuracy:.1%} ({metrics.contradiction_total} Qs) |",
        f"| Cross-Agent Recall | {metrics.cross_agent_accuracy:.1%} ({metrics.cross_agent_total} Qs) |",
        f"| Cold Start Recovery | {metrics.cold_start_accuracy:.1%} ({metrics.cold_start_total} Qs) |",
    ])

    if metrics.cost_per_correct:
        lines.extend([
            f"",
            f"## Cost Efficiency",
            f"",
            f"| Phase | $/Correct Answer |",
            f"|-------|-----------------|",
        ])
        for phase in sorted(metrics.cost_per_correct):
            cost = metrics.cost_per_correct[phase]
            lines.append(f"| {phase} | ${cost:.4f} |")

    lines.extend([
        f"",
        f"## Per-Type Breakdown",
        f"",
        f"| Question Type | Correct | Total | Accuracy |",
        f"|---------------|---------|-------|----------|",
    ])
    for qt in sorted(metrics.per_type):
        info = metrics.per_type[qt]
        lines.append(
            f"| {qt} | {info['correct']} | {info['total']} | {info['accuracy']:.1%} |"
        )

    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path
