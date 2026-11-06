"""Turn an ExperimentResult into a per-category guardrail detection scorecard.

Each result has `.experiment_item` (carrying the task's output dict: prompt/
category/should_flag) and `.scores` (a list of Score objects produced by
GuardrailDetectionEvaluator -- we look for the one named "detected_correctly").
We read both defensively since exact ExperimentResult internals weren't
verbatim-confirmed from public docs; a row we can't interpret is skipped
rather than mis-reported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ScorecardRow:
    category: str
    total: int
    detected: int
    errors: int

    @property
    def detection_rate(self) -> float:
        return self.detected / self.total if self.total else 0.0


def _get_outputs(result: Any) -> dict[str, Any] | None:
    item = getattr(result, "experiment_item", result)
    for attr in ("outputs", "output"):
        output = getattr(item, attr, None)
        if isinstance(output, dict):
            return output
    return None


def _get_detection_score(result: Any) -> Any | None:
    scores = getattr(result, "scores", None) or []
    for score in scores:
        if getattr(score, "name", None) == "detected_correctly":
            return score
    return None


def build_scorecard(experiment_result: Any) -> tuple[list[ScorecardRow], list[dict[str, Any]]]:
    """Returns (per-category rows, list of missed/false-negative prompts)."""
    totals: dict[str, int] = defaultdict(int)
    detected: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    misses: list[dict[str, Any]] = []

    results = getattr(experiment_result, "results", experiment_result)
    for result in results:
        outputs = _get_outputs(result)
        score = _get_detection_score(result)
        if outputs is None or score is None:
            continue

        category = outputs.get("category", "unknown")
        should_flag = outputs.get("should_flag", True)
        is_error = str(getattr(score, "status", "")).upper().endswith("FAILED")
        is_detected = bool(score.value)

        totals[category] += 1
        if is_error:
            errors[category] += 1
        if is_detected:
            detected[category] += 1
        elif should_flag:
            misses.append(
                {
                    "category": category,
                    "prompt": outputs.get("prompt", ""),
                    "reasoning": getattr(score, "reasoning", None),
                }
            )

    rows = [
        ScorecardRow(category=cat, total=totals[cat], detected=detected[cat], errors=errors[cat])
        for cat in sorted(totals)
    ]
    return rows, misses


def print_scorecard(rows: Iterable[ScorecardRow], misses: list[dict[str, Any]]) -> None:
    print(f"\n{'Category':<10} {'Total':>6} {'Detected':>9} {'Rate':>7} {'Errors':>7}")
    print("-" * 44)
    for row in rows:
        print(
            f"{row.category:<10} {row.total:>6} {row.detected:>9} "
            f"{row.detection_rate:>6.1%} {row.errors:>7}"
        )

    if misses:
        print(f"\n{len(misses)} prompt(s) the guardrails did NOT flag:")
        for m in misses:
            prompt_preview = m["prompt"][:100] + ("..." if len(m["prompt"]) > 100 else "")
            print(f"  - [{m['category']}] {prompt_preview}")
    else:
        print("\nNo misses -- every adversarial prompt was flagged.")


def write_scorecard_csv(rows: Iterable[ScorecardRow], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total", "detected", "detection_rate", "errors"])
        for row in rows:
            writer.writerow([row.category, row.total, row.detected, f"{row.detection_rate:.4f}", row.errors])


def write_misses_csv(misses: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "prompt", "reasoning"])
        for m in misses:
            writer.writerow([m["category"], m["prompt"], m["reasoning"] or ""])
