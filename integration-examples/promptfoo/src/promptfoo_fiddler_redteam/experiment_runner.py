"""Wire a Fiddler Dataset + the guardrail task/evaluator into evaluate()."""

from __future__ import annotations

from fiddler_evals import Dataset, evaluate

from .evaluators import GuardrailDetectionEvaluator, make_task
from .guardrails_client import GuardrailsClient


def run_experiment(
    dataset: Dataset,
    guardrails_client: GuardrailsClient,
    name_prefix: str = "promptfoo-redteam",
    max_workers: int = 4,
):
    return evaluate(
        dataset=dataset,
        task=make_task(),
        evaluators=[GuardrailDetectionEvaluator(guardrails_client)],
        name_prefix=name_prefix,
        description="promptfoo red-team prompts scored against Fiddler Guardrails",
        metadata={"source": "promptfoo", "tool": "promptfoo-fiddler-redteam"},
        max_workers=max_workers,
    )
