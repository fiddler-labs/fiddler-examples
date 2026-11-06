"""Parse a `promptfoo redteam generate` output file into normalized records.

`promptfoo redteam generate` always writes a YAML document (regardless of the
output file's extension) shaped like a promptfooconfig.yaml with a populated
`tests:` list. Verified against a real 39-case output from promptfoo 0.124.0
on 2026-10-08 -- see tests/fixtures/sample_redteam_output.yaml.

Each test case looks like:

    tests:
      - vars:
          prompt: "<the adversarial prompt text>"
        metadata:
          pluginId: "pii:direct"       # or "harmful:hate", "medical:hallucination", ...
          severity: "high"
          harmCategory: "Hate"          # only present for harmful:* plugins
        assert:
          - type: "promptfoo:redteam:pii"
            metric: "PIILeak"

We deliberately read defensively (`.get(...)` with fallbacks) since this is
an unversioned, internal file format that promptfoo could change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Plugin id prefix -> guardrail category. Only map a plugin family to a
# guardrail that actually detects what the plugin's prompts contain:
#
# - "pii" -> the sensitive-information guardrail (PII entities).
# - "safety" -> FTLPromptSafety (harmful, hateful, jailbreak content, ...).
# - "phi" is also supported (sensitive-information with PHI entities) but no
#   plugin maps to it by default.
#
# promptfoo's `medical:*` plugins are deliberately NOT mapped. They test for
# bad medical *advice* (hallucination, sycophancy, off-label use, ...), not
# PHI leakage, so scoring them with the PHI entity detector would only measure
# whether a prompt mentions a drug or condition. Unmapped plugins fall into
# "other", which the evaluator skips and the scorecard reports separately.
_CATEGORY_BY_PREFIX = {
    "pii": "pii",
    "harmful": "safety",
    "toxicity": "safety",
    "bias": "safety",
}

SCORED_CATEGORIES = ("pii", "phi", "safety")


def _category_for_plugin(plugin_id: str) -> str:
    prefix = plugin_id.split(":", 1)[0]
    return _CATEGORY_BY_PREFIX.get(prefix, "other")


@dataclass(frozen=True)
class RedteamTestCase:
    index: int
    prompt: str
    plugin_id: str
    category: str  # "pii" | "safety" | "phi" | "other"
    severity: str | None = None
    metric: str | None = None
    strategy_id: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_flag(self) -> bool:
        """promptfoo red-team prompts are synthesized to be adversarial for
        their plugin category, so by construction a well-behaved guardrail
        should flag every mapped one. `category == "other"` prompts (plugins
        with no matching guardrail) are not assumed safe: the evaluator skips
        them and the scorecard lists them separately."""
        return self.category in SCORED_CATEGORIES


def load_redteam_cases(
    path: str | Path, inject_var: str = "prompt"
) -> list[RedteamTestCase]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "tests" not in doc:
        raise ValueError(
            f"{path} doesn't look like a promptfoo redteam generate output "
            "(expected a top-level 'tests' key)."
        )

    configured_inject_var = (doc.get("redteam") or {}).get("injectVar", inject_var)

    cases: list[RedteamTestCase] = []
    for i, test in enumerate(doc["tests"]):
        test_vars = test.get("vars") or {}
        prompt = test_vars.get(configured_inject_var) or test_vars.get("prompt")
        if prompt is None:
            # Fall back to the only var present, if there's exactly one.
            if len(test_vars) == 1:
                prompt = next(iter(test_vars.values()))
            else:
                continue  # can't identify the prompt text for this case; skip it

        metadata = test.get("metadata") or {}
        plugin_id = metadata.get("pluginId", "unknown")
        asserts = test.get("assert") or []
        metric = asserts[0].get("metric") if asserts else None

        cases.append(
            RedteamTestCase(
                index=i,
                prompt=str(prompt).strip(),
                plugin_id=plugin_id,
                category=_category_for_plugin(plugin_id),
                severity=metadata.get("severity"),
                metric=metric,
                strategy_id=metadata.get("strategyId"),
                raw_metadata=metadata,
            )
        )
    return cases
