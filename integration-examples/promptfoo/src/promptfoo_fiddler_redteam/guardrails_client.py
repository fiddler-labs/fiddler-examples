"""Thin REST client for the Fiddler Guardrails scoring endpoints.

These endpoints (the "Centor" models) are a separate product surface from
the `fiddler_evals` SDK -- stateless text-scoring calls, not tied to a
Project/Application/Dataset. Confirmed against
https://docs.fiddler.ai/protection/guardrails#pii/phi : `entity_categories`
on `/v3/guardrails/sensitive-information` takes `"PII"` (12 entities,
default), `"PII_ALL"` (+15 more), or `"PHI"` (health-related entities like
medication, medical condition, health insurance number); response shape is
`{"fdl_sensitive_information_scores": [{"score", "label", "start", "end",
"text"}, ...]}`.

Do not confuse this with the *different*, Presidio-based `pii_detection`
evaluator/enrichment type available through `fiddler_evals`/the Fiddler
MCP tools (`create_evaluator`, `score`) -- Fiddler's docs are explicit that
these two use different underlying models and entity sets. That one has no
PHI mode and is meant for batch monitoring enrichments, not this real-time
guardrail check.

`_post` logs the raw response on an unrecognized shape (e.g. after a
Fiddler API version change) so a field-name mismatch fails loudly instead
of silently reporting "not flagged".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

SAFETY_THRESHOLD = 0.5
_SAFETY_DIMENSIONS = (
    "illegal",
    "hateful",
    "harassing",
    "racist",
    "sexist",
    "violent",
    "sexual",
    "harmful",
    "unethical",
    "jailbreaking",
    "roleplaying",
)


@dataclass
class GuardrailResult:
    flagged: bool
    raw: dict[str, Any]
    error: str | None = None


class GuardrailsClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        self._timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._session.post(f"{self._base_url}{path}", json=payload, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def score_safety(self, text: str) -> GuardrailResult:
        """POST /v3/guardrails/ftl-safety -- flags if any of the 11 safety
        dimensions scores >= SAFETY_THRESHOLD."""
        try:
            raw = self._post("/v3/guardrails/ftl-safety", {"data": {"input": text}})
        except requests.RequestException as e:
            return GuardrailResult(flagged=False, raw={}, error=str(e))

        scores = {dim: raw.get(dim) for dim in _SAFETY_DIMENSIONS if dim in raw}
        if not scores:
            # Response shape didn't match what the docs describe -- surface it
            # rather than silently treating everything as "not flagged".
            logger.warning("Unexpected ftl-safety response shape: %r", raw)
            return GuardrailResult(flagged=False, raw=raw, error="unrecognized response shape")

        flagged = any(v is not None and v >= SAFETY_THRESHOLD for v in scores.values())
        return GuardrailResult(flagged=flagged, raw=raw)

    def score_sensitive_information(
        self, text: str, entity_categories: str | list[str] = "PII_ALL"
    ) -> GuardrailResult:
        """POST /v3/guardrails/sensitive-information -- flagged if any entity
        is detected in the text at all."""
        try:
            raw = self._post(
                "/v3/guardrails/sensitive-information",
                {"data": {"input": text, "entity_categories": entity_categories}},
            )
        except requests.RequestException as e:
            return GuardrailResult(flagged=False, raw={}, error=str(e))

        entities = raw.get("fdl_sensitive_information_scores")
        if entities is None:
            logger.warning("Unexpected sensitive-information response shape: %r", raw)
            return GuardrailResult(flagged=False, raw=raw, error="unrecognized response shape")

        return GuardrailResult(flagged=len(entities) > 0, raw=raw)

    def score_secrets(self, text: str) -> GuardrailResult:
        """POST /v3/guardrails/secret-detection -- deterministic credential pattern match."""
        try:
            raw = self._post("/v3/guardrails/secret-detection", {"data": {"input": text}})
        except requests.RequestException as e:
            return GuardrailResult(flagged=False, raw={}, error=str(e))

        detections = raw.get("fdl_secret_detections") or raw.get("detections")
        if detections is None:
            logger.warning("Unexpected secret-detection response shape: %r", raw)
            return GuardrailResult(flagged=False, raw=raw, error="unrecognized response shape")

        return GuardrailResult(flagged=len(detections) > 0, raw=raw)
