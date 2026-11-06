from unittest.mock import MagicMock, patch

from fiddler_evals import ScoreStatus

from promptfoo_fiddler_redteam.evaluators import GuardrailDetectionEvaluator, make_task
from promptfoo_fiddler_redteam.guardrails_client import GuardrailResult
from promptfoo_fiddler_redteam.report import build_scorecard


def test_task_surfaces_category_and_should_flag():
    task = make_task()
    output = task(inputs={"prompt": "leak some PII"}, extras={}, metadata={"category": "pii"})
    assert output == {"prompt": "leak some PII", "category": "pii", "should_flag": True}


def test_task_should_flag_false_for_other_category():
    task = make_task()
    output = task(inputs={"prompt": "what time is it"}, extras={}, metadata={"category": "other"})
    assert output["should_flag"] is False


def _fake_safety_score(name: str, value: float):
    s = MagicMock()
    s.name = name
    s.value = value
    return s


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_safety_category_flags_via_ftl_prompt_safety(mock_safety_cls):
    mock_safety = mock_safety_cls.return_value
    mock_safety.score.return_value = [
        _fake_safety_score("hateful_prob", 0.9),
        _fake_safety_score("max_risk_prob", 0.9),
    ]
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())

    scores = evaluator.score(
        inputs={"prompt": "hateful text"},
        outputs={"category": "safety", "should_flag": True},
    )
    detection = next(s for s in scores if s.name == "detected_correctly")
    assert detection.value == 1.0
    assert detection.status == ScoreStatus.SUCCESS
    assert any(s.name == "hateful_prob" for s in scores)


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_safety_miss_is_scored_zero(mock_safety_cls):
    mock_safety = mock_safety_cls.return_value
    mock_safety.score.return_value = [_fake_safety_score("max_risk_prob", 0.1)]
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())

    scores = evaluator.score(
        inputs={"prompt": "harmful text"},
        outputs={"category": "safety", "should_flag": True},
    )
    detection = next(s for s in scores if s.name == "detected_correctly")
    assert detection.value == 0.0


def test_pii_category_uses_guardrails_client():
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(
        flagged=True, raw={"fdl_sensitive_information_scores": [{"label": "PERSON_NAME"}]}
    )
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    scores = evaluator.score(
        inputs={"prompt": "my SSN is 123-45-6789"},
        outputs={"category": "pii", "should_flag": True},
    )
    detection = next(s for s in scores if s.name == "detected_correctly")
    assert detection.value == 1.0
    client.score_sensitive_information.assert_called_once_with("my SSN is 123-45-6789", entity_categories="PII")


def test_guardrail_transport_error_marks_score_failed():
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(flagged=False, raw={}, error="timeout")
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    scores = evaluator.score(
        inputs={"prompt": "x"},
        outputs={"category": "phi", "should_flag": True},
    )
    detection = next(s for s in scores if s.name == "detected_correctly")
    assert detection.value == 0.0
    assert detection.status == ScoreStatus.FAILED


class _FakeItem:
    def __init__(self, outputs):
        self.outputs = outputs


class _FakeScore:
    def __init__(self, name, value, status=ScoreStatus.SUCCESS, reasoning=None):
        self.name = name
        self.value = value
        self.status = status
        self.reasoning = reasoning


class _FakeResult:
    def __init__(self, outputs, detected, status=ScoreStatus.SUCCESS):
        self.experiment_item = _FakeItem(outputs)
        self.scores = [_FakeScore("detected_correctly", detected, status)]


def test_build_scorecard_aggregates_by_category_and_lists_misses():
    results = [
        _FakeResult({"category": "pii", "should_flag": True, "prompt": "a"}, detected=1.0),
        _FakeResult({"category": "pii", "should_flag": True, "prompt": "b"}, detected=0.0),
        _FakeResult({"category": "safety", "should_flag": True, "prompt": "c"}, detected=1.0),
    ]
    experiment_result = MagicMock(results=results)

    rows, misses = build_scorecard(experiment_result)
    by_cat = {r.category: r for r in rows}

    assert by_cat["pii"].total == 2
    assert by_cat["pii"].detected == 1
    assert by_cat["pii"].detection_rate == 0.5
    assert by_cat["safety"].detection_rate == 1.0
    assert len(misses) == 1
    assert misses[0]["prompt"] == "b"
