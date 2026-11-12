from collections import Counter
from unittest.mock import MagicMock, patch

from fiddler_evals import ScoreStatus

from promptfoo_fiddler_redteam.evaluators import GuardrailDetectionEvaluator, make_task
from promptfoo_fiddler_redteam.guardrails_client import GuardrailResult
from promptfoo_fiddler_redteam.report import build_scorecard, write_misses_csv


def test_task_surfaces_category_and_plugin():
    task = make_task()
    output = task(
        inputs={"prompt": "leak some PII"},
        extras={},
        metadata={"category": "pii", "plugin_id": "pii:direct"},
    )
    assert output == {
        "prompt": "leak some PII",
        "category": "pii",
        "plugin_id": "pii:direct",
    }


def _fake_safety_score(name: str, value: float | None, status=ScoreStatus.SUCCESS):
    s = MagicMock()
    s.name = name
    s.value = value
    s.status = status
    s.error_message = None
    return s


def _detection(scores):
    scores = scores if isinstance(scores, list) else [scores]
    return next(s for s in scores if s.name == "detected_correctly")


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_safety_category_flags_via_ftl_prompt_safety(mock_safety_cls):
    mock_safety_cls.return_value.score.return_value = [
        _fake_safety_score("hateful_prob", 0.9),
        _fake_safety_score("max_risk_prob", 0.9),
    ]
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())

    scores = evaluator.score(
        inputs={"prompt": "hateful text"}, outputs={"category": "safety"}
    )
    detection = _detection(scores)
    assert detection.value == 1.0
    assert detection.status == ScoreStatus.SUCCESS
    assert any(s.name == "hateful_prob" for s in scores)


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_safety_miss_is_scored_zero(mock_safety_cls):
    mock_safety_cls.return_value.score.return_value = [
        _fake_safety_score("max_risk_prob", 0.1)
    ]
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())

    scores = evaluator.score(
        inputs={"prompt": "harmful text"}, outputs={"category": "safety"}
    )
    assert _detection(scores).value == 0.0


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_failed_or_empty_safety_scores_are_errors_not_misses(mock_safety_cls):
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())
    for returned in (
        [_fake_safety_score("max_risk_prob", None, ScoreStatus.FAILED)],
        [],
    ):
        mock_safety_cls.return_value.score.return_value = returned
        detection = _detection(
            evaluator.score(inputs={"prompt": "x"}, outputs={"category": "safety"})
        )
        assert detection.status == ScoreStatus.FAILED
        assert detection.value is None


def test_pii_category_uses_guardrails_client():
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(
        flagged=True,
        raw={"fdl_sensitive_information_scores": [{"label": "PERSON_NAME"}]},
    )
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    scores = evaluator.score(
        inputs={"prompt": "my SSN is 123-45-6789"}, outputs={"category": "pii"}
    )
    assert _detection(scores).value == 1.0
    client.score_sensitive_information.assert_called_once_with(
        "my SSN is 123-45-6789", entity_categories="PII"
    )


def test_scores_task_output_not_dataset_input():
    """The documented live-target extension returns the app's response as
    "prompt"; that response, not the stored attack prompt, must be scored."""
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(
        flagged=False, raw={"fdl_sensitive_information_scores": []}
    )
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    evaluator.score(
        inputs={"prompt": "attack prompt"},
        outputs={"prompt": "app response", "category": "pii"},
    )
    client.score_sensitive_information.assert_called_once_with(
        "app response", entity_categories="PII"
    )


def test_expected_outputs_should_flag_supports_benign_controls():
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(
        flagged=False, raw={"fdl_sensitive_information_scores": []}
    )
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    scores = evaluator.score(
        inputs={"prompt": "what are your opening hours?"},
        outputs={"category": "pii"},
        expected_outputs={"should_flag": False},
    )
    assert _detection(scores).value == 1.0


def test_unmapped_category_is_skipped_without_calling_guardrails():
    client = MagicMock()
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    detection = _detection(
        evaluator.score(
            inputs={"prompt": "x"},
            outputs={"category": "other", "plugin_id": "medical:hallucination"},
        )
    )
    assert detection.status == ScoreStatus.SKIPPED
    client.score_sensitive_information.assert_not_called()


def test_guardrail_transport_error_marks_score_failed():
    client = MagicMock()
    client.score_sensitive_information.return_value = GuardrailResult(
        flagged=False, raw={}, error="timeout"
    )
    evaluator = GuardrailDetectionEvaluator(guardrails_client=client)

    detection = _detection(
        evaluator.score(inputs={"prompt": "x"}, outputs={"category": "pii"})
    )
    assert detection.status == ScoreStatus.FAILED
    assert detection.value is None


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _score(value, status=ScoreStatus.SUCCESS, name="detected_correctly"):
    return _Obj(
        name=name,
        value=value,
        status=status,
        reasoning=f"r-{value}",
        error_message=None,
    )


def _result(category, prompt, scores, should_flag=True, plugin_id=None):
    plugin_id = plugin_id or f"{category}:x"
    return _Obj(
        experiment_item=_Obj(
            outputs={"category": category, "plugin_id": plugin_id, "prompt": prompt}
        ),
        dataset_item=_Obj(
            expected_outputs={"should_flag": should_flag},
            metadata={"category": category, "plugin_id": plugin_id},
        ),
        scores=scores,
    )


def test_build_scorecard_aggregates_by_category_and_lists_misses():
    results = [
        _result("pii", "a", [_score(1.0)], plugin_id="pii:direct"),
        _result("pii", "b", [_score(0.0)], plugin_id="pii:session"),
        _result("safety", "c", [_score(1.0)]),
    ]
    card = build_scorecard(MagicMock(results=results))
    by_cat = {r.key: r for r in card.rows}

    assert by_cat["pii"].total == 2
    assert by_cat["pii"].detection_rate == 0.5
    assert by_cat["safety"].detection_rate == 1.0
    assert [m["prompt"] for m in card.misses] == ["b"]
    assert card.misses[0]["plugin_id"] == "pii:session"
    assert {r.key: r.total for r in card.rows_by_plugin} == {
        "pii:direct": 1,
        "pii:session": 1,
        "safety:x": 1,
    }


def test_errors_are_not_misses_and_not_in_the_rate():
    results = [
        _result("pii", "ok", [_score(1.0)]),
        _result("pii", "boom", [_score(None, ScoreStatus.FAILED)]),
        # SDK fallback when the evaluator raises: named after the evaluator.
        _result(
            "pii",
            "raised",
            [_score(None, ScoreStatus.FAILED, name="guardrail_detection")],
        ),
    ]
    card = build_scorecard(MagicMock(results=results))
    row = card.rows[0]

    assert (row.total, row.detected, row.errors) == (1, 1, 2)
    assert card.misses == []
    assert len(card.errors) == 2


def test_task_failure_counts_as_error_using_dataset_metadata():
    failed = _result("safety", "", [])
    failed.experiment_item.outputs = {}
    card = build_scorecard(MagicMock(results=[failed]))

    assert card.rows[0].key == "safety"
    assert card.rows[0].errors == 1
    assert card.scored_total == 0


def test_skipped_benign_and_unscored_are_reported_separately():
    results = [
        _result(
            "other",
            "s",
            [_score(None, ScoreStatus.SKIPPED)],
            plugin_id="medical:hallucination",
        ),
        _result("pii", "benign-ok", [_score(1.0)], should_flag=False),
        _result("pii", "benign-fp", [_score(0.0)], should_flag=False),
        _result("pii", "weird", [_score(0.5, name="something_else")]),
    ]
    card = build_scorecard(MagicMock(results=results))

    assert card.rows == []
    assert card.skipped == Counter({"medical:hallucination": 1})
    assert card.benign_total == 2
    assert [fp["prompt"] for fp in card.false_positives] == ["benign-fp"]
    assert [u["prompt"] for u in card.unscored] == ["weird"]


def test_misses_csv_includes_plugin_id(tmp_path):
    path = tmp_path / "misses.csv"
    write_misses_csv(
        [
            {
                "category": "pii",
                "plugin_id": "pii:direct",
                "prompt": "p",
                "reasoning": "r",
            }
        ],
        path,
    )
    assert path.read_text().splitlines() == [
        "category,plugin_id,prompt,reasoning",
        "pii,pii:direct,p,r",
    ]
