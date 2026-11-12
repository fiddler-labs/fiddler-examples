from unittest.mock import MagicMock, patch

import pytest
import requests
from fiddler_evals import ScoreStatus

from promptfoo_fiddler_redteam import resilience
from promptfoo_fiddler_redteam.evaluators import GuardrailDetectionEvaluator
from promptfoo_fiddler_redteam.fiddler_bridge import connect
from promptfoo_fiddler_redteam.guardrails_client import GuardrailsClient
from promptfoo_fiddler_redteam.report import Scorecard, print_scorecard
from promptfoo_fiddler_redteam.resilience import (
    FIDDLER_TIMEOUT,
    describe_fiddler_error,
    error_hint,
    with_retry,
)

DROPPED = requests.exceptions.ConnectionError(
    "Failed to connect to https://x/v3/evals/score: ('Connection aborted.', "
    "RemoteDisconnected('Remote end closed connection without response'))"
)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(resilience, "RETRY_DELAY_SECONDS", 0)


def test_with_retry_recovers_from_one_dropped_connection():
    fn = MagicMock(side_effect=[DROPPED, "ok"])
    assert with_retry(fn) == "ok"
    assert fn.call_count == 2


def test_with_retry_gives_up_after_second_failure():
    fn = MagicMock(side_effect=[DROPPED, requests.exceptions.Timeout("read timed out")])
    with pytest.raises(requests.exceptions.Timeout):
        with_retry(fn)
    assert fn.call_count == 2


def test_with_retry_does_not_retry_other_errors():
    fn = MagicMock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError):
        with_retry(fn)
    assert fn.call_count == 1


@patch("promptfoo_fiddler_redteam.evaluators.FTLPromptSafety")
def test_safety_call_is_retried_once(mock_safety_cls):
    score = MagicMock(value=0.9, status=ScoreStatus.SUCCESS)
    score.name = "max_risk_prob"
    mock_safety_cls.return_value.score.side_effect = [DROPPED, [score]]
    evaluator = GuardrailDetectionEvaluator(guardrails_client=MagicMock())

    scores = evaluator.score(inputs={"prompt": "x"}, outputs={"category": "safety"})
    detection = next(s for s in scores if s.name == "detected_correctly")
    assert detection.status == ScoreStatus.SUCCESS
    assert detection.value == 1.0


def test_guardrails_client_retries_dropped_connection_and_uses_timeouts():
    client = GuardrailsClient(base_url="https://x", api_key="k")
    ok = MagicMock()
    ok.json.return_value = {"fdl_sensitive_information_scores": [{"label": "email"}]}
    client._session.post = MagicMock(side_effect=[DROPPED, ok])

    result = client.score_sensitive_information("a@b.com", entity_categories="PII")
    assert result.flagged and result.error is None
    assert client._session.post.call_count == 2
    assert client._session.post.call_args.kwargs["timeout"] == FIDDLER_TIMEOUT


@patch("promptfoo_fiddler_redteam.fiddler_bridge.Application")
@patch("promptfoo_fiddler_redteam.fiddler_bridge.Project")
@patch("promptfoo_fiddler_redteam.fiddler_bridge.init")
@patch("promptfoo_fiddler_redteam.fiddler_bridge.requests.get")
def test_connect_checks_reachability_then_passes_timeout_to_sdk(
    mock_get, mock_init, _project, _app
):
    connect(MagicMock(url="https://x", api_key="k", project="p", application="a"))
    assert mock_get.call_args.kwargs["timeout"] == FIDDLER_TIMEOUT
    assert mock_init.call_args.kwargs["timeout"] == FIDDLER_TIMEOUT


@patch("promptfoo_fiddler_redteam.fiddler_bridge.init")
@patch("promptfoo_fiddler_redteam.fiddler_bridge.requests.get")
def test_connect_stops_before_sdk_when_fiddler_is_unreachable(mock_get, mock_init):
    mock_get.side_effect = requests.exceptions.ConnectTimeout("connect timed out")
    with pytest.raises(requests.exceptions.ConnectTimeout):
        connect(MagicMock(url="https://x", api_key="k"))
    assert mock_get.call_count == 2  # one retry
    mock_init.assert_not_called()


def test_describe_fiddler_error_distinguishes_auth_from_connection():
    timeout = requests.exceptions.Timeout(
        "Request to https://x/v3/server-info timed out"
    )
    assert "Couldn't reach Fiddler at https://x" in describe_fiddler_error(
        timeout, "https://x"
    )

    unauthorized = requests.exceptions.HTTPError("nope")
    unauthorized.response = MagicMock(status_code=401)
    assert "rejected the API key" in describe_fiddler_error(unauthorized, "https://x")


def test_error_hint_and_scorecard_message_for_dropped_connection(capsys):
    assert "rejected the API key" in error_hint("error=401 Client Error: Unauthorized")
    assert "Check the logs" in error_hint("error=something odd")

    card = Scorecard(
        rows=[],
        rows_by_plugin=[],
        errors=[
            {
                "category": "safety",
                "plugin_id": "harmful:hate",
                "prompt": "p",
                "reasoning": f"category=safety should_flag=True error={DROPPED}",
            }
        ],
    )
    print_scorecard(card)
    out = capsys.readouterr().out
    assert "dropped the connection, even after a retry" in out
    assert "Check FIDDLER_URL / FIDDLER_API_KEY" not in out
