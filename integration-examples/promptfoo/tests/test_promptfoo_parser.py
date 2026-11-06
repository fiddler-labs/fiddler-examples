from pathlib import Path

from promptfoo_fiddler_redteam.promptfoo_parser import load_redteam_cases

FIXTURE = Path(__file__).parent / "fixtures" / "sample_redteam_output.yaml"


def test_loads_all_cases():
    cases = load_redteam_cases(FIXTURE)
    assert len(cases) == 39


def test_extracts_prompt_text():
    cases = load_redteam_cases(FIXTURE)
    assert all(case.prompt for case in cases)
    assert any("Muslim" in case.prompt for case in cases)


def test_categorizes_plugins():
    cases = load_redteam_cases(FIXTURE)
    by_plugin = {case.plugin_id: case for case in cases}

    assert by_plugin["pii:direct"].category == "pii"
    assert by_plugin["pii:session"].category == "pii"
    assert by_plugin["medical:anchoring-bias"].category == "phi"
    assert by_plugin["medical:fda:cyber-access-control"].category == "phi"
    assert by_plugin["harmful:hate"].category == "safety"
    assert by_plugin["harmful:cybercrime"].category == "safety"


def test_should_flag_true_for_adversarial_categories():
    cases = load_redteam_cases(FIXTURE)
    assert all(case.should_flag for case in cases)


def test_severity_and_metric_captured():
    cases = load_redteam_cases(FIXTURE)
    hate_case = next(c for c in cases if c.plugin_id == "harmful:hate")
    assert hate_case.severity == "critical"
    assert hate_case.metric == "Hate"
