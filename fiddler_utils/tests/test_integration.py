"""Integration tests for Bulk Alert Creation Utility.

These tests connect to a real Fiddler instance and exercise the full
create/update/delete lifecycle. They require a test_config.ini file
with valid credentials.

Setup:
    cp fiddler_utils/tests/test_config.ini.template fiddler_utils/tests/test_config.ini
    # Edit test_config.ini with your Fiddler URL and token

Run:
    python -m pytest fiddler_utils/tests/test_integration.py -v -s

The -s flag is recommended to see real-time progress logs.

These tests create and clean up their own alerts. They use a naming
prefix (__INTTEST__) to avoid colliding with production alerts.
"""

import configparser
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pytest

try:
    import fiddler as fdl
except ImportError:
    pytest.skip('fiddler-client not installed', allow_module_level=True)

from fiddler.constants.alert_rule import (
    AlertCondition,
    AlertThresholdAlgo,
    BinSize,
    CompareTo,
    Priority,
)
from fiddler.constants.model import ModelTask

from fiddler_utils import get_or_init, BulkAlertCreator, ModelScopeFilter
from fiddler_utils.alert_profiles import (
    AlertProfile,
    AlertSpec,
    NotificationConfig,
    ThresholdConfig,
    ThresholdStrategy,
    get_default_ml_profile,
    get_traffic_only_profile,
)
from fiddler_utils.assets.alerts import AlertManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test prefix — all test-created alerts use this to avoid collisions
# ---------------------------------------------------------------------------
TEST_PREFIX = '__INTTEST__'

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

CONFIG_PATHS = [
    Path(__file__).parent / 'test_config.ini',
    Path.cwd() / 'test_config.ini',
]


def _load_config() -> Optional[configparser.ConfigParser]:
    """Load test config from file or environment variables."""
    # Try env vars first
    url = os.environ.get('FIDDLER_URL')
    token = os.environ.get('FIDDLER_TOKEN')
    if url and token:
        config = configparser.ConfigParser()
        config['fiddler'] = {'url': url, 'token': token}
        config['test'] = {
            'project_name': os.environ.get('FIDDLER_TEST_PROJECT', ''),
            'model_name': os.environ.get('FIDDLER_TEST_MODEL', ''),
        }
        return config

    # Try config files
    for path in CONFIG_PATHS:
        if path.exists():
            config = configparser.ConfigParser()
            config.read(path)
            return config

    return None


_config = _load_config()

# Skip entire module if no config
if _config is None:
    pytest.skip(
        'No test_config.ini found and FIDDLER_URL/FIDDLER_TOKEN env vars not set. '
        'See test_config.ini.template for setup instructions.',
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def fiddler_connection():
    """Initialize Fiddler connection for the test module."""
    url = _config['fiddler']['url']
    token = _config['fiddler']['token']
    get_or_init(url=url, token=token, log_level='WARNING')
    return {'url': url}


@pytest.fixture(scope='module')
def test_model(fiddler_connection):
    """Find a suitable test model.

    Looks for a model with:
    - At least 1 input column
    - A known task type (classification or regression)

    Returns (project, model) tuple.
    """
    project_name = _config['test'].get('project_name', '').strip()
    model_name = _config['test'].get('model_name', '').strip()

    if project_name and model_name:
        project = fdl.Project.from_name(project_name)
        model = fdl.Model.from_name(model_name, project_id=project.id)
        return project, model

    # Auto-discover: find first model with inputs
    for project in fdl.Project.list():
        if project_name and project.name != project_name:
            continue
        for model_compact in fdl.Model.list(project_id=project.id):
            try:
                model = fdl.Model.get(id_=model_compact.id)
                inputs = getattr(model.spec, 'inputs', None) or []
                if len(inputs) >= 1:
                    logger.info(
                        f'Using test model: {project.name}/{model.name} '
                        f'(task={model.task}, inputs={len(inputs)})'
                    )
                    return project, model
            except Exception:
                continue

    pytest.skip('No suitable test model found. Configure model_name in test_config.ini.')


@pytest.fixture(scope='module')
def alert_mgr(fiddler_connection):
    """AlertManager instance."""
    return AlertManager()


@pytest.fixture(autouse=True)
def cleanup_test_alerts(test_model, alert_mgr):
    """Clean up any test alerts before and after each test."""
    _, model = test_model
    _cleanup_alerts(model.id, alert_mgr)
    yield
    _cleanup_alerts(model.id, alert_mgr)


def _cleanup_alerts(model_id: str, mgr: AlertManager):
    """Delete all alerts with the test prefix."""
    try:
        alerts = mgr.list_assets(model_id=model_id)
        for alert in alerts:
            if alert.name.startswith(TEST_PREFIX):
                try:
                    alert.delete()
                    logger.info(f'Cleaned up test alert: {alert.name}')
                except Exception as e:
                    logger.warning(f'Failed to clean up {alert.name}: {e}')
    except Exception as e:
        logger.warning(f'Failed to list alerts for cleanup: {e}')


def _make_test_profile(
    specs: Optional[List[AlertSpec]] = None,
    notification: Optional[NotificationConfig] = None,
) -> AlertProfile:
    """Create a test profile with the test prefix in name templates."""
    if specs is None:
        specs = [
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=10,
                    critical_value=5,
                ),
                priority=Priority.LOW,
            ),
        ]
    return AlertProfile(
        name='integration_test',
        specs=specs,
        notification=notification or NotificationConfig(),
        auto_create_baseline=False,
    )


def _count_test_alerts(model_id: str, mgr: AlertManager) -> int:
    """Count alerts with the test prefix."""
    alerts = mgr.list_assets(model_id=model_id)
    return sum(1 for a in alerts if a.name.startswith(TEST_PREFIX))


def _get_test_alert(model_id: str, name: str, mgr: AlertManager):
    """Get a specific test alert by name."""
    alerts = mgr.list_assets(model_id=model_id)
    for a in alerts:
        if a.name == name:
            return a
    return None


# ---------------------------------------------------------------------------
# Tests: Create Mode
# ---------------------------------------------------------------------------


class TestCreateMode:
    """Test the create mode end-to-end against a real Fiddler instance."""

    def test_create_single_alert(self, test_model, alert_mgr):
        """Create a single traffic alert and verify it exists."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        assert result.alerts_created == 1
        assert result.alerts_failed == 0
        assert result.models_processed == 1

        # Verify alert exists in Fiddler
        assert _count_test_alerts(model.id, alert_mgr) == 1

    def test_create_multiple_alerts(self, test_model, alert_mgr):
        """Create multiple alert types for one model."""
        project, model = test_model
        inputs = list(getattr(model.spec, 'inputs', None) or [])[:2]

        specs = [
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=10,
                    critical_value=5,
                ),
                priority=Priority.LOW,
            ),
        ]
        # Add per-column null alerts if we have inputs
        if inputs:
            specs.append(
                AlertSpec(
                    name_template=TEST_PREFIX + '{model_name} | Null | {column_name}',
                    category='data_integrity',
                    metric_id='null_violation_percentage',
                    bin_size=BinSize.DAY,
                    compare_to=CompareTo.RAW_VALUE,
                    condition=AlertCondition.GREATER,
                    threshold=ThresholdConfig(
                        strategy=ThresholdStrategy.ABSOLUTE,
                        warning_value=5.0,
                        critical_value=10.0,
                    ),
                    priority=Priority.MEDIUM,
                    columns_source='inputs',
                    requires_columns=True,
                ),
            )

        profile = _make_test_profile(specs=specs)
        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        expected_count = 1 + len(inputs)  # 1 traffic + N null alerts
        assert result.alerts_created == expected_count
        assert _count_test_alerts(model.id, alert_mgr) == expected_count

    def test_create_idempotent_rerun(self, test_model, alert_mgr):
        """Second run with same profile should skip all existing alerts."""
        project, model = test_model
        profile = _make_test_profile()
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # First run — creates
        creator = BulkAlertCreator(profile=profile, scope=scope)
        result1 = creator.run(mode='create', dry_run=False)
        assert result1.alerts_created == 1

        # Second run — skips
        result2 = creator.run(mode='create', dry_run=False)
        assert result2.alerts_created == 0
        assert result2.alerts_skipped_existing == 1

        # Still only 1 alert in Fiddler
        assert _count_test_alerts(model.id, alert_mgr) == 1

    def test_create_overwrite_deletes_and_recreates(self, test_model, alert_mgr):
        """overwrite=True should delete existing and create new."""
        project, model = test_model
        profile = _make_test_profile()
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # First run
        creator = BulkAlertCreator(profile=profile, scope=scope)
        result1 = creator.run(mode='create', dry_run=False)
        assert result1.alerts_created == 1

        # Get the original alert ID
        original_alert = _get_test_alert(
            model.id,
            TEST_PREFIX + f'{model.name} | Traffic',
            alert_mgr,
        )
        assert original_alert is not None
        original_id = original_alert.id

        # Second run with overwrite
        creator_overwrite = BulkAlertCreator(
            profile=profile, scope=scope, overwrite=True
        )
        result2 = creator_overwrite.run(mode='create', dry_run=False)
        assert result2.alerts_created == 1

        # Alert exists but with a NEW ID (was deleted and recreated)
        new_alert = _get_test_alert(
            model.id,
            TEST_PREFIX + f'{model.name} | Traffic',
            alert_mgr,
        )
        assert new_alert is not None
        assert new_alert.id != original_id


# ---------------------------------------------------------------------------
# Tests: Dry Run
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify dry run never creates, modifies, or deletes anything."""

    def test_dry_run_creates_nothing(self, test_model, alert_mgr):
        """Dry run should report counts but create zero alerts."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=True)

        # Reports as if created
        assert result.alerts_created >= 1
        assert result.models_processed >= 1

        # But nothing actually exists
        assert _count_test_alerts(model.id, alert_mgr) == 0

    def test_dry_run_update_modifies_nothing(self, test_model, alert_mgr):
        """Dry run update should not modify existing alerts."""
        project, model = test_model
        profile = _make_test_profile()
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # Create alert first (live)
        creator = BulkAlertCreator(profile=profile, scope=scope)
        creator.run(mode='create', dry_run=False)

        # Get the original threshold
        alert = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        original_threshold = alert.warning_threshold

        # Change profile thresholds
        profile.specs[0].threshold = ThresholdConfig(
            strategy=ThresholdStrategy.ABSOLUTE,
            warning_value=999,
            critical_value=998,
        )

        # Dry run update
        result = creator.run(mode='update', dry_run=True)

        # Verify the original alert is untouched
        alert_after = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        assert alert_after.warning_threshold == original_threshold


# ---------------------------------------------------------------------------
# Tests: Update Mode
# ---------------------------------------------------------------------------


class TestUpdateMode:
    """Test update mode against a real Fiddler instance."""

    def test_update_creates_missing_alerts(self, test_model, alert_mgr):
        """Update mode should create alerts that don't exist."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )

        # No existing alerts — update should create
        result = creator.run(mode='update', dry_run=False)
        assert result.alerts_created == 1
        assert _count_test_alerts(model.id, alert_mgr) == 1

    def test_update_mutable_threshold(self, test_model, alert_mgr):
        """Update mode should patch mutable threshold fields in-place."""
        project, model = test_model
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # Create with initial thresholds
        profile = _make_test_profile(specs=[
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=100,
                    critical_value=50,
                ),
                priority=Priority.LOW,
            ),
        ])
        creator = BulkAlertCreator(profile=profile, scope=scope)
        creator.run(mode='create', dry_run=False)

        original = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        original_id = original.id
        assert original.warning_threshold == 100
        assert original.critical_threshold == 50

        # Update to new thresholds
        profile.specs[0].threshold = ThresholdConfig(
            strategy=ThresholdStrategy.ABSOLUTE,
            warning_value=200,
            critical_value=150,
        )
        result = creator.run(mode='update', dry_run=False)

        assert result.alerts_updated == 1
        assert result.alerts_recreated == 0

        # Verify the alert was updated IN-PLACE (same ID)
        updated = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        assert updated.id == original_id  # Same alert, not recreated
        assert updated.warning_threshold == 200
        assert updated.critical_threshold == 150

    def test_update_immutable_triggers_recreate(self, test_model, alert_mgr):
        """Changing an immutable field (bin_size) should delete+recreate."""
        project, model = test_model
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # Create with DAY bin size
        profile = _make_test_profile(specs=[
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=100,
                    critical_value=50,
                ),
                priority=Priority.LOW,
            ),
        ])
        creator = BulkAlertCreator(profile=profile, scope=scope)
        creator.run(mode='create', dry_run=False)

        original = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        original_id = original.id

        # Change to HOUR bin size (immutable field)
        profile.specs[0].bin_size = BinSize.HOUR
        result = creator.run(mode='update', dry_run=False)

        assert result.alerts_recreated == 1
        assert result.alerts_updated == 0

        # Alert exists but with a NEW ID
        recreated = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        assert recreated is not None
        assert recreated.id != original_id
        assert recreated.bin_size == BinSize.HOUR

    def test_update_priority_triggers_recreate(self, test_model, alert_mgr):
        """Priority is classified as immutable — change should recreate."""
        project, model = test_model
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        # Create with LOW priority
        profile = _make_test_profile(specs=[
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=100,
                    critical_value=50,
                ),
                priority=Priority.LOW,
            ),
        ])
        creator = BulkAlertCreator(profile=profile, scope=scope)
        creator.run(mode='create', dry_run=False)

        original = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        original_id = original.id

        # Change to HIGH priority
        profile.specs[0].priority = Priority.HIGH
        result = creator.run(mode='update', dry_run=False)

        assert result.alerts_recreated == 1

        recreated = _get_test_alert(
            model.id, TEST_PREFIX + f'{model.name} | Traffic', alert_mgr
        )
        assert recreated.id != original_id
        assert recreated.priority == Priority.HIGH

    def test_update_no_changes_skips(self, test_model, alert_mgr):
        """When profile matches existing alert, update mode skips it."""
        project, model = test_model
        profile = _make_test_profile()
        scope = ModelScopeFilter(
            model_names=[model.name],
            project_names=[project.name],
        )

        creator = BulkAlertCreator(profile=profile, scope=scope)

        # Create
        creator.run(mode='create', dry_run=False)

        # Update with identical profile
        result = creator.run(mode='update', dry_run=False)
        assert result.alerts_skipped_existing == 1
        assert result.alerts_updated == 0
        assert result.alerts_recreated == 0


# ---------------------------------------------------------------------------
# Tests: Sigma-Based Thresholds
# ---------------------------------------------------------------------------


class TestSigmaThresholds:
    """Test STD_DEV_AUTO_THRESHOLD alert creation."""

    def test_create_sigma_alert(self, test_model, alert_mgr):
        """Create an alert with sigma-based auto-thresholds."""
        project, model = test_model
        profile = _make_test_profile(specs=[
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic Sigma',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                    warning_value=2.0,
                    critical_value=3.0,
                ),
                priority=Priority.LOW,
            ),
        ])

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        # Might fail if model has no data for sigma calculation —
        # that's a valid test outcome, just check it's handled
        if result.alerts_created == 1:
            alert = _get_test_alert(
                model.id,
                TEST_PREFIX + f'{model.name} | Traffic Sigma',
                alert_mgr,
            )
            assert alert is not None
            assert alert.threshold_type in (
                AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD,
                AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD.value,
                'standard_deviation_auto_threshold',
            )
            # warning_threshold should be None for sigma-based
            assert alert.warning_threshold is None
            assert alert.critical_threshold is None
        else:
            # Sigma failed due to no data — should be in errors
            assert result.alerts_failed > 0 or result.alerts_skipped_invalid > 0


# ---------------------------------------------------------------------------
# Tests: Scope Filtering
# ---------------------------------------------------------------------------


class TestScopeFiltering:
    """Test that scope filters correctly limit which models are processed."""

    def test_model_name_filter(self, test_model, alert_mgr):
        """Only process the specified model."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        assert result.models_processed == 1
        assert model.name in result.model_details

    def test_exclude_model_filter(self, test_model, alert_mgr):
        """Excluded model should be skipped."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                project_names=[project.name],
                exclude_model_names=[model.name],
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        assert model.name not in result.model_details

    def test_max_models_limit(self, test_model, alert_mgr):
        """max_models should cap processing."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                project_names=[project.name],
                max_models=1,
            ),
        )
        result = creator.run(mode='create', dry_run=False)

        assert result.models_processed <= 1


# ---------------------------------------------------------------------------
# Tests: Report Output
# ---------------------------------------------------------------------------


class TestReporting:
    """Test report generation."""

    def test_print_report_does_not_error(self, test_model, alert_mgr, capsys):
        """print_report should produce output without errors."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=True)
        creator.print_report(result, show_per_model=True)

        captured = capsys.readouterr()
        assert 'Bulk Alert Creation Report' in captured.out
        assert 'created' in captured.out

    def test_csv_export(self, test_model, alert_mgr, tmp_path):
        """CSV export should produce a valid file."""
        project, model = test_model
        profile = _make_test_profile()

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )
        result = creator.run(mode='create', dry_run=True)

        csv_path = str(tmp_path / 'test_report.csv')
        creator.export_report_csv(result, output_path=csv_path)

        assert Path(csv_path).exists()
        content = Path(csv_path).read_text()
        assert 'project' in content
        assert 'alert_name' in content


# ---------------------------------------------------------------------------
# Tests: Full Default Profile
# ---------------------------------------------------------------------------


class TestDefaultProfile:
    """Test with the actual default ML profile (traffic-only to limit scope)."""

    def test_traffic_only_profile_creates(self, test_model, alert_mgr):
        """The traffic_only profile should create exactly 1 alert."""
        project, model = test_model

        # Use traffic-only but with test prefix
        profile = _make_test_profile(specs=[
            AlertSpec(
                name_template=TEST_PREFIX + '{model_name} | Traffic Volume',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(strategy=ThresholdStrategy.SIGMA),
                priority=Priority.HIGH,
            ),
        ])

        creator = BulkAlertCreator(
            profile=profile,
            scope=ModelScopeFilter(
                model_names=[model.name],
                project_names=[project.name],
            ),
        )

        result = creator.run(mode='create', dry_run=False)

        # Sigma might fail if no data, but the flow should complete
        assert result.models_processed == 1
        assert result.alerts_failed + result.alerts_created == 1
