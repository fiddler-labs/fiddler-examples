"""Unit tests for bulk_alerts module.

These tests use mocks for fdl objects and test orchestration logic
without requiring a Fiddler connection.
"""

from unittest.mock import MagicMock, patch

import pytest

from fiddler.constants.alert_rule import (
    AlertCondition,
    AlertThresholdAlgo,
    BinSize,
    CompareTo,
    Priority,
)
from fiddler.constants.model import ModelTask

from fiddler_utils.alert_profiles import (
    AlertProfile,
    AlertSpec,
    ThresholdConfig,
    ThresholdStrategy,
    get_default_ml_profile,
)
from fiddler_utils.bulk_alerts import (
    BulkAlertCreator,
    BulkAlertResult,
    ModelScopeFilter,
)


_DEFAULT_INPUTS = ['age', 'income', 'score']
_DEFAULT_OUTPUTS = ['prediction']


def _make_mock_model(name='test_model', task=ModelTask.BINARY_CLASSIFICATION,
                     inputs=_DEFAULT_INPUTS, outputs=_DEFAULT_OUTPUTS,
                     model_id='model-123'):
    """Create a mock fdl.Model with spec."""
    model = MagicMock()
    model.name = name
    model.id = model_id
    model.task = task
    model.spec = MagicMock()
    model.spec.inputs = inputs
    model.spec.outputs = outputs
    return model


def _make_mock_project(name='test_project', project_id='proj-123'):
    model = MagicMock()
    model.name = name
    model.id = project_id
    return model


class TestModelScopeFilter:
    def test_default_no_restrictions(self):
        scope = ModelScopeFilter()
        assert scope.project_ids is None
        assert scope.max_models is None
        assert scope.exclude_model_names == []


class TestBulkAlertResult:
    def test_total_alerts_attempted(self):
        result = BulkAlertResult(
            alerts_created=5,
            alerts_updated=2,
            alerts_recreated=1,
            alerts_skipped_existing=3,
            alerts_skipped_invalid=1,
            alerts_failed=2,
        )
        assert result.total_alerts_attempted == 14

    def test_summary_format(self):
        result = BulkAlertResult(
            models_processed=3,
            models_skipped=1,
            alerts_created=10,
            alerts_updated=5,
            alerts_recreated=2,
        )
        summary = result.summary()
        assert 'Models: 3 processed' in summary
        assert '10 created' in summary
        assert '5 updated' in summary
        assert '2 recreated' in summary

    def test_summary_with_errors(self):
        result = BulkAlertResult(
            errors=[('model_a', 'alert_1', 'some error')],
        )
        summary = result.summary()
        assert 'Errors: 1 total' in summary
        assert 'model_a/alert_1' in summary


class TestBulkAlertCreatorShouldProcessModel:
    def setup_method(self):
        self.profile = AlertProfile(name='test')

    def test_passes_with_no_filters(self):
        creator = BulkAlertCreator(profile=self.profile)
        project = _make_mock_project()
        model = _make_mock_model()
        assert creator._should_process_model(project, model) is True

    def test_exclude_model_names(self):
        scope = ModelScopeFilter(exclude_model_names=['test_model'])
        creator = BulkAlertCreator(profile=self.profile, scope=scope)
        project = _make_mock_project()
        model = _make_mock_model(name='test_model')
        assert creator._should_process_model(project, model) is False

    def test_exclude_projects(self):
        scope = ModelScopeFilter(exclude_projects=['excluded_project'])
        creator = BulkAlertCreator(profile=self.profile, scope=scope)
        project = _make_mock_project(name='excluded_project')
        model = _make_mock_model()
        assert creator._should_process_model(project, model) is False

    def test_project_names_filter(self):
        scope = ModelScopeFilter(project_names=['allowed_project'])
        creator = BulkAlertCreator(profile=self.profile, scope=scope)

        allowed = _make_mock_project(name='allowed_project')
        denied = _make_mock_project(name='other_project')
        model = _make_mock_model()

        assert creator._should_process_model(allowed, model) is True
        assert creator._should_process_model(denied, model) is False

    def test_model_name_pattern(self):
        scope = ModelScopeFilter(model_name_pattern=r'^prod_')
        creator = BulkAlertCreator(profile=self.profile, scope=scope)
        project = _make_mock_project()

        prod_model = _make_mock_model(name='prod_model_v1')
        test_model = _make_mock_model(name='test_model_v1')

        assert creator._should_process_model(project, prod_model) is True
        assert creator._should_process_model(project, test_model) is False


class TestResolveModelTaskType:
    def setup_method(self):
        self.creator = BulkAlertCreator(profile=AlertProfile(name='test'))

    def test_binary_classification(self):
        model = _make_mock_model(task=ModelTask.BINARY_CLASSIFICATION)
        assert self.creator._resolve_model_task_type(model) == ModelTask.BINARY_CLASSIFICATION

    def test_regression(self):
        model = _make_mock_model(task=ModelTask.REGRESSION)
        assert self.creator._resolve_model_task_type(model) == ModelTask.REGRESSION

    def test_llm(self):
        model = _make_mock_model(task=ModelTask.LLM)
        assert self.creator._resolve_model_task_type(model) == ModelTask.LLM

    def test_none_returns_not_set(self):
        model = _make_mock_model()
        model.task = None
        assert self.creator._resolve_model_task_type(model) == ModelTask.NOT_SET


class TestResolveSpecsForModel:
    def test_filters_by_task_type(self):
        profile = get_default_ml_profile()
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model(task=ModelTask.REGRESSION)

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.REGRESSION, baseline_id='bl-123'
        )
        metric_ids = {d['metric_id'] for d in alert_defs}

        # Regression should have MAE, MAPE, R2 but not precision/recall
        assert 'mae' in metric_ids
        assert 'precision' not in metric_ids

    def test_expands_per_column(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(
                    name_template='{model_name} | Drift | {column_name}',
                    category='data_drift',
                    metric_id='jsd',
                    columns_source='inputs',
                    requires_baseline=True,
                    requires_columns=True,
                ),
            ],
        )
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model(inputs=['age', 'income', 'score'])

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.BINARY_CLASSIFICATION, baseline_id='bl-123'
        )

        assert len(alert_defs) == 3
        names = {d['name'] for d in alert_defs}
        assert 'test_model | Drift | age' in names
        assert 'test_model | Drift | income' in names
        assert 'test_model | Drift | score' in names

    def test_skips_drift_without_baseline(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(
                    name_template='{model_name} | Drift',
                    category='data_drift',
                    metric_id='jsd',
                    requires_baseline=True,
                ),
            ],
        )
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model()

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.BINARY_CLASSIFICATION, baseline_id=None
        )
        assert len(alert_defs) == 0

    def test_name_template_rendering(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(
                    name_template='{model_name} | Traffic Volume',
                    category='traffic',
                    metric_id='traffic',
                ),
            ],
        )
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model(name='my_model')

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.BINARY_CLASSIFICATION, baseline_id=None
        )
        assert alert_defs[0]['name'] == 'my_model | Traffic Volume'

    def test_sigma_threshold_resolution(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(
                    name_template='Test',
                    category='traffic',
                    metric_id='traffic',
                    threshold=ThresholdConfig(strategy=ThresholdStrategy.SIGMA),
                ),
            ],
        )
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model()

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.BINARY_CLASSIFICATION, baseline_id=None
        )

        assert alert_defs[0]['threshold_type'] == AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD
        assert alert_defs[0]['warning_threshold'] is None
        assert alert_defs[0]['auto_threshold_params']['warning_multiplier'] == 2.0

    def test_absolute_threshold_resolution(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(
                    name_template='Test',
                    category='integrity',
                    metric_id='null_violation_percentage',
                    threshold=ThresholdConfig(
                        strategy=ThresholdStrategy.ABSOLUTE,
                        warning_value=5.0,
                        critical_value=10.0,
                    ),
                ),
            ],
        )
        creator = BulkAlertCreator(profile=profile)
        model = _make_mock_model()

        alert_defs = creator._resolve_specs_for_model(
            model, ModelTask.BINARY_CLASSIFICATION, baseline_id=None
        )

        assert alert_defs[0]['threshold_type'] == AlertThresholdAlgo.MANUAL
        assert alert_defs[0]['warning_threshold'] == 5.0
        assert alert_defs[0]['critical_threshold'] == 10.0


class TestGetColumnsForSpec:
    def setup_method(self):
        self.creator = BulkAlertCreator(profile=AlertProfile(name='test'))

    def test_non_per_column(self):
        spec = AlertSpec(
            name_template='Test', category='t', metric_id='traffic',
            requires_columns=False,
        )
        model = _make_mock_model()
        result = self.creator._get_columns_for_spec(model, spec)
        assert result == [None]

    def test_inputs_source(self):
        spec = AlertSpec(
            name_template='Test', category='t', metric_id='jsd',
            columns_source='inputs', requires_columns=True,
        )
        model = _make_mock_model(inputs=['a', 'b', 'c'])
        result = self.creator._get_columns_for_spec(model, spec)
        assert result == ['a', 'b', 'c']

    def test_outputs_source(self):
        spec = AlertSpec(
            name_template='Test', category='t', metric_id='test',
            columns_source='outputs', requires_columns=True,
        )
        model = _make_mock_model(outputs=['pred'])
        result = self.creator._get_columns_for_spec(model, spec)
        assert result == ['pred']

    def test_empty_inputs(self):
        spec = AlertSpec(
            name_template='Test', category='t', metric_id='jsd',
            columns_source='inputs', requires_columns=True,
        )
        model = _make_mock_model(inputs=[])
        result = self.creator._get_columns_for_spec(model, spec)
        assert result == []


class TestRunModeValidation:
    def test_invalid_mode_raises(self):
        profile = AlertProfile(name='test')
        creator = BulkAlertCreator(profile=profile)
        with pytest.raises(ValueError, match="mode must be"):
            creator.run(mode='invalid')


def _make_mock_existing_alert(
    name='test_model | Traffic Volume',
    metric_id='traffic',
    bin_size=BinSize.DAY,
    compare_to=CompareTo.RAW_VALUE,
    condition=AlertCondition.LESSER,
    priority=Priority.HIGH,
    warning_threshold=None,
    critical_threshold=None,
    auto_threshold_params=None,
    columns=None,
    baseline_id=None,
    model_id='model-123',
    alert_id='alert-456',
):
    """Create a mock existing fdl.AlertRule for update mode testing."""
    alert = MagicMock()
    alert.name = name
    alert.id = alert_id
    alert.model_id = model_id
    alert.metric_id = metric_id
    alert.bin_size = bin_size
    alert.compare_to = compare_to
    alert.condition = condition
    alert.priority = priority
    alert.warning_threshold = warning_threshold
    alert.critical_threshold = critical_threshold
    alert.auto_threshold_params = auto_threshold_params
    alert.columns = columns
    alert.baseline_id = baseline_id
    alert.evaluation_delay = 0
    return alert


class TestDiffImmutableFields:
    def setup_method(self):
        self.creator = BulkAlertCreator(profile=AlertProfile(name='test'))

    def test_no_changes(self):
        existing = _make_mock_existing_alert(
            metric_id='traffic',
            bin_size=BinSize.DAY,
            compare_to=CompareTo.RAW_VALUE,
            condition=AlertCondition.LESSER,
        )
        desired = {
            'metric_id': 'traffic',
            'bin_size': BinSize.DAY,
            'compare_to': CompareTo.RAW_VALUE,
            'condition': AlertCondition.LESSER,
        }
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert changes == []

    def test_metric_id_changed(self):
        existing = _make_mock_existing_alert(metric_id='traffic')
        desired = {'metric_id': 'jsd'}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'metric_id' in changes

    def test_bin_size_changed(self):
        existing = _make_mock_existing_alert(bin_size=BinSize.DAY)
        desired = {'bin_size': BinSize.HOUR}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'bin_size' in changes

    def test_columns_changed(self):
        existing = _make_mock_existing_alert(columns=['age'])
        desired = {'columns': ['age', 'income']}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'columns' in changes

    def test_columns_both_none(self):
        existing = _make_mock_existing_alert(columns=None)
        desired = {}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'columns' not in changes


class TestDiffMutableFields:
    def setup_method(self):
        self.creator = BulkAlertCreator(profile=AlertProfile(name='test'))

    def test_no_changes(self):
        existing = _make_mock_existing_alert(
            warning_threshold=None,
            critical_threshold=None,
            auto_threshold_params={'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        )
        desired = {
            'warning_threshold': None,
            'critical_threshold': None,
            'auto_threshold_params': {'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        }
        changes = self.creator._diff_mutable_fields(existing, desired)
        assert changes == []

    def test_threshold_changed(self):
        existing = _make_mock_existing_alert(warning_threshold=5.0)
        desired = {'warning_threshold': 10.0}
        changes = self.creator._diff_mutable_fields(existing, desired)
        assert 'warning_threshold' in changes

    def test_auto_threshold_params_changed(self):
        existing = _make_mock_existing_alert(
            auto_threshold_params={'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        )
        desired = {
            'auto_threshold_params': {'warning_multiplier': 2.5, 'critical_multiplier': 3.5},
        }
        changes = self.creator._diff_mutable_fields(existing, desired)
        assert 'auto_threshold_params' in changes

    def test_priority_not_in_mutable(self):
        """Priority is in IMMUTABLE_FIELDS, not detected by _diff_mutable_fields."""
        existing = _make_mock_existing_alert(priority=Priority.HIGH)
        desired = {'priority': Priority.LOW}
        changes = self.creator._diff_mutable_fields(existing, desired)
        assert 'priority' not in changes


class TestDiffImmutableFieldsPriority:
    """Verify priority is treated as immutable (triggers delete+recreate)."""

    def setup_method(self):
        self.creator = BulkAlertCreator(profile=AlertProfile(name='test'))

    def test_priority_change_detected_as_immutable(self):
        existing = _make_mock_existing_alert(priority=Priority.HIGH)
        desired = {'priority': Priority.LOW}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'priority' in changes

    def test_same_priority_no_change(self):
        existing = _make_mock_existing_alert(priority=Priority.HIGH)
        desired = {'priority': Priority.HIGH}
        changes = self.creator._diff_immutable_fields(existing, desired)
        assert 'priority' not in changes


class TestHandleUpdateMode:
    def setup_method(self):
        self.profile = AlertProfile(name='test')
        self.creator = BulkAlertCreator(profile=self.profile)

    def test_creates_when_no_existing(self):
        """Update mode creates alerts that don't exist yet."""
        model = _make_mock_model()
        alert_def = {
            'name': 'test_model | Traffic Volume',
            'metric_id': 'traffic',
            'bin_size': BinSize.DAY,
            'compare_to': CompareTo.RAW_VALUE,
            'condition': AlertCondition.LESSER,
            'priority': Priority.HIGH,
            'category': 'traffic',
            'threshold_type': AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD,
            'warning_threshold': None,
            'critical_threshold': None,
            'auto_threshold_params': {'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        }
        result = BulkAlertResult()

        # Dry run — no existing alert
        status = self.creator._handle_update_mode(
            model, alert_def, alert_def['name'], None, True, result
        )
        assert status == 'created'
        assert result.alerts_created == 1

    def test_skips_when_nothing_changed(self):
        """Update mode skips alerts with no differences."""
        existing = _make_mock_existing_alert(
            metric_id='traffic',
            bin_size=BinSize.DAY,
            compare_to=CompareTo.RAW_VALUE,
            condition=AlertCondition.LESSER,
            priority=Priority.HIGH,
            warning_threshold=None,
            critical_threshold=None,
            auto_threshold_params={'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        )
        alert_def = {
            'name': existing.name,
            'metric_id': 'traffic',
            'bin_size': BinSize.DAY,
            'compare_to': CompareTo.RAW_VALUE,
            'condition': AlertCondition.LESSER,
            'priority': Priority.HIGH,
            'category': 'traffic',
            'threshold_type': AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD,
            'warning_threshold': None,
            'critical_threshold': None,
            'auto_threshold_params': {'warning_multiplier': 2.0, 'critical_multiplier': 3.0},
        }
        result = BulkAlertResult()
        model = _make_mock_model()

        status = self.creator._handle_update_mode(
            model, alert_def, alert_def['name'], existing, True, result
        )
        assert status == 'skipped_existing'
