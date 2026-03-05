"""Unit tests for alert_profiles module.

These tests require no Fiddler connection — they test pure data
structures, profile composition, and threshold resolution logic.
"""

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
    NotificationConfig,
    ThresholdConfig,
    ThresholdStrategy,
    get_default_ml_profile,
    get_traffic_only_profile,
)


class TestThresholdConfig:
    def test_default_is_sigma(self):
        tc = ThresholdConfig()
        assert tc.strategy == ThresholdStrategy.SIGMA
        assert tc.warning_value == 2.0
        assert tc.critical_value == 3.0

    def test_sigma_to_sdk_params(self):
        tc = ThresholdConfig(
            strategy=ThresholdStrategy.SIGMA,
            warning_value=2.0,
            critical_value=3.0,
        )
        params = tc.to_sdk_params()

        assert params['threshold_type'] == AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD
        assert params['auto_threshold_params'] == {
            'warning_multiplier': 2.0,
            'critical_multiplier': 3.0,
        }
        assert params['warning_threshold'] is None
        assert params['critical_threshold'] is None

    def test_absolute_to_sdk_params(self):
        tc = ThresholdConfig(
            strategy=ThresholdStrategy.ABSOLUTE,
            warning_value=0.15,
            critical_value=0.3,
        )
        params = tc.to_sdk_params()

        assert params['threshold_type'] == AlertThresholdAlgo.MANUAL
        assert params['warning_threshold'] == 0.15
        assert params['critical_threshold'] == 0.3
        assert params['auto_threshold_params'] is None


class TestAlertSpec:
    def test_default_values(self):
        spec = AlertSpec(
            name_template='{model_name} | Test',
            category='test',
            metric_id='test_metric',
        )
        assert spec.bin_size == BinSize.DAY
        assert spec.compare_to == CompareTo.RAW_VALUE
        assert spec.condition == AlertCondition.GREATER
        assert spec.priority == Priority.HIGH
        assert spec.columns_source == 'none'
        assert spec.requires_baseline is False
        assert spec.requires_columns is False
        assert spec.enabled is True
        assert spec.applicable_task_types == set()

    def test_per_column_spec(self):
        spec = AlertSpec(
            name_template='{model_name} | Drift | {column_name}',
            category='data_drift',
            metric_id='jsd',
            columns_source='inputs',
            requires_baseline=True,
            requires_columns=True,
        )
        assert spec.requires_columns is True
        assert spec.columns_source == 'inputs'


class TestNotificationConfig:
    def test_empty_has_no_notifications(self):
        nc = NotificationConfig()
        assert nc.has_notifications is False

    def test_emails_has_notifications(self):
        nc = NotificationConfig(emails=['test@example.com'])
        assert nc.has_notifications is True

    def test_pagerduty_has_notifications(self):
        nc = NotificationConfig(pagerduty_services=['svc-1'])
        assert nc.has_notifications is True


class TestAlertProfile:
    def test_add_spec(self):
        profile = AlertProfile(name='test')
        assert len(profile.specs) == 0

        spec = AlertSpec(
            name_template='Test',
            category='test',
            metric_id='test',
        )
        result = profile.add_spec(spec)

        assert result is profile  # Returns self
        assert len(profile.specs) == 1

    def test_remove_spec_by_metric(self):
        profile = AlertProfile(
            name='test',
            specs=[
                AlertSpec(name_template='A', category='t', metric_id='keep'),
                AlertSpec(name_template='B', category='t', metric_id='remove'),
                AlertSpec(name_template='C', category='t', metric_id='keep'),
            ],
        )
        result = profile.remove_spec_by_metric('remove')

        assert result is profile
        assert len(profile.specs) == 2
        assert all(s.metric_id == 'keep' for s in profile.specs)

    def test_filter_for_task_type_no_filter(self):
        """Specs with empty applicable_task_types apply to all."""
        spec = AlertSpec(
            name_template='All',
            category='traffic',
            metric_id='traffic',
        )
        profile = AlertProfile(name='test', specs=[spec])

        result = profile.filter_for_task_type(ModelTask.BINARY_CLASSIFICATION)
        assert len(result) == 1

        result = profile.filter_for_task_type(ModelTask.LLM)
        assert len(result) == 1

    def test_filter_for_task_type_with_filter(self):
        """Specs with applicable_task_types only match those types."""
        spec = AlertSpec(
            name_template='Classification only',
            category='performance',
            metric_id='precision',
            applicable_task_types={
                ModelTask.BINARY_CLASSIFICATION,
                ModelTask.MULTICLASS_CLASSIFICATION,
            },
        )
        profile = AlertProfile(name='test', specs=[spec])

        assert len(profile.filter_for_task_type(ModelTask.BINARY_CLASSIFICATION)) == 1
        assert len(profile.filter_for_task_type(ModelTask.MULTICLASS_CLASSIFICATION)) == 1
        assert len(profile.filter_for_task_type(ModelTask.REGRESSION)) == 0
        assert len(profile.filter_for_task_type(ModelTask.LLM)) == 0

    def test_filter_excludes_disabled(self):
        spec = AlertSpec(
            name_template='Disabled',
            category='test',
            metric_id='test',
            enabled=False,
        )
        profile = AlertProfile(name='test', specs=[spec])

        result = profile.filter_for_task_type(ModelTask.BINARY_CLASSIFICATION)
        assert len(result) == 0

    def test_merge_adds_new_specs(self):
        base = AlertProfile(
            name='base',
            specs=[
                AlertSpec(name_template='A', category='t', metric_id='traffic'),
            ],
        )
        other = AlertProfile(
            name='other',
            specs=[
                AlertSpec(name_template='B', category='t', metric_id='jsd',
                          columns_source='inputs'),
            ],
        )
        base.merge(other)

        assert len(base.specs) == 2
        assert base.specs[1].metric_id == 'jsd'

    def test_merge_does_not_duplicate(self):
        base = AlertProfile(
            name='base',
            specs=[
                AlertSpec(name_template='A', category='t', metric_id='traffic'),
            ],
        )
        other = AlertProfile(
            name='other',
            specs=[
                AlertSpec(name_template='B', category='t', metric_id='traffic'),
            ],
        )
        base.merge(other)

        assert len(base.specs) == 1  # Not duplicated

    def test_merge_takes_notification_from_other(self):
        base = AlertProfile(name='base')
        other = AlertProfile(
            name='other',
            notification=NotificationConfig(emails=['test@example.com']),
        )
        base.merge(other)

        assert base.notification.emails == ['test@example.com']

    def test_resolve_threshold_absolute_passthrough(self):
        """Absolute thresholds are returned as-is."""
        profile = AlertProfile(name='test', default_sigma_warning=2.5)
        spec = AlertSpec(
            name_template='Test',
            category='t',
            metric_id='test',
            threshold=ThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                warning_value=0.15,
                critical_value=0.3,
            ),
        )
        result = profile.resolve_threshold(spec)
        assert result.warning_value == 0.15  # Not affected by profile default

    def test_resolve_threshold_sigma_inherits_profile_defaults(self):
        """SIGMA specs with default values inherit from profile."""
        profile = AlertProfile(
            name='test',
            default_sigma_warning=2.5,
            default_sigma_critical=3.5,
        )
        spec = AlertSpec(
            name_template='Test',
            category='t',
            metric_id='test',
            threshold=ThresholdConfig(
                strategy=ThresholdStrategy.SIGMA,
                warning_value=2.0,  # Default
                critical_value=3.0,  # Default
            ),
        )
        result = profile.resolve_threshold(spec)
        assert result.warning_value == 2.5  # Inherited from profile
        assert result.critical_value == 3.5  # Inherited from profile

    def test_resolve_threshold_sigma_custom_overrides(self):
        """SIGMA specs with non-default values keep their own."""
        profile = AlertProfile(
            name='test',
            default_sigma_warning=2.5,
            default_sigma_critical=3.5,
        )
        spec = AlertSpec(
            name_template='Test',
            category='t',
            metric_id='test',
            threshold=ThresholdConfig(
                strategy=ThresholdStrategy.SIGMA,
                warning_value=1.5,  # Custom, not default
                critical_value=2.5,  # Custom, not default
            ),
        )
        result = profile.resolve_threshold(spec)
        assert result.warning_value == 1.5  # Kept
        assert result.critical_value == 2.5  # Kept


class TestDefaultProfiles:
    def test_ml_profile_has_traffic(self):
        profile = get_default_ml_profile()
        traffic_specs = [s for s in profile.specs if s.metric_id == 'traffic']
        assert len(traffic_specs) == 1
        assert traffic_specs[0].condition == AlertCondition.LESSER

    def test_ml_profile_has_drift(self):
        profile = get_default_ml_profile()
        drift_specs = [s for s in profile.specs if s.metric_id == 'jsd']
        assert len(drift_specs) == 1
        assert drift_specs[0].requires_baseline is True
        assert drift_specs[0].requires_columns is True
        assert drift_specs[0].columns_source == 'inputs'

    def test_ml_profile_has_integrity(self):
        profile = get_default_ml_profile()
        null_specs = [s for s in profile.specs if s.metric_id == 'null_violation_percentage']
        range_specs = [s for s in profile.specs if s.metric_id == 'range_violation_count']
        assert len(null_specs) == 1
        assert len(range_specs) == 1

    def test_ml_profile_classification_performance(self):
        profile = get_default_ml_profile()
        precision_specs = [s for s in profile.specs if s.metric_id == 'precision']
        assert len(precision_specs) == 1
        assert ModelTask.BINARY_CLASSIFICATION in precision_specs[0].applicable_task_types
        assert ModelTask.REGRESSION not in precision_specs[0].applicable_task_types

    def test_ml_profile_regression_performance(self):
        profile = get_default_ml_profile()
        mae_specs = [s for s in profile.specs if s.metric_id == 'mae']
        assert len(mae_specs) == 1
        assert ModelTask.REGRESSION in mae_specs[0].applicable_task_types
        assert ModelTask.BINARY_CLASSIFICATION not in mae_specs[0].applicable_task_types

    def test_ml_profile_regression_not_applied_to_classification(self):
        profile = get_default_ml_profile()
        classification_specs = profile.filter_for_task_type(
            ModelTask.BINARY_CLASSIFICATION
        )
        metric_ids = {s.metric_id for s in classification_specs}
        assert 'mae' not in metric_ids
        assert 'mape' not in metric_ids
        assert 'precision' in metric_ids

    def test_ml_profile_classification_not_applied_to_regression(self):
        profile = get_default_ml_profile()
        regression_specs = profile.filter_for_task_type(ModelTask.REGRESSION)
        metric_ids = {s.metric_id for s in regression_specs}
        assert 'precision' not in metric_ids
        assert 'recall' not in metric_ids
        assert 'mae' in metric_ids

    def test_ml_profile_llm_gets_non_filtered_specs(self):
        """LLM models get traffic + drift + integrity, not performance."""
        profile = get_default_ml_profile()
        llm_specs = profile.filter_for_task_type(ModelTask.LLM)
        metric_ids = {s.metric_id for s in llm_specs}
        assert 'traffic' in metric_ids
        assert 'jsd' in metric_ids
        assert 'null_violation_percentage' in metric_ids
        assert 'precision' not in metric_ids
        assert 'mae' not in metric_ids

    def test_traffic_only_profile(self):
        profile = get_traffic_only_profile()
        assert len(profile.specs) == 1
        assert profile.specs[0].metric_id == 'traffic'

    def test_profile_spec_count(self):
        profile = get_default_ml_profile()
        assert len(profile.specs) == 10  # 1 traffic + 3 drift/integrity + 6 performance
