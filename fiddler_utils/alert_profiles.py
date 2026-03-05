"""Alert profile definitions for bulk alert creation.

An AlertProfile defines "what alerts should exist for a model" as a
declarative specification. Profiles can be composed from individual
AlertSpec objects and customized per customer or model task type.

This module is a data definition layer — no SDK calls are made here.
All SDK interaction happens in the BulkAlertCreator orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from fiddler.constants.alert_rule import (
    AlertCondition,
    AlertThresholdAlgo,
    BinSize,
    CompareTo,
    Priority,
)
from fiddler.constants.baseline import WindowBinSize
from fiddler.constants.model import ModelTask


class ThresholdStrategy(str, Enum):
    """How thresholds are determined.

    ABSOLUTE: warning_value and critical_value are raw metric values.
    SIGMA: warning_value and critical_value are standard deviation multipliers.
        Maps to Fiddler's STD_DEV_AUTO_THRESHOLD — the platform auto-computes
        thresholds from historical data using these as sigma multipliers.
    """

    ABSOLUTE = 'absolute'
    SIGMA = 'sigma'


@dataclass
class ThresholdConfig:
    """Configuration for alert thresholds.

    For ABSOLUTE strategy:
        warning_value and critical_value are direct threshold numbers.

    For SIGMA strategy:
        warning_value is the sigma multiplier for warning (e.g., 2.0 = 2σ).
        critical_value is the sigma multiplier for critical (e.g., 3.0 = 3σ).
        The Fiddler platform auto-computes actual thresholds from historical data.
    """

    strategy: ThresholdStrategy = ThresholdStrategy.SIGMA
    warning_value: float = 2.0
    critical_value: float = 3.0

    def to_sdk_params(self) -> Dict[str, Any]:
        """Convert to SDK-compatible parameters for AlertRule construction.

        Returns:
            Dict with threshold_type, warning_threshold, critical_threshold,
            and optionally auto_threshold_params.
        """
        if self.strategy == ThresholdStrategy.SIGMA:
            return {
                'threshold_type': AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD,
                'auto_threshold_params': {
                    'warning_multiplier': self.warning_value,
                    'critical_multiplier': self.critical_value,
                },
                'warning_threshold': None,
                'critical_threshold': None,
            }
        else:
            return {
                'threshold_type': AlertThresholdAlgo.MANUAL,
                'warning_threshold': self.warning_value,
                'critical_threshold': self.critical_value,
                'auto_threshold_params': None,
            }


@dataclass
class AlertSpec:
    """Specification for a single alert type to be created.

    This is the atomic unit of the alert profile system. Each AlertSpec
    describes one kind of alert with its configuration. Per-column specs
    (requires_columns=True) are expanded into one alert per applicable column
    during bulk creation.

    Attributes:
        name_template: Name template with {model_name}, {column_name} placeholders.
        category: Alert category string (e.g., 'data_drift', 'traffic').
        metric_id: Fiddler metric ID string (e.g., 'jsd', 'traffic', 'precision').
        bin_size: Alert aggregation bin size.
        compare_to: Comparison mode — absolute or relative to time period.
        condition: When to trigger — metric GREATER or LESSER than threshold.
        threshold: Threshold configuration (strategy + values).
        priority: Alert priority level.
        columns_source: Where to get columns — 'inputs', 'outputs', 'all', 'none'.
        requires_baseline: Whether this alert needs a baseline_id (e.g., drift).
        requires_columns: Whether this alert is per-column (one alert per column).
        compare_bin_delta: For TIME_PERIOD mode, how many bins back to compare.
        enabled: Whether this spec is active in the profile.
        applicable_task_types: Which model task types this spec applies to.
            Empty set means all model types.
    """

    name_template: str
    category: str
    metric_id: str
    bin_size: BinSize = BinSize.DAY
    compare_to: CompareTo = CompareTo.RAW_VALUE
    condition: AlertCondition = AlertCondition.GREATER
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    priority: Priority = Priority.HIGH
    columns_source: str = 'none'
    requires_baseline: bool = False
    requires_columns: bool = False
    compare_bin_delta: Optional[int] = None
    enabled: bool = True
    applicable_task_types: Set[ModelTask] = field(default_factory=set)


@dataclass
class NotificationConfig:
    """Notification configuration for created alerts.

    Note: webhooks expects UUIDs (matching the SDK's type).
    """

    emails: List[str] = field(default_factory=list)
    pagerduty_services: List[str] = field(default_factory=list)
    pagerduty_severity: str = 'warning'
    webhooks: List[UUID] = field(default_factory=list)

    @property
    def has_notifications(self) -> bool:
        """Whether any notification channel is configured."""
        return bool(self.emails or self.pagerduty_services or self.webhooks)


@dataclass
class AlertProfile:
    """A complete alert profile — a named collection of AlertSpecs.

    Profiles define "what alerts should exist for a model". They can be
    composed, extended, and customized.

    Two-level sigma configuration: profile-level defaults apply to all
    SIGMA-strategy specs that use the default values (2.0/3.0). Specs
    with explicit non-default values keep their own thresholds.

    Attributes:
        name: Profile name (e.g., 'ml_standard', 'thumbtack_v0').
        description: Human-readable description.
        specs: List of AlertSpec definitions.
        notification: Notification configuration for all created alerts.
        auto_create_baseline: Whether to auto-create a rolling baseline
            if none exists (needed for drift alerts).
        baseline_config: Configuration for auto-created baselines.
        default_sigma_warning: Profile-level sigma warning multiplier.
            Applies to SIGMA specs that use the default 2.0 value.
        default_sigma_critical: Profile-level sigma critical multiplier.
            Applies to SIGMA specs that use the default 3.0 value.
    """

    name: str
    description: str = ''
    specs: List[AlertSpec] = field(default_factory=list)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    auto_create_baseline: bool = True
    baseline_config: Dict[str, Any] = field(
        default_factory=lambda: {
            'name_template': '__auto_rolling_{model_name}',
            'type': 'ROLLING',
            'window_bin_size': WindowBinSize.DAY,
            'offset_delta': 7,
        }
    )
    default_sigma_warning: float = 2.0
    default_sigma_critical: float = 3.0

    def add_spec(self, spec: AlertSpec) -> AlertProfile:
        """Add an alert spec to this profile. Returns self for chaining."""
        self.specs.append(spec)
        return self

    def remove_spec_by_metric(self, metric_id: str) -> AlertProfile:
        """Remove all specs matching a metric_id. Returns self for chaining."""
        self.specs = [s for s in self.specs if s.metric_id != metric_id]
        return self

    def filter_for_task_type(self, task_type: ModelTask) -> List[AlertSpec]:
        """Return specs applicable to a given model task type."""
        return [
            s
            for s in self.specs
            if s.enabled
            and (not s.applicable_task_types or task_type in s.applicable_task_types)
        ]

    def merge(self, other: AlertProfile) -> AlertProfile:
        """Merge another profile into this one.

        Other's specs are added if their metric_id doesn't already exist.
        Other's notification config replaces ours if it has channels configured.
        Returns self for chaining.
        """
        existing_keys = {(s.metric_id, s.columns_source) for s in self.specs}
        for spec in other.specs:
            if (spec.metric_id, spec.columns_source) not in existing_keys:
                self.specs.append(spec)
        if other.notification.has_notifications:
            self.notification = other.notification
        return self

    def resolve_threshold(self, spec: AlertSpec) -> ThresholdConfig:
        """Resolve the effective threshold for a spec, applying profile defaults.

        If the spec uses SIGMA strategy with default values (2.0/3.0),
        the profile-level sigma defaults are applied instead.
        """
        tc = spec.threshold
        if tc.strategy != ThresholdStrategy.SIGMA:
            return tc

        # Apply profile defaults if spec uses the standard defaults
        warning = tc.warning_value
        critical = tc.critical_value
        if warning == 2.0:
            warning = self.default_sigma_warning
        if critical == 3.0:
            critical = self.default_sigma_critical

        if warning == tc.warning_value and critical == tc.critical_value:
            return tc

        return ThresholdConfig(
            strategy=ThresholdStrategy.SIGMA,
            warning_value=warning,
            critical_value=critical,
        )


# ---------------------------------------------------------------------------
# Default profile factories
# ---------------------------------------------------------------------------


def get_default_ml_profile() -> AlertProfile:
    """Default alert profile for ML models.

    Creates alerts for:
    - Traffic volume (1 alert per model, sigma-based)
    - Data drift via JSD on all input columns (1 alert per input, absolute)
    - Data integrity: null violations on all input columns (absolute)
    - Data integrity: range violations on all input columns (absolute)
    - Performance metrics appropriate to task type (sigma-based)

    LLM models also receive this profile. Task-filtered specs (e.g.,
    classification-only performance metrics) are automatically excluded
    for non-matching model types.
    """
    return AlertProfile(
        name='ml_standard',
        description='Standard ML monitoring: traffic + drift + data quality + performance',
        specs=[
            # --- Traffic ---
            AlertSpec(
                name_template='{model_name} | Traffic Volume',
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
                priority=Priority.HIGH,
            ),
            # --- Data Drift (JSD, per input column) ---
            AlertSpec(
                name_template='{model_name} | Drift | {column_name}',
                category='data_drift',
                metric_id='jsd',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.GREATER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=0.15,
                    critical_value=0.3,
                ),
                priority=Priority.MEDIUM,
                columns_source='inputs',
                requires_baseline=True,
                requires_columns=True,
            ),
            # --- Data Integrity: Null Violations (per input column) ---
            AlertSpec(
                name_template='{model_name} | Null Violation | {column_name}',
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
                priority=Priority.HIGH,
                columns_source='inputs',
                requires_columns=True,
            ),
            # --- Data Integrity: Range Violations (per input column) ---
            AlertSpec(
                name_template='{model_name} | Range Violation | {column_name}',
                category='data_integrity',
                metric_id='range_violation_count',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.GREATER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.ABSOLUTE,
                    warning_value=10,
                    critical_value=50,
                ),
                priority=Priority.MEDIUM,
                columns_source='inputs',
                requires_columns=True,
            ),
            # --- Performance: Classification ---
            AlertSpec(
                name_template='{model_name} | Precision Drop',
                category='performance',
                metric_id='precision',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.HIGH,
                applicable_task_types={
                    ModelTask.BINARY_CLASSIFICATION,
                    ModelTask.MULTICLASS_CLASSIFICATION,
                },
            ),
            AlertSpec(
                name_template='{model_name} | Recall Drop',
                category='performance',
                metric_id='recall',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.MEDIUM,
                applicable_task_types={
                    ModelTask.BINARY_CLASSIFICATION,
                    ModelTask.MULTICLASS_CLASSIFICATION,
                },
            ),
            AlertSpec(
                name_template='{model_name} | F1 Score Drop',
                category='performance',
                metric_id='f1_score',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.MEDIUM,
                applicable_task_types={
                    ModelTask.BINARY_CLASSIFICATION,
                    ModelTask.MULTICLASS_CLASSIFICATION,
                },
            ),
            # --- Performance: Regression ---
            AlertSpec(
                name_template='{model_name} | MAE Spike',
                category='performance',
                metric_id='mae',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.GREATER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.HIGH,
                applicable_task_types={ModelTask.REGRESSION},
            ),
            AlertSpec(
                name_template='{model_name} | MAPE Spike',
                category='performance',
                metric_id='mape',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.GREATER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.MEDIUM,
                applicable_task_types={ModelTask.REGRESSION},
            ),
            AlertSpec(
                name_template='{model_name} | R2 Drop',
                category='performance',
                metric_id='r2',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.HIGH,
                applicable_task_types={ModelTask.REGRESSION},
            ),
        ],
    )


def get_traffic_only_profile() -> AlertProfile:
    """Minimal profile: traffic alerts only.

    Good for quick validation or environments where only volume
    monitoring is needed.
    """
    return AlertProfile(
        name='traffic_only',
        description='Traffic volume monitoring only',
        specs=[
            AlertSpec(
                name_template='{model_name} | Traffic Volume',
                category='traffic',
                metric_id='traffic',
                bin_size=BinSize.DAY,
                compare_to=CompareTo.RAW_VALUE,
                condition=AlertCondition.LESSER,
                threshold=ThresholdConfig(
                    strategy=ThresholdStrategy.SIGMA,
                ),
                priority=Priority.HIGH,
            ),
        ],
    )
