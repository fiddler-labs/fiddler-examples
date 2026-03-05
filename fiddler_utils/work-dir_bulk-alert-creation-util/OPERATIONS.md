# Bulk Alert Creation Utility — Operations Manual

This document is the comprehensive guide for understanding and using the bulk alert creation utility. For a quick start, see [README.md](README.md).

---

## Table of Contents

1. [Concepts](#1-concepts)
2. [Alert Profiles](#2-alert-profiles)
3. [Alert Specs](#3-alert-specs)
4. [Threshold Strategies](#4-threshold-strategies)
5. [Building Custom Profiles](#5-building-custom-profiles)
6. [Scope Filtering](#6-scope-filtering)
7. [Modes of Operation](#7-modes-of-operation)
8. [Notifications](#8-notifications)
9. [Baselines and Drift Alerts](#9-baselines-and-drift-alerts)
10. [How Per-Column Expansion Works](#10-how-per-column-expansion-works)
11. [How Task Filtering Works](#11-how-task-filtering-works)
12. [Reports and CSV Export](#12-reports-and-csv-export)
13. [Troubleshooting](#13-troubleshooting)
14. [Full API Reference](#14-full-api-reference)

---

## 1. Concepts

The utility has three layers:

```
AlertProfile          →  "What alerts should exist"
  └── AlertSpec[]     →  "One type of alert, with its config"
BulkAlertCreator      →  "Create/update alerts across models"
  └── ModelScopeFilter →  "Which models to process"
```

**AlertProfile** is a named collection of alert specifications. You configure it once, then run it against your environment. Think of it as a "monitoring recipe."

**AlertSpec** is a single alert type definition — e.g., "a drift alert on each input column using JSD with absolute thresholds." One spec can produce many alerts (one per column, one per model).

**BulkAlertCreator** is the orchestrator. It takes a profile, iterates over models, resolves specs into concrete alert definitions, and creates/updates them via the Fiddler SDK.

---

## 2. Alert Profiles

### Built-in Profiles

```python
from fiddler_utils.alert_profiles import get_default_ml_profile, get_traffic_only_profile

# Full ML monitoring: traffic + drift + integrity + performance
profile = get_default_ml_profile()

# Minimal: traffic volume only
profile = get_traffic_only_profile()
```

### Profile Attributes

| Attribute | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | (required) | Human-readable profile name |
| `description` | `str` | `''` | What this profile does |
| `specs` | `List[AlertSpec]` | `[]` | Alert specifications |
| `notification` | `NotificationConfig` | (empty) | Email/PagerDuty/webhook config |
| `auto_create_baseline` | `bool` | `True` | Auto-create rolling baseline for drift alerts |
| `baseline_config` | `Dict` | 7-day rolling | Config for auto-created baselines |
| `default_sigma_warning` | `float` | `2.0` | Profile-level sigma warning multiplier |
| `default_sigma_critical` | `float` | `3.0` | Profile-level sigma critical multiplier |

### Profile Methods

```python
# Add a spec
profile.add_spec(my_spec)

# Remove all specs for a metric
profile.remove_spec_by_metric('mape')

# Get specs that apply to a model task type
classification_specs = profile.filter_for_task_type(ModelTask.BINARY_CLASSIFICATION)

# Merge another profile (adds specs that don't exist yet)
profile.merge(other_profile)

# Resolve effective threshold for a spec (applies profile-level sigma defaults)
effective_threshold = profile.resolve_threshold(spec)
```

### Two-Level Sigma Defaults

When a spec uses `ThresholdStrategy.SIGMA` with the default values (2.0/3.0), the profile's `default_sigma_warning` and `default_sigma_critical` are applied instead. This lets you tune all sigma alerts at once:

```python
profile = get_default_ml_profile()

# This one change affects traffic, precision, recall, f1, mae, mape, r2 alerts
profile.default_sigma_warning = 2.5
profile.default_sigma_critical = 3.5
```

If a spec has non-default sigma values (e.g., `warning_value=1.5`), it keeps its own values regardless of the profile default.

---

## 3. Alert Specs

An `AlertSpec` defines one type of alert. Here's every attribute explained:

### Required Fields

```python
AlertSpec(
    name_template='{model_name} | Drift | {column_name}',
    # Template for the alert name. Supports {model_name} and {column_name}.
    # The rendered name is used for idempotency — same name = same alert.

    category='data_drift',
    # A label for grouping. Not enforced by the SDK, but useful for filtering.
    # Common values: 'traffic', 'data_drift', 'data_integrity', 'performance', 'custom_metric'

    metric_id='jsd',
    # The Fiddler metric this alert monitors.
    # Built-in: 'traffic', 'jsd', 'null_violation_percentage', 'range_violation_count',
    #           'type_violation_count', 'precision', 'recall', 'f1_score', 'accuracy',
    #           'auc', 'mae', 'mse', 'rmse', 'mape', 'r2'
    # Custom: any custom metric ID you've created in Fiddler
)
```

### Optional Fields with Defaults

```python
AlertSpec(
    ...,
    bin_size=BinSize.DAY,
    # Aggregation window. Options: BinSize.HOUR, BinSize.DAY, BinSize.WEEK, BinSize.MONTH
    # DAY is recommended. MONTH has known issues (see misc-utils/replace_alerts_with_mods.py)

    compare_to=CompareTo.RAW_VALUE,
    # What to compare the metric against.
    # RAW_VALUE: compare metric value against absolute thresholds
    # TIME_PERIOD: compare metric value against a historical bin (use compare_bin_delta)

    condition=AlertCondition.GREATER,
    # When to fire. GREATER = "fire when metric exceeds threshold"
    # LESSER = "fire when metric drops below threshold"
    # Rule of thumb: GREATER for bad-is-high (drift, errors, MAE)
    #                LESSER for bad-is-low (precision, recall, traffic, R2)

    threshold=ThresholdConfig(strategy=ThresholdStrategy.SIGMA),
    # How thresholds are determined. See section 4.

    priority=Priority.HIGH,
    # Alert severity. Options: Priority.HIGH, Priority.MEDIUM, Priority.LOW

    columns_source='none',
    # Where to get columns for per-column alerts.
    # 'none' = single alert, no columns
    # 'inputs' = one alert per model input column
    # 'outputs' = one alert per model output column
    # 'all' = one alert per input + output column

    requires_baseline=False,
    # If True, this alert needs a baseline_id. Used for drift alerts.
    # The orchestrator will auto-create a baseline if none exists.

    requires_columns=False,
    # If True, the alert is created once per column (per-column expansion).
    # Must be used with columns_source != 'none'.

    compare_bin_delta=None,
    # For TIME_PERIOD mode: how many bins back to compare.
    # Example: bin_size=HOUR, compare_bin_delta=24 = compare to same hour yesterday

    enabled=True,
    # Set to False to temporarily disable this spec without removing it.

    applicable_task_types=set(),
    # Which model task types this spec applies to. Empty set = all models.
    # Example: {ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION}
    # Common values: ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION,
    #                ModelTask.REGRESSION, ModelTask.RANKING, ModelTask.LLM, ModelTask.NOT_SET
)
```

---

## 4. Threshold Strategies

### SIGMA — Platform Computes Thresholds from Data

```python
threshold=ThresholdConfig(
    strategy=ThresholdStrategy.SIGMA,
    warning_value=2.0,   # 2σ from historical mean
    critical_value=3.0,  # 3σ from historical mean
)
```

The Fiddler platform calculates the mean and standard deviation of the metric's historical values, then sets the actual threshold at `mean ± (multiplier × stddev)`.

**Requirements for sigma thresholds:**
- The metric must have historical values (at least one bin period of data)
- For performance metrics (precision, recall, etc.) this means the model must have **actuals/labels** published via the Fiddler API
- If no historical data exists, the alert is created but will not fire

**How it maps to the SDK:**
```
ThresholdStrategy.SIGMA  →  threshold_type = AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD
                             auto_threshold_params = {'warning_multiplier': 2.0, 'critical_multiplier': 3.0}
                             warning_threshold = None  (platform computes)
                             critical_threshold = None  (platform computes)
```

### ABSOLUTE — You Set the Exact Threshold Values

```python
threshold=ThresholdConfig(
    strategy=ThresholdStrategy.ABSOLUTE,
    warning_value=0.15,  # Fire warning if metric reaches 0.15
    critical_value=0.3,  # Fire critical if metric reaches 0.3
)
```

You specify the exact numeric values. No historical data needed for threshold calculation (but the metric still needs data to evaluate against).

**How it maps to the SDK:**
```
ThresholdStrategy.ABSOLUTE  →  threshold_type = AlertThresholdAlgo.MANUAL
                                warning_threshold = 0.15
                                critical_threshold = 0.3
                                auto_threshold_params = None
```

### Which Strategy for Which Metric?

| Metric | Recommended Strategy | Why |
|---|---|---|
| `traffic` | SIGMA | Volume varies by model; sigma adapts to each model's baseline |
| `jsd` (drift) | ABSOLUTE | JSD is 0-1 bounded; 0.15/0.3 are well-understood thresholds |
| `null_violation_percentage` | ABSOLUTE | Percentage; 5%/10% are interpretable |
| `range_violation_count` | ABSOLUTE | Count; 10/50 are interpretable starting points |
| `precision`, `recall`, `f1_score` | SIGMA | Varies by model; sigma adapts |
| `mae`, `mape`, `r2` | SIGMA | Varies by model; sigma adapts |
| Custom metrics | Depends | Use ABSOLUTE if you know the scale; SIGMA if you don't |

---

## 5. Building Custom Profiles

### From Scratch

```python
from fiddler_utils.alert_profiles import (
    AlertProfile, AlertSpec, ThresholdConfig, ThresholdStrategy, NotificationConfig,
)
from fiddler.constants.alert_rule import BinSize, CompareTo, AlertCondition, Priority
from fiddler.constants.model import ModelTask

profile = AlertProfile(
    name='my_custom_profile',
    description='Traffic + drift only, no performance alerts',
    notification=NotificationConfig(emails=['team@company.com']),
    default_sigma_warning=2.0,
    default_sigma_critical=3.0,
    auto_create_baseline=True,
    specs=[
        AlertSpec(
            name_template='{model_name} | Traffic',
            category='traffic',
            metric_id='traffic',
            condition=AlertCondition.LESSER,
            threshold=ThresholdConfig(strategy=ThresholdStrategy.SIGMA),
            priority=Priority.HIGH,
        ),
        AlertSpec(
            name_template='{model_name} | Drift | {column_name}',
            category='data_drift',
            metric_id='jsd',
            condition=AlertCondition.GREATER,
            threshold=ThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                warning_value=0.2,
                critical_value=0.4,
            ),
            priority=Priority.MEDIUM,
            columns_source='inputs',
            requires_baseline=True,
            requires_columns=True,
        ),
    ],
)
```

### Starting from Default and Customizing

```python
profile = get_default_ml_profile()

# Remove what you don't want
profile.remove_spec_by_metric('mape')
profile.remove_spec_by_metric('r2')

# If no actuals, remove all performance metrics
for m in ['precision', 'recall', 'f1_score', 'mae', 'mape', 'r2']:
    profile.remove_spec_by_metric(m)

# Adjust sigma globally
profile.default_sigma_warning = 2.5

# Add a custom metric
profile.add_spec(AlertSpec(
    name_template='{model_name} | Lost Revenue',
    category='custom_metric',
    metric_id='lost_revenue',
    condition=AlertCondition.GREATER,
    threshold=ThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        warning_value=1000,
        critical_value=5000,
    ),
    priority=Priority.HIGH,
))

# Set notifications
profile.notification = NotificationConfig(emails=['alerts@company.com'])
```

### Composing Two Profiles

```python
base = get_default_ml_profile()
custom = AlertProfile(
    name='extra',
    specs=[
        AlertSpec(
            name_template='{model_name} | Custom',
            category='custom',
            metric_id='my_metric',
            threshold=ThresholdConfig(strategy=ThresholdStrategy.ABSOLUTE, warning_value=50, critical_value=100),
        ),
    ],
    notification=NotificationConfig(emails=['new@company.com']),
)

# Merge: adds custom's specs that don't already exist in base.
# If custom has notification config, it replaces base's.
base.merge(custom)
```

---

## 6. Scope Filtering

`ModelScopeFilter` controls which models the orchestrator processes. All filters are AND logic — a model must pass all active filters.

```python
from fiddler_utils import ModelScopeFilter

scope = ModelScopeFilter(
    # Include filters (None = no restriction)
    project_ids=['uuid-1', 'uuid-2'],         # By project UUID
    project_names=['production', 'staging'],   # By project name
    model_names=['model_a', 'model_b'],        # Specific model names
    model_name_pattern=r'^prod_',              # Regex on model name
    task_types=[ModelTask.BINARY_CLASSIFICATION, ModelTask.REGRESSION],

    # Exclude filters
    exclude_model_names=['deprecated_v1'],     # Skip these models
    exclude_model_ids=['model-uuid-xyz'],      # Skip by UUID
    exclude_projects=['sandbox'],              # Skip entire projects

    # Limit
    max_models=5,                              # Process at most N models (useful for testing)
)
```

**Common patterns:**

```python
# Process everything in one project
scope = ModelScopeFilter(project_names=['production'])

# Test on a single model before rolling out
scope = ModelScopeFilter(model_names=['my_test_model'], project_names=['production'])

# All models except test/sandbox
scope = ModelScopeFilter(exclude_projects=['sandbox', 'test'])
```

---

## 7. Modes of Operation

### `create` Mode

Creates alerts that don't exist. Skips alerts whose name already exists (idempotent).

```python
creator = BulkAlertCreator(profile=profile, scope=scope)
result = creator.run(mode='create', dry_run=False)
```

**With `overwrite=True`:** deletes existing alert and recreates it. Use when you've changed the profile and want to force a full reset. Warning: this loses alert history and triggered records.

### `update` Mode

Compares each profile spec against the existing alert and takes the minimal action:

1. **Alert doesn't exist** → creates it
2. **Mutable fields differ** (thresholds, evaluation_delay) → patches in-place via SDK `update()`
3. **Immutable fields differ** (bin_size, metric_id, condition, priority, columns) → deletes and recreates
4. **Nothing differs** → skips, but still syncs notifications

```python
# Change thresholds and notification
profile.default_sigma_warning = 2.5
profile.notification = NotificationConfig(emails=['new-team@company.com'])

creator = BulkAlertCreator(profile=profile, scope=scope)
result = creator.run(mode='update', dry_run=False)
```

**When to use which mode:**

| Scenario | Mode |
|---|---|
| First-time alert setup | `create` |
| Re-run after fixing a profile bug | `create` (idempotent, skips existing) |
| Change notification emails | `update` |
| Adjust sigma thresholds | `update` |
| Change bin_size from DAY to HOUR | `update` (will auto-recreate) |
| Full reset of all alerts | `create` with `overwrite=True` |

### Dry Run

Both modes support `dry_run=True`. The utility resolves all specs, computes diffs, and reports what it *would* do — without touching any alerts.

```python
result = creator.run(mode='create', dry_run=True)
creator.print_report(result, show_per_model=True)
# Review output, then run for real:
result = creator.run(mode='create', dry_run=False)
```

---

## 8. Notifications

```python
from fiddler_utils.alert_profiles import NotificationConfig
from uuid import UUID

profile.notification = NotificationConfig(
    emails=['alice@company.com', 'bob@company.com'],
    pagerduty_services=['service-key-1'],
    pagerduty_severity='warning',               # 'info', 'warning', 'error', 'critical'
    webhooks=[UUID('webhook-uuid-from-fiddler')],
)
```

Notifications are set on each alert after creation via `alert.set_notification_config()`. In `update` mode, notifications are synced even when the alert itself hasn't changed.

To update just notifications without changing thresholds, run `mode='update'` with the new `NotificationConfig` — existing alerts will be reported as "skipped" but notifications will be synced.

---

## 9. Baselines and Drift Alerts

Drift alerts (metric_id = `jsd`) require a baseline — a reference dataset to compare current data against.

### Auto-Baseline Creation

By default (`auto_create_baseline=True`), the utility creates a rolling baseline if none exists:

```python
# Default baseline config (customizable on the profile)
profile.baseline_config = {
    'name_template': '__auto_rolling_{model_name}',
    'type': 'ROLLING',
    'window_bin_size': WindowBinSize.DAY,
    'offset_delta': 7,  # 7-day rolling window
}
```

### Using Existing Baselines

If a model already has baselines, the utility uses the first one found. To control which baseline is used, ensure only the desired baseline exists, or disable auto-creation and set up baselines manually:

```python
profile.auto_create_baseline = False
# Now drift alerts will be SKIPPED for models without baselines
```

### Disabling Drift Alerts Entirely

```python
profile.remove_spec_by_metric('jsd')
```

---

## 10. How Per-Column Expansion Works

Specs with `requires_columns=True` create one alert per applicable column.

**Example:** A model `churn_model` with inputs `['age', 'income', 'tenure']` and a drift spec with `columns_source='inputs'`:

```
Spec template: '{model_name} | Drift | {column_name}'

Creates 3 alerts:
  - churn_model | Drift | age
  - churn_model | Drift | income
  - churn_model | Drift | tenure
```

**Column sources:**

| `columns_source` | Columns used |
|---|---|
| `'none'` | No columns — single alert per model |
| `'inputs'` | Model's input features |
| `'outputs'` | Model's output/prediction columns |
| `'all'` | Inputs + outputs combined |

A model with 50 input columns and 3 per-column specs (drift + null + range) creates 150 per-column alerts, plus any non-per-column alerts. The utility warns at 100+ columns.

---

## 11. How Task Filtering Works

Each spec can be restricted to specific model task types via `applicable_task_types`. If the set is empty, the spec applies to all models.

```python
# This spec only applies to classification models
AlertSpec(
    ...,
    applicable_task_types={ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION},
)
```

**What happens at runtime:**

| Model task | Gets traffic? | Gets drift? | Gets precision? | Gets MAE? |
|---|---|---|---|---|
| BINARY_CLASSIFICATION | Yes | Yes | Yes | No |
| REGRESSION | Yes | Yes | No | Yes |
| LLM | Yes | Yes | No | No |
| NOT_SET | Yes | Yes | No | No |

Traffic, drift, and data integrity specs have no task filter (empty set) so they apply to all models. Performance specs are filtered to their relevant task types.

---

## 12. Reports and CSV Export

### Print Report

```python
result = creator.run(mode='create', dry_run=False)
creator.print_report(result)                       # Summary only
creator.print_report(result, show_per_model=True)  # With per-model breakdown
```

Output:
```
============================================================
Bulk Alert Creation Report — Profile: ml_standard
============================================================
Models: 5 processed, 0 skipped, 0 failed
Alerts: 42 created, 3 updated, 1 recreated, 2 skipped (existing), 0 skipped (invalid), 0 failed
Baselines: 2 auto-created
Notifications: 42 configured
============================================================
```

### CSV Export

```python
csv_path = creator.export_report_csv(result, output_path='my_report.csv')
```

Columns: `project`, `model`, `task_type`, `alert_name`, `metric_id`, `status`

### BulkAlertResult Attributes

| Attribute | Type | Description |
|---|---|---|
| `models_processed` | `int` | Models successfully processed |
| `models_skipped` | `int` | Models skipped (errors, filters) |
| `models_failed` | `int` | Models where all alerts failed |
| `alerts_created` | `int` | New alerts created |
| `alerts_updated` | `int` | Existing alerts patched in-place |
| `alerts_recreated` | `int` | Alerts deleted and recreated (immutable change) |
| `alerts_skipped_existing` | `int` | Alerts skipped because they already exist |
| `alerts_skipped_invalid` | `int` | Alerts skipped due to validation errors |
| `alerts_failed` | `int` | Alerts that failed to create/update |
| `baselines_created` | `int` | Auto-created rolling baselines |
| `notifications_configured` | `int` | Alerts with notification config set |
| `errors` | `List[Tuple]` | `(model_name, alert_name, error_message)` tuples |
| `model_details` | `Dict` | Per-model breakdown |

---

## 13. Troubleshooting

### "Alert created but never fires"

**Cause:** The metric has no data. Most common with:
- Performance metrics (precision, recall, etc.) when actuals/labels are not published
- Sigma-based alerts on models with no historical data (new models)
- Drift alerts when no baseline exists or baseline has no data

**Fix:** Check that the model has the required data in Fiddler. For performance metrics, ensure actuals are being published via the Fiddler publish API.

### "All alerts skipped on re-run"

**Expected behavior** in `create` mode with `overwrite=False`. The utility sees existing alerts with matching names and skips them. To force recreation: use `overwrite=True` or switch to `mode='update'`.

### "Drift alerts skipped for model X"

**Cause:** No baseline exists and `auto_create_baseline=False`, OR auto-baseline creation failed (e.g., no production data).

**Fix:** Either set `profile.auto_create_baseline = True` or create a baseline manually in the Fiddler UI before running.

### "Too many alerts created"

**Cause:** Models with many input columns produce many per-column alerts (drift + null + range).

**Fix:** Use an explicit column list instead of `'inputs'`:
```python
spec.columns_source = ['age', 'income', 'score']  # Only these columns
```
Or remove per-column alert types entirely:
```python
profile.remove_spec_by_metric('jsd')
profile.remove_spec_by_metric('null_violation_percentage')
profile.remove_spec_by_metric('range_violation_count')
```

### "Update mode recreated an alert I only wanted to patch"

**Cause:** You changed a field that is immutable (bin_size, metric_id, condition, priority, columns, baseline_id). These cannot be patched in-place — the SDK doesn't support it. The utility automatically deletes and recreates.

**Implication:** The alert gets a new ID, and its triggered history is lost. This is inherent to the Fiddler platform, not a utility limitation.

### "Error: mode must be 'create' or 'update'"

**Fix:** Pass `mode='create'` or `mode='update'` to `creator.run()`.

---

## 14. Full API Reference

### Imports

```python
# Core utility
from fiddler_utils import BulkAlertCreator, ModelScopeFilter, BulkAlertResult

# Profile configuration
from fiddler_utils.alert_profiles import (
    AlertProfile,
    AlertSpec,
    ThresholdConfig,
    ThresholdStrategy,
    NotificationConfig,
    get_default_ml_profile,
    get_traffic_only_profile,
)

# SDK enums (used directly in AlertSpec fields)
from fiddler.constants.alert_rule import BinSize, CompareTo, AlertCondition, Priority
from fiddler.constants.model import ModelTask
```

### BulkAlertCreator

```python
creator = BulkAlertCreator(
    profile: AlertProfile,                 # Required — what alerts to create
    scope: ModelScopeFilter = None,        # Which models to process (None = all)
    overwrite: bool = False,               # Delete+recreate existing (create mode)
    skip_invalid: bool = True,             # Skip or raise on per-alert errors
    on_error: str = 'warn',               # 'warn', 'skip', or 'raise' for model errors
)

result = creator.run(
    mode: str = 'create',                  # 'create' or 'update'
    dry_run: bool = False,                 # Preview without modifying
)

creator.print_report(result, show_per_model=False)
creator.export_report_csv(result, output_path='report.csv')
```

### Example Script

See [`fiddler_utils/examples/bulk_alert_creation.py`](../examples/bulk_alert_creation.py) for a complete, customizable script ready to copy and adapt.
