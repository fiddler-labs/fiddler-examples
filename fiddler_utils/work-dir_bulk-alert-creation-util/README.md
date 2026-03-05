# Bulk Alert Creation Utility

A profile-driven utility for creating and updating monitoring alerts across all models in a Fiddler environment. Part of the `fiddler_utils` package.

---

## Quick Start

```python
from fiddler_utils import get_or_init, BulkAlertCreator
from fiddler_utils.alert_profiles import get_default_ml_profile, NotificationConfig

# 1. Connect
get_or_init(url='https://your-org.fiddler.ai', token='your-token')

# 2. Load a profile
profile = get_default_ml_profile()
profile.notification = NotificationConfig(emails=['alerts@company.com'])

# 3. Create alerts (dry run first)
creator = BulkAlertCreator(profile=profile)
result = creator.run(mode='create', dry_run=True)
creator.print_report(result)

# 4. Execute for real
result = creator.run(mode='create', dry_run=False)
creator.print_report(result, show_per_model=True)
```

For a complete, customizable script, see: [`fiddler_utils/examples/bulk_alert_creation.py`](../examples/bulk_alert_creation.py)

---

## Modes of Operation

### `create` — Initial Alert Setup

Creates alerts defined in the profile. Existing alerts (matched by name) are skipped unless `overwrite=True`.

```python
creator = BulkAlertCreator(profile=profile, overwrite=False)
result = creator.run(mode='create', dry_run=False)
```

- `overwrite=False` (default): Skip alerts that already exist. Safe for re-runs.
- `overwrite=True`: Delete existing alert and recreate from profile. Use when you need to change immutable properties (bin_size, metric_id, etc.).

### `update` — Modify Existing Alerts

Updates existing alerts to match the current profile. Smart about what it touches:

| Field Type | Examples | Update Behavior |
|---|---|---|
| **Mutable** | `warning_threshold`, `critical_threshold`, `evaluation_delay`, `auto_threshold_params` | Patched in-place via SDK `update()` |
| **Notifications** | emails, PagerDuty, webhooks | Synced via `set_notification_config()` |
| **Immutable** | `bin_size`, `metric_id`, `compare_to`, `condition`, `columns`, `baseline_id` | Delete + recreate (only way) |
| **Missing** | Alert doesn't exist yet | Created |

```python
# Change sigma thresholds and notification email
profile.default_sigma_warning = 2.5
profile.notification = NotificationConfig(emails=['new-team@company.com'])

creator = BulkAlertCreator(profile=profile)
result = creator.run(mode='update', dry_run=False)
```

---

## Customizing Profiles

### Adjust sigma thresholds globally

```python
profile = get_default_ml_profile()
profile.default_sigma_warning = 2.5   # All sigma alerts use 2.5σ warning
profile.default_sigma_critical = 3.5  # All sigma alerts use 3.5σ critical
```

### Remove specific alert types

```python
profile.remove_spec_by_metric('mape')  # No MAPE alerts
profile.remove_spec_by_metric('r2')    # No R2 alerts
```

### Add a custom alert spec

```python
from fiddler_utils.alert_profiles import AlertSpec, ThresholdConfig, ThresholdStrategy
from fiddler.constants.alert_rule import BinSize, CompareTo, AlertCondition, Priority

profile.add_spec(AlertSpec(
    name_template='{model_name} | Custom Business Metric',
    category='custom_metric',
    metric_id='my_custom_metric_id',
    bin_size=BinSize.HOUR,
    compare_to=CompareTo.RAW_VALUE,
    condition=AlertCondition.GREATER,
    threshold=ThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        warning_value=100,
        critical_value=200,
    ),
    priority=Priority.HIGH,
))
```

### Scope to specific models

```python
from fiddler_utils import ModelScopeFilter

scope = ModelScopeFilter(
    project_names=['production'],           # Only this project
    exclude_model_names=['deprecated_v1'],  # Skip this model
    max_models=1,                           # Process only 1 (for testing)
)
creator = BulkAlertCreator(profile=profile, scope=scope)
```

---

## Default ML Profile

`get_default_ml_profile()` creates these alerts:

| Alert | Per-Column? | Threshold | Applied To | Data Required |
|---|---|---|---|---|
| Traffic Volume | No | 2σ/3σ (sigma) | All models | Inference events only |
| Data Drift (JSD) | Yes (inputs) | 0.15/0.3 (absolute) | All models | Inference events + baseline |
| Null Violations | Yes (inputs) | 5%/10% (absolute) | All models | Inference events only |
| Range Violations | Yes (inputs) | 10/50 (absolute) | All models | Inference events only |
| Precision Drop | No | 2σ/3σ (sigma) | Classification only | **Actuals/labels required** |
| Recall Drop | No | 2σ/3σ (sigma) | Classification only | **Actuals/labels required** |
| F1 Score Drop | No | 2σ/3σ (sigma) | Classification only | **Actuals/labels required** |
| MAE Spike | No | 2σ/3σ (sigma) | Regression only | **Actuals/labels required** |
| MAPE Spike | No | 2σ/3σ (sigma) | Regression only | **Actuals/labels required** |
| R2 Drop | No | 2σ/3σ (sigma) | Regression only | **Actuals/labels required** |

A model with 20 input columns gets: 1 (traffic) + 20 (drift) + 20 (null) + 20 (range) + 3 (perf) = **64 alerts**.

**Important: Performance metric alerts require ground truth.** The bottom 6 alerts (precision through R2) use sigma-based thresholds computed from historical metric values. These metrics only exist if your model receives actual target values (ground truth labels) via Fiddler's publish API. If your model only logs predictions without actuals, these alerts will be created successfully but will never fire — the underlying metric will have no data for the platform to evaluate. If your models don't have actuals flowing, remove the performance specs:

```python
profile = get_default_ml_profile()
for metric in ['precision', 'recall', 'f1_score', 'mae', 'mape', 'r2']:
    profile.remove_spec_by_metric(metric)
```

---

## V0 Assumptions & Limitations

These are the known boundaries of the current version. See [FUTURE.md](FUTURE.md) for planned improvements.

1. **Name-based idempotency.** Alerts are matched by name (e.g., `ModelA | Drift | age`). If an alert with the same name is created manually, it will be treated as "existing" and skipped/updated.

2. **No two-way sync.** The utility creates and updates alerts to match the profile, but does not delete alerts that exist in Fiddler but are absent from the profile. Manual cleanup is needed for stale alerts.

3. **Baseline selection is first-found.** For drift alerts, the utility uses the first existing baseline for a model. If multiple baselines exist, it does not pick the "best" one — it picks `baselines[0]`.

4. **Delete+recreate loses alert history.** When immutable fields change (or `overwrite=True`), the existing alert is deleted and a new one is created. This means the alert's triggered history, ID, and any manual tuning are lost.

5. **No alert count caps.** Models with many input columns will get many per-column alerts. The utility warns at 100+ columns but does not cap.

6. **No REST API fallback.** All operations go through the Fiddler Python SDK. Operations not supported by the SDK (e.g., batch create, cross-model queries) are not available.

7. **LLM models get the same profile as ML models.** Task-filtered performance metrics (precision, recall, etc.) are automatically excluded for LLM models, but drift alerts on text features may not be meaningful.

---

## Files in This Directory

| File | Purpose |
|---|---|
| `README.md` | Quick start and reference (this file) |
| `OPERATIONS.md` | Full operations manual — profiles, specs, thresholds, troubleshooting |
| `PRD.md` | Technical PRD and implementation plan |
| `FUTURE.md` | Deferred features and improvement roadmap |
| `temp-customer_feedback.md` | Original Thumbtack requirements (Feb 2026 call) |
| `docs-ref/` | Fiddler documentation references |
