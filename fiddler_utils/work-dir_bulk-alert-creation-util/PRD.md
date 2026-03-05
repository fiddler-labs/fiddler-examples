# Bulk Alert Creation Utility — Technical PRD & Implementation Plan

## Context

**Problem:** Fiddler customers (starting with Thumbtack) have zero alerts despite having models onboarded. Alert creation is entirely manual, model-by-model, requiring data scientist availability that doesn't exist. Zero alerts = zero passive monitoring = Fiddler is invisible in day-to-day operations.

**Solution:** A customer-agnostic, profile-driven bulk alert creation utility that auto-creates monitoring alerts with sensible defaults across all models in a Fiddler environment. Built as an extension to the existing `fiddler_utils` package, following its established patterns.

**Immediate deliverable:** V0 Python utility for Thumbtack (~Mar 4, 2026 deadline).
**Long-term value:** Reusable module for all customers, supporting ML and LLM model types.

**This document serves as both the PRD and the implementation plan.**

---

## 1. Module Architecture

### Files to Create/Modify

```
fiddler_utils/
├── alert_profiles.py          # NEW — AlertProfile, AlertSpec, ThresholdConfig, defaults
├── bulk_alerts.py             # NEW — BulkAlertCreator orchestrator
├── assets/
│   └── alerts.py              # MODIFY — implement _create_asset() (currently NotImplementedError)
├── exceptions.py              # MODIFY — add BulkAlertCreationError
├── __init__.py                # MODIFY — export new public API
├── examples/
│   └── bulk_alert_creation.py # NEW — example usage script
└── tests/
    ├── test_alert_profiles.py # NEW — unit tests (no connection required)
    └── test_bulk_alerts.py    # NEW — unit tests with mocks
```

### Why This Structure

- `alert_profiles.py` is a **data definition module** (like `schema.py`) — declarative specs, no SDK calls
- `bulk_alerts.py` is an **orchestrator** (like `reporting.py`) — composes `AlertManager`, `BaselineManager`, `iterate_models_safe()`, and `SchemaValidator`
- NOT a `BaseAssetManager` subclass — bulk creation is cross-model orchestration, not single-model CRUD

---

## 2. Core Data Structures (`alert_profiles.py`)

### Design Principle: Reuse SDK Types Directly

The Fiddler SDK already provides strongly-typed enums with built-in validation via Pydantic v1.
We import and use them directly instead of creating parallel string constants:

```python
# SDK types we reuse (from fiddler.constants)
from fiddler.constants.alert_rule import (
    AlertCondition,        # GREATER='greater', LESSER='lesser'
    AlertThresholdAlgo,    # MANUAL='manual', STD_DEV_AUTO_THRESHOLD='standard_deviation_auto_threshold'
    BinSize,               # HOUR='Hour', DAY='Day', WEEK='Week', MONTH='Month'
    CompareTo,             # RAW_VALUE='raw_value', TIME_PERIOD='time_period'
    Priority,              # HIGH='HIGH', MEDIUM='MEDIUM', LOW='LOW'
)
from fiddler.constants.model import ModelTask      # BINARY_CLASSIFICATION, REGRESSION, LLM, etc.
from fiddler.constants.baseline import WindowBinSize  # HOUR, DAY, WEEK, MONTH
from fiddler.constants.dataset import EnvType         # For baseline environment
```

This means **zero custom enum duplication** — the SDK validates types at construction time.

### ThresholdConfig — Threshold Strategy

```python
class ThresholdStrategy(str, Enum):
    """Our only custom enum — the SDK doesn't have this concept."""
    ABSOLUTE = 'absolute'    # Fixed values (e.g., JSD > 0.15)
    SIGMA = 'sigma'          # Standard deviation auto-threshold (Fiddler's STD_DEV_AUTO_THRESHOLD)

@dataclass
class ThresholdConfig:
    strategy: ThresholdStrategy = ThresholdStrategy.SIGMA
    warning_value: float = 2.0   # Sigma multiplier OR absolute value
    critical_value: float = 3.0  # Sigma multiplier OR absolute value
```

**Critical SDK mapping for SIGMA strategy** (discovered from SDK tests):
```python
# When strategy == SIGMA, we translate to:
threshold_type = AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD
auto_threshold_params = {
    'warning_multiplier': threshold.warning_value,   # e.g., 2.0 = 2σ
    'critical_multiplier': threshold.critical_value,  # e.g., 3.0 = 3σ
}
warning_threshold = None   # Must be None — platform auto-calculates
critical_threshold = None  # Must be None — platform auto-calculates

# When strategy == ABSOLUTE, we translate to:
threshold_type = AlertThresholdAlgo.MANUAL
warning_threshold = threshold.warning_value
critical_threshold = threshold.critical_value
auto_threshold_params = None
```

### AlertSpec — Atomic Alert Definition

```python
@dataclass
class AlertSpec:
    name_template: str              # e.g., '{model_name} | Drift | {column_name}'
    category: str                   # 'data_drift', 'data_integrity', 'traffic', 'performance', 'custom_metric'
    metric_id: str                  # e.g., 'jsd', 'traffic', 'precision', 'range_violation_count'
    bin_size: BinSize = BinSize.DAY           # SDK enum directly
    compare_to: CompareTo = CompareTo.RAW_VALUE
    condition: AlertCondition = AlertCondition.GREATER
    threshold: ThresholdConfig = ThresholdConfig()
    priority: Priority = Priority.HIGH
    columns_source: str = 'none'    # 'inputs', 'outputs', 'all', 'none', or explicit list
    requires_baseline: bool = False
    requires_columns: bool = False
    compare_bin_delta: Optional[int] = None
    enabled: bool = True
    applicable_task_types: Set[ModelTask] = set()  # empty = all; uses SDK ModelTask enum
```

### AlertProfile — Collection of Specs

**Decision: Two-level sigma configuration.** Profile-level defaults apply to all SIGMA-strategy specs. Individual specs can override.

```python
@dataclass
class AlertProfile:
    name: str
    description: str = ''
    specs: List[AlertSpec] = []
    notification: NotificationConfig = NotificationConfig()
    auto_create_baseline: bool = True
    baseline_config: Dict[str, Any] = {  # For auto-created rolling baselines
        'name_template': '__auto_rolling_{model_name}',
        'type': 'ROLLING',                    # BaselineType.ROLLING
        'window_bin_size': WindowBinSize.DAY,  # SDK enum
        'offset_delta': 7,                     # 7-day rolling window
    }
    default_sigma_warning: float = 2.0   # Profile-level sigma default
    default_sigma_critical: float = 3.0  # Profile-level sigma default

    def filter_for_task_type(self, task_type: ModelTask) -> List[AlertSpec]: ...
    def add_spec(self, spec) -> AlertProfile: ...
    def remove_spec_by_metric(self, metric_id) -> AlertProfile: ...
    def merge(self, other: AlertProfile) -> AlertProfile: ...
```

**Sigma resolution logic:** When resolving thresholds for a SIGMA-strategy spec:
- If spec has explicit `threshold.warning_value` / `threshold.critical_value` → use spec values
- If spec threshold values match the default (2.0/3.0) and profile has different defaults → inherit from profile
- This lets customers do `profile.default_sigma_warning = 2.5` to change all sigma alerts at once

### NotificationConfig (ours, wraps SDK's)

```python
@dataclass
class NotificationConfig:
    emails: List[str] = []
    pagerduty_services: List[str] = []
    pagerduty_severity: str = 'warning'
    webhooks: List[UUID] = []       # Note: SDK expects List[UUID], not List[str]
```

---

## 3. Default Alert Profiles

### `get_default_ml_profile()` — Standard ML Monitoring

| Category | metric_id | Per-Column? | Baseline? | Condition | Threshold Strategy | Task Filter |
|---|---|---|---|---|---|---|
| Traffic | `traffic` | No | No | `LESSER` | SIGMA (2σ/3σ) | All |
| Data Drift | `jsd` | Yes (inputs) | Yes | `GREATER` | ABSOLUTE (0.15/0.3) | All |
| Data Integrity | `null_violation_percentage` | Yes (inputs) | No | `GREATER` | ABSOLUTE (5.0/10.0) | All |
| Data Integrity | `range_violation_count` | Yes (inputs) | No | `GREATER` | ABSOLUTE (10/50) | All |
| Performance | `precision` | No | No | `LESSER` | SIGMA (2σ/3σ) | `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION` |
| Performance | `recall` | No | No | `LESSER` | SIGMA (2σ/3σ) | `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION` |
| Performance | `f1_score` | No | No | `LESSER` | SIGMA (2σ/3σ) | `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION` |
| Performance | `mae` | No | No | `GREATER` | SIGMA (2σ/3σ) | `REGRESSION` |
| Performance | `mape` | No | No | `GREATER` | SIGMA (2σ/3σ) | `REGRESSION` |
| Performance | `r2` | No | No | `LESSER` | SIGMA (2σ/3σ) | `REGRESSION` |

**Note on SIGMA vs ABSOLUTE:**
- SIGMA specs → `threshold_type=AlertThresholdAlgo.STD_DEV_AUTO_THRESHOLD`, thresholds set to `None`, platform computes from data
- ABSOLUTE specs → `threshold_type=AlertThresholdAlgo.MANUAL`, thresholds set directly

### `get_traffic_only_profile()` — Minimal

Traffic alerts only. Good for quick validation.

### LLM Models — Full ML Profile Applied

**Decision: Apply full ML alert profile to LLM models too.** Some metrics (drift on input features, data quality, traffic) are meaningful for LLM models. Performance metrics that don't apply will be gracefully skipped via task-type filtering or will simply not trigger if the model doesn't have the relevant metric. No separate LLM profile needed for V0 — the standard ML profile covers both.

---

## 4. BulkAlertCreator Orchestrator (`bulk_alerts.py`)

### Constructor

```python
class BulkAlertCreator:
    def __init__(
        self,
        profile: AlertProfile,
        scope: Optional[ModelScopeFilter] = None,  # Filter which models to process
        overwrite: bool = False,                    # Delete+recreate existing alerts?
        skip_invalid: bool = True,                  # Skip or raise on validation failures
        on_error: str = 'warn',                     # 'warn', 'skip', 'raise'
    ): ...
```

### ModelScopeFilter

```python
@dataclass
class ModelScopeFilter:
    project_ids: Optional[List[str]] = None
    project_names: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    model_name_pattern: Optional[str] = None    # Regex
    task_types: Optional[List[str]] = None
    exclude_model_names: List[str] = []
    exclude_model_ids: List[str] = []
    exclude_projects: List[str] = []
    max_models: Optional[int] = None            # For testing
```

### Main Methods

```python
def run(self, dry_run: bool = False) -> BulkAlertResult: ...
def print_report(self, result: BulkAlertResult, show_per_model: bool = False) -> None: ...
def export_report_csv(self, result: BulkAlertResult, output_path: str = '...') -> str: ...
```

### Process Flow

```
run(dry_run)
  │
  ├─ [1] Resolve project filter (names → IDs)
  │
  ├─ [2] iterate_models_safe(project_ids, fetch_full=True, on_error)
  │       │
  │       ├─ Error? → log, skip model, continue
  │       │
  │       ├─ [3] _should_process_model() → apply scope filter
  │       │       └─ No match? → skip, continue
  │       │
  │       ├─ [4] _resolve_model_task_type(model)
  │       │
  │       ├─ [5] _get_existing_alert_names(model.id) → Set[str] for idempotency
  │       │
  │       ├─ [6] _ensure_baseline(model, dry_run)
  │       │       └─ Auto-create rolling baseline if needed for drift alerts
  │       │
  │       ├─ [7] _resolve_specs_for_model(model, task_type)
  │       │       ├─ Filter specs by task_type
  │       │       ├─ Expand per-column specs → one alert per input column
  │       │       ├─ Render name templates: {model_name}, {column_name}
  │       │       └─ Inject baseline_id for drift alerts
  │       │
  │       └─ [8] For each resolved alert:
  │               ├─ Name exists & !overwrite → skip (idempotent)
  │               ├─ Name exists & overwrite → delete existing, create new
  │               ├─ dry_run → log "Would create", count it
  │               └─ Create fdl.AlertRule → configure notifications → record result
  │
  └─ [9] Return BulkAlertResult
```

### Idempotency

- Alert names are **deterministic** from templates (e.g., `ModelA | Drift | age`)
- Re-runs with `overwrite=False` (default) safely skip existing alerts
- Re-runs with `overwrite=True` delete+recreate (needed because `bin_size`, `metric_id`, `compare_to` are immutable)

---

## 5. Prerequisite: Implement `AlertManager._create_asset()`

**File:** `fiddler_utils/assets/alerts.py`
**Current state:** Raises `NotImplementedError` with comment about metric_id mapping
**Fix:** For bulk creation (not cross-model import), metric_id is known at creation time — no mapping needed

**Key insight:** The SDK's `AlertRule.__init__` already accepts SDK enums directly (e.g., `BinSize.DAY`, `Priority.HIGH`). No string-to-enum conversion needed if we pass enums.

```python
def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> fdl.AlertRule:
    """Create an alert rule from asset data dict.

    Supports both MANUAL and STD_DEV_AUTO_THRESHOLD threshold types.
    When threshold_type is STD_DEV_AUTO_THRESHOLD, warning_threshold and
    critical_threshold should be None, and auto_threshold_params should contain
    {'warning_multiplier': float, 'critical_multiplier': float}.
    """
    kwargs = {
        'name': asset_data['name'],
        'model_id': model_id,
        'metric_id': asset_data['metric_id'],
        'bin_size': asset_data['bin_size'],           # BinSize enum or string
        'compare_to': asset_data['compare_to'],       # CompareTo enum or string
        'condition': asset_data['condition'],          # AlertCondition enum or string
        'priority': asset_data['priority'],            # Priority enum or string
        'threshold_type': asset_data.get('threshold_type', AlertThresholdAlgo.MANUAL),
        'warning_threshold': asset_data.get('warning_threshold'),
        'critical_threshold': asset_data.get('critical_threshold'),
        'auto_threshold_params': asset_data.get('auto_threshold_params'),
    }
    # Optional fields
    for field in ('columns', 'baseline_id', 'segment_id', 'compare_bin_delta',
                  'evaluation_delay', 'category'):
        if asset_data.get(field) is not None:
            kwargs[field] = asset_data[field]

    alert = fdl.AlertRule(**kwargs)
    alert.create()
    return alert
```

**SDK constructor signature for reference** (from `.venv/.../fiddler/entities/alert_rule.py`):
```python
AlertRule(
    name: str,
    model_id: UUID | str,
    metric_id: str | UUID,
    priority: Priority | str,
    compare_to: CompareTo | str,
    condition: AlertCondition | str,
    bin_size: BinSize | str,
    threshold_type: AlertThresholdAlgo | str = AlertThresholdAlgo.MANUAL,
    auto_threshold_params: dict[str, Any] | None = None,
    critical_threshold: float | None = None,
    warning_threshold: float | None = None,
    columns: list[str] | None = None,
    baseline_id: UUID | str | None = None,
    segment_id: UUID | str | None = None,
    compare_bin_delta: int | None = None,
    evaluation_delay: int = 0,
    category: str | None = None,
)
```

**Baseline creation for auto-baseline** (from `.venv/.../fiddler/entities/baseline.py`):
```python
Baseline(
    name: str,
    model_id: UUID | str,
    environment: EnvType,         # e.g., EnvType.PRODUCTION
    type_: str,                   # 'ROLLING' or 'STATIC'
    offset_delta: int | None,     # e.g., 7 for 7-day window
    window_bin_size: WindowBinSize | str | None,  # e.g., WindowBinSize.DAY
).create()
```

---

## 6. Error Handling

| Level | Behavior |
|---|---|
| Connection | Raises `ConnectionError`, aborts entirely |
| Project/Model iteration | Handled by `iterate_models_safe(on_error)` — log and continue |
| Baseline auto-creation | If fails, drift alerts skipped for that model; other alerts proceed |
| Alert creation | try/except per alert; failure recorded, continues to next |
| Notification config | If fails, alert still created; warning logged |

Add `BulkAlertCreationError(BulkOperationError)` to `exceptions.py` — raised only when `on_error='raise'`.

---

## 7. Public API (User-Facing)

### Minimal Usage (Thumbtack V0)

```python
from fiddler_utils import get_or_init, BulkAlertCreator
from fiddler_utils.alert_profiles import get_default_ml_profile, NotificationConfig

get_or_init(url=URL, token=TOKEN)

profile = get_default_ml_profile()
profile.notification = NotificationConfig(emails=['alerts@thumbtack.com'])

creator = BulkAlertCreator(profile=profile)
result = creator.run(dry_run=True)    # Preview first
creator.print_report(result)
result = creator.run(dry_run=False)   # Execute
creator.print_report(result)
```

### Scoped + Custom

```python
from fiddler_utils.bulk_alerts import ModelScopeFilter

scope = ModelScopeFilter(
    project_names=['production_models'],
    exclude_model_names=['deprecated_v1'],
    task_types=['binary_classification', 'regression'],
)

creator = BulkAlertCreator(profile=profile, scope=scope, overwrite=False)
result = creator.run(dry_run=False)
```

---

## 8. LLM Support

**V0:** LLM models receive the full ML alert profile. Metrics that don't apply (e.g., classification-only performance metrics on an LLM with no targets) are filtered out by `applicable_task_types`. LLM models still benefit from traffic, drift, and data quality alerts on their input features.

**V1+ expansion:** LLM-specific enrichment metrics (toxicity, faithfulness, PII detection, safety scores) can be added as new `AlertSpec` entries using custom metric IDs. The architecture supports this without structural changes — just new specs with `applicable_task_types={ModelTaskType.LLM.value}`.

---

## 9. Edge Cases

| Scenario | Handling |
|---|---|
| Model has 0 input columns | Per-column alerts produce empty list; model gets fewer alerts |
| No baseline & `auto_create_baseline=False` | Drift alerts skipped with warning |
| Model task is `NOT_SET` | Only non-task-filtered specs apply (traffic, integrity) |
| Model task is `LLM` | Full ML profile applied; task-filtered specs (classification/regression-only) skipped |
| Alert name > 255 chars | Truncate with `...` suffix |
| 100+ input columns | Each gets alerts; warning logged; user can filter via explicit column list |
| `dry_run=True` with `overwrite=True` | Reports what would be deleted/recreated without acting |

---

## 10. SDK Type Reference (from `.venv/.../fiddler/`)

### Enums We Import and Use Directly (no duplication)

| SDK Type | Values | File |
|---|---|---|
| `BinSize` | `HOUR='Hour'`, `DAY='Day'`, `WEEK='Week'`, `MONTH='Month'` | `constants/alert_rule.py` |
| `CompareTo` | `RAW_VALUE='raw_value'`, `TIME_PERIOD='time_period'` | `constants/alert_rule.py` |
| `AlertCondition` | `GREATER='greater'`, `LESSER='lesser'` | `constants/alert_rule.py` |
| `Priority` | `HIGH='HIGH'`, `MEDIUM='MEDIUM'`, `LOW='LOW'` | `constants/alert_rule.py` |
| `AlertThresholdAlgo` | `MANUAL='manual'`, `STD_DEV_AUTO_THRESHOLD='standard_deviation_auto_threshold'` | `constants/alert_rule.py` |
| `ModelTask` | `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `REGRESSION`, `RANKING`, `LLM`, `NOT_SET` | `constants/model.py` |
| `WindowBinSize` | `HOUR='Hour'`, `DAY='Day'`, `WEEK='Week'`, `MONTH='Month'` | `constants/baseline.py` |
| `BaselineType` | `STATIC='STATIC'`, `ROLLING='ROLLING'` | `constants/baseline.py` |
| `DataType` | `FLOAT`, `INTEGER`, `BOOLEAN`, `STRING`, `CATEGORY`, `TIMESTAMP`, `VECTOR` | `constants/model.py` |

### SDK Helper Methods on ModelTask

```python
ModelTask.is_classification()  # True for BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION
ModelTask.is_regression()      # True for REGRESSION
```

### Update-able Alert Fields (from SDK `update()` method)

Only these can be PATCHed: `critical_threshold`, `warning_threshold`, `evaluation_delay`, `auto_threshold_params`

### NotificationConfig (SDK Pydantic model)

```python
# fiddler.schemas.alert_rule.NotificationConfig
class NotificationConfig(BaseModel):
    emails: Optional[List[str]]
    pagerduty_services: Optional[List[str]]
    pagerduty_severity: Optional[str]
    webhooks: Optional[List[UUID]]   # UUIDs, not strings
```

---

## 11. Key Files to Reference During Implementation

### fiddler_utils source files

| File | Purpose |
|---|---|
| `fiddler_utils/assets/alerts.py` | Extend with `_create_asset()` |
| `fiddler_utils/assets/base.py` | `BaseAssetManager` patterns to follow |
| `fiddler_utils/assets/baselines.py` | `BaselineManager` for auto-baseline creation |
| `fiddler_utils/iteration.py` | `iterate_models_safe()` for fault-tolerant iteration |
| `fiddler_utils/schema.py` | `SchemaValidator` for column validation |
| `fiddler_utils/exceptions.py` | Exception hierarchy to extend |
| `fiddler_utils/__init__.py` | Public API exports to update |

### SDK source files (`.venv/.../fiddler/`)

| File | Purpose |
|---|---|
| `entities/alert_rule.py` | `AlertRule` class — constructor, create/update/delete/list |
| `constants/alert_rule.py` | `BinSize`, `CompareTo`, `AlertCondition`, `Priority`, `AlertThresholdAlgo` enums |
| `schemas/alert_rule.py` | `AlertRuleResp`, `NotificationConfig` Pydantic schemas |
| `entities/baseline.py` | `Baseline` class — constructor, create for auto-baseline |
| `constants/baseline.py` | `WindowBinSize`, `BaselineType` enums |
| `constants/model.py` | `ModelTask`, `DataType` enums |
| `tests/apis/test_alert_rule.py` | Test patterns showing exact SDK usage including auto-thresholds |

### Example notebooks

| File | Purpose |
|---|---|
| `misc-utils/replace_alerts_with_mods.py` | Existing bulk alert pattern (raw SDK) |
| `misc-utils/alerts_gen_testing.ipynb` | Alert type examples and SDK signatures |
| `misc-utils/alert_bulk_modification.ipynb` | Bulk modification patterns |
| `quickstart/latest/Fiddler_Quickstart_Simple_Monitoring.ipynb` | Alert creation examples |

### Reference docs (in work-dir)

| File | Purpose |
|---|---|
| `docs-ref/alert-rule.md` | AlertRule API reference — parameters, methods, examples |
| `docs-ref/alert-record.md` | AlertRecord API — triggered alert history, analysis patterns |
| `docs-ref/alerts-platform.md` | Platform docs — supported metrics, comparison types, notifications |

---

## 12. Implementation Steps

1. **Write PRD.md** — Save finalized PRD to `fiddler_utils/work-dir_bulk-alert-creation-util/PRD.md`
2. **Implement `AlertManager._create_asset()`** in `assets/alerts.py`
3. **Create `alert_profiles.py`** — all dataclasses + default profile factories + two-level sigma resolution
4. **Create `bulk_alerts.py`** — `BulkAlertCreator`, `ModelScopeFilter`, `BulkAlertResult`
5. **Add `BulkAlertCreationError`** to `exceptions.py`
6. **Update `__init__.py`** — export `BulkAlertCreator`, profile factories
7. **Create `examples/bulk_alert_creation.py`** — example script that Thumbtack can copy and customize
8. **Create `tests/test_alert_profiles.py`** — unit tests (no connection)
9. **Create `tests/test_bulk_alerts.py`** — unit tests with mocks

---

## 13. Verification

1. **Unit tests pass:** `pytest fiddler_utils/tests/test_alert_profiles.py test_bulk_alerts.py`
2. **Import works:** `from fiddler_utils import BulkAlertCreator` succeeds
3. **Dry run on real env:** Connect to a Fiddler instance, run with `dry_run=True`, verify report output
4. **Create on test model:** Run on a single test model with `max_models=1`, verify alerts created in Fiddler UI
5. **Idempotency:** Re-run same profile → all alerts skipped (existing)
6. **Overwrite:** Re-run with `overwrite=True` → alerts deleted and recreated

---

## 14. Decisions Made

| Question | Decision |
|---|---|
| Sigma configuration | **Both levels** — profile-level `default_sigma_warning`/`default_sigma_critical` + per-spec override |
| LLM models in V0 | **Full ML profile applied** to LLM models (task-filtered specs auto-excluded) |
| Deliverable format | **Module + example script** — reusable classes + Thumbtack-ready example |
| PRD location | **Written to `work-dir_bulk-alert-creation-util/PRD.md`** as persistent reference |

### Remaining Open (for Thumbtack specifically)
- Notification channel preference (email/Slack/webhook) — can be configured at runtime
- Baseline strategy — defaulting to auto-create rolling 1-week; Thumbtack can override
