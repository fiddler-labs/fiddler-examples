## Fiddler Utils - Admin Automation Library

Admin automation library providing high-level abstractions for common Fiddler administrative tasks, designed to reduce code duplication across utility scripts and notebooks.

While not part of the official Fiddler SDK, this library is available for both field engineers and customers to use, extend, and contribute to.

### Overview

`fiddler_utils` provides convenience wrappers and utilities for common Fiddler administrative tasks including:

* **Connection management** across multiple Fiddler instances
* **FQL (Fiddler Query Language)** parsing, validation, and manipulation
* **Schema** extraction, validation, and comparison
* **Asset management** (segments, custom metrics, alerts, baselines, charts, dashboards)
* **Model operations** (export/import, comparison, feature impact)
* **Environment analysis** (project/model inventory, statistics, reporting)
* Iteration utilities for safe, fault-tolerant operations

### Installation

From the `fiddler-examples` repository root:

```bash
# Install in development mode (recommended)
pip install -e .

# Or install from a requirements file
# In misc-utils/requirements.txt, add:
# -e ../
```

### Requirements

* Python >= 3.10
* fiddler-client >= 3.10

### Quick Start

#### Connection Management

```python
from fiddler_utils import get_or_init

# Initialize Fiddler client
get_or_init(url='https://acme.cloud.fiddler.ai', token='your_token')

# Subsequent calls are no-ops (client already initialized)
get_or_init()  # Does nothing, uses existing connection
```

#### Working with Multiple Instances

```python
from fiddler_utils import ConnectionManager

# Set up connections to multiple instances
mgr = ConnectionManager()
mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

# Use specific connections
with mgr.use('source'):
    source_model = fdl.Model.from_name('my_model', project_id=proj.id)
    segments = list(fdl.Segment.list(model_id=source_model.id))

with mgr.use('target'):
    target_model = fdl.Model.from_name('my_model', project_id=proj.id)
    # Import segments to target...
```

#### FQL Parsing and Validation

```python
from fiddler_utils import fql

# Extract column references from FQL expression
expr = '"age" > 30 and "geography" == \'California\''
columns = fql.extract_columns(expr)
# Returns: {'age', 'geography'}

# Validate FQL syntax
is_valid, error = fql.validate_fql_syntax(expr)
if not is_valid:
    print(f"Invalid FQL: {error}")

# Replace column names (useful when copying between models)
mapping = {'age': 'customer_age', 'geography': 'region'}
new_expr = fql.replace_column_names(expr, mapping)
# Returns: '"customer_age" > 30 and "region" == \'California\''

# Validate column references against a model
valid_columns = {'age', 'geography', 'status'}
is_valid, missing = fql.validate_column_references(expr, valid_columns)
```

#### Schema Validation and Comparison

```python
from fiddler_utils import SchemaValidator
import fiddler as fdl

# Get model columns
model = fdl.Model.from_name('my_model', project_id=project.id)
columns = SchemaValidator.get_model_columns(model)

for col_name, col_info in columns.items():
    print(f"{col_name}: {col_info.role} ({col_info.data_type})")

# Validate that columns exist in a model
columns_to_check = {'age', 'income', 'geography'}
is_valid, missing = SchemaValidator.validate_columns(
    columns_to_check,
    target_model,
    strict=False  # Just warn, don't raise exception
)

if not is_valid:
    print(f"Missing columns: {missing}")

# Compare schemas between models
comparison = SchemaValidator.compare_schemas(source_model, target_model)

print(f"Columns in both: {len(comparison.in_both)}")
print(f"Only in source: {comparison.only_in_source}")
print(f"Only in target: {comparison.only_in_target}")
print(f"Type mismatches: {comparison.type_mismatches}")
print(f"Compatible: {comparison.is_compatible}")

# Validate FQL expression against model schema
expr = '"age" > 30 and "status" == \'active\''
is_valid, missing = SchemaValidator.validate_fql_expression(
    expr,
    target_model,
    strict=True  # Raise exception if invalid
)
```

### API Reference

#### Connection Utilities

**`get_or_init(url, token, force=False)`**
Initialize Fiddler client if not already initialized. Subsequent calls are no-ops unless `force=True`.

**`reset_connection()`**
Reset connection state. Useful for testing or switching instances.

**`connection_context(url, token)`**
Context manager for temporarily switching connections:
```python
with connection_context(url=OTHER_URL, token=OTHER_TOKEN):
    # Work with different instance
    pass
# Automatically restored to previous connection
```

**`ConnectionManager`**
Manager for working with multiple named connections:
* `add(name, url, token)` - Register a connection
* `use(name)` - Context manager to activate a connection
* `list_connections()` - Get list of registered connection names

#### FQL Utilities

**`fql.extract_columns(expression)`**
Extract column names from FQL expression. Returns `Set[str]`.

**`fql.replace_column_names(expression, column_mapping)`**
Replace column names based on mapping dict. Returns modified expression.

**`fql.validate_fql_syntax(expression)`**
Basic FQL syntax validation. Returns `(is_valid, error_message)`.

**`fql.normalize_expression(expression)`**
Normalize whitespace and formatting for comparison.

**`fql.get_fql_functions(expression)`**
Extract function names used in expression. Returns `Set[str]`.

**`fql.is_simple_filter(expression)`**
Check if expression is a simple filter (no aggregations).

**`fql.split_fql_and_condition(expression)`**
Split expression on top-level 'and' operators. Returns `List[str]`.

**`fql.validate_column_references(expression, valid_columns)`**
Validate all column references exist in `valid_columns` set. Returns `(is_valid, missing_columns)`.

#### Schema Validation

**`SchemaValidator.get_model_columns(model)`**
Extract all columns from model. Returns `Dict[str, ColumnInfo]`.

**`SchemaValidator.get_column_names(model)`**
Get just column names (faster). Returns `Set[str]`.

**`SchemaValidator.validate_columns(columns, model, strict=True)`**
Validate columns exist in model. Returns `(is_valid, missing_columns)`.

**`SchemaValidator.compare_schemas(source_model, target_model, strict=False)`**
Compare two model schemas. Returns `SchemaComparison` object.

**`SchemaValidator.validate_fql_expression(expression, model, strict=True)`**
Validate FQL expression against model schema. Returns `(is_valid, missing_columns)`.

**`SchemaValidator.is_compatible(source_model, target_model, required_columns=None)`**
Check if schemas are compatible for asset transfer. Returns `bool`.

### Data Classes

**`ColumnInfo`**
Information about a model column:
* `name: str` - Column name
* `role: ColumnRole` - INPUT, OUTPUT, TARGET, METADATA, etc.
* `data_type: Optional[str]` - Data type
* `min_value, max_value` - Numeric range (if applicable)
* `categories` - Categorical values (if applicable)

**`SchemaComparison`**
Result of schema comparison:
* `only_in_source: Set[str]` - Columns only in source
* `only_in_target: Set[str]` - Columns only in target
* `in_both: Set[str]` - Common columns
* `type_mismatches: Dict` - Type mismatches for common columns
* `is_compatible: bool` - Whether schemas are compatible

### Exceptions

All exceptions inherit from `FiddlerUtilsError`:

* `ConnectionError` - Connection issues
* `ValidationError` - General validation failures
* `SchemaValidationError` - Schema validation failures
* `FQLError` - FQL parsing/validation errors
* `AssetNotFoundError` - Asset not found
* `AssetImportError` - Asset import failures
* `BulkOperationError` - Bulk operation errors

### Logging

Configure logging for fiddler_utils:

```python
from fiddler_utils import configure_logging

# Enable debug logging for fiddler_utils
configure_logging(level='DEBUG')

# Custom format
configure_logging(
    level='INFO',
    format='%(levelname)s - %(message)s'
)
```

### Asset Management

Working with segments, custom metrics, alerts, baselines, and dashboards:

```python
from fiddler_utils import (
    SegmentManager,
    CustomMetricManager,
    AlertManager,
    BaselineManager,
    ChartManager,
    DashboardManager,
)

# Export segments from source model
segment_mgr = SegmentManager()
exported_segments = segment_mgr.export_assets(model_id=source_model.id)

# Import to target model (with validation)
result = segment_mgr.import_assets(
    target_model_id=target_model.id,
    assets=exported_segments,
    validate=True,
    dry_run=False
)

print(f'Imported: {result.successful}, Failed: {result.failed}')
```

### Model Comparison

Compare models across multiple dimensions:

```python
from fiddler_utils import ModelComparator, ComparisonConfig

# Full comparison
comparator = ModelComparator(model_a, model_b)
result = comparator.compare_all()

# Schema-only comparison (fast)
config = ComparisonConfig.schema_only()
result = comparator.compare_all(config=config)

# Export results
result.to_markdown()
result.to_json('comparison.json')
df = result.to_dataframe()
```

### Environment Analysis

Analyze complete Fiddler environment:

```python
from fiddler_utils import EnvironmentReporter

# Run complete analysis
reporter = EnvironmentReporter()
reporter.analyze_environment(
    include_features=True,
    include_timestamps=True
)

# Generate formatted report
reporter.generate_report(top_n=15)

# Export to CSV
files = reporter.export_to_csv(prefix='env_stats')
```

### Development

#### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest fiddler_utils/tests/ -v

# Run with coverage
pytest fiddler_utils/tests/ --cov=fiddler_utils --cov-report=html

# Run specific test file
pytest fiddler_utils/tests/test_fql.py -v
```

#### Project Structure

```
fiddler_utils/
├── __init__.py              # Public API
├── connection.py            # Connection management
├── fql.py                   # FQL parsing utilities
├── schema.py                # Schema validation
├── comparison.py            # Model comparison
├── projects.py              # Project/environment management
├── reporting.py             # Environment reporting
├── iteration.py             # Safe iteration utilities
├── exceptions.py            # Custom exceptions
├── assets/                  # Asset managers
│   ├── __init__.py
│   ├── base.py             # BaseAssetManager
│   ├── segments.py         # SegmentManager
│   ├── metrics.py          # CustomMetricManager
│   ├── alerts.py           # AlertManager
│   ├── baselines.py        # BaselineManager
│   ├── charts.py           # ChartManager
│   ├── dashboards.py       # DashboardManager
│   ├── models.py           # ModelManager
│   └── feature_impact.py   # FeatureImpactManager
├── tests/                   # Unit tests
│   ├── test_fql.py         # FQL tests
│   ├── test_schema.py      # Schema tests
│   ├── test_comparison.py  # Comparison tests
│   └── test_models.py      # Model tests
└── examples/                # Usage examples
    ├── asset_export_import.py
    ├── chart_operations.py
    ├── dashboard_operations.py
    ├── model_comparison.py
    ├── environment_reporting.py
    ├── fql_utilities.py
    ├── model_export_import.py
    └── baseline_operations.py
```

### Examples

See the [`examples/`](examples/) directory for comprehensive usage examples:

| Example | Description |
|---------|-------------|
| **[asset_export_import.py](examples/asset_export_import.py)** | Export/import segments and custom metrics between models |
| **[chart_operations.py](examples/chart_operations.py)** | Manage charts (list, export, import, analyze) |
| **[dashboard_operations.py](examples/dashboard_operations.py)** | Create and manage dashboards with auto/custom layouts |
| **[model_comparison.py](examples/model_comparison.py)** | Compare models across configuration, schema, spec, and assets |
| **[environment_reporting.py](examples/environment_reporting.py)** | Analyze complete environment hierarchy and generate reports |
| **[fql_utilities.py](examples/fql_utilities.py)** | Parse, validate, and manipulate FQL expressions |
| **[model_export_import.py](examples/model_export_import.py)** | Export/import complete model definitions (schema + spec) |
| **[baseline_operations.py](examples/baseline_operations.py)** | Manage baselines (list, export, import, create) |

Each example is self-contained and demonstrates best practices for using fiddler_utils.

### Roadmap & Status

**✅ Phase 1: Core Infrastructure (COMPLETED)**
* ✅ Package structure and setup
* ✅ Connection management with multi-instance support
* ✅ FQL parsing utilities (extract, validate, replace, normalize)
* ✅ Schema validation and comparison
* ✅ Comprehensive unit tests

**✅ Phase 2: Asset Management (COMPLETED)**
* ✅ BaseAssetManager abstract class
* ✅ SegmentManager - segments export/import
* ✅ CustomMetricManager - custom metrics export/import
* ✅ AlertManager - alert analysis and management
* ✅ BaselineManager - baseline export/import
* ✅ ChartManager - chart operations (unofficial API)
* ✅ DashboardManager - dashboard operations (unofficial API)
* ✅ ModelManager - model export/import with full schema
* ✅ FeatureImpactManager - feature impact management

**✅ Phase 3: Model Operations & Analysis (COMPLETED)**
* ✅ ModelComparator - comprehensive model comparison
* ✅ ComparisonConfig - flexible comparison presets
* ✅ Export comparison results (JSON, Markdown, CSV, DataFrame)
* ✅ Cross-instance comparison support

**✅ Phase 4: Environment Operations (COMPLETED)**
* ✅ ProjectManager - environment hierarchy traversal
* ✅ EnvironmentReporter - high-level reporting facade
* ✅ Environment statistics and analysis
* ✅ Timestamp analysis (creation dates, update dates)
* ✅ CSV export at multiple granularity levels

**Phase 5: Additional Utilities (FUTURE)**
* Iteration utilities enhancement
* Additional reporting formats
* Performance optimizations
* Extended testing coverage

### Contributing

When adding new utilities to `fiddler_utils`:

1. **Add tests first** - Write unit tests before implementation
2. **Follow patterns** - Use existing modules as templates
3. **Document** - Add docstrings and examples
4. **Type hints** - Add type hints to all public APIs
5. **Test coverage** - Aim for 80%+ coverage

### Support

For questions or issues with `fiddler_utils`:
* Check existing utility scripts in `misc-utils/` for usage examples
* Review test files for API usage patterns
* Contact Field Engineering team

### License

Internal use only - NOT for external distribution.

---

**Version:** 0.2.0
**Status:** Production Ready (Phases 1-4 Complete)
**Last Updated:** 2025-10-30

### Summary of Capabilities

**Core Utilities:**
* Connection management (single/multi-instance)
* FQL parsing and validation
* Schema validation and comparison
* Logging configuration

**Asset Managers:**
* Segments, Custom Metrics, Alerts, Baselines
* Charts, Dashboards (unofficial API)
* Models (full export/import)
* Feature Impact

**Analysis Tools:**
* Model comparison (ModelComparator)
* Environment reporting (EnvironmentReporter)
* Project management (ProjectManager)

**8 Comprehensive Examples** covering all major features
