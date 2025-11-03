"""FQL testing and validation utilities.

This module provides utilities to test and validate FQL expressions before
committing them as Custom Metrics. Since Fiddler does not provide a dry-run
API, these utilities use temporary metrics for real validation.

Key Functions:
    - validate_metric_syntax_local(): Fast local syntax validation
    - test_metric_definition(): Test FQL by creating temporary metric
    - validate_and_preview_metric(): Full validation with preview
    - batch_test_metrics(): Test multiple definitions efficiently

Example:
    ```python
    from fiddler_utils.testing import (
        validate_metric_syntax_local,
        test_metric_definition
    )
    import fiddler as fdl

    # Fast local pre-validation
    result = validate_metric_syntax_local(
        definition='sum(if(fp(), 1, 0))',
        model=model
    )

    if result['has_errors']:
        print(f"Errors: {result['errors']}")
    elif result['has_warnings']:
        print(f"Warnings: {result['warnings']}")

    # Real testing in Fiddler (creates & deletes temp metric)
    result = test_metric_definition(
        model_id=model.id,
        definition='sum(if(fp(), 1, 0)) / sum(1)'
    )

    if result['valid']:
        print("✓ Metric definition works!")
    else:
        print(f"✗ Error: {result['error']}")
    ```
"""

from typing import Dict, List, Optional, Any, Set
import logging
import time
import random
import string

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from . import fql
from .exceptions import FQLError

logger = logging.getLogger(__name__)


def _generate_temp_metric_name(prefix: str = "__test_") -> str:
    """Generate a unique temporary metric name.

    Args:
        prefix: Prefix for the temp metric name

    Returns:
        Unique temp metric name like '__test_1704123456_a3f2'
    """
    timestamp = int(time.time())
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}{timestamp}_{random_suffix}"


def validate_metric_syntax_local(
    definition: str,
    model: Optional[fdl.Model] = None
) -> Dict[str, Any]:
    """Perform fast local validation of FQL syntax.

    This does NOT test whether the metric will actually work in Fiddler,
    but catches obvious syntax errors quickly without API calls.

    Checks performed:
    - Quote matching (double and single quotes)
    - Parentheses balance
    - Column references exist in schema (if model provided)
    - Function names are valid FQL functions

    Args:
        definition: FQL metric definition to validate
        model: Optional Model object for schema validation

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,  # True if no errors
            'has_errors': bool,
            'has_warnings': bool,
            'errors': List[str],
            'warnings': List[str],
            'metadata': Dict
        }

    Example:
        ```python
        result = validate_metric_syntax_local(
            definition='sum(if(fp(), "value", 0)',  # Missing closing paren
            model=model
        )

        if result['has_errors']:
            print("Validation failed:")
            for error in result['errors']:
                print(f"  - {error}")
        ```
    """
    errors = []
    warnings = []
    metadata = {}

    # Check 1: Basic syntax validation
    is_valid, error_msg = fql.validate_fql_syntax(definition)
    if not is_valid:
        errors.append(f"Syntax error: {error_msg}")

    # Check 2: Extract and validate columns (if model provided)
    columns = fql.extract_columns(definition)
    metadata['columns'] = list(columns)
    metadata['column_count'] = len(columns)

    if model:
        from .schema import SchemaValidator

        schema_valid, missing_cols = SchemaValidator.validate_fql_expression(
            definition, model, strict=False
        )

        if not schema_valid:
            errors.append(f"Missing columns in schema: {missing_cols}")

    # Check 3: Extract functions
    functions = fql.get_fql_functions(definition)
    metadata['functions'] = list(functions)
    metadata['function_count'] = len(functions)

    # Check 4: Verify it's an aggregation (custom metrics must be aggregated)
    is_simple = fql.is_simple_filter(definition)
    metadata['is_aggregation'] = not is_simple

    if is_simple:
        warnings.append(
            "Expression appears to be a simple filter without aggregation. "
            "Custom metrics must use aggregate functions (sum, avg, count, etc.)"
        )

    # Check 5: Complexity warnings
    if len(columns) > 10:
        warnings.append(
            f"High complexity: {len(columns)} columns referenced. "
            "Consider simplifying."
        )

    if len(functions) > 5:
        warnings.append(
            f"Deeply nested: {len(functions)} function calls. "
            "May be hard to debug if issues arise."
        )

    # Compile results
    result = {
        'valid': len(errors) == 0,
        'has_errors': len(errors) > 0,
        'has_warnings': len(warnings) > 0,
        'errors': errors,
        'warnings': warnings,
        'metadata': metadata,
    }

    if result['valid']:
        logger.debug(f"Local validation passed for expression: {definition[:100]}...")
    else:
        logger.warning(
            f"Local validation failed with {len(errors)} errors: {errors}"
        )

    return result


def test_metric_definition(
    model_id: str,
    definition: str,
    name_prefix: str = "__test_",
    cleanup: bool = True,
    wait_for_calculation: bool = False
) -> Dict[str, Any]:
    """Test an FQL metric definition by creating a temporary metric in Fiddler.

    This is the ONLY way to truly validate FQL since many functions (tp(), fp(),
    jsd(), etc.) can only be evaluated by Fiddler's backend.

    Workflow:
    1. Create metric with temporary name (__test_<timestamp>_<random>)
    2. Fiddler validates the FQL definition
    3. Optionally wait for metric to calculate
    4. Delete temporary metric (if cleanup=True)
    5. Return validation results

    Args:
        model_id: Model UUID to test against
        definition: FQL metric definition to test
        name_prefix: Prefix for temporary metric name (default: '__test_')
        cleanup: If True, delete temp metric after testing (default: True)
        wait_for_calculation: If True, wait for metric to calculate (slower)

    Returns:
        Dictionary with test results:
        {
            'valid': bool,
            'error': str | None,
            'temp_metric_id': str | None,  # For debugging
            'temp_metric_name': str,
            'cleaned_up': bool,
            'calculation_attempted': bool
        }

    Example:
        ```python
        # Test a metric definition
        result = test_metric_definition(
            model_id=model.id,
            definition='sum(if(fp(), 1, 0)) / sum(1)'
        )

        if result['valid']:
            print("✓ Definition is valid!")
            print("  Ready to create the real metric")
        else:
            print(f"✗ Definition failed: {result['error']}")
        ```
    """
    temp_name = _generate_temp_metric_name(name_prefix)

    logger.info(
        f"Testing metric definition by creating temporary metric '{temp_name}'"
    )

    result = {
        'valid': False,
        'error': None,
        'temp_metric_id': None,
        'temp_metric_name': temp_name,
        'cleaned_up': False,
        'calculation_attempted': wait_for_calculation,
    }

    temp_metric = None

    try:
        # Create temporary metric
        temp_metric = fdl.CustomMetric(
            model_id=model_id,
            name=temp_name,
            description='TEMPORARY TEST METRIC - Safe to delete',
            definition=definition,
        )

        temp_metric.create()
        result['temp_metric_id'] = temp_metric.id

        logger.info(f"✓ Temporary metric created successfully (ID: {temp_metric.id})")
        result['valid'] = True

        # Optionally wait for metric to calculate
        if wait_for_calculation:
            logger.info("Waiting for metric calculation...")
            time.sleep(2)  # Give Fiddler time to calculate
            # Note: We don't actually query the metric here, just wait

    except fdl.BadRequest as e:
        # FQL validation error from Fiddler
        error_msg = str(e)
        logger.warning(f"FQL validation failed: {error_msg}")
        result['error'] = error_msg
        result['valid'] = False

    except Exception as e:
        # Other error
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        result['error'] = error_msg
        result['valid'] = False

    finally:
        # Cleanup temporary metric
        if cleanup and temp_metric and result['temp_metric_id']:
            try:
                logger.info(f"Cleaning up temporary metric '{temp_name}'")
                temp_metric.delete()
                result['cleaned_up'] = True
                logger.debug(f"✓ Temporary metric '{temp_name}' deleted")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temporary metric '{temp_name}': {e}. "
                    "You may need to manually delete it."
                )
                result['cleaned_up'] = False

    return result


def validate_and_preview_metric(
    model_id: str,
    definition: str,
    skip_local_validation: bool = False
) -> Dict[str, Any]:
    """Complete validation workflow with both local and Fiddler testing.

    This combines fast local validation with real Fiddler testing for
    comprehensive validation.

    Workflow:
    1. Local syntax validation (fast, catches obvious errors)
    2. If local validation passes, test in Fiddler (slower, real validation)

    Args:
        model_id: Model UUID to validate against
        definition: FQL metric definition
        skip_local_validation: If True, skip local validation and go straight
                               to Fiddler testing

    Returns:
        Dictionary with comprehensive validation results:
        {
            'valid': bool,  # True only if both local and Fiddler tests pass
            'local_validation': Dict | None,
            'fiddler_test': Dict | None,
            'recommendation': str
        }

    Example:
        ```python
        result = validate_and_preview_metric(
            model_id=model.id,
            definition='sum(if(fp(), 1, 0))'
        )

        print(result['recommendation'])

        if result['valid']:
            # Safe to create real metric
            metric = fdl.CustomMetric(
                model_id=model.id,
                name='false_positive_count',
                definition='sum(if(fp(), 1, 0))'
            )
            metric.create()
        ```
    """
    result = {
        'valid': False,
        'local_validation': None,
        'fiddler_test': None,
        'recommendation': '',
    }

    # Step 1: Local validation (if not skipped)
    if not skip_local_validation:
        logger.info("Step 1: Running local validation...")
        model = fdl.Model.get(id_=model_id)
        local_result = validate_metric_syntax_local(definition, model)
        result['local_validation'] = local_result

        if local_result['has_errors']:
            result['recommendation'] = (
                "❌ Local validation failed. Fix syntax errors before testing in Fiddler."
            )
            logger.warning("Local validation failed - skipping Fiddler test")
            return result

        if local_result['has_warnings']:
            logger.info(
                f"Local validation passed with {len(local_result['warnings'])} warnings"
            )
    else:
        logger.info("Skipping local validation")

    # Step 2: Test in Fiddler
    logger.info("Step 2: Testing in Fiddler...")
    fiddler_result = test_metric_definition(
        model_id=model_id,
        definition=definition,
        cleanup=True
    )
    result['fiddler_test'] = fiddler_result

    if fiddler_result['valid']:
        result['valid'] = True
        result['recommendation'] = (
            "✅ Metric definition is valid! Safe to create the real metric."
        )
        logger.info("✓ Complete validation passed")
    else:
        result['recommendation'] = (
            f"❌ Fiddler validation failed: {fiddler_result['error']}"
        )
        logger.warning(f"Fiddler validation failed: {fiddler_result['error']}")

    return result


def batch_test_metrics(
    model_id: str,
    definitions: List[Dict[str, str]],
    delay_between_tests: float = 0.5
) -> List[Dict[str, Any]]:
    """Test multiple metric definitions efficiently.

    Args:
        model_id: Model UUID to test against
        definitions: List of dicts with 'name' and 'definition' keys
        delay_between_tests: Delay in seconds between tests (default: 0.5)

    Returns:
        List of test results, one per definition

    Example:
        ```python
        definitions = [
            {'name': 'FP Count', 'definition': 'sum(if(fp(), 1, 0))'},
            {'name': 'FN Count', 'definition': 'sum(if(fn(), 1, 0))'},
            {'name': 'Accuracy', 'definition': 'sum(if(tp() or tn(), 1, 0)) / sum(1)'},
        ]

        results = batch_test_metrics(model.id, definitions)

        valid_count = sum(1 for r in results if r['valid'])
        print(f"{valid_count}/{len(results)} definitions are valid")

        for r in results:
            status = "✓" if r['valid'] else "✗"
            print(f"{status} {r['name']}: {r.get('error', 'OK')}")
        ```
    """
    logger.info(f"Batch testing {len(definitions)} metric definitions...")

    results = []

    for i, def_dict in enumerate(definitions, 1):
        name = def_dict.get('name', f'Metric {i}')
        definition = def_dict['definition']

        logger.info(f"Testing {i}/{len(definitions)}: {name}")

        result = test_metric_definition(
            model_id=model_id,
            definition=definition,
            cleanup=True
        )

        result['name'] = name
        results.append(result)

        # Delay between tests to avoid rate limiting
        if i < len(definitions) and delay_between_tests > 0:
            time.sleep(delay_between_tests)

    valid_count = sum(1 for r in results if r['valid'])
    logger.info(
        f"Batch testing complete: {valid_count}/{len(definitions)} valid"
    )

    return results


def cleanup_orphaned_test_metrics(model_id: str, prefix: str = "__test_") -> int:
    """Clean up any orphaned temporary test metrics.

    If testing was interrupted or failed, temporary metrics may remain.
    This utility removes them.

    Args:
        model_id: Model UUID to clean
        prefix: Prefix used for temp metrics (default: '__test_')

    Returns:
        Number of metrics deleted

    Example:
        ```python
        deleted = cleanup_orphaned_test_metrics(model.id)
        if deleted > 0:
            print(f"Cleaned up {deleted} orphaned test metrics")
        ```
    """
    logger.info(f"Searching for orphaned test metrics with prefix '{prefix}'...")

    metrics = list(fdl.CustomMetric.list(model_id=model_id))
    orphaned = [m for m in metrics if m.name.startswith(prefix)]

    if not orphaned:
        logger.info("No orphaned test metrics found")
        return 0

    logger.info(f"Found {len(orphaned)} orphaned test metrics - cleaning up...")

    deleted_count = 0
    for metric in orphaned:
        try:
            logger.debug(f"Deleting orphaned metric '{metric.name}' (ID: {metric.id})")
            metric.delete()
            deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete metric '{metric.name}': {e}")

    logger.info(f"Cleanup complete: deleted {deleted_count}/{len(orphaned)} metrics")

    return deleted_count
