"""Example: Working with FQL (Fiddler Query Language) using fql utilities.

This script demonstrates how to use fiddler_utils.fql module to:
* Extract column references from FQL expressions
* Validate FQL syntax (quote matching, parentheses, etc.)
* Replace column names (useful for cross-model asset migration)
* Normalize expressions for comparison
* Detect FQL functions used
* Identify simple filters vs aggregations
* Split complex expressions
* Validate column references against model schemas

FQL is used in segments, custom metrics, and other Fiddler assets.

FQL Syntax Rules:
- Column names: Always in double quotes (e.g., "column_name")
- String values: Always in single quotes (e.g., 'value')
- Numeric values: No quotes (e.g., 42, 3.14)
- Boolean: true, false (lowercase, no quotes)
"""

import fiddler as fdl
from fiddler_utils import fql, get_or_init, SchemaValidator, configure_fiddler_logging

# ============================================================================
# Configuration
# ============================================================================

# Fiddler instance (needed for schema validation examples)
FIDDLER_URL = 'https://your-instance.fiddler.ai'
FIDDLER_TOKEN = 'your_api_token'
PROJECT_NAME = 'my_project'
MODEL_NAME = 'my_model'


# ============================================================================
# Example 1: Extract Column References
# ============================================================================


def extract_columns_example():
    """Extract column names referenced in FQL expressions."""

    print('=' * 70)
    print('EXAMPLE 1: Extract Column References')
    print('=' * 70)

    # Simple filter expression
    expr1 = '"age" > 30 and "geography" == \'California\''
    columns1 = fql.extract_columns(expr1)
    print(f'\nExpression: {expr1}')
    print(f'Columns: {columns1}')
    # Output: {'age', 'geography'}

    # Custom metric with aggregation
    expr2 = 'sum(if(fp(), 1, 0) * "transaction_value")'
    columns2 = fql.extract_columns(expr2)
    print(f'\nExpression: {expr2}')
    print(f'Columns: {columns2}')
    # Output: {'transaction_value'}

    # Complex segment definition
    expr3 = '("age" > 50 or "income" > 100000) and "status" == \'active\' and "region" in [\'US\', \'CA\']'
    columns3 = fql.extract_columns(expr3)
    print(f'\nExpression: {expr3}')
    print(f'Columns: {columns3}')
    # Output: {'age', 'income', 'status', 'region'}

    # LLM enrichment reference
    expr4 = '"pii_detection.has_pii" == true and "sentiment" < 0.3'
    columns4 = fql.extract_columns(expr4)
    print(f'\nExpression: {expr4}')
    print(f'Columns: {columns4}')
    # Output: {'pii_detection.has_pii', 'sentiment'}


# ============================================================================
# Example 2: Validate FQL Syntax
# ============================================================================


def validate_syntax_example():
    """Validate FQL syntax for common errors."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Validate FQL Syntax')
    print('=' * 70)

    # Valid expression
    valid_expr = '"age" > 30 and "status" == \'active\''
    is_valid, error = fql.validate_fql_syntax(valid_expr)
    print(f'\nExpression: {valid_expr}')
    print(f'Valid: {is_valid}')
    print(f'Error: {error}')

    # Invalid: Unbalanced quotes
    invalid_expr1 = '"age > 30'  # Missing closing quote
    is_valid, error = fql.validate_fql_syntax(invalid_expr1)
    print(f'\nExpression: {invalid_expr1}')
    print(f'Valid: {is_valid}')
    print(f'Error: {error}')

    # Invalid: Unbalanced parentheses
    invalid_expr2 = 'sum(if(fp(), 1, 0)'  # Missing closing paren
    is_valid, error = fql.validate_fql_syntax(invalid_expr2)
    print(f'\nExpression: {invalid_expr2}')
    print(f'Valid: {is_valid}')
    print(f'Error: {error}')

    # Invalid: Empty column reference
    invalid_expr3 = '"" > 30'  # Empty column name
    is_valid, error = fql.validate_fql_syntax(invalid_expr3)
    print(f'\nExpression: {invalid_expr3}')
    print(f'Valid: {is_valid}')
    print(f'Error: {error}')

    # Check multiple expressions
    test_expressions = [
        '"age" > 30',
        '"status" == \'active\'',
        'sum("revenue")',
        '("a" > 1',  # Invalid
        '"name == \'test\'',  # Invalid
    ]

    print(f'\n\nBatch validation:')
    for expr in test_expressions:
        is_valid, error = fql.validate_fql_syntax(expr)
        status = '✓' if is_valid else '✗'
        print(f'{status} {expr:40s} - {error or "OK"}')


# ============================================================================
# Example 3: Replace Column Names
# ============================================================================


def replace_column_names_example():
    """Replace column names in FQL expressions (for cross-model migration)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Replace Column Names')
    print('=' * 70)

    # Original expression
    original = '"customer_age" > 30 and "customer_region" == \'US\''
    print(f'\nOriginal: {original}')

    # Define mapping (old name -> new name)
    mapping = {
        'customer_age': 'age',
        'customer_region': 'geography',
    }

    # Apply mapping
    replaced = fql.replace_column_names(original, mapping)
    print(f'Mapping: {mapping}')
    print(f'Result: {replaced}')
    # Output: '"age" > 30 and "geography" == \'US\''

    # Complex example with multiple occurrences
    complex_expr = '"old_col" > 10 and ("old_col" < 50 or "status" == \'active\')'
    print(f'\nOriginal: {complex_expr}')

    complex_mapping = {'old_col': 'new_column'}
    replaced_complex = fql.replace_column_names(complex_expr, complex_mapping)
    print(f'Mapping: {complex_mapping}')
    print(f'Result: {replaced_complex}')
    # Output: '"new_column" > 10 and ("new_column" < 50 or "status" == \'active\')'

    # Use case: Migrating segment from one model to another
    print(f'\n💡 Use Case: Migrating assets between models with different column names')
    segment_def = '"user_age" > 25 and "user_country" == \'USA\''
    model_mapping = {
        'user_age': 'customer_age',
        'user_country': 'country_code',
    }
    migrated_def = fql.replace_column_names(segment_def, model_mapping)
    print(f'Source model segment: {segment_def}')
    print(f'Target model segment: {migrated_def}')


# ============================================================================
# Example 4: Normalize Expressions
# ============================================================================


def normalize_expression_example():
    """Normalize FQL expressions for comparison."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Normalize Expressions')
    print('=' * 70)

    # Same expression with different whitespace
    expr1 = '"age"   >  30    and    "status"== \'active\''
    expr2 = '"age" > 30 and "status" == \'active\''

    norm1 = fql.normalize_expression(expr1)
    norm2 = fql.normalize_expression(expr2)

    print(f'\nExpression 1: {expr1}')
    print(f'Normalized:   {norm1}')
    print(f'\nExpression 2: {expr2}')
    print(f'Normalized:   {norm2}')
    print(f'\nAre they equivalent? {norm1 == norm2}')

    # Useful for comparing segments/metrics from different sources
    segment_a = 'sum( if( fp( ) , 1 , 0 ) )'
    segment_b = 'sum(if(fp(), 1, 0))'

    norm_a = fql.normalize_expression(segment_a)
    norm_b = fql.normalize_expression(segment_b)

    print(f'\n💡 Use Case: Comparing definitions')
    print(f'Segment A: {segment_a}')
    print(f'Segment B: {segment_b}')
    print(f'Normalized match: {norm_a == norm_b}')


# ============================================================================
# Example 5: Detect FQL Functions
# ============================================================================


def detect_functions_example():
    """Extract function names used in FQL expressions."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Detect FQL Functions')
    print('=' * 70)

    # Simple aggregation
    expr1 = 'sum("revenue")'
    funcs1 = fql.get_fql_functions(expr1)
    print(f'\nExpression: {expr1}')
    print(f'Functions: {funcs1}')

    # Nested functions
    expr2 = 'sum(if(fp(), 1, 0) * "transaction_value")'
    funcs2 = fql.get_fql_functions(expr2)
    print(f'\nExpression: {expr2}')
    print(f'Functions: {funcs2}')
    # Output: {'sum', 'if', 'fp'}

    # Complex custom metric
    expr3 = 'avg(if(tn(), "processing_time", 0)) / max("processing_time")'
    funcs3 = fql.get_fql_functions(expr3)
    print(f'\nExpression: {expr3}')
    print(f'Functions: {funcs3}')
    # Output: {'avg', 'if', 'tn', 'max'}

    # Check if specific functions are used
    print(f'\n💡 Check for specific function usage:')
    print(f'Uses fp()? {\'fp\' in funcs2}')
    print(f'Uses aggregation? {bool({\'sum\', \'avg\', \'count\', \'max\', \'min\'} & funcs3)}')


# ============================================================================
# Example 6: Identify Simple Filters vs Aggregations
# ============================================================================


def identify_filter_type_example():
    """Distinguish between simple filters (segments) and aggregations (metrics)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Identify Filter Type')
    print('=' * 70)

    # Simple filter (can be used in segments)
    filter1 = '"age" > 30 and "status" == \'active\''
    is_simple1 = fql.is_simple_filter(filter1)
    print(f'\nExpression: {filter1}')
    print(f'Is simple filter: {is_simple1}')
    print(f'Can be used in: {"Segment" if is_simple1 else "Custom Metric"}')

    # Aggregation (must be custom metric)
    filter2 = 'sum(if(fp(), 1, 0))'
    is_simple2 = fql.is_simple_filter(filter2)
    print(f'\nExpression: {filter2}')
    print(f'Is simple filter: {is_simple2}')
    print(f'Can be used in: {"Segment" if is_simple2 else "Custom Metric"}')

    # Complex filter with logical operators (still simple)
    filter3 = '("age" > 50 or "income" > 100000) and "region" == \'US\''
    is_simple3 = fql.is_simple_filter(filter3)
    print(f'\nExpression: {filter3}')
    print(f'Is simple filter: {is_simple3}')
    print(f'Can be used in: {"Segment" if is_simple3 else "Custom Metric"}')

    # Batch check
    expressions = [
        ('"age" > 30', 'Simple age filter'),
        ('sum("revenue")', 'Revenue sum'),
        ('"status" == \'active\'', 'Status filter'),
        ('avg(if(fp(), 1, 0))', 'False positive rate'),
        ('"price" > 100 and "quantity" < 5', 'Combined filter'),
    ]

    print(f'\n\nBatch classification:')
    for expr, description in expressions:
        is_simple = fql.is_simple_filter(expr)
        asset_type = 'Segment' if is_simple else 'Custom Metric'
        print(f'{asset_type:15s} - {description:30s} | {expr}')


# ============================================================================
# Example 7: Split Complex Expressions
# ============================================================================


def split_expression_example():
    """Split complex AND-conditions into parts."""

    print('\n' + '=' * 70)
    print('EXAMPLE 7: Split Complex Expressions')
    print('=' * 70)

    # Complex segment with multiple conditions
    complex_segment = '"age" > 30 and "status" == \'active\' and "region" == \'US\''
    parts = fql.split_fql_and_condition(complex_segment)

    print(f'\nOriginal: {complex_segment}')
    print(f'Split into {len(parts)} parts:')
    for i, part in enumerate(parts, 1):
        print(f'  {i}. {part}')

    # Very complex expression
    very_complex = '("age" > 50 or "income" > 100000) and "status" == \'active\' and "region" in [\'US\', \'CA\'] and "verified" == true'
    parts2 = fql.split_fql_and_condition(very_complex)

    print(f'\nOriginal: {very_complex}')
    print(f'Split into {len(parts2)} parts:')
    for i, part in enumerate(parts2, 1):
        print(f'  {i}. {part}')

    # Use case: Break down complex segment for documentation
    print(f'\n💡 Use Case: Document complex segment logic')
    for i, part in enumerate(parts, 1):
        columns = fql.extract_columns(part)
        print(f'Condition {i}: {part}')
        print(f'  Columns: {columns}')


# ============================================================================
# Example 8: Validate Column References Against Model Schema
# ============================================================================


def validate_against_schema_example():
    """Validate that FQL expressions reference valid model columns."""

    print('\n' + '=' * 70)
    print('EXAMPLE 8: Validate Against Model Schema')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    # Get model
    project = fdl.Project.from_name(PROJECT_NAME)
    model = fdl.Model.from_name(MODEL_NAME, project_id=project.id)

    # Get model columns for reference
    model_columns = SchemaValidator.get_column_names(model)
    print(f'\nModel: {model.name}')
    print(f'Available columns: {len(model_columns)}')
    print(f'Sample columns: {list(model_columns)[:5]}')

    # Test expressions
    expr1 = '"age" > 30 and "status" == \'active\''  # Adjust column names as needed
    expr2 = '"age" > 30 and "nonexistent_column" == 1'  # Has invalid column

    # Validate first expression
    print(f'\n--- Expression 1 ---')
    print(f'Expression: {expr1}')
    is_valid1, missing1 = fql.validate_column_references(expr1, model_columns)
    if is_valid1:
        print('✓ All columns are valid')
    else:
        print(f'✗ Missing columns: {missing1}')

    # Validate second expression
    print(f'\n--- Expression 2 ---')
    print(f'Expression: {expr2}')
    is_valid2, missing2 = fql.validate_column_references(expr2, model_columns)
    if is_valid2:
        print('✓ All columns are valid')
    else:
        print(f'✗ Missing columns: {missing2}')

    # Alternative: Use SchemaValidator directly
    print(f'\n--- Using SchemaValidator ---')
    try:
        is_valid3, missing3 = SchemaValidator.validate_fql_expression(
            expr2, model, strict=False
        )
        print(f'Valid: {is_valid3}')
        print(f'Missing: {missing3}')
    except Exception as e:
        print(f'Validation error: {e}')


# ============================================================================
# Example 9: Complete Workflow - Migrate Segment
# ============================================================================


def complete_workflow_example():
    """Complete workflow: Extract, validate, transform, and re-validate FQL."""

    print('\n' + '=' * 70)
    print('EXAMPLE 9: Complete Workflow - Migrate Segment')
    print('=' * 70)

    # Source segment definition
    source_segment = '"customer_age" > 30 and "customer_region" == \'California\''

    print(f'Source segment definition:')
    print(f'  {source_segment}')

    # Step 1: Validate syntax
    print(f'\n[Step 1] Validate syntax...')
    is_valid, error = fql.validate_fql_syntax(source_segment)
    if not is_valid:
        print(f'  ✗ Invalid syntax: {error}')
        return
    print(f'  ✓ Syntax valid')

    # Step 2: Extract columns
    print(f'\n[Step 2] Extract columns...')
    columns = fql.extract_columns(source_segment)
    print(f'  Columns: {columns}')

    # Step 3: Check if it's a simple filter
    print(f'\n[Step 3] Check filter type...')
    is_simple = fql.is_simple_filter(source_segment)
    if not is_simple:
        print(f'  ⚠️ This is an aggregation - must be a custom metric')
    else:
        print(f'  ✓ Simple filter - can be used as segment')

    # Step 4: Map columns to target model
    print(f'\n[Step 4] Map columns to target model...')
    column_mapping = {
        'customer_age': 'age',
        'customer_region': 'geography',
    }
    print(f'  Mapping: {column_mapping}')

    target_segment = fql.replace_column_names(source_segment, column_mapping)
    print(f'  Target segment: {target_segment}')

    # Step 5: Validate against target model columns (simulated)
    print(f'\n[Step 5] Validate against target model...')
    target_columns = {'age', 'geography', 'income', 'status'}  # Simulated
    target_segment_columns = fql.extract_columns(target_segment)
    is_valid_target, missing = fql.validate_column_references(
        target_segment, target_columns
    )

    if is_valid_target:
        print(f'  ✓ All columns exist in target model')
        print(f'  ✅ Ready to import!')
    else:
        print(f'  ✗ Missing columns in target: {missing}')
        print(f'  ❌ Cannot import - fix schema first')

    # Summary
    print(f'\n--- Migration Summary ---')
    print(f'Source: {source_segment}')
    print(f'Target: {target_segment}')
    print(f'Status: {"✅ Ready" if is_valid_target else "❌ Blocked"}')


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all FQL utility examples."""

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 24 + 'FQL UTILITIES EXAMPLES' + ' ' * 22 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: Extract columns
    extract_columns_example()

    # Example 2: Validate syntax
    validate_syntax_example()

    # Example 3: Replace column names
    replace_column_names_example()

    # Example 4: Normalize expressions
    normalize_expression_example()

    # Example 5: Detect functions
    detect_functions_example()

    # Example 6: Identify filter type
    identify_filter_type_example()

    # Example 7: Split expressions
    split_expression_example()

    # Example 8: Validate against schema (requires connection)
    # validate_against_schema_example()

    # Example 9: Complete workflow
    # complete_workflow_example()

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* FQL column names must be in double quotes: "column_name"')
    print('* FQL string values must be in single quotes: \'value\'')
    print('* Use extract_columns() to find all column references')
    print('* Use validate_fql_syntax() to catch basic syntax errors')
    print('* Use replace_column_names() for cross-model asset migration')
    print('* Use is_simple_filter() to determine if expression is a segment or metric')
    print('* Always validate against model schema before importing assets')
    print('=' * 70)


if __name__ == '__main__':
    main()
