"""Fiddler Query Language (FQL) parsing and manipulation utilities.

This module provides utilities for working with FQL expressions used in
segments, custom metrics, and other Fiddler assets.

FQL Syntax Rules:
- Column names: Always in double quotes (e.g., "column_name")
- String values: Always in single quotes (e.g., 'value')
- Numeric values: No quotes (e.g., 42, 3.14)
"""

import re
from typing import Set, Dict, Optional, List
import logging

from .exceptions import FQLError

logger = logging.getLogger(__name__)


def extract_columns(expression: str) -> Set[str]:
    """Extract column references from an FQL expression.

    Args:
        expression: FQL expression string

    Returns:
        Set of column names referenced in the expression

    Example:
        >>> extract_columns('"age" > 30 and "geography" == \'California\'')
        {'age', 'geography'}

        >>> extract_columns('sum(if(fp(), 1, 0) * "transaction_value")')
        {'transaction_value'}
    """
    if not expression:
        return set()

    # Extract all double-quoted identifiers (column names in FQL)
    column_pattern = r'"([^"]+)"'
    columns = set(re.findall(column_pattern, expression))

    logger.debug(f'Extracted {len(columns)} columns from expression: {columns}')
    return columns


def replace_column_names(expression: str, column_mapping: Dict[str, str]) -> str:
    """Replace column names in an FQL expression based on a mapping.

    This is useful when copying assets between models with different
    column names.

    Args:
        expression: FQL expression string
        column_mapping: Dict mapping old column names to new column names

    Returns:
        Expression with column names replaced

    Example:
        >>> expr = '"old_col" > 30 and "status" == \'active\''
        >>> mapping = {'old_col': 'new_col'}
        >>> replace_column_names(expr, mapping)
        '"new_col" > 30 and "status" == \'active\''
    """
    if not expression or not column_mapping:
        return expression

    result = expression
    for old_name, new_name in column_mapping.items():
        # Use word boundaries to avoid partial matches
        pattern = f'"{re.escape(old_name)}"'
        replacement = f'"{new_name}"'
        result = re.sub(pattern, replacement, result)

    logger.debug(f'Applied {len(column_mapping)} column replacements')
    return result


def validate_fql_syntax(expression: str) -> tuple[bool, Optional[str]]:
    """Perform basic FQL syntax validation.

    This does not guarantee the expression will work in Fiddler,
    but catches common syntax errors.

    Args:
        expression: FQL expression to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None

    Example:
        >>> validate_fql_syntax('"age" > 30')
        (True, None)

        >>> validate_fql_syntax('"unclosed > 30')
        (False, 'Unclosed double quote at position 0')
    """
    if not expression:
        return True, None

    errors = []

    # Check for balanced double quotes
    if expression.count('"') % 2 != 0:
        return False, 'Unbalanced double quotes (column names)'

    # Check for balanced single quotes
    if expression.count("'") % 2 != 0:
        return False, 'Unbalanced single quotes (string values)'

    # Check for balanced parentheses
    open_count = expression.count('(')
    close_count = expression.count(')')
    if open_count != close_count:
        return (
            False,
            f"Unbalanced parentheses (found {open_count} '(' and {close_count} ')')",
        )

    # Check for empty column references
    empty_columns = re.findall(r'""', expression)
    if empty_columns:
        return False, 'Empty column reference found'

    # Check for likely typos with nested quotes
    nested_double = re.findall(r'"[^"]*"[^"]*"', expression)
    if nested_double:
        logger.warning('Potential nested double quotes detected')

    return True, None


def normalize_expression(expression: str) -> str:
    """Normalize an FQL expression for comparison.

    This standardizes whitespace and formatting to make it easier
    to compare two expressions for equality.

    Args:
        expression: FQL expression to normalize

    Returns:
        Normalized expression

    Example:
        >>> normalize_expression('"age"   >  30')
        '"age" > 30'
    """
    if not expression:
        return expression

    # Normalize whitespace around operators
    result = re.sub(r'\s*([<>=!]+)\s*', r' \1 ', expression)

    # Normalize whitespace around parentheses
    result = re.sub(r'\s*\(\s*', ' (', result)
    result = re.sub(r'\s*\)\s*', ') ', result)

    # Normalize whitespace around commas
    result = re.sub(r'\s*,\s*', ', ', result)

    # Remove extra whitespace
    result = re.sub(r'\s+', ' ', result)

    return result.strip()


def get_fql_functions(expression: str) -> Set[str]:
    """Extract FQL function names used in an expression.

    Args:
        expression: FQL expression

    Returns:
        Set of function names found (e.g., {'sum', 'if', 'fp'})

    Example:
        >>> get_fql_functions('sum(if(fp(), 1, 0))')
        {'sum', 'if', 'fp'}
    """
    if not expression:
        return set()

    # Match function names: word characters followed by (
    function_pattern = r'(\w+)\s*\('
    functions = set(re.findall(function_pattern, expression))

    logger.debug(f'Found FQL functions: {functions}')
    return functions


def is_simple_filter(expression: str) -> bool:
    """Check if expression is a simple filter (no aggregations).

    Simple filters can be used in segments. Complex aggregations
    are typically used in custom metrics.

    Args:
        expression: FQL expression

    Returns:
        True if expression appears to be a simple filter

    Example:
        >>> is_simple_filter('"age" > 30 and "status" == \'active\'')
        True

        >>> is_simple_filter('sum(if(fp(), 1, 0))')
        False
    """
    # Check for common aggregation functions
    agg_functions = {'sum', 'avg', 'count', 'min', 'max', 'mean', 'std'}
    used_functions = get_fql_functions(expression)

    has_aggregation = bool(agg_functions & used_functions)
    return not has_aggregation


def split_fql_and_condition(expression: str) -> List[str]:
    """Split an FQL expression on 'and' operators at the top level.

    Useful for breaking down complex segment definitions.

    Args:
        expression: FQL expression

    Returns:
        List of sub-expressions

    Example:
        >>> split_fql_and_condition('"age" > 30 and "status" == \'active\'')
        ['"age" > 30', '"status" == \'active\'']

    Note:
        This is a simple implementation that may not handle all cases
        (e.g., 'and' inside function calls). Use with caution.
    """
    if not expression or ' and ' not in expression.lower():
        return [expression] if expression else []

    # Simple split on ' and ' (case-insensitive)
    # More sophisticated parsing would require a full parser
    parts = re.split(r'\s+and\s+', expression, flags=re.IGNORECASE)

    return [part.strip() for part in parts if part.strip()]


def validate_column_references(
    expression: str, valid_columns: Set[str]
) -> tuple[bool, List[str]]:
    """Validate that all column references in expression exist in valid_columns.

    Args:
        expression: FQL expression
        valid_columns: Set of valid column names

    Returns:
        Tuple of (all_valid, missing_columns)

    Example:
        >>> expr = '"age" > 30 and "unknown_col" == 1'
        >>> validate_column_references(expr, {'age', 'status'})
        (False, ['unknown_col'])
    """
    referenced_columns = extract_columns(expression)
    missing_columns = [col for col in referenced_columns if col not in valid_columns]

    is_valid = len(missing_columns) == 0

    if not is_valid:
        logger.warning(
            f'Found {len(missing_columns)} missing column references: {missing_columns}'
        )

    return is_valid, missing_columns
