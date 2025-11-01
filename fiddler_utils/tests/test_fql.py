"""Tests for FQL parsing and manipulation utilities."""

import pytest
from fiddler_utils import fql


class TestExtractColumns:
    """Tests for fql.extract_columns function."""

    def test_simple_expression(self):
        """Test extracting columns from simple expression."""
        expr = '"age" > 30'
        columns = fql.extract_columns(expr)
        assert columns == {'age'}

    def test_multiple_columns(self):
        """Test extracting multiple columns."""
        expr = '"age" > 30 and "geography" == \'California\''
        columns = fql.extract_columns(expr)
        assert columns == {'age', 'geography'}

    def test_complex_expression(self):
        """Test extracting columns from complex expression."""
        expr = '("age" < 60 and "age" > 30) and "geography" == \'Hawaii\''
        columns = fql.extract_columns(expr)
        # Note: age appears twice but should only be in set once
        assert columns == {'age', 'geography'}

    def test_function_calls(self):
        """Test extracting columns from expressions with functions."""
        expr = 'sum(if(fp(), 1, 0) * "transaction_value")'
        columns = fql.extract_columns(expr)
        assert columns == {'transaction_value'}

    def test_empty_expression(self):
        """Test empty expression returns empty set."""
        assert fql.extract_columns('') == set()
        assert fql.extract_columns(None) == set()

    def test_no_columns(self):
        """Test expression with no column references."""
        expr = '1 + 2'
        columns = fql.extract_columns(expr)
        assert columns == set()

    def test_ignores_single_quotes(self):
        """Test that single-quoted strings (values) are not extracted."""
        expr = "\"status\" == 'active' and 'inactive' != 'pending'"
        columns = fql.extract_columns(expr)
        assert columns == {'status'}  # Should only get column, not values

    def test_column_with_underscores(self):
        """Test column names with underscores."""
        expr = '"user_id" == 123 and "product_category" == \'electronics\''
        columns = fql.extract_columns(expr)
        assert columns == {'user_id', 'product_category'}


class TestReplaceColumnNames:
    """Tests for fql.replace_column_names function."""

    def test_simple_replacement(self):
        """Test simple column name replacement."""
        expr = '"old_col" > 30'
        mapping = {'old_col': 'new_col'}
        result = fql.replace_column_names(expr, mapping)
        assert result == '"new_col" > 30'

    def test_multiple_replacements(self):
        """Test replacing multiple column names."""
        expr = '"col1" > 30 and "col2" == \'value\''
        mapping = {'col1': 'new_col1', 'col2': 'new_col2'}
        result = fql.replace_column_names(expr, mapping)
        assert result == '"new_col1" > 30 and "new_col2" == \'value\''

    def test_partial_mapping(self):
        """Test replacement with partial mapping (some columns not in mapping)."""
        expr = '"col1" > 30 and "col2" == \'value\''
        mapping = {'col1': 'new_col1'}
        result = fql.replace_column_names(expr, mapping)
        assert result == '"new_col1" > 30 and "col2" == \'value\''

    def test_duplicate_column_references(self):
        """Test that all occurrences of a column are replaced."""
        expr = '"age" > 30 and "age" < 60'
        mapping = {'age': 'customer_age'}
        result = fql.replace_column_names(expr, mapping)
        assert result == '"customer_age" > 30 and "customer_age" < 60'

    def test_empty_mapping(self):
        """Test that empty mapping returns original expression."""
        expr = '"col" > 30'
        result = fql.replace_column_names(expr, {})
        assert result == expr

    def test_empty_expression(self):
        """Test empty expression."""
        result = fql.replace_column_names('', {'col': 'new_col'})
        assert result == ''


class TestValidateFQLSyntax:
    """Tests for fql.validate_fql_syntax function."""

    def test_valid_expression(self):
        """Test valid FQL expression."""
        expr = '"age" > 30 and "status" == \'active\''
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is True
        assert error is None

    def test_unbalanced_double_quotes(self):
        """Test detection of unbalanced double quotes."""
        expr = '"unclosed > 30'
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is False
        assert 'double quote' in error.lower()

    def test_unbalanced_single_quotes(self):
        """Test detection of unbalanced single quotes."""
        expr = '"col" == \'unclosed'
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is False
        assert 'single quote' in error.lower()

    def test_unbalanced_parentheses_open(self):
        """Test detection of unclosed parentheses."""
        expr = '("age" > 30 and "status" == \'active\''
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is False
        assert 'parenthes' in error.lower()

    def test_unbalanced_parentheses_close(self):
        """Test detection of extra closing parentheses."""
        expr = '"age" > 30) and "status" == \'active\''
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is False
        assert 'parenthes' in error.lower()

    def test_empty_column_reference(self):
        """Test detection of empty column references."""
        expr = '"" > 30'
        is_valid, error = fql.validate_fql_syntax(expr)
        assert is_valid is False
        assert 'empty column' in error.lower()

    def test_empty_expression(self):
        """Test that empty expression is valid."""
        is_valid, error = fql.validate_fql_syntax('')
        assert is_valid is True
        assert error is None


class TestNormalizeExpression:
    """Tests for fql.normalize_expression function."""

    def test_normalize_operators(self):
        """Test normalization of operator spacing."""
        expr = '"age">30'
        normalized = fql.normalize_expression(expr)
        assert normalized == '"age" > 30'

    def test_normalize_multiple_spaces(self):
        """Test removal of extra whitespace."""
        expr = '"age"   >  30'
        normalized = fql.normalize_expression(expr)
        assert normalized == '"age" > 30'

    def test_normalize_parentheses(self):
        """Test normalization of parentheses spacing."""
        expr = '"age">30and("status"==\'active\')'
        normalized = fql.normalize_expression(expr)
        # Should have consistent spacing
        assert '( ' in normalized or ' (' in normalized

    def test_normalize_commas(self):
        """Test normalization of comma spacing."""
        expr = 'func("col1","col2","col3")'
        normalized = fql.normalize_expression(expr)
        assert ', ' in normalized


class TestGetFQLFunctions:
    """Tests for fql.get_fql_functions function."""

    def test_extract_single_function(self):
        """Test extracting single function."""
        expr = 'sum("value")'
        functions = fql.get_fql_functions(expr)
        assert functions == {'sum'}

    def test_extract_multiple_functions(self):
        """Test extracting multiple functions."""
        expr = 'sum(if(fp(), "value", 0))'
        functions = fql.get_fql_functions(expr)
        assert functions == {'sum', 'if', 'fp'}

    def test_nested_functions(self):
        """Test extracting nested function calls."""
        expr = 'avg(max("col1", "col2"))'
        functions = fql.get_fql_functions(expr)
        assert functions == {'avg', 'max'}

    def test_no_functions(self):
        """Test expression with no functions."""
        expr = '"age" > 30'
        functions = fql.get_fql_functions(expr)
        assert functions == set()

    def test_empty_expression(self):
        """Test empty expression."""
        functions = fql.get_fql_functions('')
        assert functions == set()


class TestIsSimpleFilter:
    """Tests for fql.is_simple_filter function."""

    def test_simple_filter(self):
        """Test detection of simple filter."""
        expr = '"age" > 30 and "status" == \'active\''
        assert fql.is_simple_filter(expr) is True

    def test_aggregation_sum(self):
        """Test detection of aggregation with sum."""
        expr = 'sum(if(fp(), 1, 0))'
        assert fql.is_simple_filter(expr) is False

    def test_aggregation_avg(self):
        """Test detection of aggregation with avg."""
        expr = 'avg("transaction_value")'
        assert fql.is_simple_filter(expr) is False

    def test_aggregation_count(self):
        """Test detection of aggregation with count."""
        expr = 'count(*)'
        assert fql.is_simple_filter(expr) is False

    def test_filter_with_non_agg_function(self):
        """Test that non-aggregation functions don't mark as complex."""
        expr = 'if("status" == \'active\', 1, 0)'
        # This should still be considered simple since 'if' is not an aggregation
        # However, our current implementation might mark this as complex
        # depending on function list
        result = fql.is_simple_filter(expr)
        assert result is True  # 'if' alone doesn't make it an aggregation


class TestSplitFQLAndCondition:
    """Tests for fql.split_fql_and_condition function."""

    def test_simple_split(self):
        """Test splitting simple AND condition."""
        expr = '"age" > 30 and "status" == \'active\''
        parts = fql.split_fql_and_condition(expr)
        assert len(parts) == 2
        assert '"age" > 30' in parts
        assert '"status" == \'active\'' in parts

    def test_multiple_ands(self):
        """Test splitting multiple AND conditions."""
        expr = '"a" > 1 and "b" > 2 and "c" > 3'
        parts = fql.split_fql_and_condition(expr)
        assert len(parts) == 3

    def test_no_and(self):
        """Test expression with no AND."""
        expr = '"age" > 30'
        parts = fql.split_fql_and_condition(expr)
        assert len(parts) == 1
        assert parts[0] == expr

    def test_case_insensitive(self):
        """Test case-insensitive AND detection."""
        expr = '"a" > 1 AND "b" > 2 And "c" > 3'
        parts = fql.split_fql_and_condition(expr)
        assert len(parts) == 3

    def test_empty_expression(self):
        """Test empty expression."""
        parts = fql.split_fql_and_condition('')
        assert parts == []


class TestValidateColumnReferences:
    """Tests for fql.validate_column_references function."""

    def test_valid_references(self):
        """Test validation with all valid columns."""
        expr = '"age" > 30 and "status" == \'active\''
        valid_columns = {'age', 'status', 'other'}
        is_valid, missing = fql.validate_column_references(expr, valid_columns)
        assert is_valid is True
        assert missing == []

    def test_missing_columns(self):
        """Test detection of missing columns."""
        expr = '"age" > 30 and "unknown_col" == 1'
        valid_columns = {'age', 'status'}
        is_valid, missing = fql.validate_column_references(expr, valid_columns)
        assert is_valid is False
        assert 'unknown_col' in missing

    def test_multiple_missing_columns(self):
        """Test detection of multiple missing columns."""
        expr = '"col1" > 1 and "col2" > 2 and "col3" > 3'
        valid_columns = {'col1'}
        is_valid, missing = fql.validate_column_references(expr, valid_columns)
        assert is_valid is False
        assert set(missing) == {'col2', 'col3'}

    def test_empty_expression(self):
        """Test validation with empty expression."""
        is_valid, missing = fql.validate_column_references('', {'age'})
        assert is_valid is True
        assert missing == []
