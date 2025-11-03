"""Unit tests for FQL testing utilities.

Tests the functions in fiddler_utils/testing.py that handle
validating and testing FQL expressions before creating Custom Metrics.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from fiddler_utils.testing import (
    _generate_temp_metric_name,
    validate_metric_syntax_local,
    test_metric_definition,
    validate_and_preview_metric,
    batch_test_metrics,
    cleanup_orphaned_test_metrics,
)


class TestGenerateTempMetricName:
    """Test temporary metric name generation."""

    @patch('fiddler_utils.testing.time')
    @patch('fiddler_utils.testing.random')
    def test_generate_temp_name(self, mock_random, mock_time):
        """Test temp metric name generation."""
        mock_time.time.return_value = 1704123456
        mock_random.choices.return_value = list('a3f2')

        result = _generate_temp_metric_name()

        assert result == '__test_1704123456_a3f2'

    @patch('fiddler_utils.testing.time')
    @patch('fiddler_utils.testing.random')
    def test_generate_temp_name_custom_prefix(self, mock_random, mock_time):
        """Test temp metric name with custom prefix."""
        mock_time.time.return_value = 1704123456
        mock_random.choices.return_value = list('a3f2')

        result = _generate_temp_metric_name(prefix='__custom_')

        assert result == '__custom_1704123456_a3f2'


class TestValidateMetricSyntaxLocal:
    """Test local FQL syntax validation."""

    @patch('fiddler_utils.testing.fql')
    def test_valid_simple_expression(self, mock_fql):
        """Test validation of valid simple aggregation."""
        mock_fql.validate_fql_syntax.return_value = (True, '')
        mock_fql.extract_columns.return_value = {'age', 'status'}
        mock_fql.get_fql_functions.return_value = {'sum', 'if'}
        mock_fql.is_simple_filter.return_value = False

        result = validate_metric_syntax_local(
            definition='sum(if(fp(), 1, 0))'
        )

        assert result['valid'] is True
        assert result['has_errors'] is False
        assert result['metadata']['has_aggregations'] is True

    @patch('fiddler_utils.testing.fql')
    def test_syntax_error(self, mock_fql):
        """Test detection of syntax errors."""
        mock_fql.validate_fql_syntax.return_value = (False, 'Unbalanced parentheses')

        result = validate_metric_syntax_local(
            definition='sum(if(fp(), 1, 0'  # Missing closing paren
        )

        assert result['valid'] is False
        assert result['has_errors'] is True
        assert 'Syntax error' in result['errors'][0]

    @patch('fiddler_utils.testing.fql')
    @patch('fiddler_utils.testing.SchemaValidator')
    def test_missing_columns(self, mock_validator, mock_fql):
        """Test detection of missing columns."""
        mock_fql.validate_fql_syntax.return_value = (True, '')
        mock_fql.extract_columns.return_value = {'nonexistent_column'}
        mock_fql.get_fql_functions.return_value = {'sum'}
        mock_fql.is_simple_filter.return_value = False

        mock_validator.validate_fql_expression.return_value = (
            False,
            {'nonexistent_column'}
        )

        mock_model = Mock()

        result = validate_metric_syntax_local(
            definition='sum("nonexistent_column")',
            model=mock_model
        )

        assert result['valid'] is False
        assert result['has_errors'] is True
        assert any('Missing columns' in err for err in result['errors'])

    @patch('fiddler_utils.testing.fql')
    def test_simple_filter_warning(self, mock_fql):
        """Test warning for non-aggregated expressions."""
        mock_fql.validate_fql_syntax.return_value = (True, '')
        mock_fql.extract_columns.return_value = {'age'}
        mock_fql.get_fql_functions.return_value = set()
        mock_fql.is_simple_filter.return_value = True  # Not an aggregation

        result = validate_metric_syntax_local(
            definition='"age" > 30'
        )

        assert result['valid'] is True
        assert result['has_warnings'] is True
        assert any('aggregation' in w.lower() for w in result['warnings'])

    @patch('fiddler_utils.testing.fql')
    def test_complexity_warnings(self, mock_fql):
        """Test warnings for complex expressions."""
        mock_fql.validate_fql_syntax.return_value = (True, '')
        # 11 columns (triggers warning at > 10)
        mock_fql.extract_columns.return_value = set(f'col{i}' for i in range(11))
        # 6 functions (triggers warning at > 5)
        mock_fql.get_fql_functions.return_value = set(f'func{i}' for i in range(6))
        mock_fql.is_simple_filter.return_value = False

        result = validate_metric_syntax_local(
            definition='complex expression'
        )

        assert result['valid'] is True
        assert result['has_warnings'] is True
        assert len(result['warnings']) == 2  # Complexity + nested functions


class TestTestMetricDefinition:
    """Test real FQL validation via temporary metrics."""

    @patch('fiddler_utils.testing.fdl')
    def test_valid_metric_definition(self, mock_fdl):
        """Test successful metric validation."""
        # Mock successful metric creation
        mock_metric = Mock()
        mock_metric.id = 'temp-metric-123'
        mock_fdl.CustomMetric.return_value = mock_metric

        result = test_metric_definition(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))',
            cleanup=True
        )

        # Verify
        assert result['valid'] is True
        assert result['error'] is None
        assert result['temp_metric_id'] == 'temp-metric-123'
        assert result['cleaned_up'] is True

        # Verify metric was created and deleted
        mock_metric.create.assert_called_once()
        mock_metric.delete.assert_called_once()

    @patch('fiddler_utils.testing.fdl')
    def test_invalid_metric_syntax(self, mock_fdl):
        """Test handling of invalid FQL syntax."""
        # Mock BadRequest error from Fiddler
        mock_fdl.BadRequest = Exception
        mock_metric = Mock()
        mock_metric.create.side_effect = mock_fdl.BadRequest("Invalid FQL syntax")
        mock_fdl.CustomMetric.return_value = mock_metric

        result = test_metric_definition(
            model_id='model-456',
            definition='invalid fql',
            cleanup=True
        )

        assert result['valid'] is False
        assert 'Invalid FQL syntax' in result['error']
        assert result['temp_metric_id'] is None

    @patch('fiddler_utils.testing.fdl')
    @patch('fiddler_utils.testing.time')
    def test_metric_with_wait(self, mock_time, mock_fdl):
        """Test metric validation with calculation wait."""
        mock_metric = Mock()
        mock_metric.id = 'temp-metric-123'
        mock_fdl.CustomMetric.return_value = mock_metric

        result = test_metric_definition(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))',
            cleanup=True,
            wait_for_calculation=True
        )

        assert result['valid'] is True
        assert result['calculation_attempted'] is True
        mock_time.sleep.assert_called_with(2)

    @patch('fiddler_utils.testing.fdl')
    def test_cleanup_failure(self, mock_fdl):
        """Test handling of cleanup failure."""
        mock_metric = Mock()
        mock_metric.id = 'temp-metric-123'
        mock_metric.delete.side_effect = Exception("Delete failed")
        mock_fdl.CustomMetric.return_value = mock_metric

        result = test_metric_definition(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))',
            cleanup=True
        )

        assert result['valid'] is True
        assert result['cleaned_up'] is False

    @patch('fiddler_utils.testing.fdl')
    def test_no_cleanup(self, mock_fdl):
        """Test skipping cleanup."""
        mock_metric = Mock()
        mock_metric.id = 'temp-metric-123'
        mock_fdl.CustomMetric.return_value = mock_metric

        result = test_metric_definition(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))',
            cleanup=False
        )

        assert result['valid'] is True
        mock_metric.delete.assert_not_called()


class TestValidateAndPreviewMetric:
    """Test complete validation workflow."""

    @patch('fiddler_utils.testing.test_metric_definition')
    @patch('fiddler_utils.testing.validate_metric_syntax_local')
    @patch('fiddler_utils.testing.fdl')
    def test_complete_validation_success(
        self,
        mock_fdl,
        mock_local_validate,
        mock_test_metric
    ):
        """Test successful complete validation."""
        # Mock local validation success
        mock_local_validate.return_value = {
            'valid': True,
            'has_errors': False,
            'has_warnings': False,
            'errors': [],
            'warnings': []
        }

        # Mock Fiddler test success
        mock_test_metric.return_value = {
            'valid': True,
            'error': None
        }

        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        result = validate_and_preview_metric(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))'
        )

        assert result['valid'] is True
        assert 'valid' in result['recommendation'].lower()
        assert result['local_validation'] is not None
        assert result['fiddler_test'] is not None

    @patch('fiddler_utils.testing.validate_metric_syntax_local')
    @patch('fiddler_utils.testing.fdl')
    def test_local_validation_failure(self, mock_fdl, mock_local_validate):
        """Test when local validation fails."""
        mock_local_validate.return_value = {
            'valid': False,
            'has_errors': True,
            'errors': ['Syntax error: Unbalanced parens']
        }

        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        result = validate_and_preview_metric(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0'
        )

        assert result['valid'] is False
        assert result['fiddler_test'] is None  # Skipped
        assert 'syntax errors' in result['recommendation'].lower()

    @patch('fiddler_utils.testing.test_metric_definition')
    @patch('fiddler_utils.testing.validate_metric_syntax_local')
    @patch('fiddler_utils.testing.fdl')
    def test_fiddler_validation_failure(
        self,
        mock_fdl,
        mock_local_validate,
        mock_test_metric
    ):
        """Test when Fiddler validation fails."""
        mock_local_validate.return_value = {
            'valid': True,
            'has_errors': False,
            'has_warnings': False
        }

        mock_test_metric.return_value = {
            'valid': False,
            'error': 'Division by zero not allowed'
        }

        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        result = validate_and_preview_metric(
            model_id='model-456',
            definition='sum(fp()) / 0'
        )

        assert result['valid'] is False
        assert 'Division by zero' in result['recommendation']

    @patch('fiddler_utils.testing.test_metric_definition')
    def test_skip_local_validation(self, mock_test_metric):
        """Test skipping local validation."""
        mock_test_metric.return_value = {
            'valid': True,
            'error': None
        }

        result = validate_and_preview_metric(
            model_id='model-456',
            definition='sum(if(fp(), 1, 0))',
            skip_local_validation=True
        )

        assert result['local_validation'] is None
        assert result['fiddler_test'] is not None


class TestBatchTestMetrics:
    """Test batch testing of multiple metrics."""

    @patch('fiddler_utils.testing.test_metric_definition')
    @patch('fiddler_utils.testing.time')
    def test_batch_test_all_valid(self, mock_time, mock_test_metric):
        """Test batch testing with all valid metrics."""
        # Mock all metrics as valid
        mock_test_metric.return_value = {
            'valid': True,
            'error': None
        }

        definitions = [
            {'name': 'FP Count', 'definition': 'sum(if(fp(), 1, 0))'},
            {'name': 'FN Count', 'definition': 'sum(if(fn(), 1, 0))'},
            {'name': 'Accuracy', 'definition': 'sum(if(tp() or tn(), 1, 0)) / sum(1)'},
        ]

        results = batch_test_metrics(
            model_id='model-456',
            definitions=definitions,
            delay_between_tests=0.5
        )

        # Verify
        assert len(results) == 3
        assert all(r['valid'] for r in results)
        assert results[0]['name'] == 'FP Count'
        assert results[1]['name'] == 'FN Count'
        assert results[2]['name'] == 'Accuracy'

        # Verify delays (2 delays for 3 metrics)
        assert mock_time.sleep.call_count == 2

    @patch('fiddler_utils.testing.test_metric_definition')
    @patch('fiddler_utils.testing.time')
    def test_batch_test_mixed_results(self, mock_time, mock_test_metric):
        """Test batch testing with some failures."""
        # Mock alternating valid/invalid
        mock_test_metric.side_effect = [
            {'valid': True, 'error': None},
            {'valid': False, 'error': 'Syntax error'},
            {'valid': True, 'error': None},
        ]

        definitions = [
            {'name': 'Metric 1', 'definition': 'valid'},
            {'name': 'Metric 2', 'definition': 'invalid'},
            {'name': 'Metric 3', 'definition': 'valid'},
        ]

        results = batch_test_metrics(
            model_id='model-456',
            definitions=definitions
        )

        assert len(results) == 3
        assert results[0]['valid'] is True
        assert results[1]['valid'] is False
        assert results[2]['valid'] is True

    @patch('fiddler_utils.testing.test_metric_definition')
    @patch('fiddler_utils.testing.time')
    def test_batch_test_no_delay(self, mock_time, mock_test_metric):
        """Test batch testing with no delays."""
        mock_test_metric.return_value = {'valid': True, 'error': None}

        definitions = [
            {'name': 'M1', 'definition': 'd1'},
            {'name': 'M2', 'definition': 'd2'},
        ]

        batch_test_metrics(
            model_id='model-456',
            definitions=definitions,
            delay_between_tests=0
        )

        # No delays should be called
        mock_time.sleep.assert_not_called()


class TestCleanupOrphanedTestMetrics:
    """Test cleanup of orphaned test metrics."""

    @patch('fiddler_utils.testing.fdl')
    def test_cleanup_with_orphans(self, mock_fdl):
        """Test cleanup when orphaned metrics exist."""
        # Mock metrics list
        metric1 = Mock()
        metric1.name = '__test_1234_abcd'
        metric1.id = 'metric-1'

        metric2 = Mock()
        metric2.name = 'normal_metric'
        metric2.id = 'metric-2'

        metric3 = Mock()
        metric3.name = '__test_5678_efgh'
        metric3.id = 'metric-3'

        mock_fdl.CustomMetric.list.return_value = [metric1, metric2, metric3]

        result = cleanup_orphaned_test_metrics(model_id='model-456')

        # Verify
        assert result == 2
        metric1.delete.assert_called_once()
        metric2.delete.assert_not_called()
        metric3.delete.assert_called_once()

    @patch('fiddler_utils.testing.fdl')
    def test_cleanup_no_orphans(self, mock_fdl):
        """Test cleanup when no orphaned metrics exist."""
        metric1 = Mock()
        metric1.name = 'normal_metric_1'

        metric2 = Mock()
        metric2.name = 'normal_metric_2'

        mock_fdl.CustomMetric.list.return_value = [metric1, metric2]

        result = cleanup_orphaned_test_metrics(model_id='model-456')

        assert result == 0
        metric1.delete.assert_not_called()
        metric2.delete.assert_not_called()

    @patch('fiddler_utils.testing.fdl')
    def test_cleanup_delete_failure(self, mock_fdl):
        """Test cleanup when some deletions fail."""
        metric1 = Mock()
        metric1.name = '__test_1234_abcd'
        metric1.delete.return_value = None  # Success

        metric2 = Mock()
        metric2.name = '__test_5678_efgh'
        metric2.delete.side_effect = Exception("Delete failed")  # Failure

        mock_fdl.CustomMetric.list.return_value = [metric1, metric2]

        result = cleanup_orphaned_test_metrics(model_id='model-456')

        # Should only count successful deletions
        assert result == 1

    @patch('fiddler_utils.testing.fdl')
    def test_cleanup_custom_prefix(self, mock_fdl):
        """Test cleanup with custom prefix."""
        metric1 = Mock()
        metric1.name = '__custom_1234_abcd'

        metric2 = Mock()
        metric2.name = '__test_5678_efgh'  # Different prefix

        mock_fdl.CustomMetric.list.return_value = [metric1, metric2]

        result = cleanup_orphaned_test_metrics(
            model_id='model-456',
            prefix='__custom_'
        )

        assert result == 1
        metric1.delete.assert_called_once()
        metric2.delete.assert_not_called()
