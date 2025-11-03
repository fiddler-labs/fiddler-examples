"""Unit tests for reference management utilities.

Tests the functions in fiddler_utils/assets/references.py that handle
finding and migrating Custom Metric references in Charts and Alerts.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from fiddler_utils.assets.references import (
    find_alerts_using_metric,
    find_charts_using_metric,
    find_all_metric_references,
    migrate_alert_metric_reference,
    migrate_chart_metric_reference,
    safe_update_metric,
)
from fiddler_utils.exceptions import AssetNotFoundError


class TestFindAlertsUsingMetric:
    """Test finding alerts that reference a custom metric."""

    @patch('fiddler_utils.assets.references.fdl')
    def test_find_alerts_with_matches(self, mock_fdl):
        """Test finding alerts that reference a metric."""
        # Setup mock metric
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_metric.model_id = 'model-456'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        # Setup mock alerts
        alert1 = Mock()
        alert1.name = 'Alert 1'
        alert1.metric_id = 'metric-123'  # Matches by ID

        alert2 = Mock()
        alert2.name = 'Alert 2'
        alert2.metric_id = 'test_metric'  # Matches by name

        alert3 = Mock()
        alert3.name = 'Alert 3'
        alert3.metric_id = 'other-metric'  # Doesn't match

        mock_fdl.AlertRule.list.return_value = [alert1, alert2, alert3]

        # Execute
        result = find_alerts_using_metric('metric-123')

        # Verify
        assert len(result) == 2
        assert alert1 in result
        assert alert2 in result
        assert alert3 not in result

    @patch('fiddler_utils.assets.references.fdl')
    def test_find_alerts_no_matches(self, mock_fdl):
        """Test when no alerts reference the metric."""
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_metric.model_id = 'model-456'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        alert1 = Mock()
        alert1.metric_id = 'other-metric'
        mock_fdl.AlertRule.list.return_value = [alert1]

        result = find_alerts_using_metric('metric-123')

        assert len(result) == 0

    @patch('fiddler_utils.assets.references.fdl')
    def test_find_alerts_metric_not_found(self, mock_fdl):
        """Test error when metric doesn't exist."""
        mock_fdl.CustomMetric.get.side_effect = Exception("Not found")

        with pytest.raises(AssetNotFoundError):
            find_alerts_using_metric('nonexistent-metric')


class TestFindChartsUsingMetric:
    """Test finding charts that reference a custom metric."""

    @patch('fiddler_utils.assets.references.ChartManager')
    @patch('fiddler_utils.assets.references.fdl')
    def test_find_charts_with_matches(self, mock_fdl, mock_chart_manager_class):
        """Test finding charts that reference a metric."""
        # Setup mock metric
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        # Setup mock charts
        chart1 = {
            'title': 'Chart 1',
            'data_source': {
                'queries': [{
                    'metric_type': 'custom',
                    'metric': 'metric-123'  # Matches by ID
                }]
            }
        }

        chart2 = {
            'title': 'Chart 2',
            'data_source': {
                'queries': [{
                    'metric_type': 'custom',
                    'metric_name': 'test_metric'  # Matches by name
                }]
            }
        }

        chart3 = {
            'title': 'Chart 3',
            'data_source': {
                'queries': [{
                    'metric_type': 'builtin',
                    'metric': 'precision'
                }]
            }
        }

        # Setup mock ChartManager
        mock_chart_mgr = Mock()
        mock_chart_mgr.list_charts.return_value = [chart1, chart2, chart3]
        mock_chart_manager_class.return_value = mock_chart_mgr

        # Execute
        result = find_charts_using_metric(
            metric_id='metric-123',
            project_id='project-789',
            url='https://test.fiddler.ai',
            token='test-token'
        )

        # Verify
        assert len(result) == 2
        assert chart1 in result
        assert chart2 in result
        assert chart3 not in result

    @patch('fiddler_utils.assets.references.ChartManager')
    @patch('fiddler_utils.assets.references.fdl')
    def test_find_charts_api_unavailable(self, mock_fdl, mock_chart_manager_class):
        """Test handling when Chart API is unavailable."""
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        mock_chart_mgr = Mock()
        mock_chart_mgr.list_charts.side_effect = Exception("API unavailable")
        mock_chart_manager_class.return_value = mock_chart_mgr

        result = find_charts_using_metric(
            metric_id='metric-123',
            project_id='project-789'
        )

        assert len(result) == 0  # Returns empty list on error


class TestFindAllMetricReferences:
    """Test comprehensive reference discovery."""

    @patch('fiddler_utils.assets.references.find_charts_using_metric')
    @patch('fiddler_utils.assets.references.find_alerts_using_metric')
    @patch('fiddler_utils.assets.references.fdl')
    def test_find_all_references(self, mock_fdl, mock_find_alerts, mock_find_charts):
        """Test finding all references (charts and alerts)."""
        # Setup mock metric
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        # Setup mock charts and alerts
        mock_charts = [{'title': 'Chart 1'}, {'title': 'Chart 2'}]
        mock_alerts = [Mock(name='Alert 1'), Mock(name='Alert 2'), Mock(name='Alert 3')]

        mock_find_charts.return_value = mock_charts
        mock_find_alerts.return_value = mock_alerts

        # Execute
        result = find_all_metric_references(
            metric_id='metric-123',
            project_id='project-789'
        )

        # Verify
        assert result['metric_id'] == 'metric-123'
        assert result['metric_name'] == 'test_metric'
        assert result['chart_count'] == 2
        assert result['alert_count'] == 3
        assert result['total_count'] == 5
        assert result['has_references'] is True
        assert result['charts'] == mock_charts
        assert result['alerts'] == mock_alerts

    @patch('fiddler_utils.assets.references.find_charts_using_metric')
    @patch('fiddler_utils.assets.references.find_alerts_using_metric')
    @patch('fiddler_utils.assets.references.fdl')
    def test_find_all_references_none_found(self, mock_fdl, mock_find_alerts, mock_find_charts):
        """Test when no references are found."""
        mock_metric = Mock()
        mock_metric.id = 'metric-123'
        mock_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_metric

        mock_find_charts.return_value = []
        mock_find_alerts.return_value = []

        result = find_all_metric_references(
            metric_id='metric-123',
            project_id='project-789'
        )

        assert result['total_count'] == 0
        assert result['has_references'] is False


class TestMigrateAlertMetricReference:
    """Test migrating alert references to new metric UUID."""

    @patch('fiddler_utils.assets.references.fdl')
    def test_migrate_alert_success(self, mock_fdl):
        """Test successful alert migration."""
        # Setup mock new metric
        mock_new_metric = Mock()
        mock_new_metric.id = 'new-metric-456'
        mock_new_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_new_metric

        # Setup mock alert
        mock_alert = Mock()
        mock_alert.name = 'Test Alert'
        mock_alert.metric_id = 'old-metric-123'
        mock_alert.model_id = 'model-789'
        mock_alert.bin_size = Mock(value='DAY')
        mock_alert.priority = Mock(value='HIGH')
        mock_alert.compare_to = 'TIME_PERIOD'
        mock_alert.condition = 'GREATER'
        mock_alert.warning_threshold = 0.1
        mock_alert.critical_threshold = 0.2

        # Mock successful alert creation
        mock_new_alert = Mock()
        mock_fdl.AlertRule.return_value = mock_new_alert

        # Execute
        result = migrate_alert_metric_reference(
            alert=mock_alert,
            old_metric_id='old-metric-123',
            new_metric_id='new-metric-456'
        )

        # Verify
        assert result is True
        mock_alert.delete.assert_called_once()
        mock_new_alert.create.assert_called_once()

    @patch('fiddler_utils.assets.references.fdl')
    def test_migrate_alert_wrong_metric(self, mock_fdl):
        """Test when alert doesn't reference the expected metric."""
        mock_alert = Mock()
        mock_alert.metric_id = 'different-metric'

        result = migrate_alert_metric_reference(
            alert=mock_alert,
            old_metric_id='old-metric-123',
            new_metric_id='new-metric-456'
        )

        assert result is False

    @patch('fiddler_utils.assets.references.fdl')
    def test_migrate_alert_failure(self, mock_fdl):
        """Test handling of migration failure."""
        mock_new_metric = Mock()
        mock_new_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_new_metric

        mock_alert = Mock()
        mock_alert.metric_id = 'old-metric-123'
        mock_alert.delete.side_effect = Exception("Delete failed")

        result = migrate_alert_metric_reference(
            alert=mock_alert,
            old_metric_id='old-metric-123',
            new_metric_id='new-metric-456'
        )

        assert result is False


class TestMigrateChartMetricReference:
    """Test migrating chart references to new metric UUID."""

    @patch('fiddler_utils.assets.references.ChartManager')
    @patch('fiddler_utils.assets.references.fdl')
    def test_migrate_chart_success(self, mock_fdl, mock_chart_manager_class):
        """Test successful chart migration."""
        # Setup mock new metric
        mock_new_metric = Mock()
        mock_new_metric.id = 'new-metric-456'
        mock_new_metric.name = 'test_metric'
        mock_fdl.CustomMetric.get.return_value = mock_new_metric

        # Setup mock chart
        mock_chart = {
            'uuid': 'chart-uuid-123',
            'title': 'Test Chart',
            'data_source': {
                'queries': [{
                    'metric_type': 'custom',
                    'metric': 'old-metric-123'
                }]
            }
        }

        # Setup mock ChartManager
        mock_chart_mgr = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.put.return_value = mock_response
        mock_chart_mgr._get_client.return_value = mock_client
        mock_chart_manager_class.return_value = mock_chart_mgr

        # Execute
        result = migrate_chart_metric_reference(
            chart=mock_chart,
            old_metric_id='old-metric-123',
            new_metric_id='new-metric-456'
        )

        # Verify
        assert result is True
        assert mock_chart['data_source']['queries'][0]['metric'] == 'new-metric-456'
        assert mock_chart['data_source']['queries'][0]['metric_name'] == 'test_metric'

    @patch('fiddler_utils.assets.references.ChartManager')
    @patch('fiddler_utils.assets.references.fdl')
    def test_migrate_chart_wrong_metric(self, mock_fdl, mock_chart_manager_class):
        """Test when chart doesn't reference the expected metric."""
        mock_chart = {
            'data_source': {
                'queries': [{
                    'metric_type': 'custom',
                    'metric': 'different-metric'
                }]
            }
        }

        result = migrate_chart_metric_reference(
            chart=mock_chart,
            old_metric_id='old-metric-123',
            new_metric_id='new-metric-456'
        )

        assert result is False


class TestSafeUpdateMetric:
    """Test safe metric update with automatic reference migration."""

    @patch('fiddler_utils.assets.references.find_all_metric_references')
    @patch('fiddler_utils.assets.references.migrate_chart_metric_reference')
    @patch('fiddler_utils.assets.references.migrate_alert_metric_reference')
    @patch('fiddler_utils.assets.references.fdl')
    @patch('fiddler_utils.assets.references.time')
    def test_safe_update_with_migration(
        self,
        mock_time,
        mock_fdl,
        mock_migrate_alert,
        mock_migrate_chart,
        mock_find_refs
    ):
        """Test safe update with automatic reference migration."""
        # Setup mock metric
        mock_metric = Mock()
        mock_metric.id = 'old-metric-123'
        mock_metric.name = 'test_metric'
        mock_metric.model_id = 'model-789'
        mock_metric.description = 'Test description'

        # Setup mock model for validation
        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        # Setup mock new metric
        mock_new_metric = Mock()
        mock_new_metric.id = 'new-metric-456'
        mock_new_metric.name = 'test_metric'
        mock_fdl.CustomMetric.return_value = mock_new_metric

        # Setup mock references
        mock_refs = {
            'metric_id': 'old-metric-123',
            'metric_name': 'test_metric',
            'charts': [{'title': 'Chart 1'}],
            'alerts': [Mock(name='Alert 1')],
            'chart_count': 1,
            'alert_count': 1,
            'total_count': 2,
            'has_references': True
        }
        mock_find_refs.return_value = mock_refs

        # Setup successful migrations
        mock_migrate_chart.return_value = True
        mock_migrate_alert.return_value = True

        # Setup validation mock
        with patch('fiddler_utils.assets.references.CustomMetricManager') as mock_mgr_class:
            mock_mgr = Mock()
            mock_mgr.validate_metric_definition.return_value = (True, '')
            mock_mgr_class.return_value = mock_mgr

            # Execute
            new_metric, report = safe_update_metric(
                metric=mock_metric,
                new_definition='sum(if(tp(), 1, 0))',
                auto_migrate=True,
                project_id='project-123',
                url='https://test.fiddler.ai',
                token='test-token',
                validate=True
            )

        # Verify
        assert new_metric == mock_new_metric
        assert report['old_metric_id'] == 'old-metric-123'
        assert report['new_metric_id'] == 'new-metric-456'
        assert report['migrated_count'] == 2
        assert report['failed_count'] == 0

        # Verify metric was deleted and recreated
        mock_metric.delete.assert_called_once()
        mock_new_metric.create.assert_called_once()

    @patch('fiddler_utils.assets.references.find_all_metric_references')
    @patch('fiddler_utils.assets.references.fdl')
    @patch('fiddler_utils.assets.references.time')
    def test_safe_update_no_references(self, mock_time, mock_fdl, mock_find_refs):
        """Test safe update when no references exist."""
        mock_metric = Mock()
        mock_metric.id = 'old-metric-123'
        mock_metric.name = 'test_metric'
        mock_metric.model_id = 'model-789'
        mock_metric.description = ''

        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        mock_new_metric = Mock()
        mock_new_metric.id = 'new-metric-456'
        mock_fdl.CustomMetric.return_value = mock_new_metric

        # No references found
        mock_refs = {
            'charts': [],
            'alerts': [],
            'has_references': False
        }
        mock_find_refs.return_value = mock_refs

        with patch('fiddler_utils.assets.references.CustomMetricManager') as mock_mgr_class:
            mock_mgr = Mock()
            mock_mgr.validate_metric_definition.return_value = (True, '')
            mock_mgr_class.return_value = mock_mgr

            new_metric, report = safe_update_metric(
                metric=mock_metric,
                new_definition='sum(if(tp(), 1, 0))',
                auto_migrate=True,
                project_id='project-123',
                validate=True
            )

        assert report['migrated_count'] == 0
        assert report['failed_count'] == 0

    @patch('fiddler_utils.assets.references.fdl')
    def test_safe_update_validation_failure(self, mock_fdl):
        """Test safe update with validation failure."""
        mock_metric = Mock()
        mock_metric.model_id = 'model-789'

        mock_model = Mock()
        mock_fdl.Model.get.return_value = mock_model

        with patch('fiddler_utils.assets.references.CustomMetricManager') as mock_mgr_class:
            mock_mgr = Mock()
            mock_mgr.validate_metric_definition.return_value = (False, 'Invalid FQL')
            mock_mgr_class.return_value = mock_mgr

            with pytest.raises(Exception):  # Should raise FQLError
                safe_update_metric(
                    metric=mock_metric,
                    new_definition='invalid fql',
                    validate=True,
                    project_id='project-123'
                )

    @patch('fiddler_utils.assets.references.fdl')
    @patch('fiddler_utils.assets.references.time')
    def test_safe_update_skip_migration(self, mock_time, mock_fdl):
        """Test safe update with auto_migrate=False."""
        mock_metric = Mock()
        mock_metric.id = 'old-metric-123'
        mock_metric.name = 'test_metric'
        mock_metric.model_id = 'model-789'
        mock_metric.description = ''

        mock_new_metric = Mock()
        mock_new_metric.id = 'new-metric-456'
        mock_fdl.CustomMetric.return_value = mock_new_metric

        new_metric, report = safe_update_metric(
            metric=mock_metric,
            new_definition='sum(if(tp(), 1, 0))',
            auto_migrate=False,
            validate=False
        )

        assert report['auto_migrate_enabled'] is False
        assert report['migrated_count'] == 0
