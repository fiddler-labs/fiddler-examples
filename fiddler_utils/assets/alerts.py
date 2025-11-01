"""Alert rule management utilities.

This module provides the AlertManager class for working with Fiddler alert rules.

Note: Alert management is simplified in Phase 2. Full alert cloning with notification
configs will be added in a future phase.
"""

from typing import List, Dict, Set, Any, Optional
import logging

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .base import BaseAssetManager, AssetType
from .. import fql

logger = logging.getLogger(__name__)


class AlertManager(BaseAssetManager[fdl.AlertRule]):
    """Manager for Fiddler alert rules.

    Alert rules monitor metrics and trigger notifications when thresholds are exceeded.
    This manager provides basic CRUD operations and analysis.

    Note: Export/import of alerts is complex due to metric_id dependencies.
    Current implementation focuses on listing and analysis operations.

    Example:
        ```python
        from fiddler_utils.assets import AlertManager
        import fiddler as fdl

        mgr = AlertManager()

        # List all alerts for a model
        alerts = mgr.list_assets(model_id=model.id)
        print(f"Found {len(alerts)} alerts")

        # Find monthly alerts (common issue)
        monthly_alerts = mgr.find_alerts_by_bin_size(
            model_id=model.id,
            bin_size=fdl.BinSize.MONTH
        )

        # Analyze alert configuration
        summary = mgr.get_alert_summary(model_id=model.id)
        print(f"By priority: {summary['by_priority']}")
        print(f"By bin size: {summary['by_bin_size']}")
        ```
    """

    def _get_asset_type(self) -> AssetType:
        """Get asset type."""
        return AssetType.ALERT_RULE

    def _list_assets(self, model_id: str) -> List[fdl.AlertRule]:
        """List all alert rules for a model."""
        return list(fdl.AlertRule.list(model_id=model_id))

    def _get_asset_name(self, asset: fdl.AlertRule) -> str:
        """Get alert rule name."""
        return asset.name

    def _extract_referenced_columns(self, asset: fdl.AlertRule) -> Set[str]:
        """Extract column references from alert rule.

        Args:
            asset: AlertRule object

        Returns:
            Set of column names (from columns property if set)
        """
        # Alerts reference columns through their 'columns' property
        if hasattr(asset, 'columns') and asset.columns:
            return set(asset.columns)
        return set()

    def _extract_asset_data(self, asset: fdl.AlertRule) -> Dict[str, Any]:
        """Extract alert rule data for export.

        Args:
            asset: AlertRule object

        Returns:
            Dictionary with alert rule data
        """
        data = {
            'name': asset.name,
            'metric_id': asset.metric_id,  # Note: This needs mapping when importing
            'bin_size': asset.bin_size.value
            if hasattr(asset.bin_size, 'value')
            else str(asset.bin_size),
            'priority': asset.priority.value
            if hasattr(asset.priority, 'value')
            else str(asset.priority),
            'compare_to': asset.compare_to.value
            if hasattr(asset.compare_to, 'value')
            else str(asset.compare_to),
            'condition': asset.condition.value
            if hasattr(asset.condition, 'value')
            else str(asset.condition),
            'warning_threshold': asset.warning_threshold,
            'metadata': {
                'id': asset.id,
                'model_id': asset.model_id,
            },
        }

        # Optional fields
        if (
            hasattr(asset, 'critical_threshold')
            and asset.critical_threshold is not None
        ):
            data['critical_threshold'] = asset.critical_threshold
        if hasattr(asset, 'segment_id') and asset.segment_id:
            data['segment_id'] = asset.segment_id
        if hasattr(asset, 'columns') and asset.columns:
            data['columns'] = asset.columns
        if hasattr(asset, 'compare_bin_delta') and asset.compare_bin_delta:
            data['compare_bin_delta'] = asset.compare_bin_delta
        if hasattr(asset, 'evaluation_delay') and asset.evaluation_delay:
            data['evaluation_delay'] = asset.evaluation_delay
        if hasattr(asset, 'category') and asset.category:
            data['category'] = asset.category
        if hasattr(asset, 'baseline_id') and asset.baseline_id:
            data['baseline_id'] = asset.baseline_id
        if hasattr(asset, 'threshold_type') and asset.threshold_type:
            data['threshold_type'] = (
                asset.threshold_type.value
                if hasattr(asset.threshold_type, 'value')
                else str(asset.threshold_type)
            )

        return data

    def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> fdl.AlertRule:
        """Create an alert rule from data.

        Note: This is a simplified implementation. metric_id needs to be
        mapped to the target model's corresponding metric.

        Args:
            model_id: Target model ID
            asset_data: Alert rule data dictionary

        Returns:
            Created AlertRule object

        Raises:
            NotImplementedError: Alert import requires metric ID mapping
        """
        raise NotImplementedError(
            'Alert import is not yet fully implemented. '
            'Metric IDs need to be mapped from source to target model. '
            'Use AlertManager for listing and analysis operations. '
            'Manual alert creation is recommended for now.'
        )

    def find_alerts_by_bin_size(
        self, model_id: str, bin_size: fdl.BinSize
    ) -> List[fdl.AlertRule]:
        """Find all alerts with a specific bin size.

        Args:
            model_id: Model ID
            bin_size: Bin size to filter by (e.g., fdl.BinSize.MONTH)

        Returns:
            List of matching alert rules

        Example:
            ```python
            mgr = AlertManager()

            # Find problematic monthly alerts
            monthly_alerts = mgr.find_alerts_by_bin_size(
                model_id=model.id,
                bin_size=fdl.BinSize.MONTH
            )

            print(f"Found {len(monthly_alerts)} monthly alerts")
            for alert in monthly_alerts:
                print(f"  - {alert.name}")
            ```
        """
        alerts = self._list_assets(model_id)
        return [a for a in alerts if a.bin_size == bin_size]

    def find_alerts_by_metric(
        self, model_id: str, metric_id: str
    ) -> List[fdl.AlertRule]:
        """Find all alerts monitoring a specific metric.

        Args:
            model_id: Model ID
            metric_id: Metric ID to filter by

        Returns:
            List of alert rules monitoring this metric

        Example:
            ```python
            mgr = AlertManager()
            alerts = mgr.find_alerts_by_metric(model.id, 'accuracy')
            print(f"{len(alerts)} alerts monitoring accuracy")
            ```
        """
        alerts = self._list_assets(model_id)
        return [a for a in alerts if a.metric_id == metric_id]

    def get_alert_summary(self, model_id: str) -> Dict[str, Any]:
        """Get summary statistics about alerts in a model.

        Args:
            model_id: Model ID

        Returns:
            Dictionary with alert statistics

        Example:
            ```python
            mgr = AlertManager()
            summary = mgr.get_alert_summary(model.id)

            print(f"Total alerts: {summary['total']}")
            print(f"By priority: {summary['by_priority']}")
            print(f"By bin size: {summary['by_bin_size']}")
            print(f"By metric: {summary['by_metric']}")
            ```
        """
        alerts = self._list_assets(model_id)

        by_priority = {}
        by_bin_size = {}
        by_metric = {}

        for alert in alerts:
            # Count by priority
            priority = str(alert.priority)
            by_priority[priority] = by_priority.get(priority, 0) + 1

            # Count by bin size
            bin_size = str(alert.bin_size)
            by_bin_size[bin_size] = by_bin_size.get(bin_size, 0) + 1

            # Count by metric
            metric_id = alert.metric_id
            by_metric[metric_id] = by_metric.get(metric_id, 0) + 1

        summary = {
            'total': len(alerts),
            'by_priority': by_priority,
            'by_bin_size': by_bin_size,
            'by_metric': by_metric,
            'unique_metrics': len(by_metric),
        }

        logger.info(
            f'Alert summary for model {model_id}: '
            f'{summary["total"]} total, '
            f'{summary["unique_metrics"]} unique metrics monitored'
        )

        return summary

    def find_alerts_with_segment(
        self, model_id: str, segment_id: Optional[str] = None
    ) -> List[fdl.AlertRule]:
        """Find alerts that use segments.

        Args:
            model_id: Model ID
            segment_id: Optional specific segment ID to filter by.
                       If None, returns all alerts using any segment.

        Returns:
            List of alert rules using segments

        Example:
            ```python
            mgr = AlertManager()

            # Find all alerts using segments
            segmented_alerts = mgr.find_alerts_with_segment(model.id)
            print(f"{len(segmented_alerts)} alerts use segments")

            # Find alerts using a specific segment
            specific_alerts = mgr.find_alerts_with_segment(
                model.id,
                segment_id='seg-123'
            )
            ```
        """
        alerts = self._list_assets(model_id)

        if segment_id:
            return [
                a
                for a in alerts
                if hasattr(a, 'segment_id') and a.segment_id == segment_id
            ]
        else:
            return [a for a in alerts if hasattr(a, 'segment_id') and a.segment_id]

    def get_alert_by_name(self, model_id: str, name: str) -> fdl.AlertRule:
        """Get a specific alert by name.

        Args:
            model_id: Model ID
            name: Alert name

        Returns:
            AlertRule object

        Raises:
            AssetNotFoundError: If alert not found

        Example:
            ```python
            mgr = AlertManager()
            alert = mgr.get_alert_by_name(model.id, 'Accuracy Drop Alert')
            print(f"Alert monitors metric: {alert.metric_id}")
            print(f"Bin size: {alert.bin_size}")
            ```
        """
        from ..exceptions import AssetNotFoundError

        alerts = self._list_assets(model_id)
        for alert in alerts:
            if alert.name == name:
                return alert

        raise AssetNotFoundError(
            f"Alert '{name}' not found in model {model_id}",
            asset_type='AlertRule',
            asset_id=name,
        )
