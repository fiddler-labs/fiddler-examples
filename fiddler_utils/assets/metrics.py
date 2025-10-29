"""Custom metric management utilities.

This module provides the CustomMetricManager class for working with Fiddler custom metrics.
"""

from typing import List, Dict, Set, Any
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


class CustomMetricManager(BaseAssetManager[fdl.CustomMetric]):
    """Manager for Fiddler custom metrics.

    Custom metrics define business-specific KPIs using FQL expressions.
    This manager handles CRUD operations, export/import, and validation.

    Example:
        ```python
        from fiddler_utils.assets import CustomMetricManager
        import fiddler as fdl

        mgr = CustomMetricManager()

        # Export custom metrics from source model
        exported = mgr.export_assets(
            model_id=source_model.id,
            names=['revenue_lost', 'accuracy_by_segment']
        )

        # Import to target model with validation
        result = mgr.import_assets(
            target_model_id=target_model.id,
            assets=exported,
            validate=True
        )

        print(f"Imported {result.successful} custom metrics")
        if result.skipped > 0:
            print(f"Skipped {result.skipped} due to validation errors")
        ```
    """

    def _get_asset_type(self) -> AssetType:
        """Get asset type."""
        return AssetType.CUSTOM_METRIC

    def _list_assets(self, model_id: str) -> List[fdl.CustomMetric]:
        """List all custom metrics for a model."""
        return list(fdl.CustomMetric.list(model_id=model_id))

    def _get_asset_name(self, asset: fdl.CustomMetric) -> str:
        """Get custom metric name."""
        return asset.name

    def _extract_referenced_columns(self, asset: fdl.CustomMetric) -> Set[str]:
        """Extract column references from metric definition.

        Args:
            asset: CustomMetric object

        Returns:
            Set of column names referenced in the FQL definition
        """
        return fql.extract_columns(asset.definition)

    def _extract_asset_data(self, asset: fdl.CustomMetric) -> Dict[str, Any]:
        """Extract custom metric data for export.

        Args:
            asset: CustomMetric object

        Returns:
            Dictionary with custom metric data
        """
        return {
            'name': asset.name,
            'description': asset.description if hasattr(asset, 'description') else '',
            'definition': asset.definition,
            # Store metadata for reference
            'metadata': {
                'id': asset.id,
                'model_id': asset.model_id,
                # Check if this is an aggregation metric
                'is_aggregation': not fql.is_simple_filter(asset.definition),
                'functions_used': list(fql.get_fql_functions(asset.definition)),
            },
        }

    def _create_asset(
        self, model_id: str, asset_data: Dict[str, Any]
    ) -> fdl.CustomMetric:
        """Create a custom metric from data.

        Args:
            model_id: Target model ID
            asset_data: Custom metric data dictionary

        Returns:
            Created CustomMetric object
        """
        metric = fdl.CustomMetric(
            model_id=model_id,
            name=asset_data['name'],
            description=asset_data.get('description', ''),
            definition=asset_data['definition'],
        )
        metric.create()
        logger.info(f'Created custom metric: {metric.name} (ID: {metric.id})')
        return metric

    def get_metric_by_name(self, model_id: str, name: str) -> fdl.CustomMetric:
        """Get a specific custom metric by name.

        Args:
            model_id: Model ID
            name: Custom metric name

        Returns:
            CustomMetric object

        Raises:
            AssetNotFoundError: If custom metric not found

        Example:
            ```python
            mgr = CustomMetricManager()
            metric = mgr.get_metric_by_name(model.id, 'revenue_lost')
            print(metric.definition)
            ```
        """
        from ..exceptions import AssetNotFoundError

        metrics = self._list_assets(model_id)
        for metric in metrics:
            if metric.name == name:
                return metric

        raise AssetNotFoundError(
            f"Custom metric '{name}' not found in model {model_id}",
            asset_type='CustomMetric',
            asset_id=name,
        )

    def analyze_metric_complexity(self, model_id: str) -> Dict[str, List[str]]:
        """Analyze complexity of custom metrics in a model.

        Categorizes metrics by complexity level based on FQL functions used.

        Args:
            model_id: Model ID

        Returns:
            Dictionary mapping complexity level to list of metric names

        Example:
            ```python
            mgr = CustomMetricManager()
            complexity = mgr.analyze_metric_complexity(model.id)

            print(f"Simple metrics: {complexity['simple']}")
            print(f"Aggregation metrics: {complexity['aggregation']}")
            print(f"Complex metrics: {complexity['complex']}")
            ```
        """
        metrics = self._list_assets(model_id)

        categorized = {
            'simple': [],  # No functions
            'aggregation': [],  # Basic aggregation (sum, avg, count)
            'complex': [],  # Multiple functions or nested
        }

        for metric in metrics:
            functions = fql.get_fql_functions(metric.definition)

            if not functions:
                categorized['simple'].append(metric.name)
            elif len(functions) == 1:
                categorized['aggregation'].append(metric.name)
            else:
                categorized['complex'].append(metric.name)

        logger.info(
            f'Complexity analysis for model {model_id}: '
            f'{len(categorized["simple"])} simple, '
            f'{len(categorized["aggregation"])} aggregation, '
            f'{len(categorized["complex"])} complex'
        )

        return categorized

    def find_metrics_using_column(
        self, model_id: str, column_name: str
    ) -> List[fdl.CustomMetric]:
        """Find all custom metrics that reference a specific column.

        Useful for impact analysis when changing or removing columns.

        Args:
            model_id: Model ID
            column_name: Column name to search for

        Returns:
            List of CustomMetric objects that reference the column

        Example:
            ```python
            mgr = CustomMetricManager()
            metrics = mgr.find_metrics_using_column(model.id, 'transaction_value')

            print(f"Found {len(metrics)} metrics using 'transaction_value':")
            for metric in metrics:
                print(f"  - {metric.name}")
            ```
        """
        metrics = self._list_assets(model_id)
        using_column = []

        for metric in metrics:
            columns = self._extract_referenced_columns(metric)
            if column_name in columns:
                using_column.append(metric)

        logger.info(
            f"Found {len(using_column)} metrics using column '{column_name}' "
            f'in model {model_id}'
        )

        return using_column

    def validate_metric_definition(
        self, definition: str, model: fdl.Model
    ) -> tuple[bool, str]:
        """Validate a custom metric definition.

        Checks both FQL syntax and column references.

        Args:
            definition: FQL metric definition
            model: Fiddler model to validate against

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string

        Example:
            ```python
            mgr = CustomMetricManager()
            definition = 'sum(if(fp(), 1, 0)) / sum(1)'

            is_valid, error = mgr.validate_metric_definition(definition, model)
            if not is_valid:
                print(f"Invalid definition: {error}")
            ```
        """
        # Check syntax
        is_valid, error = fql.validate_fql_syntax(definition)
        if not is_valid:
            return False, f'Syntax error: {error}'

        # Check columns
        from ..schema import SchemaValidator

        is_valid, missing = SchemaValidator.validate_fql_expression(
            definition, model, strict=False
        )

        if not is_valid:
            return False, f'Missing columns: {missing}'

        return True, ''

    def update_metric_definition(
        self, metric: fdl.CustomMetric, new_definition: str, validate: bool = True
    ) -> fdl.CustomMetric:
        """Update a custom metric's FQL definition.

        Note: This requires deleting and recreating the metric,
        as the Fiddler SDK doesn't support in-place updates.

        Args:
            metric: Existing custom metric
            new_definition: New FQL definition
            validate: If True, validate new definition before updating

        Returns:
            Updated CustomMetric object

        Example:
            ```python
            mgr = CustomMetricManager()
            metric = mgr.get_metric_by_name(model.id, 'my_metric')

            # Update definition
            updated = mgr.update_metric_definition(
                metric,
                new_definition='sum(if(tp(), 1, 0)) / sum(1)'
            )
            ```
        """
        if validate:
            model = fdl.Model.get(id_=metric.model_id)
            is_valid, error = self.validate_metric_definition(new_definition, model)
            if not is_valid:
                from ..exceptions import FQLError

                raise FQLError(
                    f'Invalid metric definition: {error}', expression=new_definition
                )

        # Delete old metric
        old_id = metric.id
        old_name = metric.name
        old_description = metric.description if hasattr(metric, 'description') else ''

        logger.info(f'Deleting custom metric {old_name} (ID: {old_id}) for update')
        metric.delete()

        # Create new metric with updated definition
        new_metric = fdl.CustomMetric(
            model_id=metric.model_id,
            name=old_name,
            description=old_description,
            definition=new_definition,
        )
        new_metric.create()

        logger.info(
            f'Recreated custom metric {new_metric.name} (ID: {new_metric.id}) '
            f'with updated definition'
        )

        return new_metric
