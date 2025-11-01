"""Model comparison utilities for comparing Fiddler models.

This module provides ModelComparator for comprehensive model-to-model comparison
across configuration, schema, specs, and assets (segments, custom metrics, alerts, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timezone
import json
import logging
import pandas as pd

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .schema import SchemaValidator, SchemaComparison

logger = logging.getLogger(__name__)


@dataclass
class ValueDifference:
    """Represents a difference between two values.

    Attributes:
        source_value: Value from the source/first model
        target_value: Value from the target/second model
        context: Optional context about the difference (e.g., field name, model IDs)
    """
    source_value: Any
    target_value: Any
    context: Optional[str] = None

    def __str__(self) -> str:
        """String representation showing source vs target."""
        if self.context:
            return f"{self.context}: {self.source_value} → {self.target_value}"
        return f"{self.source_value} → {self.target_value}"


@dataclass
class ConfigurationComparison:
    """Comparison of basic model configuration."""
    task_match: bool = False
    event_id_col_match: bool = False
    event_ts_col_match: bool = False
    task_params_match: bool = False
    differences: Dict[str, ValueDifference] = field(default_factory=dict)

    def has_differences(self) -> bool:
        """Check if there are any configuration differences."""
        return len(self.differences) > 0


@dataclass
class SpecComparison:
    """Comparison of model spec (inputs, outputs, targets, etc.)."""
    inputs_match: bool = False
    outputs_match: bool = False
    targets_match: bool = False
    decisions_match: bool = False
    metadata_match: bool = False
    custom_features_match: bool = False

    only_in_source: Dict[str, List[str]] = field(default_factory=dict)
    only_in_target: Dict[str, List[str]] = field(default_factory=dict)
    in_both: Dict[str, List[str]] = field(default_factory=dict)

    def has_differences(self) -> bool:
        """Check if there are any spec differences."""
        return not all([
            self.inputs_match,
            self.outputs_match,
            self.targets_match,
            self.decisions_match,
            self.metadata_match,
            self.custom_features_match,
        ])


@dataclass
class AssetComparison:
    """Comparison of model assets (segments, custom metrics, alerts, etc.)."""
    asset_type: str  # 'segments', 'custom_metrics', 'alerts', 'baselines', 'charts'
    only_in_source: List[str] = field(default_factory=list)
    only_in_target: List[str] = field(default_factory=list)
    in_both: List[str] = field(default_factory=list)
    definition_differences: Dict[str, ValueDifference] = field(default_factory=dict)

    def has_differences(self) -> bool:
        """Check if there are any asset differences."""
        return (
            len(self.only_in_source) > 0 or
            len(self.only_in_target) > 0 or
            len(self.definition_differences) > 0
        )

    @property
    def total_differences(self) -> int:
        """Total number of differences."""
        return (
            len(self.only_in_source) +
            len(self.only_in_target) +
            len(self.definition_differences)
        )


@dataclass
class ComparisonConfig:
    """Configuration for model comparison operations.

    This replaces multiple boolean parameters with a single configuration object,
    making it easier to manage comparison options and add new options in the future.

    Attributes:
        include_configuration: Compare basic configuration (task, event columns, etc.)
        include_schema: Compare model schemas
        include_spec: Compare model specs (inputs, outputs, targets, etc.)
        include_segments: Compare segments
        include_custom_metrics: Compare custom metrics
        include_alerts: Compare alert rules
        include_baselines: Compare baselines
        include_charts: Compare charts

    Example:
        ```python
        from fiddler_utils import ModelComparator, ComparisonConfig

        # Compare only schema and spec (skip assets)
        config = ComparisonConfig(
            include_configuration=True,
            include_schema=True,
            include_spec=True,
            include_segments=False,
            include_custom_metrics=False,
            include_alerts=False,
            include_baselines=False,
            include_charts=False
        )

        comparator = ModelComparator(source_model, target_model)
        result = comparator.compare_all(config=config)
        ```
    """
    include_configuration: bool = True
    include_schema: bool = True
    include_spec: bool = True
    include_segments: bool = True
    include_custom_metrics: bool = True
    include_alerts: bool = True
    include_baselines: bool = True
    include_charts: bool = True

    @classmethod
    def all(cls) -> 'ComparisonConfig':
        """Create config that includes all comparisons (default behavior).

        Returns:
            ComparisonConfig with all options enabled
        """
        return cls()

    @classmethod
    def schema_only(cls) -> 'ComparisonConfig':
        """Create config for schema-only comparison.

        Returns:
            ComparisonConfig with only schema comparison enabled
        """
        return cls(
            include_configuration=False,
            include_schema=True,
            include_spec=False,
            include_segments=False,
            include_custom_metrics=False,
            include_alerts=False,
            include_baselines=False,
            include_charts=False
        )

    @classmethod
    def no_assets(cls) -> 'ComparisonConfig':
        """Create config that skips asset comparisons (segments, metrics, alerts, etc.).

        Returns:
            ComparisonConfig with model structure only (no assets)
        """
        return cls(
            include_configuration=True,
            include_schema=True,
            include_spec=True,
            include_segments=False,
            include_custom_metrics=False,
            include_alerts=False,
            include_baselines=False,
            include_charts=False
        )


@dataclass
class ComparisonResult:
    """Complete comparison result for two models."""
    source_model_name: str
    target_model_name: str
    compared_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    configuration: Optional[ConfigurationComparison] = None
    schema: Optional[SchemaComparison] = None
    spec: Optional[SpecComparison] = None
    segments: Optional[AssetComparison] = None
    custom_metrics: Optional[AssetComparison] = None
    alerts: Optional[AssetComparison] = None
    baselines: Optional[AssetComparison] = None
    charts: Optional[AssetComparison] = None

    def has_differences(self) -> bool:
        """Check if there are any differences across all comparisons."""
        # Check configuration
        if self.configuration and self.configuration.has_differences():
            return True

        # Check schema (SchemaComparison uses different API)
        if self.schema and (
            len(self.schema.only_in_source) > 0 or
            len(self.schema.only_in_target) > 0 or
            len(self.schema.type_mismatches) > 0
        ):
            return True

        # Check spec
        if self.spec and self.spec.has_differences():
            return True

        # Check assets
        for asset in [self.segments, self.custom_metrics, self.alerts, self.baselines, self.charts]:
            if asset and asset.has_differences():
                return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of differences."""
        differences_by_category: Dict[str, int] = {}

        if self.configuration:
            differences_by_category['configuration'] = len(self.configuration.differences)

        if self.schema:
            differences_by_category['schema'] = (
                len(self.schema.only_in_source) +
                len(self.schema.only_in_target) +
                len(self.schema.type_mismatches)
            )

        if self.spec:
            differences_by_category['spec'] = sum(
                len(items) for items in self.spec.only_in_source.values()
            ) + sum(
                len(items) for items in self.spec.only_in_target.values()
            )

        for asset_type, asset_comp in [
            ('segments', self.segments),
            ('custom_metrics', self.custom_metrics),
            ('alerts', self.alerts),
            ('baselines', self.baselines),
            ('charts', self.charts),
        ]:
            if asset_comp:
                differences_by_category[asset_type] = asset_comp.total_differences

        summary = {
            'source_model': self.source_model_name,
            'target_model': self.target_model_name,
            'compared_at': self.compared_at,
            'has_differences': self.has_differences(),
            'differences_by_category': differences_by_category,
        }

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Convert to JSON string or save to file.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=indent, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f'Saved comparison result to {filepath}')

        return json_str

    def to_markdown(self) -> str:
        """Generate formatted Markdown report.

        Returns:
            Markdown formatted comparison report
        """
        lines = []
        lines.append("# Model Comparison Report")
        lines.append("")
        lines.append(f"**Model A:** {self.source_model_name}")
        lines.append(f"**Model B:** {self.target_model_name}")
        lines.append(f"**Compared at:** {self.compared_at}")
        lines.append("")

        summary = self.get_summary()
        if summary['has_differences']:
            lines.append("⚠️ **Differences found**")
        else:
            lines.append("✅ **Models are identical**")
        lines.append("")

        # Configuration comparison
        if self.configuration and self.configuration.has_differences():
            lines.append("## ⚙️ Configuration Differences")
            lines.append("")
            for key, diff in self.configuration.differences.items():
                lines.append(f"- **{key}**")
                lines.append(f"  - Model A: `{diff.source_value}`")
                lines.append(f"  - Model B: `{diff.target_value}`")
            lines.append("")

        # Schema comparison
        if self.schema and (
            len(self.schema.only_in_source) > 0 or
            len(self.schema.only_in_target) > 0 or
            len(self.schema.type_mismatches) > 0
        ):
            lines.append("## 📋 Schema Differences")
            lines.append("")
            if self.schema.only_in_source:
                lines.append(f"**Only in Model A:** {', '.join(sorted(self.schema.only_in_source))}")
            if self.schema.only_in_target:
                lines.append(f"**Only in Model B:** {', '.join(sorted(self.schema.only_in_target))}")
            if self.schema.type_mismatches:
                lines.append("**Type mismatches:**")
                for col, (type_a, type_b) in self.schema.type_mismatches.items():
                    lines.append(f"  - {col}: `{type_a}` vs `{type_b}`")
            lines.append("")

        # Spec comparison
        if self.spec and self.spec.has_differences():
            lines.append("## 🔧 Spec Differences")
            lines.append("")
            for spec_type in ['inputs', 'outputs', 'targets', 'decisions', 'metadata', 'custom_features']:
                only_source = self.spec.only_in_source.get(spec_type, [])
                only_target = self.spec.only_in_target.get(spec_type, [])
                if only_source or only_target:
                    lines.append(f"**{spec_type.capitalize()}:**")
                    if only_source:
                        lines.append(f"  - Only in Model A: {', '.join(sorted(only_source))}")
                    if only_target:
                        lines.append(f"  - Only in Model B: {', '.join(sorted(only_target))}")
            lines.append("")

        # Asset comparisons
        for asset_name, asset_comp, emoji in [
            ('Segments', self.segments, '🔍'),
            ('Custom Metrics', self.custom_metrics, '📊'),
            ('Alerts', self.alerts, '🔔'),
            ('Baselines', self.baselines, '📈'),
            ('Charts', self.charts, '📉'),
        ]:
            if asset_comp and asset_comp.has_differences():
                lines.append(f"## {emoji} {asset_name} Differences")
                lines.append("")
                if asset_comp.only_in_source:
                    lines.append(f"**Only in Model A:** {', '.join(sorted(asset_comp.only_in_source))}")
                if asset_comp.only_in_target:
                    lines.append(f"**Only in Model B:** {', '.join(sorted(asset_comp.only_in_target))}")
                if asset_comp.definition_differences:
                    lines.append("**Definition differences:**")
                    for name, diff in asset_comp.definition_differences.items():
                        lines.append(f"  - **{name}**")
                        lines.append(f"    - Model A: `{diff.source_value}`")
                        lines.append(f"    - Model B: `{diff.target_value}`")
                lines.append("")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per difference.

        Returns:
            Pandas DataFrame with difference details
        """
        rows = []

        # Configuration differences
        if self.configuration:
            for key, diff in self.configuration.differences.items():
                rows.append({
                    'category': 'configuration',
                    'item': key,
                    'difference_type': 'value_mismatch',
                    'source_model_value': str(diff.source_value),
                    'target_model_value': str(diff.target_value),
                })

        # Schema differences
        if self.schema:
            for col in self.schema.only_in_source:
                rows.append({
                    'category': 'schema',
                    'item': col,
                    'difference_type': 'only_in_source_model',
                    'source_model_value': 'present',
                    'target_model_value': 'missing',
                })
            for col in self.schema.only_in_target:
                rows.append({
                    'category': 'schema',
                    'item': col,
                    'difference_type': 'only_in_target_model',
                    'source_model_value': 'missing',
                    'target_model_value': 'present',
                })
            for col, (type_a, type_b) in self.schema.type_mismatches.items():
                rows.append({
                    'category': 'schema',
                    'item': col,
                    'difference_type': 'type_mismatch',
                    'source_model_value': str(type_a),
                    'target_model_value': str(type_b),
                })

        # Spec differences
        if self.spec:
            for spec_type in ['inputs', 'outputs', 'targets', 'decisions', 'metadata', 'custom_features']:
                for item in self.spec.only_in_source.get(spec_type, []):
                    rows.append({
                        'category': f'spec_{spec_type}',
                        'item': item,
                        'difference_type': 'only_in_source',
                        'source_model_value': 'present',
                        'target_model_value': 'missing',
                    })
                for item in self.spec.only_in_target.get(spec_type, []):
                    rows.append({
                        'category': f'spec_{spec_type}',
                        'item': item,
                        'difference_type': 'only_in_target',
                        'source_model_value': 'missing',
                        'target_model_value': 'present',
                    })

        # Asset differences
        for asset_name, asset_comp in [
            ('segments', self.segments),
            ('custom_metrics', self.custom_metrics),
            ('alerts', self.alerts),
            ('baselines', self.baselines),
            ('charts', self.charts),
        ]:
            if asset_comp:
                for item in asset_comp.only_in_source:
                    rows.append({
                        'category': asset_name,
                        'item': item,
                        'difference_type': 'only_in_source',
                        'source_model_value': 'present',
                        'target_model_value': 'missing',
                    })
                for item in asset_comp.only_in_target:
                    rows.append({
                        'category': asset_name,
                        'item': item,
                        'difference_type': 'only_in_target',
                        'source_model_value': 'missing',
                        'target_model_value': 'present',
                    })
                for name, diff in asset_comp.definition_differences.items():
                    rows.append({
                        'category': asset_name,
                        'item': name,
                        'difference_type': 'definition_mismatch',
                        'source_model_value': str(diff.source_value),
                        'target_model_value': str(diff.target_value),
                    })

        return pd.DataFrame(rows)


class ModelComparator:
    """Compare two Fiddler models across multiple dimensions.

    This class orchestrates model comparison using existing fiddler_utils
    components (SchemaValidator, asset managers) and provides structured output.

    Example:
        ```python
        from fiddler_utils import ModelComparator, connection_context

        # Fetch models (potentially from different instances)
        with connection_context(URL_A, TOKEN_A):
            source_model = fdl.Model.from_name(project_id=proj_a.id, name='model_v1')

        with connection_context(URL_B, TOKEN_B):
            target_model = fdl.Model.from_name(project_id=proj_b.id, name='model_v2')

        # Compare models
        comparator = ModelComparator(source_model, target_model)
        result = comparator.compare_all()

        # Display and export
        print(result.to_markdown())
        result.to_json('comparison_result.json')
        result.to_dataframe().to_csv('differences.csv', index=False)
        ```
    """

    def __init__(self, source_model: fdl.Model, target_model: fdl.Model):
        """Initialize comparator with two models.

        Args:
            source_model: First model to compare
            target_model: Second model to compare
        """
        self.source_model = source_model
        self.target_model = target_model
        self.result = ComparisonResult(
            source_model_name=f"{source_model.name}",
            target_model_name=f"{target_model.name}",
        )

    def compare_configuration(self) -> ConfigurationComparison:
        """Compare basic model configuration.

        Returns:
            ConfigurationComparison with differences
        """
        comp = ConfigurationComparison()

        # Compare task
        task_a = getattr(self.source_model, 'task', None)
        task_b = getattr(self.target_model, 'task', None)
        comp.task_match = task_a == task_b
        if not comp.task_match:
            comp.differences['task'] = ValueDifference(task_a, task_b, 'task')

        # Compare event ID column
        event_id_a = getattr(self.source_model, 'event_id_col', None)
        event_id_b = getattr(self.target_model, 'event_id_col', None)
        comp.event_id_col_match = event_id_a == event_id_b
        if not comp.event_id_col_match:
            comp.differences['event_id_col'] = ValueDifference(event_id_a, event_id_b, 'event_id_col')

        # Compare event timestamp column
        event_ts_a = getattr(self.source_model, 'event_ts_col', None)
        event_ts_b = getattr(self.target_model, 'event_ts_col', None)
        comp.event_ts_col_match = event_ts_a == event_ts_b
        if not comp.event_ts_col_match:
            comp.differences['event_ts_col'] = ValueDifference(event_ts_a, event_ts_b, 'event_ts_col')

        # Compare task parameters (if present)
        task_params_a = getattr(self.source_model, 'task_params', None)
        task_params_b = getattr(self.target_model, 'task_params', None)
        comp.task_params_match = task_params_a == task_params_b
        if not comp.task_params_match:
            comp.differences['task_params'] = ValueDifference(task_params_a, task_params_b, 'task_params')

        logger.info(f"[ModelComparator] Configuration comparison: {len(comp.differences)} differences")
        return comp

    def compare_schemas(self) -> SchemaComparison:
        """Compare model schemas using SchemaValidator.

        Returns:
            SchemaComparison from fiddler_utils.schema
        """
        comp = SchemaValidator.compare_schemas(
            source_model=self.source_model,
            target_model=self.target_model,
            strict=True
        )

        logger.info(f"[ModelComparator] Schema comparison: {len(comp.only_in_source) + len(comp.only_in_target) + len(comp.type_mismatches)} differences")
        return comp

    def compare_specs(self) -> SpecComparison:
        """Compare model specs (inputs, outputs, targets, etc.).

        Returns:
            SpecComparison with differences
        """
        comp = SpecComparison()

        spec_a = self.source_model.spec
        spec_b = self.target_model.spec

        # Compare each spec attribute
        for attr in ['inputs', 'outputs', 'targets', 'decisions', 'metadata']:
            attr_a = getattr(spec_a, attr, None)
            attr_b = getattr(spec_b, attr, None)

            # Convert None to empty list for comparison
            list_a = [] if attr_a is None else list(attr_a)
            list_b = [] if attr_b is None else list(attr_b)

            set_a = set(list_a)
            set_b = set(list_b)

            match = set_a == set_b
            setattr(comp, f'{attr}_match', match)

            if not match:
                only_a = sorted(set_a - set_b)
                only_b = sorted(set_b - set_a)
                both = sorted(set_a & set_b)

                if only_a:
                    comp.only_in_source[attr] = only_a
                if only_b:
                    comp.only_in_target[attr] = only_b
                comp.in_both[attr] = both

        # Compare custom features using SchemaValidator utility
        custom_a = getattr(spec_a, 'custom_features', None)
        custom_b = getattr(spec_b, 'custom_features', None)

        # Extract names using shared utility (handles multiple formats)
        names_a = SchemaValidator.extract_custom_feature_names(custom_a)
        names_b = SchemaValidator.extract_custom_feature_names(custom_b)

        comp.custom_features_match = names_a == names_b
        if not comp.custom_features_match:
            only_a = sorted(names_a - names_b)
            only_b = sorted(names_b - names_a)
            both = sorted(names_a & names_b)

            if only_a:
                comp.only_in_source['custom_features'] = only_a
            if only_b:
                comp.only_in_target['custom_features'] = only_b
            comp.in_both['custom_features'] = both

        logger.info(f"[ModelComparator] Spec comparison: {len(comp.only_in_source) + len(comp.only_in_target)} differences")
        return comp

    @staticmethod
    def _get_asset_key(asset: Any) -> str:
        """Extract consistent comparison key from asset.

        This standardizes asset key extraction across all asset types,
        handling edge cases like missing names or non-string names.

        Args:
            asset: Asset object (Segment, CustomMetric, AlertRule, etc.)

        Returns:
            String key for comparison (typically the name)

        Raises:
            ValueError: If asset has no usable key

        Example:
            ```python
            key = ModelComparator._get_asset_key(segment)
            # Returns: "high_value_customers"
            ```
        """
        # Try to get name attribute
        if hasattr(asset, 'name'):
            name = asset.name
            if name is not None:
                return str(name)  # Ensure it's a string

        # Fallback to id if name not available
        if hasattr(asset, 'id'):
            asset_id = asset.id
            if asset_id is not None:
                logger.warning(
                    f"[ModelComparator] Asset has no name, using ID as key: {asset_id}"
                )
                return str(asset_id)

        # If neither name nor id available, raise error
        raise ValueError(
            f"Asset has no usable key (no 'name' or 'id' attribute): {type(asset).__name__}"
        )

    def _compare_assets_generic(
        self,
        asset_type: str,
        list_class: Any,
        value_extractor: Callable[[Any], str]
    ) -> AssetComparison:
        """Generic asset comparison helper to reduce code duplication.

        Args:
            asset_type: Asset type name (e.g., 'segments', 'custom_metrics')
            list_class: Fiddler class with .list() method (e.g., fdl.Segment)
            value_extractor: Function to extract comparable value from asset
                           Signature: (asset) -> str

        Returns:
            AssetComparison with differences
        """
        comp = AssetComparison(asset_type=asset_type)

        try:
            # Fetch assets for both models
            assets_a = list(list_class.list(model_id=self.source_model.id))
            assets_b = list(list_class.list(model_id=self.target_model.id))

            # Create dictionaries keyed by name (using standardized key extraction)
            dict_a = {self._get_asset_key(asset): value_extractor(asset) for asset in assets_a}
            dict_b = {self._get_asset_key(asset): value_extractor(asset) for asset in assets_b}

            # Find differences
            names_a = set(dict_a.keys())
            names_b = set(dict_b.keys())

            comp.only_in_source = sorted(names_a - names_b)
            comp.only_in_target = sorted(names_b - names_a)
            comp.in_both = sorted(names_a & names_b)

            # Check value differences for assets in both
            for name in comp.in_both:
                if dict_a[name] != dict_b[name]:
                    comp.definition_differences[name] = ValueDifference(
                        dict_a[name], dict_b[name], f"{asset_type[:-1]} '{name}'"
                    )

            logger.info(f"[ModelComparator] {asset_type.capitalize()} comparison: {comp.total_differences} differences")
        except Exception as e:
            logger.warning(f"[ModelComparator] Error comparing {asset_type}: {e}")

        return comp

    def compare_segments(self) -> AssetComparison:
        """Compare model segments.

        Returns:
            AssetComparison for segments
        """
        return self._compare_assets_generic(
            asset_type='segments',
            list_class=fdl.Segment,
            value_extractor=lambda seg: seg.definition
        )

    def compare_custom_metrics(self) -> AssetComparison:
        """Compare model custom metrics.

        Returns:
            AssetComparison for custom metrics
        """
        return self._compare_assets_generic(
            asset_type='custom_metrics',
            list_class=fdl.CustomMetric,
            value_extractor=lambda metric: metric.definition
        )

    def compare_alerts(self) -> AssetComparison:
        """Compare model alerts.

        Returns:
            AssetComparison for alerts
        """
        # For alerts, compare configuration as string representation
        def alert_value(alert: Any) -> str:
            return str((alert.name, alert.metric_id, alert.condition, alert.priority))

        return self._compare_assets_generic(
            asset_type='alerts',
            list_class=fdl.AlertRule,
            value_extractor=alert_value
        )

    def compare_baselines(self) -> AssetComparison:
        """Compare model baselines.

        Returns:
            AssetComparison for baselines
        """
        return self._compare_assets_generic(
            asset_type='baselines',
            list_class=fdl.Baseline,
            value_extractor=lambda bl: str(bl.type)
        )

    def compare_charts(self) -> AssetComparison:
        """Compare model charts.

        Note: Charts are handled differently than other assets because
        the Fiddler SDK doesn't expose a Chart class with a .list() method.
        Charts are project-level assets, not model-level, so comparison
        is limited.

        Returns:
            AssetComparison for charts
        """
        comp = AssetComparison(asset_type='charts')

        try:
            # Charts are project-level, not model-level assets
            # We can only check if both models exist in the same project
            if self.source_model.project_id != self.target_model.project_id:
                logger.warning(
                    "[ModelComparator] Charts comparison skipped: "
                    "models are in different projects"
                )
            else:
                logger.info(
                    "[ModelComparator] Charts are project-level assets. "
                    "Use ChartManager for detailed chart comparison."
                )
        except Exception as e:
            logger.warning(f"[ModelComparator] Error comparing charts: {e}")

        return comp

    def compare_all(
        self,
        config: Optional[ComparisonConfig] = None,
        # Backward compatibility: individual boolean parameters (deprecated)
        include_configuration: Optional[bool] = None,
        include_schema: Optional[bool] = None,
        include_spec: Optional[bool] = None,
        include_segments: Optional[bool] = None,
        include_custom_metrics: Optional[bool] = None,
        include_alerts: Optional[bool] = None,
        include_baselines: Optional[bool] = None,
        include_charts: Optional[bool] = None,
    ) -> ComparisonResult:
        """Run all comparisons and return comprehensive result.

        Args:
            config: ComparisonConfig object specifying what to compare.
                   If provided, individual boolean parameters are ignored.
            include_configuration: (Deprecated) Compare basic configuration.
                                 Use config parameter instead.
            include_schema: (Deprecated) Compare schemas. Use config parameter instead.
            include_spec: (Deprecated) Compare specs. Use config parameter instead.
            include_segments: (Deprecated) Compare segments. Use config parameter instead.
            include_custom_metrics: (Deprecated) Compare custom metrics. Use config parameter instead.
            include_alerts: (Deprecated) Compare alerts. Use config parameter instead.
            include_baselines: (Deprecated) Compare baselines. Use config parameter instead.
            include_charts: (Deprecated) Compare charts. Use config parameter instead.

        Returns:
            ComparisonResult with all requested comparisons

        Example:
            ```python
            # Recommended: Use ComparisonConfig
            config = ComparisonConfig.no_assets()
            result = comparator.compare_all(config=config)

            # Backward compatible: Individual booleans still work
            result = comparator.compare_all(
                include_configuration=True,
                include_schema=True,
                include_segments=False
            )
            ```
        """
        # Use config if provided, otherwise create from individual parameters
        if config is None:
            # Backward compatibility: create config from individual boolean parameters
            config = ComparisonConfig(
                include_configuration=include_configuration if include_configuration is not None else True,
                include_schema=include_schema if include_schema is not None else True,
                include_spec=include_spec if include_spec is not None else True,
                include_segments=include_segments if include_segments is not None else True,
                include_custom_metrics=include_custom_metrics if include_custom_metrics is not None else True,
                include_alerts=include_alerts if include_alerts is not None else True,
                include_baselines=include_baselines if include_baselines is not None else True,
                include_charts=include_charts if include_charts is not None else True,
            )

        logger.info(f"[ModelComparator] Starting comprehensive comparison: '{self.source_model.name}' vs '{self.target_model.name}'")

        if config.include_configuration:
            self.result.configuration = self.compare_configuration()

        if config.include_schema:
            self.result.schema = self.compare_schemas()

        if config.include_spec:
            self.result.spec = self.compare_specs()

        if config.include_segments:
            self.result.segments = self.compare_segments()

        if config.include_custom_metrics:
            self.result.custom_metrics = self.compare_custom_metrics()

        if config.include_alerts:
            self.result.alerts = self.compare_alerts()

        if config.include_baselines:
            self.result.baselines = self.compare_baselines()

        if config.include_charts:
            self.result.charts = self.compare_charts()

        logger.info(f"[ModelComparator] Comparison complete: {'differences found' if self.result.has_differences() else 'models are identical'}")
        return self.result
