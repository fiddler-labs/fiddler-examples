"""Model management utilities for Fiddler models.

This module provides ModelManager for exporting and importing complete model
definitions using the Model constructor pattern (not from_data()), enabling
deterministic, programmatic model recreation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from uuid import UUID
import logging
import json
from datetime import datetime

try:
    import fiddler as fdl
    from fiddler.libs.http_client import RequestClient
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from ..exceptions import (
    ValidationError,
    AssetImportError,
    FiddlerUtilsError,
)
from .. import connection

logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles UUID objects.

    Converts UUID objects to strings during JSON serialization.
    This is the standard approach recommended by Python community.
    """
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


@dataclass
class ColumnExportData:
    """Schema definition for a single column.

    Attributes:
        name: Column name
        data_type: Fiddler DataType as string (INTEGER, FLOAT, STRING, CATEGORY, BOOLEAN, etc.)
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        categories: List of valid categories for CATEGORY type
    """
    name: str
    data_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None


@dataclass
class ModelExportData:
    """Complete model export package.

    Contains all information needed to recreate a model using the Model constructor,
    including schema, spec, task configuration, and optionally related assets.

    Attributes:
        name: Model name
        version: Model version label (optional)
        columns: List of column definitions with data types, ranges, categories
        spec: Model spec with role assignments (inputs, outputs, targets, metadata, decisions)
        custom_features: List of custom features for LLM models (enrichments)
        task: ModelTask enum as string
        task_params: Task-specific parameters (target_class_order, binary_threshold, etc.)
        event_id_col: Event ID column name
        event_ts_col: Event timestamp column name
        baselines: List of baseline definitions
        has_artifacts: Warning flag if model has uploaded artifacts
        related_assets: Dict of related assets (segments, custom_metrics, alerts)
        source_project_id: Source project ID (reference only)
        source_model_id: Source model ID (reference only)
        exported_at: Export timestamp
    """
    name: str
    version: Optional[str]
    columns: List[ColumnExportData]
    spec: Dict[str, List[str]]  # inputs, outputs, targets, metadata, decisions
    custom_features: List[Dict[str, Any]]
    task: str
    task_params: Optional[Dict[str, Any]]
    event_id_col: Optional[str]
    event_ts_col: Optional[str]
    baselines: List[Dict[str, Any]]
    has_artifacts: bool
    related_assets: Dict[str, List[Dict]]
    source_project_id: str
    source_model_id: str
    exported_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Note: The returned dict may contain UUID objects. Use to_json()
        for JSON serialization, which handles UUIDs automatically.
        """
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelExportData':
        """Create from dictionary (JSON deserialization)."""
        # Convert column dicts to ColumnExportData objects
        columns = [ColumnExportData(**col) for col in data['columns']]
        data['columns'] = columns
        return cls(**data)

    def to_json(self, filepath: str, indent: int = 2):
        """Save to JSON file with proper UUID handling.

        Uses custom UUIDEncoder to convert UUID objects to strings.
        This is the standard approach for handling UUIDs in JSON.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent, cls=UUIDEncoder)

    @classmethod
    def from_json(cls, filepath: str) -> 'ModelExportData':
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ModelManager:
    """Manager for exporting and importing Fiddler models.

    This class provides utilities to export complete model definitions (schema, spec,
    task configuration, baselines, and related assets) and import them to other projects
    or instances using the Model constructor pattern.

    Example:
        ```python
        from fiddler_utils import ModelManager

        # Export model with all related assets
        mgr = ModelManager()
        export_data = mgr.export_model(
            model_id=source_model.id,
            include_baselines=True,
            include_related_assets=True
        )

        # Save to file
        export_data.to_json('model_export.json')

        # Import to different project
        imported_model = mgr.import_model(
            target_project_id=target_project.id,
            model_data=export_data,
            import_mode='create_new'
        )
        ```
    """

    def __init__(self):
        """Initialize ModelManager."""
        self.baseline_manager = None  # Will be created when needed

    def _get_baseline_manager(self):
        """Lazy load BaselineManager to avoid circular imports."""
        if self.baseline_manager is None:
            from .baselines import BaselineManager
            self.baseline_manager = BaselineManager()
        return self.baseline_manager

    def export_model(
        self,
        model_id: str,
        include_baselines: bool = True,
        include_related_assets: bool = True,
    ) -> ModelExportData:
        """Export complete model definition.

        Args:
            model_id: Model UUID to export
            include_baselines: Include baseline definitions
            include_related_assets: Include segments, custom metrics, and alerts

        Returns:
            ModelExportData with complete model definition

        Example:
            ```python
            export_data = mgr.export_model(
                model_id='abc123',
                include_baselines=True,
                include_related_assets=True
            )

            # Save to file
            export_data.to_json('my_model.json')
            ```
        """
        logger.info(f'Exporting model {model_id}')

        # Get model
        model = fdl.Model.get(id_=model_id)
        logger.info(f"Retrieved model '{model.name}' (version: {model.version or 'none'})")

        # Extract schema columns
        columns = self._extract_columns(model)
        logger.debug(f'Extracted {len(columns)} columns from schema')

        # Extract spec
        spec_data = self._extract_spec(model.spec)
        logger.debug(f"Extracted spec: {len(spec_data.get('inputs', []))} inputs, "
                    f"{len(spec_data.get('outputs', []))} outputs, "
                    f"{len(spec_data.get('targets', []))} targets")

        # Extract custom features (LLM enrichments)
        custom_features = self._extract_custom_features(model.spec)
        if custom_features:
            logger.debug(f'Extracted {len(custom_features)} custom features')

        # Extract task configuration
        task_params = self._extract_task_params(model.task_params) if model.task_params else None

        # Export baselines
        baselines = []
        if include_baselines:
            baseline_mgr = self._get_baseline_manager()
            baselines = baseline_mgr.export_baselines(model_id)
            logger.info(f'Exported {len(baselines)} baselines')

        # Detect artifacts
        has_artifacts = self._detect_artifacts(model)
        if has_artifacts:
            logger.warning(f"Model '{model.name}' has uploaded artifacts. "
                          "Artifacts cannot be exported and must be re-uploaded after import.")

        # Export related assets
        related_assets = {}
        if include_related_assets:
            related_assets = self._export_related_assets(model_id)

        # Create export data
        export_data = ModelExportData(
            name=model.name,
            version=model.version,
            columns=columns,
            spec=spec_data,
            custom_features=custom_features,
            task=model.task.name if hasattr(model.task, 'name') else str(model.task),  # Get enum name
            task_params=task_params,
            event_id_col=model.event_id_col,
            event_ts_col=model.event_ts_col,
            baselines=baselines,
            has_artifacts=has_artifacts,
            related_assets=related_assets,
            source_project_id=str(model.project_id),  # Convert UUID to str (field type is str)
            source_model_id=str(model.id),  # Convert UUID to str (field type is str)
            exported_at=datetime.utcnow().isoformat()
        )

        logger.info(f"Successfully exported model '{model.name}'")
        return export_data

    def _extract_columns(self, model: fdl.Model) -> List[ColumnExportData]:
        """Extract schema columns as ColumnExportData list.

        Args:
            model: Fiddler Model object

        Returns:
            List of ColumnExportData with full column definitions
        """
        columns = []
        schema_columns = getattr(model.schema, 'columns', [])

        for col in schema_columns:
            col_data = ColumnExportData(
                name=col.name,
                data_type=str(col.data_type).replace('DataType.', ''),  # Remove prefix
                min_value=getattr(col, 'min', None),
                max_value=getattr(col, 'max', None),
                categories=getattr(col, 'categories', None)
            )
            columns.append(col_data)

        return columns

    def _extract_spec(self, spec: fdl.ModelSpec) -> Dict[str, List[str]]:
        """Extract model spec as dictionary.

        Args:
            spec: Fiddler ModelSpec object

        Returns:
            Dict with inputs, outputs, targets, metadata, decisions lists
        """
        return {
            'inputs': list(spec.inputs or []),
            'outputs': list(spec.outputs or []),
            'targets': list(spec.targets or []),
            'metadata': list(spec.metadata or []),
            'decisions': list(spec.decisions or []) if hasattr(spec, 'decisions') and spec.decisions else [],
        }

    def _extract_custom_features(self, spec: fdl.ModelSpec) -> List[Dict[str, Any]]:
        """Extract custom features (LLM enrichments) from spec.

        Args:
            spec: Fiddler ModelSpec object

        Returns:
            List of custom feature definitions as dicts
        """
        if not hasattr(spec, 'custom_features') or not spec.custom_features:
            return []

        custom_features_data = []
        for feature in spec.custom_features:
            if isinstance(feature, fdl.TextEmbedding):
                custom_features_data.append({
                    'type': 'TextEmbedding',
                    'name': feature.name,
                    'source_column': feature.source_column,
                    'column': feature.column,
                    'n_tags': getattr(feature, 'n_tags', 10)
                })
            elif isinstance(feature, fdl.Enrichment):
                custom_features_data.append({
                    'type': 'Enrichment',
                    'name': feature.name,
                    'enrichment': feature.enrichment,
                    'columns': list(feature.columns),
                    'config': feature.config if hasattr(feature, 'config') else {},
                    'allow_list': getattr(feature, 'allow_list', None)
                })
            else:
                logger.warning(f'Unknown custom feature type: {type(feature)}')

        return custom_features_data

    def _extract_task_params(self, task_params: fdl.ModelTaskParams) -> Dict[str, Any]:
        """Extract task parameters as dictionary.

        Args:
            task_params: Fiddler ModelTaskParams object

        Returns:
            Dict with task-specific parameters
        """
        params = {}

        if hasattr(task_params, 'target_class_order') and task_params.target_class_order:
            params['target_class_order'] = list(task_params.target_class_order)

        if hasattr(task_params, 'binary_classification_threshold'):
            params['binary_classification_threshold'] = task_params.binary_classification_threshold

        return params

    def _detect_artifacts(self, model: fdl.Model) -> bool:
        """Check if model has uploaded artifacts.

        Args:
            model: Fiddler Model object

        Returns:
            True if model has artifacts
        """
        # Check for deployment params or other artifact indicators
        return hasattr(model, 'deployment_params') and model.deployment_params is not None

    def _export_related_assets(self, model_id: str) -> Dict[str, List[Dict]]:
        """Export related assets using existing asset managers.

        Leverages SegmentManager, CustomMetricManager, and AlertManager
        to export assets, then converts AssetExportData to dict for JSON serialization.

        Args:
            model_id: Model UUID

        Returns:
            Dict with lists of exported assets by type
        """
        related = {
            'segments': [],
            'custom_metrics': [],
            'alerts': []
        }

        # Export segments
        from .segments import SegmentManager
        segment_mgr = SegmentManager()
        segments = segment_mgr.export_assets(model_id=model_id)
        # Use asdict to serialize AssetExportData, converting enums and sets properly
        related['segments'] = [
            {
                **asdict(s),
                'asset_type': s.asset_type.name,  # Convert enum to string name
                'referenced_columns': list(s.referenced_columns)
            }
            for s in segments
        ]
        logger.info(f'Exported {len(segments)} segments')

        # Export custom metrics
        from .metrics import CustomMetricManager
        metric_mgr = CustomMetricManager()
        metrics = metric_mgr.export_assets(model_id=model_id)
        related['custom_metrics'] = [
            {
                **asdict(m),
                'asset_type': m.asset_type.name,  # Convert enum to string name
                'referenced_columns': list(m.referenced_columns)
            }
            for m in metrics
        ]
        logger.info(f'Exported {len(metrics)} custom metrics')

        # Export alerts
        from .alerts import AlertManager
        alert_mgr = AlertManager()
        alerts = alert_mgr.export_assets(model_id=model_id)
        related['alerts'] = [
            {
                **asdict(a),
                'asset_type': a.asset_type.name,  # Convert enum to string name
                'referenced_columns': list(a.referenced_columns)
            }
            for a in alerts
        ]
        logger.info(f'Exported {len(alerts)} alerts')

        return related

    def import_model(
        self,
        target_project_id: str,
        model_data: ModelExportData,
        import_mode: str = 'create_new',
        version_label: Optional[str] = None,
        create_baselines: bool = True,
        import_related_assets: bool = True,
    ) -> fdl.Model:
        """Import model to target project.

        Args:
            target_project_id: Target project UUID
            model_data: ModelExportData from export_model()
            import_mode: One of 'create_new', 'create_version', 'update_existing'
            version_label: Version label for 'create_version' mode
            create_baselines: Recreate baselines after model creation
            import_related_assets: Import segments, custom metrics, alerts

        Returns:
            Created/updated Fiddler Model object

        Raises:
            ValidationError: If model name conflict in 'create_new' mode
            ValidationError: If invalid import_mode
            AssetImportError: If model creation fails

        Example:
            ```python
            # Create new model in target project
            model = mgr.import_model(
                target_project_id='xyz789',
                model_data=export_data,
                import_mode='create_new'
            )

            # Create new version of existing model
            model_v2 = mgr.import_model(
                target_project_id='xyz789',
                model_data=export_data,
                import_mode='create_version',
                version_label='v2'
            )
            ```
        """
        valid_modes = ['create_new', 'create_version', 'update_existing']
        if import_mode not in valid_modes:
            raise ValidationError(
                f"Invalid import_mode '{import_mode}'. Must be one of: {valid_modes}"
            )

        if import_mode == 'create_version' and not version_label:
            raise ValidationError(
                "version_label is required when import_mode='create_version'"
            )

        logger.info(f"Importing model '{model_data.name}' with mode '{import_mode}'")

        # Reconstruct schema
        columns = self._reconstruct_columns(model_data.columns)
        schema = fdl.ModelSchema(columns=columns)
        logger.debug(f'Reconstructed schema with {len(columns)} columns')

        # Reconstruct spec
        custom_features = self._reconstruct_custom_features(model_data.custom_features)

        # Build spec kwargs - only include custom_features if present
        spec_kwargs = {
            'inputs': model_data.spec['inputs'],
            'outputs': model_data.spec['outputs'],
            'targets': model_data.spec['targets'],
            'metadata': model_data.spec['metadata'],
            'decisions': model_data.spec.get('decisions', []),
        }

        # Only add custom_features if there are any (ModelSpec doesn't accept None)
        if custom_features:
            spec_kwargs['custom_features'] = custom_features

        spec = fdl.ModelSpec(**spec_kwargs)
        logger.debug('Reconstructed model spec')

        # Reconstruct task params
        task_params = self._reconstruct_task_params(model_data.task_params) if model_data.task_params else None

        # Parse task enum
        task = self._parse_task_enum(model_data.task)

        # Execute import based on mode
        if import_mode == 'create_new':
            model = self._import_create_new(
                target_project_id, model_data, schema, spec, task, task_params
            )
        elif import_mode == 'create_version':
            model = self._import_create_version(
                target_project_id, model_data, schema, spec, task, task_params, version_label
            )
        elif import_mode == 'update_existing':
            model = self._import_update_existing(
                target_project_id, model_data, schema, spec, task, task_params
            )

        logger.info(f"Successfully imported model '{model.name}' (ID: {model.id})")

        # Import baselines
        if create_baselines and model_data.baselines:
            self._import_baselines(model.id, model_data.baselines)

        # Import related assets
        if import_related_assets and model_data.related_assets:
            self._import_related_assets(model.id, model_data.related_assets)

        # Warn about artifacts
        if model_data.has_artifacts:
            logger.warning(
                f"Source model had uploaded artifacts. "
                f"You must manually re-upload artifacts using: "
                f"model.add_artifact(model_dir='path/to/package')"
            )

        return model

    def _reconstruct_columns(self, columns_data: List[ColumnExportData]) -> List[fdl.Column]:
        """Reconstruct fdl.Column objects from export data.

        Args:
            columns_data: List of ColumnExportData

        Returns:
            List of fdl.Column objects
        """
        columns = []

        for col_data in columns_data:
            # Parse data type enum
            data_type = getattr(fdl.DataType, col_data.data_type)

            if data_type == fdl.DataType.CATEGORY:
                # Category columns require categories, min/max must be None
                col = fdl.Column(
                    name=col_data.name,
                    data_type=data_type,
                    categories=col_data.categories,
                    min=None,
                    max=None
                )
            else:
                # Numeric columns require min/max, categories must be None
                col = fdl.Column(
                    name=col_data.name,
                    data_type=data_type,
                    min=col_data.min_value,
                    max=col_data.max_value,
                    categories=None
                )

            columns.append(col)

        return columns

    def _reconstruct_custom_features(self, custom_features_data: List[Dict[str, Any]]) -> List:
        """Reconstruct custom features (LLM enrichments) from export data.

        Args:
            custom_features_data: List of custom feature dicts

        Returns:
            List of fdl.TextEmbedding and fdl.Enrichment objects
        """
        if not custom_features_data:
            return []

        custom_features = []

        for feature_data in custom_features_data:
            feature_type = feature_data['type']

            if feature_type == 'TextEmbedding':
                custom_features.append(fdl.TextEmbedding(
                    name=feature_data['name'],
                    source_column=feature_data['source_column'],
                    column=feature_data['column'],
                    n_tags=feature_data.get('n_tags', 10)
                ))
            elif feature_type == 'Enrichment':
                enrichment = fdl.Enrichment(
                    name=feature_data['name'],
                    enrichment=feature_data['enrichment'],
                    columns=feature_data['columns'],
                    config=feature_data.get('config', {})
                )
                # Add allow_list if present
                if feature_data.get('allow_list'):
                    enrichment.allow_list = feature_data['allow_list']
                custom_features.append(enrichment)
            else:
                logger.warning(f"Unknown custom feature type '{feature_type}', skipping")

        return custom_features

    def _reconstruct_task_params(self, task_params_data: Dict[str, Any]) -> fdl.ModelTaskParams:
        """Reconstruct ModelTaskParams from export data.

        Args:
            task_params_data: Task params dict

        Returns:
            fdl.ModelTaskParams object
        """
        params = fdl.ModelTaskParams()

        if 'target_class_order' in task_params_data:
            params.target_class_order = task_params_data['target_class_order']

        if 'binary_classification_threshold' in task_params_data:
            params.binary_classification_threshold = task_params_data['binary_classification_threshold']

        return params

    def _parse_task_enum(self, task_str: str) -> fdl.ModelTask:
        """Parse ModelTask enum from string.

        Args:
            task_str: Task as string (e.g., 'BINARY_CLASSIFICATION', 'ModelTask.BINARY_CLASSIFICATION')

        Returns:
            fdl.ModelTask enum value
        """
        # Remove 'ModelTask.' prefix if present
        task_name = task_str.replace('ModelTask.', '')

        # Convert to uppercase (enum members are uppercase)
        task_name = task_name.upper()

        return getattr(fdl.ModelTask, task_name)

    def _import_create_new(
        self,
        target_project_id: str,
        model_data: ModelExportData,
        schema: fdl.ModelSchema,
        spec: fdl.ModelSpec,
        task: fdl.ModelTask,
        task_params: Optional[fdl.ModelTaskParams]
    ) -> fdl.Model:
        """Import as new model (fail if name exists).

        Args:
            target_project_id: Target project UUID
            model_data: Model export data
            schema: Reconstructed schema
            spec: Reconstructed spec
            task: ModelTask enum
            task_params: Reconstructed task params

        Returns:
            Created Model object

        Raises:
            ValidationError: If model name already exists
        """
        # Check if model exists
        try:
            existing = fdl.Model.from_name(name=model_data.name, project_id=target_project_id)
            raise ValidationError(
                f"Model '{model_data.name}' already exists in target project. "
                f"Use import_mode='create_version' to create a new version, or choose a different name."
            )
        except:
            # Model doesn't exist, proceed
            pass

        # Create new model
        model = fdl.Model(
            name=model_data.name,
            project_id=target_project_id,
            schema=schema,
            spec=spec,
            task=task,
            task_params=task_params,
            event_id_col=model_data.event_id_col,
            event_ts_col=model_data.event_ts_col,
            version=model_data.version  # Preserve version label if present
        )

        try:
            model.create()
            logger.info(f"Created new model '{model.name}'")
        except Exception as e:
            raise AssetImportError(f"Failed to create model: {str(e)}")

        return model

    def _import_create_version(
        self,
        target_project_id: str,
        model_data: ModelExportData,
        schema: fdl.ModelSchema,
        spec: fdl.ModelSpec,
        task: fdl.ModelTask,
        task_params: Optional[fdl.ModelTaskParams],
        version_label: str
    ) -> fdl.Model:
        """Import as new version of existing model.

        Args:
            target_project_id: Target project UUID
            model_data: Model export data
            schema: Reconstructed schema
            spec: Reconstructed spec
            task: ModelTask enum
            task_params: Reconstructed task params
            version_label: Version label for new version

        Returns:
            Created Model version

        Raises:
            AssetImportError: If base model doesn't exist or version creation fails
        """
        # Get existing model (without version to get base model)
        try:
            existing_model = fdl.Model.from_name(name=model_data.name, project_id=target_project_id)
        except Exception as e:
            raise AssetImportError(
                f"Cannot create version: base model '{model_data.name}' not found in target project. "
                f"Use import_mode='create_new' first."
            )

        # Duplicate and modify
        try:
            new_version = existing_model.duplicate(version=version_label)
            new_version.schema = schema
            new_version.spec = spec
            new_version.task = task
            new_version.task_params = task_params
            new_version.event_id_col = model_data.event_id_col
            new_version.event_ts_col = model_data.event_ts_col
            new_version.create()
            logger.info(f"Created new version '{version_label}' of model '{model_data.name}'")
        except Exception as e:
            raise AssetImportError(f"Failed to create model version: {str(e)}")

        return new_version

    def _import_update_existing(
        self,
        target_project_id: str,
        model_data: ModelExportData,
        schema: fdl.ModelSchema,
        spec: fdl.ModelSpec,
        task: fdl.ModelTask,
        task_params: Optional[fdl.ModelTaskParams]
    ) -> fdl.Model:
        """Import by updating existing model (dangerous).

        Args:
            target_project_id: Target project UUID
            model_data: Model export data
            schema: Reconstructed schema
            spec: Reconstructed spec
            task: ModelTask enum
            task_params: Reconstructed task params

        Returns:
            Updated Model object

        Raises:
            AssetImportError: If model doesn't exist or update fails
        """
        # Get existing model
        try:
            existing_model = fdl.Model.from_name(
                name=model_data.name,
                project_id=target_project_id,
                version=model_data.version
            )
        except Exception as e:
            raise AssetImportError(
                f"Cannot update: model '{model_data.name}' (version: {model_data.version}) not found. "
                f"Use import_mode='create_new' instead."
            )

        # Update properties
        logger.warning(
            f"Updating existing model '{model_data.name}' - this may cause data loss or conflicts!"
        )

        try:
            existing_model.schema = schema
            existing_model.spec = spec
            existing_model.task = task
            existing_model.task_params = task_params
            existing_model.event_id_col = model_data.event_id_col
            existing_model.event_ts_col = model_data.event_ts_col
            existing_model.update()
            logger.info(f"Updated existing model '{model_data.name}'")
        except Exception as e:
            raise AssetImportError(f"Failed to update model: {str(e)}")

        return existing_model

    def _import_baselines(self, target_model_id: str, baselines_data: List[Dict[str, Any]]):
        """Import baselines to target model.

        Args:
            target_model_id: Target model UUID
            baselines_data: List of baseline definitions
        """
        if not baselines_data:
            return

        print(f"\nImporting {len(baselines_data)} baselines...")
        baseline_mgr = self._get_baseline_manager()
        success_count = 0
        skip_count = 0
        fail_count = 0

        for baseline_data in baselines_data:
            try:
                result = baseline_mgr.import_baseline(target_model_id, baseline_data)
                if result is None:
                    print(f"  ⊗ Skipped: {baseline_data['name']} ({baseline_data['type']})")
                    skip_count += 1
                else:
                    print(f"  ✓ Created: {baseline_data['name']} ({baseline_data['type']})")
                    success_count += 1
            except Exception as e:
                print(f"  ✗ Failed: {baseline_data['name']} - {str(e)}")
                fail_count += 1

        print(f"  Summary: {success_count} created, {skip_count} skipped, {fail_count} failed")

    def _import_related_assets(self, target_model_id: str, related_assets: Dict[str, List[Dict]]):
        """Import related assets using existing asset managers.

        Reconstructs AssetExportData from serialized dicts and passes them to
        the appropriate managers (SegmentManager, CustomMetricManager, AlertManager).

        Args:
            target_model_id: Target model UUID
            related_assets: Dict of related assets by type (from export)
        """
        from .base import AssetExportData, AssetType

        # Import segments
        if related_assets.get('segments'):
            print(f"\nImporting {len(related_assets['segments'])} segments...")
            try:
                from .segments import SegmentManager
                segment_mgr = SegmentManager()

                # Reconstruct AssetExportData from dict (asdict was used during export)
                segment_assets = [
                    AssetExportData(
                        asset_type=AssetType[s['asset_type']] if isinstance(s['asset_type'], str) else s['asset_type'],
                        name=s['name'],
                        data=s['data'],
                        referenced_columns=set(s.get('referenced_columns', []))
                    )
                    for s in related_assets['segments']
                ]

                result = segment_mgr.import_assets(
                    target_model_id=target_model_id,
                    assets=segment_assets,
                    validate=False,
                    dry_run=False
                )
                print(f"  ✓ Created: {result.successful}, Skipped: {result.skipped}, Failed: {result.failed}")
                if result.failed > 0:
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"    Error: {error}")
            except Exception as e:
                print(f"  ✗ Failed to import segments: {str(e)}")
                logger.warning(f'Failed to import segments: {e}')

        # Import custom metrics
        if related_assets.get('custom_metrics'):
            print(f"\nImporting {len(related_assets['custom_metrics'])} custom metrics...")
            try:
                from .metrics import CustomMetricManager
                metric_mgr = CustomMetricManager()

                metric_assets = [
                    AssetExportData(
                        asset_type=AssetType[m['asset_type']] if isinstance(m['asset_type'], str) else m['asset_type'],
                        name=m['name'],
                        data=m['data'],
                        referenced_columns=set(m.get('referenced_columns', []))
                    )
                    for m in related_assets['custom_metrics']
                ]

                result = metric_mgr.import_assets(
                    target_model_id=target_model_id,
                    assets=metric_assets,
                    validate=False,
                    dry_run=False
                )
                print(f"  ✓ Created: {result.successful}, Skipped: {result.skipped}, Failed: {result.failed}")
                if result.failed > 0:
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"    Error: {error}")
            except Exception as e:
                print(f"  ✗ Failed to import custom metrics: {str(e)}")
                logger.warning(f'Failed to import custom metrics: {e}')

        # Import alerts
        if related_assets.get('alerts'):
            print(f"\nImporting {len(related_assets['alerts'])} alerts...")
            print(f"  ⚠️  Alert import is not yet supported - metric IDs require manual mapping")
            print(f"  📋 Exported alert definitions are saved in the export data for reference")
            logger.info(f"Skipped {len(related_assets['alerts'])} alerts (not yet supported)")

    def display_model_info(
        self,
        model_id: str,
        show_columns: bool = True,
        show_related_assets: bool = True
    ):
        """Display comprehensive model information.

        Args:
            model_id: Model UUID
            show_columns: Show detailed column information
            show_related_assets: Show counts of related assets

        Example:
            ```python
            mgr.display_model_info(model.id)
            # Output:
            # Model: bank_churn_model
            # Version: v1
            # Task: BINARY_CLASSIFICATION
            # ...
            ```
        """
        model = fdl.Model.get(id_=model_id)

        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")

        if model.version:
            print(f"Version: {model.version}")

        print(f"Task: {model.task}")
        print(f"Project ID: {model.project_id}")
        print(f"Model ID: {model.id}")

        # Get default dashboard UUID
        try:
            dashboard_uuid = self.get_default_dashboard_uuid(model.id)
            if dashboard_uuid:
                print(f"Default Dashboard UUID: {dashboard_uuid}")
        except Exception as e:
            logger.debug(f"Could not retrieve default dashboard UUID: {e}")

        # Event columns
        if model.event_id_col or model.event_ts_col:
            print(f"\nEvent Columns:")
            if model.event_id_col:
                print(f"  ID Column: {model.event_id_col}")
            if model.event_ts_col:
                print(f"  Timestamp Column: {model.event_ts_col}")

        # Task parameters
        if model.task_params:
            print(f"\nTask Parameters:")
            if hasattr(model.task_params, 'target_class_order') and model.task_params.target_class_order:
                print(f"  Target Class Order: {model.task_params.target_class_order}")
            if hasattr(model.task_params, 'binary_classification_threshold'):
                print(f"  Binary Threshold: {model.task_params.binary_classification_threshold}")

        # Schema summary
        schema_columns = getattr(model.schema, 'columns', [])
        print(f"\nSchema: {len(schema_columns)} columns")
        print(f"  Inputs: {len(model.spec.inputs or [])}")
        print(f"  Outputs: {len(model.spec.outputs or [])}")
        print(f"  Targets: {len(model.spec.targets or [])}")
        print(f"  Metadata: {len(model.spec.metadata or [])}")

        if model.spec.decisions:
            print(f"  Decisions: {len(model.spec.decisions)}")

        if hasattr(model.spec, 'custom_features') and model.spec.custom_features:
            print(f"  Custom Features: {len(model.spec.custom_features)}")

        # Detailed column information
        if show_columns:
            print(f"\n{'-'*60}")
            print(f"Columns:")
            print(f"{'-'*60}")

            # Group columns by role
            column_roles = {}
            for col_name in model.spec.inputs or []:
                column_roles[col_name] = 'INPUT'
            for col_name in model.spec.outputs or []:
                column_roles[col_name] = 'OUTPUT'
            for col_name in model.spec.targets or []:
                column_roles[col_name] = 'TARGET'
            for col_name in model.spec.metadata or []:
                column_roles[col_name] = 'METADATA'
            for col_name in model.spec.decisions or []:
                column_roles[col_name] = 'DECISION'

            # Print column details
            for col in schema_columns:
                role = column_roles.get(col.name, 'UNKNOWN')
                data_type = str(col.data_type).replace('DataType.', '')

                info_parts = [f"{col.name:<30}", f"{role:<10}", f"{data_type:<12}"]

                # Add range/categories info
                if hasattr(col, 'min') and col.min is not None:
                    info_parts.append(f"min={col.min}")
                if hasattr(col, 'max') and col.max is not None:
                    info_parts.append(f"max={col.max}")
                if hasattr(col, 'categories') and col.categories:
                    cat_preview = str(col.categories[:3])[:-1]
                    if len(col.categories) > 3:
                        cat_preview += f", ... ({len(col.categories)} total)]"
                    else:
                        cat_preview += "]"
                    info_parts.append(f"categories={cat_preview}")

                print("  " + "  ".join(info_parts))

        # Artifacts warning
        if self._detect_artifacts(model):
            print(f"\n{'!'*60}")
            print(f"⚠  Model has uploaded artifacts (model package)")
            print(f"{'!'*60}")

        # Related assets counts
        if show_related_assets:
            print(f"\n{'-'*60}")
            print(f"Related Assets:")
            print(f"{'-'*60}")

            # Count baselines
            try:
                baseline_mgr = self._get_baseline_manager()
                baselines = baseline_mgr.list_baselines(model_id)
                print(f"  Baselines: {len(baselines)}")
            except Exception as e:
                print(f"  Baselines: Unable to count ({str(e)[:50]})")

            # Count segments
            try:
                segments = list(fdl.Segment.list(model_id=model_id))
                print(f"  Segments: {len(segments)}")
            except Exception as e:
                print(f"  Segments: Unable to count")

            # Count custom metrics
            try:
                metrics = list(fdl.CustomMetric.list(model_id=model_id))
                print(f"  Custom Metrics: {len(metrics)}")
            except Exception as e:
                print(f"  Custom Metrics: Unable to count")

            # Count alerts
            try:
                alerts = list(fdl.AlertRule.list(model_id=model_id))
                print(f"  Alerts: {len(alerts)}")
            except Exception as e:
                print(f"  Alerts: Unable to count")

        print(f"{'='*60}\n")

    def get_default_dashboard_uuid(self, model_id: str) -> Optional[str]:
        """Get the default dashboard UUID for a model.

        Uses the REST API endpoint since the Python SDK doesn't expose this.

        Args:
            model_id: Model UUID

        Returns:
            Default dashboard UUID, or None if not found

        Raises:
            FiddlerUtilsError: If API request fails

        Example:
            ```python
            dashboard_uuid = mgr.get_default_dashboard_uuid(model.id)
            print(f'Default dashboard: {dashboard_uuid}')
            ```
        """
        # Get current connection info
        url = connection._current_url
        token = connection._current_token

        if not url or not token:
            raise FiddlerUtilsError(
                'No active Fiddler connection. Call fdl.init() first or use ConnectionManager.'
            )

        # Create RequestClient for API call
        client = RequestClient(
            base_url=url,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
        )

        try:
            # Call the default-dashboard endpoint
            endpoint = f'/v3/models/{model_id}/default-dashboard'
            response = client.get(url=endpoint)

            if response.get('kind') == 'NORMAL' and 'data' in response:
                dashboard_uuid = response['data'].get('dashboard_uuid')
                logger.debug(f'Retrieved default dashboard UUID for model {model_id}: {dashboard_uuid}')
                return dashboard_uuid
            else:
                logger.warning(f'Unexpected response format from default-dashboard endpoint: {response}')
                return None

        except Exception as e:
            logger.error(f'Failed to get default dashboard for model {model_id}: {e}')
            raise FiddlerUtilsError(f'Failed to get default dashboard: {str(e)}')

    def compare_models(
        self,
        source_model_id: str,
        target_model_id: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Compare two models and report differences.

        Args:
            source_model_id: Source model UUID
            target_model_id: Target model UUID
            detailed: Show detailed column-by-column comparison

        Returns:
            Dict with comparison results

        Example:
            ```python
            comparison = mgr.compare_models(
                source_model_id='abc123',
                target_model_id='xyz789',
                detailed=True
            )
            ```
        """
        source = fdl.Model.get(id_=source_model_id)
        target = fdl.Model.get(id_=target_model_id)

        print(f"\n{'='*60}")
        print(f"Model Comparison")
        print(f"{'='*60}")
        print(f"Source: {source.name} (version: {source.version or 'none'})")
        print(f"Target: {target.name} (version: {target.version or 'none'})")
        print(f"{'='*60}\n")

        comparison = {
            'models_identical': True,
            'differences': []
        }

        # Compare basic properties
        if source.name != target.name:
            comparison['models_identical'] = False
            comparison['differences'].append(f"Name: '{source.name}' → '{target.name}'")
            print(f"✗ Name differs: '{source.name}' → '{target.name}'")

        if source.version != target.version:
            comparison['models_identical'] = False
            comparison['differences'].append(f"Version: '{source.version}' → '{target.version}'")
            print(f"✗ Version differs: '{source.version}' → '{target.version}'")

        if source.task != target.task:
            comparison['models_identical'] = False
            comparison['differences'].append(f"Task: {source.task} → {target.task}")
            print(f"✗ Task differs: {source.task} → {target.task}")

        # Compare schema using SchemaValidator
        try:
            from ..schema import SchemaValidator

            schema_comparison = SchemaValidator.compare_schemas(source, target)

            if schema_comparison.added_columns:
                comparison['models_identical'] = False
                comparison['differences'].append(f"Added columns: {schema_comparison.added_columns}")
                print(f"✗ Added columns: {schema_comparison.added_columns}")

            if schema_comparison.removed_columns:
                comparison['models_identical'] = False
                comparison['differences'].append(f"Removed columns: {schema_comparison.removed_columns}")
                print(f"✗ Removed columns: {schema_comparison.removed_columns}")

            if schema_comparison.changed_columns:
                comparison['models_identical'] = False
                for col_name, changes in schema_comparison.changed_columns.items():
                    comparison['differences'].append(f"Column '{col_name}' changed: {changes}")
                    print(f"✗ Column '{col_name}' changed:")
                    for change_type, change_val in changes.items():
                        print(f"    {change_type}: {change_val}")

            if schema_comparison.is_compatible:
                print(f"✓ Schemas are compatible")
            else:
                print(f"✗ Schemas are NOT compatible")

        except Exception as e:
            logger.warning(f"Could not compare schemas: {e}")
            print(f"⚠  Could not perform schema comparison: {e}")

        # Compare spec (role assignments)
        source_inputs = set(source.spec.inputs or [])
        target_inputs = set(target.spec.inputs or [])
        if source_inputs != target_inputs:
            comparison['models_identical'] = False
            added = target_inputs - source_inputs
            removed = source_inputs - target_inputs
            if added:
                comparison['differences'].append(f"Added inputs: {added}")
                print(f"✗ Added input columns: {added}")
            if removed:
                comparison['differences'].append(f"Removed inputs: {removed}")
                print(f"✗ Removed input columns: {removed}")

        source_outputs = set(source.spec.outputs or [])
        target_outputs = set(target.spec.outputs or [])
        if source_outputs != target_outputs:
            comparison['models_identical'] = False
            comparison['differences'].append(f"Outputs differ: {source_outputs} → {target_outputs}")
            print(f"✗ Output columns differ: {source_outputs} → {target_outputs}")

        source_targets = set(source.spec.targets or [])
        target_targets = set(target.spec.targets or [])
        if source_targets != target_targets:
            comparison['models_identical'] = False
            comparison['differences'].append(f"Targets differ: {source_targets} → {target_targets}")
            print(f"✗ Target columns differ: {source_targets} → {target_targets}")

        # Summary
        print(f"\n{'-'*60}")
        if comparison['models_identical']:
            print(f"✓ Models are identical")
        else:
            print(f"✗ Models differ in {len(comparison['differences'])} ways")
        print(f"{'-'*60}\n")

        return comparison
