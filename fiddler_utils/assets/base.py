"""Base class for asset managers.

This module provides the abstract base class for managing different types
of Fiddler assets (segments, custom metrics, alerts, charts, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any, TypeVar, Generic
from enum import Enum
import logging

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from ..exceptions import (
    AssetNotFoundError,
    AssetImportError,
    ValidationError,
)
from ..schema import SchemaValidator
from .. import fql

logger = logging.getLogger(__name__)

# Type variable for asset types
T = TypeVar('T')


class AssetType(str, Enum):
    """Enum for different asset types."""

    SEGMENT = 'segment'
    CUSTOM_METRIC = 'custom_metric'
    ALERT_RULE = 'alert_rule'
    CHART = 'chart'
    DASHBOARD = 'dashboard'
    BASELINE = 'baseline'


@dataclass
class AssetExportData:
    """Exported asset data with metadata."""

    asset_type: AssetType
    name: str
    data: Dict[str, Any]
    referenced_columns: Set[str]
    dependencies: List[str] = None  # IDs of dependent assets

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ImportResult:
    """Result of importing assets."""

    successful: int = 0
    skipped_existing: int = 0  # Assets skipped because they already exist
    skipped_invalid: int = 0  # Assets skipped due to validation failures
    failed: int = 0
    errors: List[tuple] = None  # List of (asset_name, error_message)

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def skipped(self) -> int:
        """Total skipped (existing + invalid) for backward compatibility."""
        return self.skipped_existing + self.skipped_invalid

    @property
    def total(self) -> int:
        return self.successful + self.skipped + self.failed

    def __repr__(self) -> str:
        return (
            f'ImportResult(successful={self.successful}, '
            f'skipped_existing={self.skipped_existing}, '
            f'skipped_invalid={self.skipped_invalid}, '
            f'failed={self.failed})'
        )


@dataclass
class ValidationResult:
    """Result of validating an asset."""

    is_valid: bool
    missing_columns: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.missing_columns is None:
            self.missing_columns = []
        if self.errors is None:
            self.errors = []

    def __repr__(self) -> str:
        if self.is_valid:
            return 'ValidationResult(valid=True)'
        return (
            f'ValidationResult(valid=False, '
            f'missing_columns={len(self.missing_columns)}, '
            f'errors={len(self.errors)})'
        )


class BaseAssetManager(ABC, Generic[T]):
    """Abstract base class for managing Fiddler assets.

    Subclasses should implement asset-specific logic for:
    - Listing assets
    - Extracting asset data
    - Creating assets
    - Validating assets

    Example:
        ```python
        class SegmentManager(BaseAssetManager[fdl.Segment]):
            def _list_assets(self, model_id: str) -> List[fdl.Segment]:
                return list(fdl.Segment.list(model_id=model_id))
            # ... implement other abstract methods
        ```
    """

    def __init__(self):
        self.asset_type = self._get_asset_type()

    @abstractmethod
    def _get_asset_type(self) -> AssetType:
        """Get the asset type this manager handles."""
        pass

    @abstractmethod
    def _list_assets(self, model_id: str) -> List[T]:
        """List all assets of this type for a model.

        Args:
            model_id: Fiddler model ID

        Returns:
            List of asset objects
        """
        pass

    @abstractmethod
    def _extract_asset_data(self, asset: T) -> Dict[str, Any]:
        """Extract data from an asset for export.

        Args:
            asset: Asset object

        Returns:
            Dictionary of asset data
        """
        pass

    @abstractmethod
    def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> T:
        """Create a new asset from data.

        Args:
            model_id: Target model ID
            asset_data: Asset data dictionary

        Returns:
            Created asset object
        """
        pass

    @abstractmethod
    def _get_asset_name(self, asset: T) -> str:
        """Get the name of an asset."""
        pass

    @abstractmethod
    def _extract_referenced_columns(self, asset: T) -> Set[str]:
        """Extract column references from asset definition.

        Args:
            asset: Asset object

        Returns:
            Set of column names referenced by this asset
        """
        pass

    def list_assets(self, model_id: str, names: Optional[List[str]] = None) -> List[T]:
        """List assets for a model, optionally filtered by name.

        Args:
            model_id: Fiddler model ID
            names: Optional list of asset names to filter by

        Returns:
            List of asset objects

        Example:
            ```python
            mgr = SegmentManager()
            # Get all segments
            all_segments = mgr.list_assets(model_id=model.id)

            # Get specific segments
            some_segments = mgr.list_assets(
                model_id=model.id,
                names=['segment1', 'segment2']
            )
            ```
        """
        logger.info(f'Listing {self.asset_type.value}s for model {model_id}')
        assets = self._list_assets(model_id)

        if names:
            assets = [a for a in assets if self._get_asset_name(a) in names]
            logger.info(
                f'Filtered to {len(assets)} {self.asset_type.value}s '
                f'matching names: {names}'
            )

        return assets

    def asset_exists_by_name(self, model_id: str, name: str) -> Optional[T]:
        """Check if an asset with the given name exists.

        Args:
            model_id: Fiddler model ID
            name: Asset name to check

        Returns:
            The existing asset if found, None otherwise

        Example:
            ```python
            mgr = SegmentManager()
            existing = mgr.asset_exists_by_name(model.id, 'my_segment')
            if existing:
                print(f"Segment exists with ID: {existing.id}")
            ```
        """
        assets = self._list_assets(model_id)
        for asset in assets:
            if self._get_asset_name(asset) == name:
                return asset
        return None

    def export_assets(
        self, model_id: str, names: Optional[List[str]] = None
    ) -> List[AssetExportData]:
        """Export assets from a model.

        Args:
            model_id: Source model ID
            names: Optional list of asset names to export (None = all)

        Returns:
            List of AssetExportData objects

        Example:
            ```python
            mgr = SegmentManager()
            exported = mgr.export_assets(
                model_id=source_model.id,
                names=['important_segment']
            )

            for asset in exported:
                print(f"Exported: {asset.name}")
                print(f"  Columns: {asset.referenced_columns}")
            ```
        """
        assets = self.list_assets(model_id, names=names)

        logger.info(
            f'Exporting {len(assets)} {self.asset_type.value}s from model {model_id}'
        )

        exported = []
        for asset in assets:
            name = self._get_asset_name(asset)
            data = self._extract_asset_data(asset)
            columns = self._extract_referenced_columns(asset)

            export_data = AssetExportData(
                asset_type=self.asset_type,
                name=name,
                data=data,
                referenced_columns=columns,
            )
            exported.append(export_data)

            logger.debug(
                f'Exported {self.asset_type.value}: {name} (columns: {columns})'
            )

        return exported

    def validate_asset(
        self, asset_data: AssetExportData, target_model: fdl.Model
    ) -> ValidationResult:
        """Validate that an asset is compatible with a target model.

        Args:
            asset_data: Exported asset data
            target_model: Target Fiddler model

        Returns:
            ValidationResult object

        Example:
            ```python
            mgr = SegmentManager()
            result = mgr.validate_asset(exported_asset, target_model)

            if not result.is_valid:
                print(f"Missing columns: {result.missing_columns}")
            ```
        """
        # Validate columns exist in target model
        is_valid, missing = SchemaValidator.validate_columns(
            asset_data.referenced_columns, target_model, strict=False
        )

        result = ValidationResult(is_valid=is_valid, missing_columns=missing, errors=[])

        if not is_valid:
            logger.warning(
                f"Validation failed for {self.asset_type.value} '{asset_data.name}': "
                f'missing columns: {missing}'
            )

        return result

    def import_assets(
        self,
        target_model_id: str,
        assets: List[AssetExportData],
        validate: bool = True,
        dry_run: bool = False,
        skip_invalid: bool = True,
        overwrite: bool = False,
    ) -> ImportResult:
        """Import assets to a target model.

        Args:
            target_model_id: Target model ID
            assets: List of AssetExportData to import
            validate: If True, validate assets before import
            dry_run: If True, validate only (don't create assets)
            skip_invalid: If True, skip invalid assets. If False, raise exception.
            overwrite: If True, delete and recreate existing assets with same name.
                      If False (default), skip existing assets.

        Returns:
            ImportResult with summary

        Raises:
            ValidationError: If skip_invalid=False and validation fails

        Example:
            ```python
            mgr = SegmentManager()

            # Import, skipping existing assets (default)
            result = mgr.import_assets(
                target_model_id=target_model.id,
                assets=exported_assets
            )
            print(f"Imported: {result.successful}")
            print(f"Skipped (existing): {result.skipped_existing}")
            print(f"Skipped (invalid): {result.skipped_invalid}")

            # Import with overwrite (delete and recreate existing)
            result = mgr.import_assets(
                target_model_id=target_model.id,
                assets=exported_assets,
                overwrite=True
            )
            ```
        """
        logger.info(
            f'{"[DRY RUN] " if dry_run else ""}Importing '
            f'{len(assets)} {self.asset_type.value}s to model {target_model_id}'
            f'{" (overwrite=True)" if overwrite else ""}'
        )

        result = ImportResult()

        # Get target model for validation
        if validate or dry_run:
            target_model = fdl.Model.get(id_=target_model_id)

        for asset_data in assets:
            asset_name = asset_data.name

            # Check if asset already exists
            existing_asset = self.asset_exists_by_name(target_model_id, asset_name)

            if existing_asset and not overwrite:
                # Skip existing asset when overwrite=False
                result.skipped_existing += 1
                logger.info(
                    f"Skipping existing {self.asset_type.value} '{asset_name}' "
                    f"(use overwrite=True to replace)"
                )
                continue

            # Validate if requested
            if validate or dry_run:
                validation = self.validate_asset(asset_data, target_model)

                if not validation.is_valid:
                    result.skipped_invalid += 1
                    error_msg = f'Missing columns: {validation.missing_columns}'
                    result.errors.append((asset_name, error_msg))

                    logger.warning(
                        f"Skipping {self.asset_type.value} '{asset_name}': {error_msg}"
                    )

                    if not skip_invalid:
                        raise ValidationError(
                            f"Validation failed for {self.asset_type.value} '{asset_name}'",
                            errors=validation.missing_columns,
                        )
                    continue

            # Skip actual creation if dry run
            if dry_run:
                result.successful += 1
                action = "overwrite" if existing_asset else "create"
                logger.info(
                    f'[DRY RUN] Would {action} {self.asset_type.value}: {asset_name}'
                )
                continue

            # Delete existing asset if overwrite=True
            if existing_asset and overwrite:
                logger.info(
                    f"Deleting existing {self.asset_type.value} '{asset_name}' "
                    f"(ID: {existing_asset.id})"
                )
                try:
                    existing_asset.delete()
                except Exception as e:
                    result.failed += 1
                    error_msg = f'Failed to delete existing asset: {str(e)}'
                    result.errors.append((asset_name, error_msg))
                    logger.error(
                        f"Failed to delete {self.asset_type.value} '{asset_name}': {error_msg}"
                    )
                    if not skip_invalid:
                        raise AssetImportError(
                            f'Failed to delete existing {self.asset_type.value}',
                            asset_name=asset_name,
                            reason=error_msg,
                        )
                    continue

            # Create the asset
            try:
                self._create_asset(target_model_id, asset_data.data)
                result.successful += 1
                action = "Overwrote" if (existing_asset and overwrite) else "Imported"
                logger.info(
                    f'{action} {self.asset_type.value}: {asset_name}'
                )

            except Exception as e:
                result.failed += 1
                error_msg = str(e)
                result.errors.append((asset_name, error_msg))
                logger.error(
                    f"Failed to import {self.asset_type.value} '{asset_name}': {error_msg}"
                )

                if not skip_invalid:
                    raise AssetImportError(
                        f'Failed to import {self.asset_type.value}',
                        asset_name=asset_name,
                        reason=error_msg,
                    )

        logger.info(
            f'Import complete: {result.successful} successful, '
            f'{result.skipped_existing} skipped (existing), '
            f'{result.skipped_invalid} skipped (invalid), '
            f'{result.failed} failed'
        )

        return result

    def copy_assets(
        self,
        source_model_id: str,
        target_model_id: str,
        names: Optional[List[str]] = None,
        validate: bool = True,
        dry_run: bool = False,
        overwrite: bool = False,
    ) -> ImportResult:
        """Copy assets from source model to target model.

        This is a convenience method that combines export and import.

        Args:
            source_model_id: Source model ID
            target_model_id: Target model ID
            names: Optional list of asset names to copy (None = all)
            validate: If True, validate before import
            dry_run: If True, validate only (don't create)
            overwrite: If True, replace existing assets with same name

        Returns:
            ImportResult

        Example:
            ```python
            mgr = SegmentManager()
            result = mgr.copy_assets(
                source_model_id=source_model.id,
                target_model_id=target_model.id,
                names=['critical_segment'],
                overwrite=True  # Replace if exists
            )
            ```
        """
        # Export from source
        exported = self.export_assets(source_model_id, names=names)

        # Import to target
        return self.import_assets(
            target_model_id=target_model_id,
            assets=exported,
            validate=validate,
            dry_run=dry_run,
            overwrite=overwrite,
        )
