"""Segment management utilities.

This module provides the SegmentManager class for working with Fiddler segments.
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


class SegmentManager(BaseAssetManager[fdl.Segment]):
    """Manager for Fiddler segments.

    Segments define subsets of data based on FQL filter expressions.
    This manager handles CRUD operations, export/import, and validation.

    Example:
        ```python
        from fiddler_utils.assets import SegmentManager
        import fiddler as fdl

        mgr = SegmentManager()

        # Export segments from source model
        exported = mgr.export_assets(
            model_id=source_model.id,
            names=['important_customers', 'high_risk']
        )

        # Import to target model with validation
        result = mgr.import_assets(
            target_model_id=target_model.id,
            assets=exported,
            validate=True,
            dry_run=False
        )

        print(f"Imported {result.successful} segments")
        ```
    """

    def _get_asset_type(self) -> AssetType:
        """Get asset type."""
        return AssetType.SEGMENT

    def _list_assets(self, model_id: str) -> List[fdl.Segment]:
        """List all segments for a model."""
        return list(fdl.Segment.list(model_id=model_id))

    def _get_asset_name(self, asset: fdl.Segment) -> str:
        """Get segment name."""
        return asset.name

    def _extract_referenced_columns(self, asset: fdl.Segment) -> Set[str]:
        """Extract column references from segment definition.

        Args:
            asset: Segment object

        Returns:
            Set of column names referenced in the FQL definition
        """
        return fql.extract_columns(asset.definition)

    def _extract_asset_data(self, asset: fdl.Segment) -> Dict[str, Any]:
        """Extract segment data for export.

        Args:
            asset: Segment object

        Returns:
            Dictionary with segment data
        """
        return {
            'name': asset.name,
            'description': asset.description if hasattr(asset, 'description') else '',
            'definition': asset.definition,
            # Store metadata for reference
            'metadata': {
                'id': asset.id,
                'model_id': asset.model_id,
            },
        }

    def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> fdl.Segment:
        """Create a segment from data.

        Args:
            model_id: Target model ID
            asset_data: Segment data dictionary

        Returns:
            Created Segment object
        """
        segment = fdl.Segment(
            model_id=model_id,
            name=asset_data['name'],
            description=asset_data.get('description', ''),
            definition=asset_data['definition'],
        )
        segment.create()
        logger.info(f'Created segment: {segment.name} (ID: {segment.id})')
        return segment

    def get_segment_by_name(self, model_id: str, name: str) -> fdl.Segment:
        """Get a specific segment by name.

        Args:
            model_id: Model ID
            name: Segment name

        Returns:
            Segment object

        Raises:
            AssetNotFoundError: If segment not found

        Example:
            ```python
            mgr = SegmentManager()
            segment = mgr.get_segment_by_name(model.id, 'high_value_customers')
            print(segment.definition)
            ```
        """
        from ..exceptions import AssetNotFoundError

        segments = self._list_assets(model_id)
        for segment in segments:
            if segment.name == name:
                return segment

        raise AssetNotFoundError(
            f"Segment '{name}' not found in model {model_id}",
            asset_type='Segment',
            asset_id=name,
        )

    def update_segment_definition(
        self, segment: fdl.Segment, new_definition: str, validate: bool = True
    ) -> fdl.Segment:
        """Update a segment's FQL definition.

        Note: This requires deleting and recreating the segment,
        as the Fiddler SDK doesn't support in-place updates.

        Args:
            segment: Existing segment
            new_definition: New FQL definition
            validate: If True, validate new definition before updating

        Returns:
            Updated segment object

        Example:
            ```python
            mgr = SegmentManager()
            segment = mgr.get_segment_by_name(model.id, 'my_segment')

            # Update definition
            updated = mgr.update_segment_definition(
                segment,
                new_definition='"age" > 40 and "status" == \'active\''
            )
            ```
        """
        if validate:
            # Validate syntax
            is_valid, error = fql.validate_fql_syntax(new_definition)
            if not is_valid:
                from ..exceptions import FQLError

                raise FQLError(
                    f'Invalid FQL syntax: {error}', expression=new_definition
                )

            # Validate columns exist in model
            model = fdl.Model.get(id_=segment.model_id)
            from ..schema import SchemaValidator

            is_valid, missing = SchemaValidator.validate_fql_expression(
                new_definition, model, strict=True
            )

        # Delete old segment
        old_id = segment.id
        old_name = segment.name
        old_description = segment.description if hasattr(segment, 'description') else ''

        logger.info(f'Deleting segment {old_name} (ID: {old_id}) for update')
        segment.delete()

        # Create new segment with updated definition
        new_segment = fdl.Segment(
            model_id=segment.model_id,
            name=old_name,
            description=old_description,
            definition=new_definition,
        )
        new_segment.create()

        logger.info(
            f'Recreated segment {new_segment.name} (ID: {new_segment.id}) '
            f'with updated definition'
        )

        return new_segment

    def find_segments_with_same_definition(
        self, model_id: str
    ) -> Dict[str, List[fdl.Segment]]:
        """Find segments that have identical FQL definitions but different names.

        This is useful for identifying redundant segments that define the same
        data subset but with different names.

        Note: Duplicate segment names are not possible in Fiddler (enforced by API),
        so this function only checks for duplicate definitions.

        Args:
            model_id: Model ID

        Returns:
            Dictionary mapping normalized definition to list of segments with that definition.
            Only includes definitions that have 2+ segments.

        Example:
            ```python
            mgr = SegmentManager()
            duplicates = mgr.find_segments_with_same_definition(model.id)

            if duplicates:
                print("Found segments with identical definitions:")
                for defn, segments in duplicates.items():
                    print(f"\nDefinition: {defn}")
                    for seg in segments:
                        print(f"  - {seg.name} (ID: {seg.id})")
            else:
                print("No duplicate definitions found")
            ```
        """
        segments = self._list_assets(model_id)
        def_map = {}

        for segment in segments:
            # Normalize definition for comparison
            norm_def = fql.normalize_expression(segment.definition)
            if norm_def not in def_map:
                def_map[norm_def] = []
            def_map[norm_def].append(segment)

        # Keep only definitions with 2+ segments
        duplicates = {
            defn: segs
            for defn, segs in def_map.items()
            if len(segs) > 1
        }

        if duplicates:
            logger.warning(
                f'Found {len(duplicates)} FQL definitions with multiple segments '
                f'in model {model_id}'
            )

        return duplicates
