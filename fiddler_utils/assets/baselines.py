"""Baseline management utilities for Fiddler baselines.

This module provides BaselineManager for exporting and importing baseline
definitions (static and rolling baselines).
"""

from typing import List, Dict, Any, Optional
import logging

import fiddler as fdl

from ..exceptions import AssetImportError, ValidationError

logger = logging.getLogger(__name__)


class BaselineManager:
    """Manager for Fiddler baseline operations.

    Handles export and import of baseline definitions (both static and rolling).
    Note: Static baselines require datasets to be published before creation.

    Example:
        ```python
        from fiddler_utils.assets import BaselineManager

        mgr = BaselineManager()

        # Export baselines from source model
        baselines = mgr.export_baselines(source_model_id)

        # Import to target model
        for baseline_data in baselines:
            mgr.import_baseline(target_model_id, baseline_data)
        ```
    """

    def list_baselines(self, model_id: str) -> List[fdl.Baseline]:
        """List all baselines for a model.

        Args:
            model_id: Model UUID

        Returns:
            List of Baseline objects

        Raises:
            AttributeError: If fdl.Baseline.list() is not available
            Exception: If listing fails for other reasons

        Note:
            This method requires fdl.Baseline.list() to be available in the SDK.
        """
        if not hasattr(fdl.Baseline, 'list'):
            raise AttributeError(
                'fdl.Baseline.list() is not available. '
                'Baseline enumeration requires this method to be present in the Fiddler SDK.'
            )

        baselines = list(fdl.Baseline.list(model_id=model_id))
        logger.debug(f'Listed {len(baselines)} baselines for model {model_id}')

        return baselines

    def export_baselines(self, model_id: str) -> List[Dict[str, Any]]:
        """Export baseline definitions from a model.

        Args:
            model_id: Model UUID

        Returns:
            List of baseline definition dictionaries

        Example:
            ```python
            baselines = mgr.export_baselines(source_model.id)
            # Returns:
            # [
            #     {
            #         'name': 'rolling_baseline_1week',
            #         'type': 'ROLLING',
            #         'environment': 'PRODUCTION',
            #         'window_bin_size': 'DAY',
            #         'offset_delta': 7
            #     },
            #     ...
            # ]
            ```
        """
        baselines = self.list_baselines(model_id)
        baseline_data_list = []

        for baseline in baselines:
            baseline_data = {
                'name': baseline.name,
                'type': str(baseline.type).replace('BaselineType.', ''),
                'environment': str(baseline.environment).replace('EnvType.', ''),
            }

            # Add rolling baseline specific fields
            if baseline.type == fdl.BaselineType.ROLLING:
                baseline_data['window_bin_size'] = str(baseline.window_bin_size).replace('WindowBinSize.', '')
                baseline_data['offset_delta'] = baseline.offset_delta
            else:
                # Static baseline - might have dataset reference
                baseline_data['window_bin_size'] = None
                baseline_data['offset_delta'] = None

            baseline_data_list.append(baseline_data)
            logger.debug(f"Exported baseline '{baseline.name}' ({baseline_data['type']})")

        return baseline_data_list

    def import_baseline(
        self,
        target_model_id: str,
        baseline_data: Dict[str, Any],
        skip_if_exists: bool = True
    ) -> Optional[fdl.Baseline]:
        """Import a baseline definition to target model.

        Args:
            target_model_id: Target model UUID
            baseline_data: Baseline definition dict from export_baselines()
            skip_if_exists: Skip if baseline with same name already exists

        Returns:
            Created Baseline object, or None if skipped/failed

        Raises:
            AssetImportError: If baseline creation fails

        Note:
            * Static baselines require the dataset to be published first
            * Rolling baselines can be created immediately
            * Baseline names must be unique within a model

        Example:
            ```python
            baseline_data = {
                'name': 'rolling_baseline_1week',
                'type': 'ROLLING',
                'environment': 'PRODUCTION',
                'window_bin_size': 'DAY',
                'offset_delta': 7
            }
            baseline = mgr.import_baseline(target_model.id, baseline_data)
            ```
        """
        baseline_name = baseline_data['name']
        baseline_type = baseline_data['type']

        # Check if baseline already exists
        if skip_if_exists:
            try:
                existing = fdl.Baseline.from_name(name=baseline_name, model_id=target_model_id)
                logger.info(f"Baseline '{baseline_name}' already exists, skipping")
                return None
            except:
                # Baseline doesn't exist, proceed
                pass

        # Parse enums (convert to uppercase to match enum names)
        baseline_type_enum = getattr(fdl.BaselineType, baseline_type.upper())
        environment_enum = getattr(fdl.EnvType, baseline_data['environment'].upper())

        try:
            if baseline_type_enum == fdl.BaselineType.ROLLING:
                # Create rolling baseline
                window_bin_size_enum = getattr(fdl.WindowBinSize, baseline_data['window_bin_size'].upper())

                baseline = fdl.Baseline(
                    model_id=target_model_id,
                    name=baseline_name,
                    type_=baseline_type_enum,
                    environment=environment_enum,
                    window_bin_size=window_bin_size_enum,
                    offset_delta=baseline_data['offset_delta']
                )
                baseline.create()
                logger.info(f"Created rolling baseline '{baseline_name}'")

            else:
                # Static baseline - requires dataset
                logger.warning(
                    f"Cannot automatically create static baseline '{baseline_name}'. "
                    f"Static baselines require a dataset to be published first. "
                    f"You must manually publish the baseline dataset and create the baseline."
                )
                return None

            return baseline

        except Exception as e:
            raise AssetImportError(f"Failed to import baseline '{baseline_name}': {str(e)}")

    def create_rolling_baseline(
        self,
        model_id: str,
        name: str,
        window_bin_size: str = 'DAY',
        offset_delta: int = 7,
        environment: str = 'PRODUCTION'
    ) -> fdl.Baseline:
        """Helper to create a rolling baseline.

        Args:
            model_id: Model UUID
            name: Baseline name
            window_bin_size: Window bin size (DAY, HOUR, WEEK, MONTH)
            offset_delta: How far back (multiple of window_bin_size)
            environment: Environment (PRODUCTION or PRE_PRODUCTION)

        Returns:
            Created Baseline object

        Example:
            ```python
            # Create 7-day rolling baseline
            baseline = mgr.create_rolling_baseline(
                model_id=model.id,
                name='rolling_1week',
                window_bin_size='DAY',
                offset_delta=7
            )
            ```
        """
        window_bin_size_enum = getattr(fdl.WindowBinSize, window_bin_size)
        environment_enum = getattr(fdl.EnvType, environment)

        baseline = fdl.Baseline(
            model_id=model_id,
            name=name,
            type_=fdl.BaselineType.ROLLING,
            environment=environment_enum,
            window_bin_size=window_bin_size_enum,
            offset_delta=offset_delta
        )

        baseline.create()
        logger.info(f"Created rolling baseline '{name}' ({offset_delta} {window_bin_size})")

        return baseline
