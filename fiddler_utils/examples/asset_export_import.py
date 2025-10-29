"""Example: Export and import assets between Fiddler models.

This script demonstrates how to use fiddler_utils asset managers to:
1. Export segments and custom metrics from a source model
2. Validate assets against a target model schema
3. Import assets to a target model

This replaces ~300 lines of manual code with a clean, reusable API.
"""

import fiddler as fdl
from fiddler_utils import (
    get_or_init,
    ConnectionManager,
    SegmentManager,
    CustomMetricManager,
    SchemaValidator,
)

# ============================================================================
# Configuration
# ============================================================================

# Source instance
SOURCE_URL = 'https://source.fiddler.ai'
SOURCE_TOKEN = 'your_source_token'
SOURCE_PROJECT = 'my_project'
SOURCE_MODEL = 'my_model_v1'

# Target instance (can be same as source)
TARGET_URL = 'https://target.fiddler.ai'
TARGET_TOKEN = 'your_target_token'
TARGET_PROJECT = 'my_project'
TARGET_MODEL = 'my_model_v2'

# Assets to export (empty = all)
SEGMENTS_TO_EXPORT = []  # e.g., ['high_value', 'critical_issues']
METRICS_TO_EXPORT = []  # e.g., ['revenue_lost', 'accuracy']


# ============================================================================
# Main Script
# ============================================================================


def main():
    print('=' * 70)
    print('FIDDLER ASSET EXPORT/IMPORT DEMO')
    print('=' * 70)

    # Setup connection manager for handling multiple instances
    conn_mgr = ConnectionManager()
    conn_mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
    conn_mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

    # Initialize managers
    segment_mgr = SegmentManager()
    metric_mgr = CustomMetricManager()

    # ------------------------------------------------------------------------
    # Step 1: Connect to source and export assets
    # ------------------------------------------------------------------------
    print('\n[STEP 1] Exporting from source model...')
    print('-' * 70)

    with conn_mgr.use('source'):
        # Get source model
        source_proj = fdl.Project.from_name(SOURCE_PROJECT)
        source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=source_proj.id)

        print(f'Source: {source_model.name} (ID: {source_model.id})')

        # Export segments
        exported_segments = segment_mgr.export_assets(
            model_id=source_model.id, names=SEGMENTS_TO_EXPORT or None
        )
        print(f'\n✓ Exported {len(exported_segments)} segments:')
        for seg in exported_segments:
            print(f'  - {seg.name}')
            print(f'    Definition: {seg.data["definition"]}')
            print(f'    Columns: {seg.referenced_columns}')

        # Export custom metrics
        exported_metrics = metric_mgr.export_assets(
            model_id=source_model.id, names=METRICS_TO_EXPORT or None
        )
        print(f'\n✓ Exported {len(exported_metrics)} custom metrics:')
        for metric in exported_metrics:
            print(f'  - {metric.name}')
            print(f'    Definition: {metric.data["definition"]}')
            print(f'    Columns: {metric.referenced_columns}')

    # ------------------------------------------------------------------------
    # Step 2: Connect to target and validate schema
    # ------------------------------------------------------------------------
    print('\n[STEP 2] Validating target model schema...')
    print('-' * 70)

    with conn_mgr.use('target'):
        # Get target model
        target_proj = fdl.Project.from_name(TARGET_PROJECT)
        target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_proj.id)

        print(f'Target: {target_model.name} (ID: {target_model.id})')

        # Compare schemas
        source_model_for_comparison = (
            fdl.Model.get(id_=source_model.id) if SOURCE_URL == TARGET_URL else None
        )

        if source_model_for_comparison:
            comparison = SchemaValidator.compare_schemas(
                source_model_for_comparison, target_model
            )
            print(f'\nSchema Comparison:')
            print(f'  Common columns: {len(comparison.in_both)}')
            print(f'  Only in source: {len(comparison.only_in_source)}')
            print(f'  Only in target: {len(comparison.only_in_target)}')
            print(f'  Compatible: {comparison.is_compatible}')

            if comparison.only_in_source:
                print(f'\n⚠ Columns in source but missing in target:')
                for col in list(comparison.only_in_source)[:5]:
                    print(f'    - {col}')
                if len(comparison.only_in_source) > 5:
                    print(f'    ... and {len(comparison.only_in_source) - 5} more')

    # ------------------------------------------------------------------------
    # Step 3: Dry run import (validation only)
    # ------------------------------------------------------------------------
    print('\n[STEP 3] Dry run - validating assets...')
    print('-' * 70)

    with conn_mgr.use('target'):
        # Dry run for segments
        seg_result_dry = segment_mgr.import_assets(
            target_model_id=target_model.id,
            assets=exported_segments,
            validate=True,
            dry_run=True,
        )

        print(f'\nSegment dry run results:')
        print(f'  Would import: {seg_result_dry.successful}')
        print(f'  Would skip: {seg_result_dry.skipped}')

        if seg_result_dry.errors:
            print(f'\n  Validation errors:')
            for name, error in seg_result_dry.errors[:3]:
                print(f'    - {name}: {error}')

        # Dry run for custom metrics
        metric_result_dry = metric_mgr.import_assets(
            target_model_id=target_model.id,
            assets=exported_metrics,
            validate=True,
            dry_run=True,
        )

        print(f'\nCustom metric dry run results:')
        print(f'  Would import: {metric_result_dry.successful}')
        print(f'  Would skip: {metric_result_dry.skipped}')

        if metric_result_dry.errors:
            print(f'\n  Validation errors:')
            for name, error in metric_result_dry.errors[:3]:
                print(f'    - {name}: {error}')

    # ------------------------------------------------------------------------
    # Step 4: Actual import
    # ------------------------------------------------------------------------
    print('\n[STEP 4] Importing assets to target model...')
    print('-' * 70)

    proceed = input('\nProceed with actual import? (y/n): ').lower().strip()

    if proceed == 'y':
        with conn_mgr.use('target'):
            # Import segments
            seg_result = segment_mgr.import_assets(
                target_model_id=target_model.id,
                assets=exported_segments,
                validate=True,
                dry_run=False,
                skip_invalid=True,
            )

            print(f'\nSegment import results:')
            print(f'  ✓ Successful: {seg_result.successful}')
            print(f'  ⊘ Skipped: {seg_result.skipped}')
            print(f'  ✗ Failed: {seg_result.failed}')

            # Import custom metrics
            metric_result = metric_mgr.import_assets(
                target_model_id=target_model.id,
                assets=exported_metrics,
                validate=True,
                dry_run=False,
                skip_invalid=True,
            )

            print(f'\nCustom metric import results:')
            print(f'  ✓ Successful: {metric_result.successful}')
            print(f'  ⊘ Skipped: {metric_result.skipped}')
            print(f'  ✗ Failed: {metric_result.failed}')

            # Summary
            total_success = seg_result.successful + metric_result.successful
            total_skipped = seg_result.skipped + metric_result.skipped
            total_failed = seg_result.failed + metric_result.failed

            print('\n' + '=' * 70)
            print(f'IMPORT COMPLETE')
            print('=' * 70)
            print(f'Total successful: {total_success}')
            print(f'Total skipped: {total_skipped}')
            print(f'Total failed: {total_failed}')
    else:
        print('\nImport cancelled by user.')

    print('\n' + '=' * 70)


# ============================================================================
# Alternative: Simple one-liner for same-instance copy
# ============================================================================


def simple_copy_example():
    """Example of simplest possible asset copy within same instance."""

    # Initialize connection
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN)

    # Get models
    proj = fdl.Project.from_name(SOURCE_PROJECT)
    source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=proj.id)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=proj.id)

    # Copy segments in one line
    segment_mgr = SegmentManager()
    result = segment_mgr.copy_assets(
        source_model_id=source_model.id,
        target_model_id=target_model.id,
        names=['important_segment'],  # or None for all
    )

    print(f'Copied {result.successful} segments, skipped {result.skipped}')


if __name__ == '__main__':
    # Run full demo
    main()

    # Or run simple example
    # simple_copy_example()
