"""Example: Working with Fiddler baselines using BaselineManager.

This script demonstrates how to use BaselineManager to:
* List all baselines for a model
* Export baseline definitions for backup or migration
* Import baselines to another model
* Analyze baseline configurations
* Handle different baseline types (static, rolling, pre-production)

Baselines are used for drift detection and comparison in Fiddler monitoring.
"""

import fiddler as fdl
from fiddler_utils import (
    BaselineManager,
    get_or_init,
    ConnectionManager,
    SchemaValidator,
    configure_fiddler_logging,
)

# ============================================================================
# Configuration
# ============================================================================

# Source model
SOURCE_URL = 'https://source.fiddler.ai'
SOURCE_TOKEN = 'your_source_token'
SOURCE_PROJECT = 'my_project'
SOURCE_MODEL = 'my_model_v1'

# Target model (can be same or different instance)
TARGET_URL = 'https://target.fiddler.ai'
TARGET_TOKEN = 'your_target_token'
TARGET_PROJECT = 'my_project'
TARGET_MODEL = 'my_model_v2'

# Baselines to export (empty = all)
BASELINES_TO_EXPORT = []  # e.g., ['7_day_rolling', 'production_baseline']


# ============================================================================
# Example 1: List and Analyze Baselines
# ============================================================================


def list_and_analyze_baselines():
    """List all baselines for a model and analyze their configuration."""

    print('=' * 70)
    print('EXAMPLE 1: List and Analyze Baselines')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    # Get model
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    print(f'\nModel: {model.name} (ID: {model.id})')

    # List all baselines
    baseline_mgr = BaselineManager()
    baselines = baseline_mgr.list_assets(model_id=model.id)

    print(f'\n✓ Found {len(baselines)} baselines:')

    if not baselines:
        print('  (No baselines configured)')
        return

    # Display baseline details
    for baseline in baselines:
        print(f'\n  📈 {baseline.name}')
        print(f'     Type: {baseline.type}')
        print(f'     Environment: {baseline.environment}')
        print(f'     ID: {baseline.id}')

        # Type-specific details
        if baseline.type == fdl.BaselineType.STATIC:
            if hasattr(baseline, 'dataset_id'):
                print(f'     Dataset ID: {baseline.dataset_id}')

        elif baseline.type == fdl.BaselineType.ROLLING:
            if hasattr(baseline, 'window_bin_size'):
                print(f'     Window: {baseline.window_bin_size}')
            if hasattr(baseline, 'offset_delta'):
                print(f'     Offset: {baseline.offset_delta} bins')

        elif baseline.type == fdl.BaselineType.PRE_PRODUCTION:
            if hasattr(baseline, 'dataset_id'):
                print(f'     Dataset ID: {baseline.dataset_id}')

    # Analyze baseline distribution
    print(f'\n📊 Baseline Analysis:')
    type_counts = {}
    env_counts = {}

    for baseline in baselines:
        baseline_type = str(baseline.type)
        type_counts[baseline_type] = type_counts.get(baseline_type, 0) + 1

        env = str(baseline.environment)
        env_counts[env] = env_counts.get(env, 0) + 1

    print(f'  By type:')
    for bl_type, count in type_counts.items():
        print(f'    {bl_type}: {count}')

    print(f'  By environment:')
    for env, count in env_counts.items():
        print(f'    {env}: {count}')


# ============================================================================
# Example 2: Export Baselines
# ============================================================================


def export_baselines():
    """Export baseline definitions from a model."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Export Baselines')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    # Get model
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    print(f'\nExporting baselines from: {model.name}')

    # Export baselines
    baseline_mgr = BaselineManager()
    exported_baselines = baseline_mgr.export_assets(
        model_id=model.id,
        names=BASELINES_TO_EXPORT or None  # None = export all
    )

    print(f'\n✓ Exported {len(exported_baselines)} baselines:')

    for baseline_data in exported_baselines:
        print(f'\n  - {baseline_data.name}')
        print(f'      Type: {baseline_data.data.get("type")}')
        print(f'      Environment: {baseline_data.data.get("environment")}')

        # Show referenced columns (if any)
        if baseline_data.referenced_columns:
            print(f'      Columns: {baseline_data.referenced_columns}')

    return exported_baselines


# ============================================================================
# Example 3: Import Baselines to Target Model
# ============================================================================


def import_baselines_to_target(exported_baselines):
    """Import baselines to a target model."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Import Baselines to Target Model')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to target
    get_or_init(url=TARGET_URL, token=TARGET_TOKEN, log_level='ERROR')

    # Get target model
    target_proj = fdl.Project.from_name(TARGET_PROJECT)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_proj.id)

    print(f'\nTarget model: {target_model.name}')
    print(f'  Project: {target_proj.name}')
    print(f'  Model ID: {target_model.id}')

    # Dry run first
    print('\n[DRY RUN] Validating baselines...')
    baseline_mgr = BaselineManager()

    dry_result = baseline_mgr.import_assets(
        target_model_id=target_model.id,
        assets=exported_baselines,
        validate=True,
        dry_run=True,
        skip_invalid=False
    )

    print(f'  Would import: {dry_result.successful}')
    print(f'  Would skip: {dry_result.skipped}')
    print(f'  Validation errors: {len(dry_result.errors)}')

    if dry_result.errors:
        print(f'\n  ⚠️ Validation issues:')
        for name, error in dry_result.errors[:3]:
            print(f'    - {name}: {error}')

    # Ask for confirmation
    if dry_result.errors:
        print(f'\n⚠️ Some baselines have validation errors.')
        print(f'   Continue with skip_invalid=True? (Only valid baselines will import)')

    proceed = input('\nProceed with actual import? (y/n): ').lower().strip()

    if proceed != 'y':
        print('\nImport cancelled.')
        return

    # Actual import
    print('\n[ACTUAL IMPORT] Importing baselines...')
    result = baseline_mgr.import_assets(
        target_model_id=target_model.id,
        assets=exported_baselines,
        validate=True,
        dry_run=False,
        skip_invalid=True  # Skip invalid, import valid ones
    )

    print(f'\n✓ Import complete:')
    print(f'  Successful: {result.successful}')
    print(f'  Skipped: {result.skipped}')
    print(f'  Failed: {result.failed}')

    if result.errors:
        print(f'\n  Errors:')
        for name, error in result.errors:
            print(f'    - {name}: {error}')

    return result


# ============================================================================
# Example 4: Copy Baselines Within Same Instance
# ============================================================================


def copy_baselines_same_instance():
    """Copy baselines from one model to another (same instance)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Copy Baselines (Same Instance)')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    # Get models
    project = fdl.Project.from_name(SOURCE_PROJECT)
    source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    print(f'\nCopying baselines:')
    print(f'  From: {source_model.name}')
    print(f'  To: {target_model.name}')

    # Use copy_assets method (convenience for same-instance operations)
    baseline_mgr = BaselineManager()
    result = baseline_mgr.copy_assets(
        source_model_id=source_model.id,
        target_model_id=target_model.id,
        names=None,  # Copy all baselines
        validate=True
    )

    print(f'\n✓ Copy complete:')
    print(f'  Successful: {result.successful}')
    print(f'  Skipped: {result.skipped}')
    print(f'  Failed: {result.failed}')

    if result.errors:
        print(f'\n  Errors:')
        for name, error in result.errors:
            print(f'    - {name}: {error}')


# ============================================================================
# Example 5: Cross-Instance Baseline Migration
# ============================================================================


def cross_instance_migration():
    """Complete workflow: Export from source instance, import to target instance."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Cross-Instance Baseline Migration')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Setup connection manager
    conn_mgr = ConnectionManager(log_level='ERROR')
    conn_mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
    conn_mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

    baseline_mgr = BaselineManager()

    # Step 1: Export from source
    print('\n[Step 1] Exporting from source instance...')
    with conn_mgr.use('source'):
        source_proj = fdl.Project.from_name(SOURCE_PROJECT)
        source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=source_proj.id)

        print(f'  Source: {source_model.name}')
        exported = baseline_mgr.export_assets(model_id=source_model.id)
        print(f'  ✓ Exported {len(exported)} baselines')

    # Step 2: Validate target schema
    print('\n[Step 2] Validating target model schema...')
    with conn_mgr.use('target'):
        target_proj = fdl.Project.from_name(TARGET_PROJECT)
        target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_proj.id)

        print(f'  Target: {target_model.name}')

        # Check if schemas are compatible
        # (Baseline import doesn't require schema validation like segments/metrics,
        #  but it's good practice to verify the models are similar)

    # Step 3: Import to target
    print('\n[Step 3] Importing to target instance...')
    with conn_mgr.use('target'):
        result = baseline_mgr.import_assets(
            target_model_id=target_model.id,
            assets=exported,
            validate=True,
            dry_run=False,
            skip_invalid=True
        )

        print(f'  ✓ Import complete:')
        print(f'    Successful: {result.successful}')
        print(f'    Skipped: {result.skipped}')
        print(f'    Failed: {result.failed}')

    print(f'\n✅ Migration complete!')


# ============================================================================
# Example 6: Create New Baselines Programmatically
# ============================================================================


def create_baselines_example():
    """Create different types of baselines programmatically."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Create Baselines Programmatically')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    # Get model
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    print(f'\nCreating baselines for model: {model.name}')

    # Example 1: Rolling baseline (7-day)
    print('\n  Creating 7-day rolling baseline...')
    try:
        rolling_baseline = fdl.Baseline(
            model_id=model.id,
            name='7_day_rolling',
            type_=fdl.BaselineType.ROLLING,
            environment=fdl.EnvType.PRODUCTION,
            window_bin_size=fdl.WindowBinSize.DAY,
            offset_delta=7
        )
        rolling_baseline.create()
        print(f'    ✓ Created: {rolling_baseline.name}')
    except Exception as e:
        print(f'    ✗ Failed: {e}')

    # Example 2: Rolling baseline (30-day)
    print('\n  Creating 30-day rolling baseline...')
    try:
        rolling_baseline_30 = fdl.Baseline(
            model_id=model.id,
            name='30_day_rolling',
            type_=fdl.BaselineType.ROLLING,
            environment=fdl.EnvType.PRODUCTION,
            window_bin_size=fdl.WindowBinSize.DAY,
            offset_delta=30
        )
        rolling_baseline_30.create()
        print(f'    ✓ Created: {rolling_baseline_30.name}')
    except Exception as e:
        print(f'    ✗ Failed: {e}')

    # Note: Static and PRE_PRODUCTION baselines require dataset_id
    # which would come from publishing a baseline dataset
    print(f'\n💡 Note: Static and PRE_PRODUCTION baselines require dataset_id')
    print(f'   from publishing baseline data with environment=EnvType.PRE_PRODUCTION')


# ============================================================================
# Example 7: Baseline Comparison Across Models
# ============================================================================


def compare_baselines_across_models():
    """Compare baseline configurations across multiple models."""

    print('\n' + '=' * 70)
    print('EXAMPLE 7: Compare Baselines Across Models')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    project = fdl.Project.from_name(SOURCE_PROJECT)

    # Get multiple models
    model_names = [SOURCE_MODEL, TARGET_MODEL]  # Add more as needed
    baseline_mgr = BaselineManager()

    print('\nComparing baseline configurations:')

    baseline_summary = {}
    for model_name in model_names:
        try:
            model = fdl.Model.from_name(model_name, project_id=project.id)
            baselines = baseline_mgr.list_assets(model_id=model.id)

            baseline_summary[model_name] = {
                'count': len(baselines),
                'types': {},
                'names': [bl.name for bl in baselines]
            }

            # Count by type
            for bl in baselines:
                bl_type = str(bl.type)
                baseline_summary[model_name]['types'][bl_type] = \
                    baseline_summary[model_name]['types'].get(bl_type, 0) + 1

        except Exception as e:
            print(f'  ✗ Failed to get baselines for {model_name}: {e}')
            continue

    # Display comparison
    for model_name, summary in baseline_summary.items():
        print(f'\n  {model_name}:')
        print(f'    Total baselines: {summary["count"]}')
        if summary['types']:
            print(f'    By type:')
            for bl_type, count in summary['types'].items():
                print(f'      {bl_type}: {count}')
        print(f'    Names: {", ".join(summary["names"][:5])}')
        if len(summary['names']) > 5:
            print(f'      ... and {len(summary["names"]) - 5} more')

    # Find common and different baselines
    if len(baseline_summary) >= 2:
        model_list = list(baseline_summary.keys())
        model_a, model_b = model_list[0], model_list[1]

        names_a = set(baseline_summary[model_a]['names'])
        names_b = set(baseline_summary[model_b]['names'])

        common = names_a & names_b
        only_a = names_a - names_b
        only_b = names_b - names_a

        print(f'\nComparison between {model_a} and {model_b}:')
        print(f'  Common baselines: {len(common)}')
        if common:
            print(f'    {", ".join(list(common)[:5])}')
        print(f'  Only in {model_a}: {len(only_a)}')
        if only_a:
            print(f'    {", ".join(list(only_a)[:5])}')
        print(f'  Only in {model_b}: {len(only_b)}')
        if only_b:
            print(f'    {", ".join(list(only_b)[:5])}')


# ============================================================================
# Main
# ============================================================================


def main():
    """Run baseline management examples."""

    # Suppress verbose logs for all examples
    configure_fiddler_logging(level='ERROR')

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 20 + 'BASELINE MANAGER EXAMPLES' + ' ' * 23 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: List and analyze
    list_and_analyze_baselines()

    # Example 2: Export baselines
    # exported_baselines = export_baselines()

    # Example 3: Import to target
    # import_baselines_to_target(exported_baselines)

    # Example 4: Copy within same instance
    # copy_baselines_same_instance()

    # Example 5: Cross-instance migration
    # cross_instance_migration()

    # Example 6: Create baselines programmatically
    # create_baselines_example()

    # Example 7: Compare across models
    # compare_baselines_across_models()

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* Baselines are used for drift detection and comparison')
    print('* Types: STATIC, ROLLING, PRE_PRODUCTION')
    print('* Rolling baselines reference production data automatically')
    print('* Static/pre-production baselines require dataset_id')
    print('* Baseline export/import preserves configuration, not data')
    print('* Use dry_run=True to validate before importing')
    print('* Cross-instance migration requires ConnectionManager')
    print('=' * 70)


if __name__ == '__main__':
    main()
