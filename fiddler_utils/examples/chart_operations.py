"""Example: Working with Fiddler charts using ChartManager.

This script demonstrates how to use ChartManager to:
1. List all charts in a project
2. Export charts for backup/migration
3. Import charts to another project/model
4. Analyze chart usage
5. Delete old charts

IMPORTANT: The chart API is unofficial and may change without notice.
"""

import fiddler as fdl
from fiddler_utils import ChartManager

# ============================================================================
# Configuration
# ============================================================================

# Source instance
SOURCE_URL = 'https://source.fiddler.ai'
SOURCE_TOKEN = 'your_source_token'
SOURCE_PROJECT = 'my_project'
SOURCE_MODEL = 'my_model'

# Target instance (can be same as source)
TARGET_URL = 'https://target.fiddler.ai'
TARGET_TOKEN = 'your_target_token'
TARGET_PROJECT = 'my_project_copy'
TARGET_MODEL = 'my_model_copy'


# ============================================================================
# Example 1: List and Analyze Charts
# ============================================================================


def list_and_analyze_charts():
    """List all charts and get analytics."""

    print('=' * 70)
    print('EXAMPLE 1: List and Analyze Charts')
    print('=' * 70)

    # Suppress verbose Fiddler client logs (recommended)

    # Initialize chart manager
    # IMPORTANT: ChartManager requires explicit URL and token
    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    # Also initialize fiddler client for model access
    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # List all charts in project
    all_charts = chart_mgr.list_charts(project_id=project.id)
    print(f'\nFound {len(all_charts)} total charts in project')

    for chart in all_charts[:5]:  # Show first 5
        print(f'\n  Chart: {chart["title"]}')
        print(f'    Type: {chart.get("query_type")}')
        print(f'    ID: {chart.get("id")}')

    # Filter charts by model
    model_charts = chart_mgr.list_charts(project_id=project.id, model_id=model.id)
    print(f"\n{len(model_charts)} charts reference model '{model.name}'")

    # Analyze chart usage
    analysis = chart_mgr.analyze_charts(project_id=project.id)
    print(f'\nChart Analysis:')
    print(f'  Total charts: {analysis["total"]}')
    print(f'  By type: {analysis["by_type"]}')
    print(f'  Unique models: {analysis["unique_models"]}')

    print('\n' + '=' * 70)


# ============================================================================
# Example 2: Export Charts for Backup
# ============================================================================


def export_charts_for_backup():
    """Export all charts from a project."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Export Charts for Backup')
    print('=' * 70)

    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # Export all charts for a model
    exported_charts = chart_mgr.export_charts(project_id=project.id, model_id=model.id)

    print(f'\nExported {len(exported_charts)} charts')

    for chart in exported_charts:
        print(f'\n  {chart["title"]}')
        print(f'    Type: {chart.get("query_type")}')
        print(f'    Description: {chart.get("description", "N/A")[:50]}')

    # Could save to file for backup
    # import json
    # with open('charts_backup.json', 'w') as f:
    #     json.dump(exported_charts, f, indent=2)

    print('\n' + '=' * 70)
    return exported_charts


# ============================================================================
# Example 3: Import Charts to Another Model
# ============================================================================


def import_charts_to_target(exported_charts):
    """Import charts to a target model."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Import Charts to Target Model')
    print('=' * 70)

    # Initialize chart manager for target
    chart_mgr = ChartManager(url=TARGET_URL, token=TARGET_TOKEN)

    # Initialize fiddler client for target
    fdl.init(url=TARGET_URL, token=TARGET_TOKEN)
    target_project = fdl.Project.from_name(TARGET_PROJECT)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_project.id)

    print(f'\nTarget: {target_model.name} in project {target_project.name}')

    # Dry run first to check what would happen
    print('\nRunning dry run...')
    dry_result = chart_mgr.import_charts(
        target_project_id=target_project.id
        target_model_id=target_model.id
        charts=exported_charts
        dry_run=True
    )

    print(f'  Would import: {dry_result["successful"]} charts')
    print(f'  Would fail: {dry_result["failed"]} charts')

    if dry_result['failed'] > 0:
        print(f'\n  Errors:')
        for title, error in dry_result['errors'][:3]:
            print(f'    - {title}: {error}')

    # Ask for confirmation
    proceed = input('\nProceed with actual import? (y/n): ').lower().strip()

    if proceed == 'y':
        print('\nImporting charts...')
        result = chart_mgr.import_charts(
            target_project_id=target_project.id
            target_model_id=target_model.id
            charts=exported_charts
            dry_run=False
        )

        print(f'\nImport complete:')
        print(f'  ✓ Successful: {result["successful"]}')
        print(f'  ✗ Failed: {result["failed"]}')

        if result['failed'] > 0:
            print(f'\n  Errors:')
            for title, error in result['errors']:
                print(f'    - {title}: {error}')
    else:
        print('\nImport cancelled.')

    print('\n' + '=' * 70)


# ============================================================================
# Example 4: Find and Delete Specific Charts
# ============================================================================


def find_and_delete_old_charts():
    """Find charts by pattern and optionally delete them."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Find and Delete Old Charts')
    print('=' * 70)

    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)

    # List all charts
    all_charts = chart_mgr.list_charts(project_id=project.id)

    # Find charts with specific pattern in title
    pattern = 'test'  # Find charts with "test" in title
    matching_charts = [
        c for c in all_charts if pattern.lower() in c.get('title', '').lower()
    ]

    print(f"\nFound {len(matching_charts)} charts matching pattern '{pattern}'")

    if not matching_charts:
        print('No charts to delete.')
        print('\n' + '=' * 70)
        return

    for chart in matching_charts:
        print(f'\n  - {chart["title"]} (ID: {chart["id"]})')

    # Ask for confirmation
    proceed = (
        input(f'\nDelete these {len(matching_charts)} charts? (y/n): ').lower().strip()
    )

    if proceed == 'y':
        deleted_count = 0
        for chart in matching_charts:
            try:
                chart_mgr.delete_chart(chart['id'])
                print(f'  ✓ Deleted: {chart["title"]}')
                deleted_count += 1
            except Exception as e:
                print(f'  ✗ Failed to delete {chart["title"]}: {e}')

        print(f'\nDeleted {deleted_count} out of {len(matching_charts)} charts')
    else:
        print('\nDeletion cancelled.')

    print('\n' + '=' * 70)


# ============================================================================
# Example 5: Get Specific Chart and Inspect
# ============================================================================


def inspect_specific_chart():
    """Get and inspect a specific chart's configuration."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Inspect Specific Chart')
    print('=' * 70)

    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)

    # Get chart by title
    chart_title = 'Performance Over Time'  # Replace with actual chart title

    try:
        chart = chart_mgr.get_chart_by_title(project_id=project.id, title=chart_title)

        print(f'\nChart: {chart["title"]}')
        print(f'  ID: {chart["id"]}')
        print(f'  Type: {chart.get("query_type")}')
        print(f'  Description: {chart.get("description", "N/A")}')

        # Inspect data source
        data_source = chart.get('data_source', {})
        print(f'\n  Data Source:')
        print(f'    Query Type: {data_source.get("query_type")}')

        if data_source.get('query_type') == 'MONITORING':
            queries = data_source.get('queries', [])
            print(f'    Queries: {len(queries)}')

            for i, query in enumerate(queries[:3]):  # Show first 3
                print(f'\n    Query {i + 1}:')
                print(f'      Metric: {query.get("metric")}')
                print(f'      Metric Type: {query.get("metric_type")}')
                print(f'      Viz Type: {query.get("viz_type")}')
                if query.get('columns'):
                    print(f'      Columns: {query.get("columns")}')

    except Exception as e:
        print(f"\nChart '{chart_title}' not found: {e}")

    print('\n' + '=' * 70)


# ============================================================================
# Example 6: Copy Charts Between Models (Same Instance)
# ============================================================================


def copy_charts_between_models():
    """Copy charts from one model to another on the same instance."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Copy Charts Between Models')
    print('=' * 70)

    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    print(f"Copying charts from '{source_model.name}' to '{target_model.name}'")

    # Export from source model
    exported = chart_mgr.export_charts(project_id=project.id, model_id=source_model.id)

    print(f'\nExported {len(exported)} charts')

    # Import to target model
    result = chart_mgr.import_charts(
        target_project_id=project.id
        target_model_id=target_model.id
        charts=exported
        dry_run=False
    )

    print(f'\nCopy complete:')
    print(f'  ✓ Successful: {result["successful"]}')
    print(f'  ✗ Failed: {result["failed"]}')

    print('\n' + '=' * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""

    # Suppress verbose logs for cleaner output (recommended)

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 20 + 'CHART MANAGER EXAMPLES' + ' ' * 26 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: List and analyze
    list_and_analyze_charts()

    # Example 2: Export for backup
    exported_charts = export_charts_for_backup()

    # Example 3: Import to target
    # import_charts_to_target(exported_charts)

    # Example 4: Find and delete
    # find_and_delete_old_charts()

    # Example 5: Inspect specific chart
    # inspect_specific_chart()

    # Example 6: Copy between models
    # copy_charts_between_models()

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* Chart API is unofficial - use with caution')
    print('* ChartManager requires explicit URL and token')
    print('* ID resolution (baselines, metrics, segments) may fail if')
    print("  assets don't exist in target with same names")
    print('* Test with dry_run=True before actual imports')
    print('=' * 70)


if __name__ == '__main__':
    main()
