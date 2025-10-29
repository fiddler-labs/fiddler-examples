"""Example: Working with Fiddler dashboards using DashboardManager.

This script demonstrates how to use DashboardManager to:
1. List all dashboards in a project
2. Create dashboards from charts
3. Export dashboards for backup/migration
4. Import dashboards to another project/model
5. Set default dashboards
6. Analyze dashboard usage

IMPORTANT: The dashboard API is unofficial and may change without notice.
"""

import fiddler as fdl
from fiddler_utils import DashboardManager, ChartManager

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
# Example 1: List and Analyze Dashboards
# ============================================================================


def list_and_analyze_dashboards():
    """List all dashboards and get analytics."""

    print('=' * 70)
    print('EXAMPLE 1: List and Analyze Dashboards')
    print('=' * 70)

    # Initialize dashboard manager
    # IMPORTANT: DashboardManager requires explicit URL and token
    dashboard_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    # Also initialize fiddler client for project/model access
    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # List all dashboards in project
    all_dashboards = dashboard_mgr.list_dashboards(project_id=project.id)
    print(f'\nFound {len(all_dashboards)} total dashboards in project')

    for dashboard in all_dashboards[:5]:  # Show first 5
        print(f'\n  Dashboard: {dashboard["title"]}')
        print(f'    UUID: {dashboard.get("uuid")}')
        print(f'    Model: {dashboard.get("model_name")}')
        print(f'    Charts: {len(dashboard.get("layouts", []))}')

    # Filter dashboards by model
    model_dashboards = dashboard_mgr.list_dashboards(
        project_id=project.id, model_id=model.id
    )
    print(f"\n{len(model_dashboards)} dashboards for model '{model.name}'")

    # Analyze dashboard usage
    analysis = dashboard_mgr.analyze_dashboards(project_id=project.id)
    print(f'\nDashboard Analysis:')
    print(f'  Total dashboards: {analysis["total"]}')
    print(f'  By model: {analysis["by_model"]}')
    print(f'  Average charts per dashboard: {analysis["avg_charts_per_dashboard"]:.1f}')

    print('\n' + '=' * 70)


# ============================================================================
# Example 2: Create Dashboard from Charts
# ============================================================================


def create_dashboard_from_charts():
    """Create a new dashboard from existing charts."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Create Dashboard from Charts')
    print('=' * 70)

    dashboard_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)
    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # Get all charts for the model
    charts = chart_mgr.list_charts(project_id=project.id, model_id=model.id)

    print(f"\nFound {len(charts)} charts for model '{model.name}'")

    if not charts:
        print('No charts available. Create some charts first!')
        print('\n' + '=' * 70)
        return

    # Select first few charts
    selected_charts = charts[:4]  # Take first 4 charts
    chart_ids = [c['id'] for c in selected_charts]

    print(f'\nSelected {len(selected_charts)} charts:')
    for chart in selected_charts:
        print(f'  - {chart["title"]}')

    # Create dashboard with automatic grid layout
    print('\nCreating dashboard with automatic 2-column layout...')

    try:
        dashboard = dashboard_mgr.create_dashboard(
            project_id=project.id,
            model_id=model.id,
            title='Example Dashboard (Auto Layout)',
            chart_ids=chart_ids,
            set_as_default=False,
        )

        print(f'\n✓ Created dashboard: {dashboard.get("title")}')
        print(f'  UUID: {dashboard.get("uuid")}')
        print(f'  Charts: {len(dashboard.get("layouts", []))}')

    except Exception as e:
        print(f'\n✗ Failed to create dashboard: {e}')

    print('\n' + '=' * 70)


# ============================================================================
# Example 3: Create Dashboard with Custom Layout
# ============================================================================


def create_dashboard_with_custom_layout():
    """Create a dashboard with custom chart positioning."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Create Dashboard with Custom Layout')
    print('=' * 70)

    dashboard_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)
    chart_mgr = ChartManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # Get charts
    charts = chart_mgr.list_charts(project_id=project.id, model_id=model.id)

    if len(charts) < 3:
        print('Need at least 3 charts for this example')
        print('\n' + '=' * 70)
        return

    # Define custom layout
    # Chart 1: Wide at top (2 columns, 1 row)
    # Chart 2: Bottom left (1 column, 1 row)
    # Chart 3: Bottom right (1 column, 1 row)
    custom_layout = [
        {
            'chart_uuid': charts[0]['id'],
            'grid_props': {
                'position_x': 0,
                'position_y': 0,
                'width': 2,  # Wide - takes 2 columns
                'height': 1,
            },
        },
        {
            'chart_uuid': charts[1]['id'],
            'grid_props': {'position_x': 0, 'position_y': 1, 'width': 1, 'height': 1},
        },
        {
            'chart_uuid': charts[2]['id'],
            'grid_props': {'position_x': 1, 'position_y': 1, 'width': 1, 'height': 1},
        },
    ]

    chart_ids = [charts[0]['id'], charts[1]['id'], charts[2]['id']]

    print(f'\nLayout configuration:')
    print(f'  Row 0: [{charts[0]["title"][:30]}...] (wide)')
    print(f'  Row 1: [{charts[1]["title"][:25]}...] | [{charts[2]["title"][:25]}...]')

    try:
        dashboard = dashboard_mgr.create_dashboard(
            project_id=project.id,
            model_id=model.id,
            title='Example Dashboard (Custom Layout)',
            chart_ids=chart_ids,
            layout=custom_layout,
            set_as_default=False,
        )

        print(f'\n✓ Created dashboard with custom layout')
        print(f'  UUID: {dashboard.get("uuid")}')

    except Exception as e:
        print(f'\n✗ Failed to create dashboard: {e}')

    print('\n' + '=' * 70)


# ============================================================================
# Example 4: Export and Import Dashboard
# ============================================================================


def export_and_import_dashboard():
    """Export a dashboard and import to another model."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Export and Import Dashboard')
    print('=' * 70)

    # Source
    source_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    source_project = fdl.Project.from_name(SOURCE_PROJECT)
    source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=source_project.id)

    # List and select dashboard
    dashboards = source_mgr.list_dashboards(project_id=source_project.id)

    if not dashboards:
        print('No dashboards available to export!')
        print('\n' + '=' * 70)
        return

    dashboard_to_export = dashboards[0]
    print(f'Exporting dashboard: {dashboard_to_export["title"]}')

    # Export
    exported = source_mgr.export_dashboard(
        project_id=source_project.id, dashboard_id=dashboard_to_export['uuid']
    )

    print(f'  Charts in dashboard: {len(exported.get("layouts", []))}')

    # Target
    target_mgr = DashboardManager(url=TARGET_URL, token=TARGET_TOKEN)

    fdl.init(url=TARGET_URL, token=TARGET_TOKEN)
    target_project = fdl.Project.from_name(TARGET_PROJECT)
    target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_project.id)

    print(f'\nImporting to target model: {target_model.name}')

    # Note: This requires charts with same titles to exist in target!
    print('\nNOTE: Import requires charts with matching titles in target model.')
    print('This example will skip the actual import.')

    # To actually import:
    # try:
    #     imported = target_mgr.import_dashboard(
    #         target_project_id=target_project.id,
    #         target_model_id=target_model.id,
    #         dashboard_data=exported
    #     )
    #     print(f"\n✓ Imported dashboard: {imported.get('title')}")
    # except Exception as e:
    #     print(f"\n✗ Import failed: {e}")

    print('\n' + '=' * 70)


# ============================================================================
# Example 5: Set Default Dashboard
# ============================================================================


def set_default_dashboard():
    """Set a dashboard as the default for a model."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Set Default Dashboard')
    print('=' * 70)

    dashboard_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    # List dashboards for model
    dashboards = dashboard_mgr.list_dashboards(project_id=project.id, model_id=model.id)

    if not dashboards:
        print('No dashboards available!')
        print('\n' + '=' * 70)
        return

    dashboard = dashboards[0]
    print(f"Setting '{dashboard['title']}' as default for model '{model.name}'")

    try:
        dashboard_mgr.set_default_dashboard(
            model_id=model.id, dashboard_uuid=dashboard['uuid']
        )
        print(f'\n✓ Successfully set as default dashboard')

    except Exception as e:
        print(f'\n✗ Failed to set default: {e}')

    print('\n' + '=' * 70)


# ============================================================================
# Example 6: Find and Delete Old Dashboards
# ============================================================================


def find_and_delete_old_dashboards():
    """Find dashboards by pattern and optionally delete them."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Find and Delete Old Dashboards')
    print('=' * 70)

    dashboard_mgr = DashboardManager(url=SOURCE_URL, token=SOURCE_TOKEN)

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)
    project = fdl.Project.from_name(SOURCE_PROJECT)

    # List all dashboards
    all_dashboards = dashboard_mgr.list_dashboards(project_id=project.id)

    # Find dashboards with specific pattern in title
    pattern = 'test'  # Find dashboards with "test" in title
    matching_dashboards = [
        d for d in all_dashboards if pattern.lower() in d.get('title', '').lower()
    ]

    print(f"\nFound {len(matching_dashboards)} dashboards matching pattern '{pattern}'")

    if not matching_dashboards:
        print('No dashboards to delete.')
        print('\n' + '=' * 70)
        return

    for dashboard in matching_dashboards:
        print(f'\n  - {dashboard["title"]} (UUID: {dashboard["uuid"]})')

    # Ask for confirmation
    proceed = (
        input(f'\nDelete these {len(matching_dashboards)} dashboards? (y/n): ')
        .lower()
        .strip()
    )

    if proceed == 'y':
        deleted_count = 0
        for dashboard in matching_dashboards:
            try:
                dashboard_mgr.delete_dashboard(dashboard['uuid'])
                print(f'  ✓ Deleted: {dashboard["title"]}')
                deleted_count += 1
            except Exception as e:
                print(f'  ✗ Failed to delete {dashboard["title"]}: {e}')

        print(f'\nDeleted {deleted_count} out of {len(matching_dashboards)} dashboards')
    else:
        print('\nDeletion cancelled.')

    print('\n' + '=' * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 18 + 'DASHBOARD MANAGER EXAMPLES' + ' ' * 24 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: List and analyze
    list_and_analyze_dashboards()

    # Example 2: Create with auto layout
    # create_dashboard_from_charts()

    # Example 3: Create with custom layout
    # create_dashboard_with_custom_layout()

    # Example 4: Export and import
    # export_and_import_dashboard()

    # Example 5: Set default
    # set_default_dashboard()

    # Example 6: Find and delete
    # find_and_delete_old_dashboards()

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* Dashboard API is unofficial - use with caution')
    print('* DashboardManager requires explicit URL and token')
    print('* Importing dashboards requires charts with same titles')
    print('  in target project')
    print('* Chart positions are defined by grid_props:')
    print('  - position_x/position_y: Grid cell position (0-indexed)')
    print('  - width/height: Number of cells to span')
    print('* Default dashboard is shown when opening model in UI')
    print('=' * 70)


if __name__ == '__main__':
    main()
