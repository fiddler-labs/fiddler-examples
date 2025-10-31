"""Example: Environment analysis and reporting using EnvironmentReporter.

This script demonstrates how to use EnvironmentReporter and ProjectManager to:
* Discover and analyze the complete Fiddler environment hierarchy
* Generate comprehensive statistics across projects, models, and features
* Analyze timestamp patterns (creation dates, update dates)
* Export environment data to CSV for further analysis
* Generate formatted reports for documentation and health checks

This replaces the env_stats.ipynb notebook with a reusable, scriptable API.
"""

import fiddler as fdl
from fiddler_utils import (
    EnvironmentReporter,
    ProjectManager,
    get_or_init,
    configure_fiddler_logging,
)

# ============================================================================
# Configuration
# ============================================================================

# Fiddler instance connection
FIDDLER_URL = 'https://your-instance.fiddler.ai'
FIDDLER_TOKEN = 'your_api_token'

# Export configuration
EXPORT_DIR = 'environment_exports'
EXPORT_PREFIX = 'env_analysis'


# ============================================================================
# Example 1: Quick Environment Summary (EnvironmentReporter)
# ============================================================================


def quick_environment_summary():
    """Generate a quick environment summary using EnvironmentReporter."""

    print('=' * 70)
    print('EXAMPLE 1: Quick Environment Summary')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    # Create reporter and run analysis
    print('\nAnalyzing environment (this may take a few moments)...')
    reporter = EnvironmentReporter()
    reporter.analyze_environment(
        include_features=True,
        include_timestamps=True,
        include_assets=False  # Set to True for asset counts (slower)
    )

    # Generate formatted report
    reporter.generate_report(
        show_projects=True,
        show_models=True,
        show_timestamps=True,
        show_newest_oldest=True,
        top_n=15
    )

    # Export to CSV
    print('\nExporting to CSV...')
    files = reporter.export_to_csv(
        output_dir=EXPORT_DIR,
        prefix=EXPORT_PREFIX
    )
    print(f'✓ Exported {len(files)} files:')
    for file in files:
        print(f'  - {file}')


# ============================================================================
# Example 2: Detailed Environment Analysis (ProjectManager)
# ============================================================================


def detailed_environment_analysis():
    """Use ProjectManager for granular control over environment analysis."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Detailed Environment Analysis (ProjectManager)')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    # Create project manager
    mgr = ProjectManager()

    # Step 1: List all projects
    print('\n[Step 1] Listing projects...')
    projects = mgr.list_projects()
    print(f'✓ Found {len(projects)} projects:')
    for proj in projects[:10]:  # Show first 10
        print(f'  - {proj.name} (ID: {proj.id})')

    # Step 2: Get environment hierarchy
    print('\n[Step 2] Building environment hierarchy...')
    hierarchy = mgr.get_environment_hierarchy(
        include_features=True,
        include_timestamps=True
    )

    print(f'\n✓ Hierarchy complete:')
    print(f'  Total projects: {hierarchy.total_projects}')
    print(f'  Total models: {hierarchy.total_models}')
    print(f'  Total features: {hierarchy.total_features}')

    # Step 3: Calculate statistics
    print('\n[Step 3] Calculating statistics...')
    stats = mgr.get_environment_statistics(hierarchy)

    print(f'\n📊 Environment Statistics:')
    print(f'  Models per project (mean): {stats.models_per_project_mean:.1f}')
    print(f'  Models per project (median): {stats.models_per_project_median:.1f}')
    print(f'  Features per model (mean): {stats.features_per_model_mean:.1f}')
    print(f'  Features per model (median): {stats.features_per_model_median:.1f}')

    # Show top projects
    if stats.top_projects_by_models:
        print(f'\n🏆 Top 5 Projects by Model Count:')
        for i, (name, count) in enumerate(stats.top_projects_by_models[:5], 1):
            print(f'  {i}. {name}: {count} models')

    # Show top models
    if stats.top_models_by_features:
        print(f'\n🏆 Top 5 Models by Feature Count:')
        for i, (proj, model, count) in enumerate(stats.top_models_by_features[:5], 1):
            print(f'  {i}. {proj}/{model}: {count} features')

    # Step 4: Timestamp analysis
    print('\n[Step 4] Analyzing timestamps...')
    ts_analysis = mgr.get_timestamp_analysis(hierarchy)

    print(f'\n📅 Timestamp Analysis:')
    print(f'  Coverage: {ts_analysis.timestamp_coverage_pct:.1f}%')
    print(f'    ({ts_analysis.models_with_timestamps} models have timestamps)')

    if ts_analysis.earliest_created and ts_analysis.latest_created:
        print(f'  Date range:')
        print(f'    Earliest: {ts_analysis.earliest_created.strftime("%Y-%m-%d")}')
        print(f'    Latest: {ts_analysis.latest_created.strftime("%Y-%m-%d")}')

    if ts_analysis.avg_days_between_create_update:
        print(f'  Avg days between create/update: '
              f'{ts_analysis.avg_days_between_create_update:.1f}')

    # Show newest models
    if ts_analysis.newest_models:
        print(f'\n🆕 5 Newest Models:')
        for i, model in enumerate(ts_analysis.newest_models[:5], 1):
            created = model.created_at.strftime('%Y-%m-%d') if model.created_at else 'N/A'
            print(f'  {i}. {model.name} ({created})')

    # Step 5: Export to CSV
    print('\n[Step 5] Exporting to CSV...')
    files = mgr.export_environment_to_csv(
        output_dir=EXPORT_DIR,
        prefix=f'{EXPORT_PREFIX}_detailed'
    )
    print(f'✓ Exported {len(files)} files:')
    for file in files:
        print(f'  - {file}')

    return hierarchy, stats, ts_analysis


# ============================================================================
# Example 3: Project-Level Analysis
# ============================================================================


def project_level_analysis():
    """Analyze specific projects in detail."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Project-Level Analysis')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    mgr = ProjectManager()

    # List specific projects
    project_names = ['production', 'staging', 'dev']  # Adjust as needed
    print(f'\nAnalyzing specific projects: {project_names}')

    projects = mgr.list_projects(names=project_names)
    print(f'✓ Found {len(projects)} matching projects')

    # Get stats for each project
    for project in projects:
        print(f'\n📁 Project: {project.name}')
        stats = mgr.get_project_stats(project.id)
        print(f'  Models: {stats["model_count"]}')
        print(f'  Total features: {stats["feature_count"]}')


# ============================================================================
# Example 4: Model Inventory
# ============================================================================


def model_inventory():
    """Get a complete inventory of all models across all projects."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Complete Model Inventory')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    mgr = ProjectManager()

    # List all models (compact - fast)
    print('\nFetching all models (compact mode)...')
    all_models = mgr.list_all_models(fetch_full=False)

    print(f'✓ Found {len(all_models)} total models')

    # Group by project
    from collections import defaultdict
    models_by_project = defaultdict(list)
    for model in all_models:
        project_id = getattr(model, 'project_id', 'unknown')
        models_by_project[project_id].append(model)

    print(f'\nModels grouped by project:')
    for project_id, models in list(models_by_project.items())[:10]:
        print(f'  Project {project_id}: {len(models)} models')

    # Fetch full model details for first few (slower)
    print(f'\nFetching full details for first 5 models...')
    sample_models = mgr.list_all_models(fetch_full=True)[:5]

    for model in sample_models:
        inputs_count = len(model.spec.inputs) if model.spec.inputs else 0
        print(f'  - {model.name}:')
        print(f'      Task: {model.task}')
        print(f'      Inputs: {inputs_count}')


# ============================================================================
# Example 5: Export to DataFrame for Custom Analysis
# ============================================================================


def export_for_custom_analysis():
    """Export environment data to DataFrames for custom analysis."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Export for Custom Analysis')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    mgr = ProjectManager()

    # Get hierarchy
    print('\nCollecting environment data...')
    hierarchy = mgr.get_environment_hierarchy(
        include_features=True,
        include_timestamps=True
    )

    # Export at different granularities
    print('\n📊 Exporting DataFrames at different levels:')

    # Project level
    project_df = mgr.export_environment_to_dataframe(level='project', hierarchy=hierarchy)
    print(f'\n1. Project-level DataFrame:')
    print(f'   Shape: {project_df.shape}')
    print(f'   Columns: {list(project_df.columns)}')
    if not project_df.empty:
        print(f'\n   Sample (first 3 rows):')
        print(project_df.head(3).to_string(index=False))

    # Model level
    model_df = mgr.export_environment_to_dataframe(level='model', hierarchy=hierarchy)
    print(f'\n2. Model-level DataFrame:')
    print(f'   Shape: {model_df.shape}')
    print(f'   Columns: {list(model_df.columns)}')
    if not model_df.empty:
        print(f'\n   Sample (first 3 rows):')
        print(model_df.head(3).to_string(index=False))

    # Feature level (detailed)
    feature_df = mgr.export_environment_to_dataframe(level='feature', hierarchy=hierarchy)
    print(f'\n3. Feature-level DataFrame:')
    print(f'   Shape: {feature_df.shape}')
    print(f'   Columns: {list(feature_df.columns)}')
    if not feature_df.empty:
        print(f'\n   Sample (first 3 rows):')
        print(feature_df.head(3).to_string(index=False))

    # Save DataFrames
    import os
    os.makedirs(EXPORT_DIR, exist_ok=True)

    project_csv = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}_projects.csv')
    model_csv = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}_models.csv')
    feature_csv = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}_features.csv')

    project_df.to_csv(project_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    feature_df.to_csv(feature_csv, index=False)

    print(f'\n✓ Saved DataFrames:')
    print(f'  - {project_csv}')
    print(f'  - {model_csv}')
    print(f'  - {feature_csv}')

    return project_df, model_df, feature_df


# ============================================================================
# Example 6: Console-Friendly Display
# ============================================================================


def console_friendly_display():
    """Generate a console-friendly environment summary."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Console-Friendly Display')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    mgr = ProjectManager()

    # Get hierarchy and display
    print('\nCollecting environment data...')
    hierarchy = mgr.get_environment_hierarchy(include_features=True, include_timestamps=True)

    # Use built-in display method
    mgr.display_environment_summary(hierarchy, show_top_n=10)


# ============================================================================
# Example 7: Incremental Analysis (Minimal Data Collection)
# ============================================================================


def minimal_analysis():
    """Quick analysis with minimal data collection (faster)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 7: Minimal Analysis (Fast)')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=FIDDLER_URL, token=FIDDLER_TOKEN, log_level='ERROR')

    # Create reporter with minimal data collection
    print('\nRunning minimal analysis (features and timestamps disabled)...')
    reporter = EnvironmentReporter()
    reporter.analyze_environment(
        include_features=False,  # Skip feature extraction (faster)
        include_timestamps=False,  # Skip timestamp extraction (faster)
        include_assets=False
    )

    # Generate report (will skip feature and timestamp sections)
    reporter.generate_report(
        show_projects=True,
        show_models=False,  # Skip model details
        show_timestamps=False,
        show_newest_oldest=False,
        top_n=10
    )


# ============================================================================
# Main
# ============================================================================


def main():
    """Run environment reporting examples."""

    # Suppress verbose logs for all examples
    configure_fiddler_logging(level='ERROR')

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 16 + 'ENVIRONMENT REPORTER EXAMPLES' + ' ' * 23 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: Quick summary (recommended starting point)
    quick_environment_summary()

    # Example 2: Detailed analysis with ProjectManager
    # hierarchy, stats, ts_analysis = detailed_environment_analysis()

    # Example 3: Project-level analysis
    # project_level_analysis()

    # Example 4: Model inventory
    # model_inventory()

    # Example 5: Export for custom analysis
    # project_df, model_df, feature_df = export_for_custom_analysis()

    # Example 6: Console-friendly display
    # console_friendly_display()

    # Example 7: Minimal analysis (fast)
    # minimal_analysis()

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* EnvironmentReporter provides high-level facade for common workflows')
    print('* ProjectManager offers granular control for advanced use cases')
    print('* Use include_features=False for faster analysis when features not needed')
    print('* Use include_timestamps=False to skip timestamp analysis')
    print('* Export to CSV for further analysis in Excel, Python, or R')
    print('* DataFrames available at project, model, and feature granularity')
    print('* This replaces the env_stats.ipynb notebook with reusable API')
    print('=' * 70)


if __name__ == '__main__':
    main()
