"""Example: Compare Fiddler models using ModelComparator.

This script demonstrates how to use ModelComparator to comprehensively compare
two Fiddler models across multiple dimensions:
* Configuration (task, event columns, task params)
* Schema (column names and types)
* Model spec (inputs, outputs, targets, metadata, decisions, custom features)
* Assets (segments, custom metrics, alerts, baselines, charts)

The comparison can be performed:
* Within the same instance (comparing versions or similar models)
* Across different instances (comparing dev vs prod deployments)
* With flexible configuration (compare all, schema only, exclude assets, etc.)

Results can be exported to multiple formats for documentation and analysis.
"""

import fiddler as fdl
from fiddler_utils import (
    ModelComparator
    ComparisonConfig
    ConnectionManager
)

# ============================================================================
# Configuration
# ============================================================================

# Source model (Model A)
SOURCE_URL = 'https://source.fiddler.ai'
SOURCE_TOKEN = 'your_source_token'
SOURCE_PROJECT = 'my_project'
SOURCE_MODEL = 'my_model_v1'

# Target model (Model B) - can be same or different instance
TARGET_URL = 'https://target.fiddler.ai'  # Can be same as SOURCE_URL
TARGET_TOKEN = 'your_target_token'  # Can be same as SOURCE_TOKEN
TARGET_PROJECT = 'my_project'
TARGET_MODEL = 'my_model_v2'

# Output configuration
EXPORT_DIR = 'comparison_results'
EXPORT_PREFIX = 'model_comparison'


# ============================================================================
# Example 1: Basic Comparison (Same Instance)
# ============================================================================


def compare_models_same_instance():
    """Compare two models in the same Fiddler instance."""

    print('=' * 70)
    print('EXAMPLE 1: Compare Models (Same Instance)')
    print('=' * 70)

    # Suppress verbose logs

    # Connect to Fiddler instance
    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)

    # Get both models
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model_a = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    model_b = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    print(f'\nComparing:')
    print(f'  Model A: {model_a.name} (ID: {model_a.id})')
    print(f'  Model B: {model_b.name} (ID: {model_b.id})')

    # Create comparator and run full comparison
    print('\nRunning comprehensive comparison...')
    comparator = ModelComparator(model_a, model_b)
    result = comparator.compare_all()

    # Display summary
    summary = result.get_summary()
    print(f'\n✓ Comparison complete')
    print(f'  Has differences: {summary["has_differences"]}')
    print(f'  Differences by category:')
    for category, count in summary['differences_by_category'].items():
        if count > 0:
            print(f'    - {category}: {count} difference(s)')

    # Show detailed markdown report
    print('\n' + '=' * 70)
    print('DETAILED REPORT:')
    print('=' * 70)
    print(result.to_markdown())

    return result


# ============================================================================
# Example 2: Cross-Instance Comparison
# ============================================================================


def compare_models_cross_instance():
    """Compare models across different Fiddler instances (e.g., dev vs prod)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Compare Models (Cross-Instance)')
    print('=' * 70)

    # Suppress verbose logs

    # Set up connection manager for multiple instances
    conn_mgr = ConnectionManager(log_level='ERROR')
    conn_mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
    conn_mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

    # Fetch source model
    print('\nFetching source model...')
    with conn_mgr.use('source'):
        source_proj = fdl.Project.from_name(SOURCE_PROJECT)
        source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=source_proj.id)
        print(f'  ✓ Source: {source_model.name} from {SOURCE_URL}')

    # Fetch target model
    print('Fetching target model...')
    with conn_mgr.use('target'):
        target_proj = fdl.Project.from_name(TARGET_PROJECT)
        target_model = fdl.Model.from_name(TARGET_MODEL, project_id=target_proj.id)
        print(f'  ✓ Target: {target_model.name} from {TARGET_URL}')

    # Compare models (no active connection needed - models are already fetched)
    print('\nRunning comparison...')
    comparator = ModelComparator(source_model, target_model)
    result = comparator.compare_all()

    # Display results
    print(f'\n✓ Comparison complete')
    if result.has_differences():
        print('  ⚠️ Models have differences')
        summary = result.get_summary()
        for category, count in summary['differences_by_category'].items():
            if count > 0:
                print(f'    - {category}: {count}')
    else:
        print('  ✅ Models are identical')

    return result


# ============================================================================
# Example 3: Schema-Only Comparison
# ============================================================================


def compare_schemas_only():
    """Quick schema-only comparison (fast, no asset loading)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Schema-Only Comparison (Fast)')
    print('=' * 70)

    # Suppress verbose logs

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)

    project = fdl.Project.from_name(SOURCE_PROJECT)
    model_a = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    model_b = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    print(f'\nComparing schemas only (fast check):')
    print(f'  Model A: {model_a.name}')
    print(f'  Model B: {model_b.name}')

    # Use schema_only preset
    config = ComparisonConfig.schema_only()

    comparator = ModelComparator(model_a, model_b)
    result = comparator.compare_all(config=config)

    # Check schema compatibility
    if result.schema:
        print(f'\nSchema Comparison:')
        print(f'  Common columns: {len(result.schema.in_both)}')
        print(f'  Only in Model A: {len(result.schema.only_in_source)}')
        print(f'  Only in Model B: {len(result.schema.only_in_target)}')
        print(f'  Type mismatches: {len(result.schema.type_mismatches)}')
        print(f'  Compatible: {result.schema.is_compatible}')

        if result.schema.only_in_source:
            print(f'\n  ⚠️ Columns in A but missing in B:')
            for col in sorted(result.schema.only_in_source)[:10]:
                print(f'    - {col}')

        if result.schema.only_in_target:
            print(f'\n  ⚠️ Columns in B but missing in A:')
            for col in sorted(result.schema.only_in_target)[:10]:
                print(f'    - {col}')

    return result


# ============================================================================
# Example 4: Comparison Without Assets
# ============================================================================


def compare_without_assets():
    """Compare model structure only (configuration, schema, spec) - skip assets."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Compare Without Assets (Structure Only)')
    print('=' * 70)

    # Suppress verbose logs

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)

    project = fdl.Project.from_name(SOURCE_PROJECT)
    model_a = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    model_b = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    print(f'\nComparing model structure (no assets):')
    print(f'  Model A: {model_a.name}')
    print(f'  Model B: {model_b.name}')

    # Use no_assets preset (compares configuration, schema, spec only)
    config = ComparisonConfig.no_assets()

    comparator = ModelComparator(model_a, model_b)
    result = comparator.compare_all(config=config)

    # Show configuration differences
    if result.configuration and result.configuration.has_differences():
        print(f'\n⚙️ Configuration Differences:')
        for key, diff in result.configuration.differences.items():
            print(f'  {key}:')
            print(f'    Model A: {diff.source_value}')
            print(f'    Model B: {diff.target_value}')

    # Show spec differences
    if result.spec and result.spec.has_differences():
        print(f'\n🔧 Spec Differences:')
        for spec_type in ['inputs', 'outputs', 'targets', 'metadata', 'custom_features']:
            only_a = result.spec.only_in_source.get(spec_type, [])
            only_b = result.spec.only_in_target.get(spec_type, [])
            if only_a or only_b:
                print(f'  {spec_type.capitalize()}:')
                if only_a:
                    print(f'    Only in A: {", ".join(only_a[:5])}')
                if only_b:
                    print(f'    Only in B: {", ".join(only_b[:5])}')

    return result


# ============================================================================
# Example 5: Export Results to Multiple Formats
# ============================================================================


def export_comparison_results(result):
    """Export comparison results to multiple formats."""

    import os

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Export Comparison Results')
    print('=' * 70)

    # Create output directory
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Export to JSON
    json_file = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}.json')
    result.to_json(json_file, indent=2)
    print(f'\n✓ Exported to JSON: {json_file}')

    # Export to Markdown
    markdown_file = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}.md')
    with open(markdown_file, 'w') as f:
        f.write(result.to_markdown())
    print(f'✓ Exported to Markdown: {markdown_file}')

    # Export to CSV (DataFrame)
    csv_file = os.path.join(EXPORT_DIR, f'{EXPORT_PREFIX}.csv')
    df = result.to_dataframe()
    df.to_csv(csv_file, index=False)
    print(f'✓ Exported to CSV: {csv_file}')
    print(f'  Total differences: {len(df)} rows')

    # Show sample of DataFrame
    if not df.empty:
        print(f'\nSample differences (first 5):')
        print(df.head().to_string(index=False))

    return {
        'json': json_file
        'markdown': markdown_file
        'csv': csv_file
    }


# ============================================================================
# Example 6: Custom Comparison Configuration
# ============================================================================


def custom_comparison_config():
    """Use custom ComparisonConfig to selectively compare specific aspects."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Custom Comparison Configuration')
    print('=' * 70)

    # Suppress verbose logs

    fdl.init(url=SOURCE_URL, token=SOURCE_TOKEN)

    project = fdl.Project.from_name(SOURCE_PROJECT)
    model_a = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)
    model_b = fdl.Model.from_name(TARGET_MODEL, project_id=project.id)

    # Create custom config: compare schema, spec, segments, and custom metrics only
    # Skip: configuration, alerts, baselines, charts
    config = ComparisonConfig(
        include_configuration=False
        include_schema=True
        include_spec=True
        include_segments=True
        include_custom_metrics=True
        include_alerts=False
        include_baselines=False
        include_charts=False
    )

    print(f'\nCustom comparison (schema + spec + segments + metrics):')
    print(f'  Model A: {model_a.name}')
    print(f'  Model B: {model_b.name}')

    comparator = ModelComparator(model_a, model_b)
    result = comparator.compare_all(config=config)

    # Show what was compared
    print(f'\n✓ Comparison complete')
    print(f'  Compared:')
    print(f'    ✅ Schema')
    print(f'    ✅ Spec')
    print(f'    ✅ Segments')
    print(f'    ✅ Custom Metrics')
    print(f'  Skipped:')
    print(f'    ⊘ Configuration')
    print(f'    ⊘ Alerts')
    print(f'    ⊘ Baselines')
    print(f'    ⊘ Charts')

    # Show segment differences if any
    if result.segments and result.segments.has_differences():
        print(f'\n🔍 Segment Differences:')
        print(f'  Only in A: {len(result.segments.only_in_source)}')
        print(f'  Only in B: {len(result.segments.only_in_target)}')
        print(f'  Definition differences: {len(result.segments.definition_differences)}')

    # Show custom metric differences if any
    if result.custom_metrics and result.custom_metrics.has_differences():
        print(f'\n📊 Custom Metric Differences:')
        print(f'  Only in A: {len(result.custom_metrics.only_in_source)}')
        print(f'  Only in B: {len(result.custom_metrics.only_in_target)}')
        print(f'  Definition differences: {len(result.custom_metrics.definition_differences)}')

    return result


# ============================================================================
# Example 7: Interpreting Comparison Results
# ============================================================================


def interpret_comparison_results(result):
    """Demonstrate how to programmatically interpret and act on comparison results."""

    print('\n' + '=' * 70)
    print('EXAMPLE 7: Interpreting Comparison Results')
    print('=' * 70)

    # Check if models are identical
    if not result.has_differences():
        print('\n✅ Models are identical - no action needed')
        return

    print('\n⚠️ Models have differences - analysis:')

    # Check schema compatibility for asset migration
    if result.schema:
        if result.schema.is_compatible:
            print('\n✅ Schema: Compatible for asset migration')
            print(f'   All {len(result.schema.in_both)} source columns exist in target')
        else:
            print('\n❌ Schema: NOT compatible for asset migration')
            if result.schema.only_in_source:
                print(f'   Missing columns in target: {len(result.schema.only_in_source)}')
                print(f'   Examples: {list(result.schema.only_in_source)[:3]}')

    # Check configuration differences
    if result.configuration and result.configuration.has_differences():
        print(f'\n⚙️ Configuration: {len(result.configuration.differences)} difference(s)')
        critical_config = ['task', 'event_id_col', 'event_ts_col']
        for key in critical_config:
            if key in result.configuration.differences:
                print(f'   ⚠️ CRITICAL: {key} differs')

    # Check spec differences
    if result.spec and result.spec.has_differences():
        print(f'\n🔧 Model Spec: Has differences')
        if result.spec.only_in_source.get('inputs'):
            print(f'   Inputs only in A: {len(result.spec.only_in_source["inputs"])}')
        if result.spec.only_in_target.get('inputs'):
            print(f'   Inputs only in B: {len(result.spec.only_in_target["inputs"])}')

    # Check asset differences
    asset_summary = []
    if result.segments and result.segments.has_differences():
        asset_summary.append(f'Segments: {result.segments.total_differences}')
    if result.custom_metrics and result.custom_metrics.has_differences():
        asset_summary.append(f'Custom Metrics: {result.custom_metrics.total_differences}')
    if result.alerts and result.alerts.has_differences():
        asset_summary.append(f'Alerts: {result.alerts.total_differences}')
    if result.baselines and result.baselines.has_differences():
        asset_summary.append(f'Baselines: {result.baselines.total_differences}')

    if asset_summary:
        print(f'\n📦 Asset Differences:')
        for summary in asset_summary:
            print(f'   {summary}')

    # Provide recommendations
    print(f'\n💡 Recommendations:')
    if result.schema and result.schema.is_compatible:
        print('   ✓ Safe to copy assets from Model A to Model B')
    else:
        print('   ⚠️ Fix schema differences before copying assets')

    if result.configuration and result.configuration.has_differences():
        print('   ⚠️ Review configuration differences before deployment')

    if any([result.segments and result.segments.has_differences()
            result.custom_metrics and result.custom_metrics.has_differences()]):
        print('   ℹ️ Consider syncing assets between models')


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all comparison examples."""

    # Suppress verbose logs for all examples

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 20 + 'MODEL COMPARATOR EXAMPLES' + ' ' * 23 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: Basic same-instance comparison
    result = compare_models_same_instance()

    # Example 2: Cross-instance comparison
    # result = compare_models_cross_instance()

    # Example 3: Schema-only comparison (fast)
    # schema_result = compare_schemas_only()

    # Example 4: Compare without assets (structure only)
    # structure_result = compare_without_assets()

    # Example 5: Export results
    # files = export_comparison_results(result)
    # print(f"\nExported files: {files}")

    # Example 6: Custom comparison configuration
    # custom_result = custom_comparison_config()

    # Example 7: Interpret results programmatically
    # interpret_comparison_results(result)

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* ModelComparator compares models across multiple dimensions')
    print('* Use ComparisonConfig presets for common scenarios:')
    print('  - ComparisonConfig.all() - full comparison (default)')
    print('  - ComparisonConfig.schema_only() - fast schema check')
    print('  - ComparisonConfig.no_assets() - structure only')
    print('* Export results to JSON, Markdown, or CSV for documentation')
    print('* Check schema.is_compatible before migrating assets')
    print('* Models can be from same or different Fiddler instances')
    print('=' * 70)


if __name__ == '__main__':
    main()
