"""Example: Export and import complete model definitions using ModelManager.

This script demonstrates how to use ModelManager to export and import complete
model definitions, including:
* Schema (column definitions with types, ranges, categories)
* Model spec (inputs, outputs, targets, metadata, decisions)
* Custom features (for LLM models)
* Task configuration (task type, task parameters)
* Event ID and timestamp column settings
* Baselines (optionally)

This approach uses the Model constructor pattern (not from_data()), enabling
deterministic, programmatic model recreation without requiring datasets.

Use cases:
* Model migration between environments (dev → staging → prod)
* Model versioning and backup
* Model templates and cloning
* Infrastructure-as-code workflows
"""

import fiddler as fdl
from fiddler_utils import (
    ModelManager,
    get_or_init,
    ConnectionManager,
    configure_fiddler_logging,
)
import json
import os

# ============================================================================
# Configuration
# ============================================================================

# Source model
SOURCE_URL = 'https://source.fiddler.ai'
SOURCE_TOKEN = 'your_source_token'
SOURCE_PROJECT = 'my_project'
SOURCE_MODEL = 'my_model_v1'

# Target environment (can be same or different instance)
TARGET_URL = 'https://target.fiddler.ai'
TARGET_TOKEN = 'your_target_token'
TARGET_PROJECT = 'my_project'
TARGET_MODEL = 'my_model_v1_copy'

# Export configuration
EXPORT_DIR = 'model_exports'
EXPORT_FILE = 'model_export.json'


# ============================================================================
# Example 1: Export Model Definition
# ============================================================================


def export_model_definition():
    """Export a complete model definition to a ModelExportData object."""

    print('=' * 70)
    print('EXAMPLE 1: Export Model Definition')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to source
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    # Get model
    project = fdl.Project.from_name(SOURCE_PROJECT)
    model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    print(f'\nExporting model: {model.name}')
    print(f'  Project: {project.name}')
    print(f'  Model ID: {model.id}')

    # Export model
    print('\nExporting model definition...')
    model_mgr = ModelManager()
    export_data = model_mgr.export_model(
        model_id=model.id,
        include_baselines=True  # Include baseline definitions
    )

    # Display export summary
    print(f'\n✓ Export complete')
    print(f'  Model: {export_data.name}')
    print(f'  Version: {export_data.version or "N/A"}')
    print(f'  Task: {export_data.task}')
    print(f'  Columns: {len(export_data.columns)}')

    # Show column details
    print(f'\n  Column breakdown:')
    spec = export_data.spec
    print(f'    Inputs: {len(spec.get("inputs", []))}')
    print(f'    Outputs: {len(spec.get("outputs", []))}')
    print(f'    Targets: {len(spec.get("targets", []))}')
    print(f'    Metadata: {len(spec.get("metadata", []))}')

    # Show custom features
    if export_data.custom_features:
        print(f'    Custom features: {len(export_data.custom_features)}')

    # Show baselines
    if export_data.baselines:
        print(f'  Baselines: {len(export_data.baselines)}')

    # Warnings
    if export_data.has_artifacts:
        print(f'\n  ⚠️ WARNING: Model has uploaded artifacts')
        print(f'     Artifacts cannot be exported and must be re-uploaded')

    return export_data


# ============================================================================
# Example 2: Save Export to JSON
# ============================================================================


def save_export_to_json(export_data):
    """Save model export to JSON file."""

    print('\n' + '=' * 70)
    print('EXAMPLE 2: Save Export to JSON')
    print('=' * 70)

    # Create export directory
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Save to JSON
    export_path = os.path.join(EXPORT_DIR, EXPORT_FILE)

    print(f'\nSaving to: {export_path}')

    model_mgr = ModelManager()
    model_mgr.save_export(export_data, export_path)

    print(f'✓ Saved successfully')

    # Display file info
    file_size = os.path.getsize(export_path)
    print(f'  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)')

    # Show sample of JSON structure
    with open(export_path, 'r') as f:
        data = json.load(f)
        print(f'\n  JSON structure:')
        for key in data.keys():
            print(f'    - {key}')

    return export_path


# ============================================================================
# Example 3: Load Export from JSON
# ============================================================================


def load_export_from_json(export_path):
    """Load model export from JSON file."""

    print('\n' + '=' * 70)
    print('EXAMPLE 3: Load Export from JSON')
    print('=' * 70)

    print(f'\nLoading from: {export_path}')

    model_mgr = ModelManager()
    export_data = model_mgr.load_export(export_path)

    print(f'✓ Loaded successfully')
    print(f'  Model: {export_data.name}')
    print(f'  Columns: {len(export_data.columns)}')
    print(f'  Exported at: {export_data.exported_at}')

    return export_data


# ============================================================================
# Example 4: Import Model to Target Environment
# ============================================================================


def import_model_to_target(export_data):
    """Import model to target environment."""

    print('\n' + '=' * 70)
    print('EXAMPLE 4: Import Model to Target Environment')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to target
    get_or_init(url=TARGET_URL, token=TARGET_TOKEN, log_level='ERROR')

    # Get or create target project
    print(f'\nTarget environment:')
    print(f'  URL: {TARGET_URL}')
    print(f'  Project: {TARGET_PROJECT}')
    print(f'  Model: {TARGET_MODEL}')

    target_project = fdl.Project.get_or_create(name=TARGET_PROJECT)
    print(f'  Project ID: {target_project.id}')

    # Import model
    print(f'\nImporting model...')
    model_mgr = ModelManager()

    try:
        imported_model = model_mgr.import_model(
            project_id=target_project.id,
            export_data=export_data,
            model_name=TARGET_MODEL,  # Override name
            include_baselines=True
        )

        print(f'\n✓ Import successful')
        print(f'  Model ID: {imported_model.id}')
        print(f'  Model name: {imported_model.name}')

        # Verify imported model
        print(f'\n  Verification:')
        print(f'    Task: {imported_model.task}')
        print(f'    Inputs: {len(imported_model.spec.inputs or [])}')
        print(f'    Outputs: {len(imported_model.spec.outputs or [])}')

        return imported_model

    except Exception as e:
        print(f'\n✗ Import failed: {e}')
        return None


# ============================================================================
# Example 5: Cross-Instance Migration
# ============================================================================


def cross_instance_migration():
    """Complete workflow: Export from source, import to target (different instances)."""

    print('\n' + '=' * 70)
    print('EXAMPLE 5: Cross-Instance Migration')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Setup connection manager
    conn_mgr = ConnectionManager(log_level='ERROR')
    conn_mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
    conn_mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

    model_mgr = ModelManager()

    # Step 1: Export from source
    print('\n[Step 1] Exporting from source...')
    with conn_mgr.use('source'):
        source_proj = fdl.Project.from_name(SOURCE_PROJECT)
        source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=source_proj.id)

        print(f'  Source model: {source_model.name}')
        export_data = model_mgr.export_model(source_model.id, include_baselines=True)
        print(f'  ✓ Exported {len(export_data.columns)} columns')

    # Step 2: Import to target
    print('\n[Step 2] Importing to target...')
    with conn_mgr.use('target'):
        target_proj = fdl.Project.get_or_create(name=TARGET_PROJECT)

        print(f'  Target project: {target_proj.name}')
        imported_model = model_mgr.import_model(
            project_id=target_proj.id,
            export_data=export_data,
            model_name=TARGET_MODEL,
            include_baselines=True
        )
        print(f'  ✓ Imported as: {imported_model.name}')

    print(f'\n✅ Migration complete!')
    print(f'   {SOURCE_URL}/{SOURCE_PROJECT}/{SOURCE_MODEL}')
    print(f'   → {TARGET_URL}/{TARGET_PROJECT}/{TARGET_MODEL}')


# ============================================================================
# Example 6: Model Template / Cloning
# ============================================================================


def clone_model_as_template():
    """Clone a model as a template for creating similar models."""

    print('\n' + '=' * 70)
    print('EXAMPLE 6: Clone Model as Template')
    print('=' * 70)

    # Suppress verbose logs
    configure_fiddler_logging(level='ERROR')

    # Connect to Fiddler
    get_or_init(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR')

    project = fdl.Project.from_name(SOURCE_PROJECT)
    source_model = fdl.Model.from_name(SOURCE_MODEL, project_id=project.id)

    print(f'\nCloning model: {source_model.name}')

    # Export
    model_mgr = ModelManager()
    export_data = model_mgr.export_model(source_model.id, include_baselines=False)

    # Create multiple clones
    clones = ['model_v2', 'model_v3', 'model_dev']

    print(f'\nCreating {len(clones)} clones...')
    for clone_name in clones:
        try:
            clone = model_mgr.import_model(
                project_id=project.id,
                export_data=export_data,
                model_name=clone_name,
                include_baselines=False
            )
            print(f'  ✓ Created: {clone.name}')
        except Exception as e:
            print(f'  ✗ Failed to create {clone_name}: {e}')


# ============================================================================
# Example 7: Inspect Export Data
# ============================================================================


def inspect_export_data(export_data):
    """Inspect the contents of a model export."""

    print('\n' + '=' * 70)
    print('EXAMPLE 7: Inspect Export Data')
    print('=' * 70)

    print(f'\nModel: {export_data.name}')
    print(f'Version: {export_data.version or "N/A"}')
    print(f'Task: {export_data.task}')
    print(f'Event ID column: {export_data.event_id_col}')
    print(f'Event TS column: {export_data.event_ts_col}')

    # Column details
    print(f'\nColumns ({len(export_data.columns)}):')
    for i, col in enumerate(export_data.columns[:10], 1):
        range_info = ''
        if col.min_value is not None and col.max_value is not None:
            range_info = f' [{col.min_value}, {col.max_value}]'
        cat_info = ''
        if col.categories:
            cat_info = f' (categories: {len(col.categories)})'

        print(f'  {i:2d}. {col.name:30s} {col.data_type:15s}{range_info}{cat_info}')

    if len(export_data.columns) > 10:
        print(f'  ... and {len(export_data.columns) - 10} more')

    # Spec breakdown
    print(f'\nModel Spec:')
    spec = export_data.spec
    for spec_type in ['inputs', 'outputs', 'targets', 'metadata', 'decisions']:
        cols = spec.get(spec_type, [])
        if cols:
            print(f'  {spec_type.capitalize()}: {len(cols)}')
            if len(cols) <= 5:
                print(f'    {", ".join(cols)}')
            else:
                print(f'    {", ".join(cols[:5])}, ... ({len(cols) - 5} more)')

    # Custom features (LLM models)
    if export_data.custom_features:
        print(f'\nCustom Features: {len(export_data.custom_features)}')
        for feat in export_data.custom_features[:5]:
            feat_type = feat.get('type', 'unknown')
            feat_name = feat.get('name', 'unnamed')
            print(f'  - {feat_name} ({feat_type})')

    # Baselines
    if export_data.baselines:
        print(f'\nBaselines: {len(export_data.baselines)}')
        for bl in export_data.baselines:
            bl_name = bl.get('name', 'unnamed')
            bl_type = bl.get('type', 'unknown')
            print(f'  - {bl_name} ({bl_type})')

    # Warnings
    if export_data.has_artifacts:
        print(f'\n⚠️ WARNING: This model has uploaded artifacts')
        print(f'   Artifacts must be manually uploaded to imported models')


# ============================================================================
# Main
# ============================================================================


def main():
    """Run model export/import examples."""

    # Suppress verbose logs for all examples
    configure_fiddler_logging(level='ERROR')

    print('\n')
    print('╔' + '=' * 68 + '╗')
    print('║' + ' ' * 18 + 'MODEL MANAGER EXAMPLES' + ' ' * 28 + '║')
    print('╚' + '=' * 68 + '╝')

    # Example 1: Export model
    export_data = export_model_definition()

    # Example 2: Save to JSON
    export_path = save_export_to_json(export_data)

    # Example 3: Load from JSON
    loaded_data = load_export_from_json(export_path)

    # Example 4: Import to target
    # imported_model = import_model_to_target(loaded_data)

    # Example 5: Cross-instance migration
    # cross_instance_migration()

    # Example 6: Clone as template
    # clone_model_as_template()

    # Example 7: Inspect export data
    inspect_export_data(export_data)

    print('\n' + '=' * 70)
    print('IMPORTANT NOTES:')
    print('=' * 70)
    print('* ModelManager exports complete model definitions (not datasets)')
    print('* Uses Model constructor pattern for deterministic recreation')
    print('* Export includes: schema, spec, task config, baselines')
    print('* Artifacts (uploaded model packages) cannot be exported')
    print('* Ideal for model versioning, migration, and templates')
    print('* JSON format enables version control and infrastructure-as-code')
    print('* Cross-instance migration supported with ConnectionManager')
    print('=' * 70)


if __name__ == '__main__':
    main()
