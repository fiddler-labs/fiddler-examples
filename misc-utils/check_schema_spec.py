"""
Utility to check that the schema and spec are in sync.

Spec defines which columns play what role (inputs, outputs, targets, etc.)
Schema defines the structure and data types for all columns.

This script validates that all columns referenced in a model's spec are
present in the model's schema, and reports any extra columns in the schema
that are not referenced in the spec.

Uses fiddler-utils for robust validation. Install with:

    pip install "fiddler-utils @ git+https://github.com/fiddler-labs/fiddler-utils.git@v1.1.0"

Environment variables:
    FIDDLER_URL           Fiddler instance URL
    FIDDLER_TOKEN         Fiddler API token
    FIDDLER_MODEL_ID      Model UUID (takes precedence over project/model name)
    FIDDLER_PROJECT_NAME  Project name (used when FIDDLER_MODEL_ID is not set)
    FIDDLER_MODEL_NAME    Model name (used when FIDDLER_MODEL_ID is not set)
"""

import os

import fiddler as fdl

from fiddler_utils import SchemaValidator, get_or_init


def main() -> None:
    url = os.environ.get("FIDDLER_URL", "https://your-org.cloud.fiddler.ai")
    token = os.environ.get("FIDDLER_TOKEN", "YOUR_FIDDLER_TOKEN")
    model_id = os.environ.get("FIDDLER_MODEL_ID", "")
    project_name = os.environ.get("FIDDLER_PROJECT_NAME", "your-project-name")
    model_name = os.environ.get("FIDDLER_MODEL_NAME", "your-model-name")

    get_or_init(url=url, token=token, log_level="ERROR")

    if model_id:
        model = fdl.Model.get(id_=model_id)
    else:
        project = fdl.Project.from_name(name=project_name)
        model = fdl.Model.from_name(name=model_name, project_id=project.id)

    # Validate spec/schema consistency using fiddler-utils
    comparison = SchemaValidator.validate_spec_schema_consistency(model)

    # Report results
    total_schema_cols = len(comparison.only_in_target) + len(comparison.in_both)
    total_spec_cols = len(comparison.only_in_source) + len(comparison.in_both)

    print(f"\nTotal columns in schema: {total_schema_cols}")
    print(f"Total columns in spec: {total_spec_cols}")
    print(f"Columns in both: {len(comparison.in_both)}")

    if comparison.only_in_source:
        print("\n✗ Columns present in spec but missing from schema:")
        for col in sorted(comparison.only_in_source):
            print(f"   - {col}")
    else:
        print("\n✓ All spec columns are present in schema.")

    if comparison.only_in_target:
        print("\n→ Columns present in schema but not referenced in spec:")
        for col in sorted(comparison.only_in_target):
            print(f"   - {col}")
    else:
        print("\n✓ No extra columns in schema.")

    # Summary
    print("\n" + "=" * 70)
    if comparison.is_compatible:
        print("✓ SCHEMA/SPEC VALIDATION PASSED")
        print("   All columns referenced in spec are present in schema.")
    else:
        print("✗ SCHEMA/SPEC VALIDATION FAILED")
        print(f"   {len(comparison.only_in_source)} column(s) in spec are missing from schema.")
    print("=" * 70)


if __name__ == "__main__":
    main()
