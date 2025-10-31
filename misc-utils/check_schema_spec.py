"""
Utility to check that the schema and spec are in sync.

Spec defines which columns play what role (inputs, outputs, targets, etc.)
Schema defines the structure and data types for all columns.

This script validates that all columns referenced in a model's spec are
present in the model's schema, and reports any extra columns in the schema
that are not referenced in the spec.

Uses fiddler_utils.SchemaValidator for robust validation.
"""

import fiddler as fdl
from fiddler_utils import get_or_init, SchemaValidator

URL = "https://customer.fiddler.ai"  # Replace with your Fiddler instance URL
AUTH_TOKEN = ""  # Replace with your Fiddler instance API token

# Get model by ID
MODEL_ID = ""

# Or get by project and model name
PROJECT_NAME = ""
MODEL_NAME = ""

# Initialize connection
get_or_init(url=URL, token=AUTH_TOKEN, log_level="ERROR")

# Get model
if MODEL_ID:
    model = fdl.Model.get(id_=MODEL_ID)
else:
    project = fdl.Project.from_name(name=PROJECT_NAME)
    model = fdl.Model.from_name(name=MODEL_NAME, project_id=project.id)

# Validate spec/schema consistency using fiddler_utils
comparison = SchemaValidator.validate_spec_schema_consistency(model)

# Report results
total_schema_cols = len(comparison.only_in_target) + len(comparison.in_both)
total_spec_cols = len(comparison.only_in_source) + len(comparison.in_both)

print(f"\nTotal columns in schema: {total_schema_cols}")
print(f"Total columns in spec: {total_spec_cols}")
print(f"Columns in both: {len(comparison.in_both)}")

if comparison.only_in_source:
    print("\n⚠️ Columns present in spec but missing from schema:")
    for col in sorted(comparison.only_in_source):
        print(f"   - {col}")
else:
    print("\n✅ All spec columns are present in schema.")

if comparison.only_in_target:
    print("\nℹ️ Columns present in schema but not referenced in spec:")
    for col in sorted(comparison.only_in_target):
        print(f"   - {col}")
else:
    print("\n✅ No extra columns in schema.")

# Summary
print("\n" + "=" * 70)
if comparison.is_compatible:
    print("✅ SCHEMA/SPEC VALIDATION PASSED")
    print("   All columns referenced in spec are present in schema.")
else:
    print("⚠️  SCHEMA/SPEC VALIDATION FAILED")
    print(
        f"   {len(comparison.only_in_source)} column(s) in spec are missing from schema."
    )
print("=" * 70)
