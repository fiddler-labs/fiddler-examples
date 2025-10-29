"""
Utility to check that the schema and spec are in sync.
Spec defines which columns play what role (inputs, outputs, targets, etc.)
Schema defines the structure and data types for all columns.
"""

import fiddler as fdl
from fiddler_utils import get_or_init, SchemaValidator

URL = "https://customer.fiddler.ai"
AUTH_TOKEN = ""

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

# Extract columns from spec using fiddler_utils
spec_columns = SchemaValidator.get_column_names(model)

# Extract columns from schema
schema_columns = set(col.name for col in getattr(model.schema, "columns", []))

# Compare
missing_in_schema = spec_columns - schema_columns
extra_in_schema = schema_columns - spec_columns

# Report results
print(f"\nTotal columns in schema: {len(schema_columns)}")
print(f"Total columns in spec: {len(spec_columns)}")

if missing_in_schema:
    print("\n⚠️ Columns present in spec but missing from schema:")
    for col in sorted(missing_in_schema):
        print(f" - {col}")
else:
    print("\n✅ All spec columns are present in schema.")

if extra_in_schema:
    print("\nℹ️ Columns present in schema but not referenced in spec:")
    for col in sorted(extra_in_schema):
        print(f" - {col}")
else:
    print("\n✅ No extra columns in schema.")
