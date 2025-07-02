"""
Utility to check that the schema and spec are in sync.
Spec defines which columns play what role (inputs, outputs, targets, etc.)
Schema defines the structure and data types for all columns.
"""

import fiddler as fdl

URL = "https://customer.fiddler.ai"
AUTH_TOKEN = ""
MODEL_ID = ""

fdl.init(url=URL, token=AUTH_TOKEN)
model = fdl.Model.get(id_=MODEL_ID)
schema_columns = set(col.name for col in getattr(model.schema, 'columns', []))
spec = model.spec
spec_columns = set()
for key in ["inputs", "outputs", "targets", "decisions", "metadata", "custom_features"]:
    spec_columns.update(getattr(spec, key, []) or [])

missing_in_schema = spec_columns - schema_columns
extra_in_schema = schema_columns - spec_columns

print("\nTotal columns in schema:", len(schema_columns))
print("Total columns in spec:", len(spec_columns))
if missing_in_schema:
    print("\n⚠️ Columns present in spec but missing from schema:")
    for col in sorted(missing_in_schema):
        print(" -", col)
else:
    print("\n✅ All spec columns are present in schema.")
if extra_in_schema:
    print("\nℹ️ Columns present in schema but not referenced in spec:")
    for col in sorted(extra_in_schema):
        print(" -", col)
else:
    print("\n✅ No extra columns in schema.")