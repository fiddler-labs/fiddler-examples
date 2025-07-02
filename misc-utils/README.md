# Fiddler Utility Notebooks

## Directory Purpose

This directory serves as a centralized collection of utility notebooks. Unlike production code, these notebooks address specific one-time or occasional administrative needs that arise when supporting Fiddler deployments.
Field engineers often develop scripts to solve unique challenges that may not warrant inclusion in the main product but are invaluable when similar situations arise with other customers.

### General pre-requisites

- Fiddler instance URL and valid API token
- Administrative access to view / edit (may not be needed for every tool)

### Contributing Guidelines

When adding a new notebook:

- Include clear prerequisites
- Document any conditional logic
- Add a summary to this README
- Ensure sensitive information (tokens, URLs) is not committed

---

## Notebook and Script Catalog

### env_stats.ipynb

A utility for analyzing and extracting hierarchical information about a Fiddler environment:

- Collects comprehensive statistics on projects, models, and features
- Captures model creation and update timestamps (`created_at`, `updated_at`)
- Generates summary metrics like average models per project and features per model
- Creates exportable dataframes for further analysis
- Visualizes the top models by feature count
- Provides temporal insights including newest/oldest models and recent updates

**Outputs:**

- Provides high-level environment statistics (totals, averages, etc.)
- Shows ranked lists of projects by model count and models by feature count
- Enhanced statistics including timestamp analysis (date ranges, update patterns, temporal coverage)
- `env_stats__overview.csv` - Main export table with project, model, version_name, created_at, updated_at
- `env_stats__flattened_hierarchy.csv` - Detailed feature-level data enriched with timestamps

### user_invite_automation.ipynb

A utility notebook for bulk user invitation management in Fiddler. It allows:

- Loading email addresses from a CSV file
- Sending invitations to multiple users at once with specified roles (Org Admin/Org Member)
- Deleting invitations in bulk
- Extracting and saving invitation URLs from the Fiddler UI

**Prerequisites:** CSV file with email addresses in a column named 'mails'

**Notes:**

- Contains example of API response format with invitation links, useful for reference
- Separate code blocks for invitation creation and deletion for safety

### alert_bulk_modification.ipynb

A comprehensive utility for modifying alert thresholds across all models and projects. Specifically:

- Identifies and fixes percentage-based data integrity alerts with incorrect threshold scales
- Scans all projects and models for alerts with percentage metrics
- Converts thresholds from decimal format (0-1) to percentage format (0-100)
- Provides detailed logging and summary statistics

**Prerequisites:** Administrative access to modify alerts

**Execution logic:**

- Only modifies alerts with raw value comparisons (skips time-period comparison alerts)
- Only updates thresholds that are between 0-1 (already assumes these are improperly scaled)
- Handles both critical and warning thresholds

### feature_importance_reupload.ipynb

A utility for uploading pre-calculated feature importance values to a Fiddler model:

- Loads feature importance values from a CSV file
- Validates features against model input schema
- Uploads the values to a specified model and version

**Prerequisites:**

- CSV with columns 'feature_name' and 'feature_imp'
- Project name, model name, and version

**Validation logic:**

- Checks for missing features (model inputs not in CSV)
- Checks for extra features (CSV features not in model)
- Removes extra features before upload to prevent errors

### alerts_gen_testing.ipynb

A comprehensive testing utility to verify Fiddler's alert functionality and email delivery:

- Creates a test project and model with a synthetic customer churn dataset
- Configures all available Fiddler alert types with appropriate thresholds
- Generates and publishes events that intentionally violate alert rules
- Tests email delivery through corporate networks/firewalls

**Prerequisites:**

- Email address(es) for receiving test alerts

**Alert types demonstrated:**

- Data Integrity (range, null, and type violations)
- Data Drift (distribution shift via Jensen-Shannon Distance)
- Traffic (low volume detection)
- Performance (model precision degradation)
- Custom Metric (business-relevant metrics like revenue loss)

**Usage notes:**

- Includes troubleshooting tips for email delivery issues
- Creates a complete self-contained testing environment
- Publishes violation events in batches to avoid timeouts

### copying_segments.ipynb

A utility notebook for copying segment definitions between model versions:

- Retrieves all segments from a source model version
- Creates identical segments in a target model version
- Preserves segment names, descriptions, and filter definitions
- Handles bulk transfer of multiple segments

**Prerequisites:**

- Source and target model versions must exist
- Fiddler URL and valid API token
- Administrative access to create segments

**Usage notes:**

- Useful when promoting models to new versions
- Maintains segment configuration consistency across versions
- Automates what would otherwise be manual segment recreation

### replace_alerts_with_mods.py

A utility script for modifying immutable alert properties across a Fiddler deployment:

- Programmatically converts monthly bin alerts to daily bin alerts across all projects and models
- Handles immutable alert properties by creating new alerts with modified parameters
- Preserves all other alert configuration details during conversion
- Provides comprehensive logging and progress tracking

**Prerequisites:**

- Fiddler URL and valid API token
- Administrative access to create and delete alerts

**Key features:**

- Traverses entire Fiddler hierarchy (projects → models → alerts)
- Creates duplicate alerts with "_FIXED" suffix in name and modified bin size
- Deletes original alerts after successful duplication
- Includes robust error handling and progress tracking with tqdm
- Detailed logging with separate app and Fiddler library log files

**Usage notes:**

- Addresses the limitation that certain alert attributes (like bin_size) cannot be modified after creation
- Can be adapted for other immutable property changes beyond bin size
- Useful for batch modifications that would otherwise require manual recreation of alerts
- Maintains alert continuity by creating replacement alerts before deleting originals

### create_dashboard_from_saved_charts.ipynb

A utility for creating and organizing dashboards using existing charts:

- Creates dashboards with specific chart arrangements using Fiddler's API
- Positions charts in a customizable grid layout
- Maps chart titles to their UUIDs for proper dashboard configuration
- Sets the created dashboard as the default for a model

**Prerequisites:**

- Existing model with pre-created charts
- Fiddler URL and valid API token

**Key features:**

- Supports precise positioning of charts in a grid (position_x, position_y, width, height)
- Allows export/import of dashboard configurations via YAML
- Converts chart titles to the required UUID format automatically
- Sets time filters and timezone preferences

**Usage notes:**

- Useful for standardizing dashboards across similar models
- Provides programmatic control over dashboard layout
- Can be adapted for bulk dashboard creation

### check_schema_spec.py

A utility script for validating consistency between model schema and specification:

- Compares columns defined in the model schema with those referenced in the model specification
- Identifies columns present in spec but missing from schema (potential issues)
- Identifies columns present in schema but not referenced in spec (informational)
- Provides detailed reporting on schema-spec alignment

**Prerequisites:**

- Fiddler URL and valid API token
- Model ID of the model to validate

**Key features:**

- Validates all spec categories: inputs, outputs, targets, decisions, metadata, custom_features
- Provides clear visual indicators (✅ for success, ⚠️ for warnings, ℹ️ for info)
- Reports total column counts for both schema and spec
- Lists specific column names for any discrepancies found

**Usage notes:**

- Essential for troubleshooting model configuration issues
- Helps ensure data consistency and proper model monitoring setup
- Can be run as a health check after model updates or schema changes
- Useful for debugging publishing or monitoring issues related to column mismatches

### adding_charts.ipynb

A utility for programmatically adding charts to Fiddler projects and models:

- Uses Fiddler's API to create charts without using the UI
- Extracts chart definitions from browser network requests
- Handles mapping of project, model, and baseline IDs automatically
- Supports custom metrics and segment-specific charts

**Prerequisites:**

- Fiddler URL and valid API token
- Existing project and model
- Chart definition (obtained from browser network inspection)

**Key features:**

- Preserves chart metadata including title, description, and visualization type
- Configures time ranges and bin sizes for temporal charts
- Maps segment names to their corresponding IDs
- Supports custom metrics by name reference

**Usage notes:**

- Useful for bulk chart creation across multiple models
- Enables chart replication between environments
- Chart definitions can be captured from existing charts via browser dev tools
- Chart definitions can be modified programmatically before creation

---

These notebooks demonstrate practical solutions for common Fiddler administrative tasks that might be encountered by customer success and field AI engineers.
