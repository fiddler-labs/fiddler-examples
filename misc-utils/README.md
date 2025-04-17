# Fiddler Utility Notebooks

## Repository Purpose

This repository serves as a centralized collection of utility notebooks for customer success engineers / field AI engineers. Unlike production code, these notebooks address specific one-time or occasional administrative needs that arise when supporting Fiddler deployments.
Field engineers often develop scripts to solve unique challenges that may not warrant inclusion in the main product but are invaluable when similar situations arise with other customers.

### General pre-requisites :
- Fiddler instance URL and valid API token
- Administrative access to view / edit ( may not be needed for every tool)


### Contributing Guidelines

When adding a new notebook:
- Include clear prerequisites
- Document any conditional logic
- Add a summary to this README
- Ensure sensitive information (tokens, URLs) is not committed

---

# Notebook Catalog

## env_stats.ipynb
A utility for analyzing and extracting hierarchical information about a Fiddler environment:
- Collects comprehensive statistics on projects, models, and features
- Generates summary metrics like average models per project and features per model
- Creates exportable dataframes for further analysis
- Visualizes the top models by feature count

**Output formats:**
- Generates CSV export of feature/model/project relationships
- Provides high-level environment statistics (totals, averages, etc.)
- Shows ranked lists of projects by model count and models by feature count

## user_invite_automation.ipynb
A utility notebook for bulk user invitation management in Fiddler. It allows:
- Loading email addresses from a CSV file
- Sending invitations to multiple users at once with specified roles (Org Admin/Org Member)
- Deleting invitations in bulk
- Extracting and saving invitation URLs from the Fiddler UI

**Prerequisites:** CSV file with email addresses in a column named 'mails'

**Notes:**
- Contains example of API response format with invitation links, useful for reference
- Separate code blocks for invitation creation and deletion for safety

## alert_bulk_modification.ipynb
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

## feature_importance_reupload.ipynb
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

## alerts_gen_testing.ipynb
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

## copying_segments.ipynb
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


---

These notebooks demonstrate practical solutions for common Fiddler administrative tasks that might be encountered by customer success and field AI engineers.

