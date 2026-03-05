"""Bulk Alert Creation & Update — Example Script

This script demonstrates how to use the BulkAlertCreator to create
and update monitoring alerts across all models in a Fiddler environment.

Supports two modes:
- 'create': Create new alerts (skip existing unless overwrite=True)
- 'update': Update existing alerts to match the profile (patch mutable
             fields in-place, delete+recreate for immutable changes)

Customize the configuration section below for your environment, then
run with dry_run=True first to preview what would be created/updated.

Usage:
    python bulk_alert_creation.py
"""

from fiddler_utils import (
    get_or_init,
    BulkAlertCreator,
    ModelScopeFilter,
    configure_logging,
)
from fiddler_utils.alert_profiles import (
    get_default_ml_profile,
    NotificationConfig,
)

# ---------------------------------------------------------------------------
# Configuration — customize for your environment
# ---------------------------------------------------------------------------

URL = 'https://your-org.fiddler.ai'
TOKEN = 'your-api-token'

# Notification recipients (uncomment and fill in as needed)
NOTIFICATION_EMAILS = [
    # 'alerts@company.com',
    # 'ml-team@company.com',
]

# Optional: scope to specific projects or exclude models
SCOPE = ModelScopeFilter(
    # project_names=['production_models'],    # Only these projects
    # exclude_model_names=['test_model_v0'],  # Skip these models
    # max_models=1,                           # Process only 1 model (for testing)
)

# Operation mode: 'create' or 'update'
MODE = 'create'

# Set to False to actually create/update alerts
DRY_RUN = True

# Set to True to delete and recreate existing alerts (create mode only)
OVERWRITE = False

# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def main():
    # Configure logging
    configure_logging(level='INFO')

    # Connect to Fiddler
    get_or_init(url=URL, token=TOKEN, log_level='ERROR')

    # Load default ML profile (traffic + drift + integrity + performance)
    profile = get_default_ml_profile()

    # Configure notifications (if any)
    if NOTIFICATION_EMAILS:
        profile.notification = NotificationConfig(emails=NOTIFICATION_EMAILS)

    # Optional: adjust sigma thresholds globally
    # profile.default_sigma_warning = 2.5  # 2.5σ for warning
    # profile.default_sigma_critical = 3.0  # 3σ for critical

    # Optional: remove specific alert types
    # profile.remove_spec_by_metric('mape')  # Remove MAPE alerts

    # Create the bulk alert creator
    creator = BulkAlertCreator(
        profile=profile,
        scope=SCOPE,
        overwrite=OVERWRITE,
        skip_invalid=True,
        on_error='warn',
    )

    # Run
    print(f'\n{"=" * 60}')
    print(f'Operation: {MODE.upper()} {"(DRY RUN)" if DRY_RUN else "(LIVE)"}')
    print(f'Profile: {profile.name}')
    if MODE == 'create':
        print(f'Overwrite: {OVERWRITE}')
    print(f'{"=" * 60}\n')

    result = creator.run(mode=MODE, dry_run=DRY_RUN)

    # Print report
    creator.print_report(result, show_per_model=True)

    # Export CSV report
    if result.models_processed > 0:
        csv_path = creator.export_report_csv(result)
        print(f'CSV report saved to: {csv_path}')


if __name__ == '__main__':
    main()
