"""Reference management utilities for Custom Metrics.

This module provides utilities to find and manage references to Custom Metrics
across Charts and Alert Rules. This is critical when updating metrics since
deleting and recreating a metric generates a new UUID, breaking all references.

Key Functions:
    - find_charts_using_metric(): Find all charts referencing a metric
    - find_alerts_using_metric(): Find all alerts monitoring a metric
    - find_all_metric_references(): Comprehensive reference discovery
    - safe_update_metric(): Update metric with automatic reference migration

Example:
    ```python
    from fiddler_utils.assets.references import (
        find_all_metric_references,
        safe_update_metric
    )
    import fiddler as fdl

    # Find all references before updating
    metric = fdl.CustomMetric.from_name(name='my_metric', model_id=model.id)
    refs = find_all_metric_references(metric.id, project.id)
    print(f"Found {refs['total_count']} references")
    print(f"  Charts: {len(refs['charts'])}")
    print(f"  Alerts: {len(refs['alerts'])}")

    # Safe update with automatic reference migration
    new_metric, report = safe_update_metric(
        metric=metric,
        new_definition='sum(if(tp(), 1, 0))',
        auto_migrate=True
    )
    print(f"Updated metric. Migrated {report['migrated_count']} references")
    ```
"""

from typing import List, Dict, Tuple, Optional, Any
import logging
import time

try:
    import fiddler as fdl
    from fiddler.libs.http_client import RequestClient
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from ..exceptions import AssetNotFoundError

logger = logging.getLogger(__name__)


def find_alerts_using_metric(metric_id: str) -> List[fdl.AlertRule]:
    """Find all alert rules that monitor a specific custom metric.

    Args:
        metric_id: UUID of the custom metric

    Returns:
        List of AlertRule objects that reference this metric

    Example:
        ```python
        alerts = find_alerts_using_metric(metric.id)
        if alerts:
            print(f"⚠️  Deleting this metric will break {len(alerts)} alerts:")
            for alert in alerts:
                print(f"  - {alert.name}")
        ```
    """
    # Get the custom metric to find its model
    try:
        metric = fdl.CustomMetric.get(id_=metric_id)
    except Exception as e:
        logger.error(f"Failed to get metric {metric_id}: {e}")
        raise AssetNotFoundError(f"Custom metric with ID {metric_id} not found")

    # List all alerts for this model
    all_alerts = list(fdl.AlertRule.list(model_id=metric.model_id))

    # Filter to alerts that reference this metric
    using_metric = []
    for alert in all_alerts:
        # Check if alert's metric_id matches (could be metric name or ID)
        alert_metric_id = alert.metric_id

        # metric_id could be the metric name or UUID
        if alert_metric_id == metric_id or alert_metric_id == metric.name:
            using_metric.append(alert)

    logger.info(
        f"Found {len(using_metric)} alerts using metric {metric.name} "
        f"(ID: {metric_id})"
    )

    return using_metric


def find_charts_using_metric(
    metric_id: str,
    project_id: str,
    url: Optional[str] = None,
    token: Optional[str] = None
) -> List[Dict]:
    """Find all charts that reference a specific custom metric.

    Note: This uses the unofficial Fiddler Chart API which may change.

    Args:
        metric_id: UUID of the custom metric
        project_id: Project ID to search within
        url: Fiddler instance URL (optional if fiddler already initialized)
        token: API token (optional if fiddler already initialized)

    Returns:
        List of chart dictionaries that reference this metric

    Example:
        ```python
        charts = find_charts_using_metric(
            metric_id=metric.id,
            project_id=project.id,
            url='https://demo.fiddler.ai',
            token='your-api-token'
        )

        if charts:
            print(f"⚠️  Deleting this metric will break {len(charts)} charts:")
            for chart in charts:
                print(f"  - {chart.get('title', 'Untitled')}")
        ```
    """
    from .charts import ChartManager

    # Get the custom metric to find its model and name
    try:
        metric = fdl.CustomMetric.get(id_=metric_id)
    except Exception as e:
        logger.error(f"Failed to get metric {metric_id}: {e}")
        raise AssetNotFoundError(f"Custom metric with ID {metric_id} not found")

    # Create chart manager
    chart_mgr = ChartManager(url=url, token=token)

    # List all charts in the project
    try:
        all_charts = chart_mgr.list_charts(project_id=project_id)
    except Exception as e:
        logger.warning(
            f"Could not list charts in project {project_id}: {e}. "
            "Chart API may not be available or may require url/token parameters."
        )
        return []

    # Filter to charts that reference this metric
    using_metric = []
    for chart in all_charts:
        # Check if chart references this metric in any query
        data_source = chart.get('data_source', {})
        queries = data_source.get('queries', [])

        for query in queries:
            # Check if query references this custom metric
            if query.get('metric_type') == 'custom':
                query_metric_id = query.get('metric')
                query_metric_name = query.get('metric_name')

                # Match by ID or name
                if query_metric_id == metric_id or query_metric_name == metric.name:
                    using_metric.append(chart)
                    break  # Only count chart once even if multiple queries use it

    logger.info(
        f"Found {len(using_metric)} charts using metric {metric.name} "
        f"(ID: {metric_id})"
    )

    return using_metric


def find_all_metric_references(
    metric_id: str,
    project_id: str,
    url: Optional[str] = None,
    token: Optional[str] = None
) -> Dict[str, Any]:
    """Find all references to a custom metric across Charts and Alerts.

    Comprehensive discovery of all assets that depend on this metric.

    Args:
        metric_id: UUID of the custom metric
        project_id: Project ID to search within
        url: Fiddler instance URL (optional)
        token: API token (optional)

    Returns:
        Dictionary with reference information:
        {
            'metric_id': str,
            'metric_name': str,
            'charts': List[Dict],
            'alerts': List[AlertRule],
            'chart_count': int,
            'alert_count': int,
            'total_count': int,
            'has_references': bool
        }

    Example:
        ```python
        refs = find_all_metric_references(metric.id, project.id)

        if refs['has_references']:
            print(f"⚠️  WARNING: Metric '{refs['metric_name']}' is used in:")
            print(f"   {refs['chart_count']} charts")
            print(f"   {refs['alert_count']} alerts")
            print(f"   Total: {refs['total_count']} references")
            print()
            print("Deleting this metric will break these assets!")
        else:
            print("✓ No references found. Safe to delete.")
        ```
    """
    # Get metric info
    try:
        metric = fdl.CustomMetric.get(id_=metric_id)
    except Exception as e:
        logger.error(f"Failed to get metric {metric_id}: {e}")
        raise AssetNotFoundError(f"Custom metric with ID {metric_id} not found")

    logger.info(f"Finding all references to metric '{metric.name}' (ID: {metric_id})")

    # Find charts
    charts = find_charts_using_metric(metric_id, project_id, url, token)

    # Find alerts
    alerts = find_alerts_using_metric(metric_id)

    # Compile results
    result = {
        'metric_id': metric_id,
        'metric_name': metric.name,
        'charts': charts,
        'alerts': alerts,
        'chart_count': len(charts),
        'alert_count': len(alerts),
        'total_count': len(charts) + len(alerts),
        'has_references': len(charts) + len(alerts) > 0,
    }

    logger.info(
        f"Reference discovery complete: {result['chart_count']} charts, "
        f"{result['alert_count']} alerts"
    )

    return result


def migrate_alert_metric_reference(
    alert: fdl.AlertRule,
    old_metric_id: str,
    new_metric_id: str
) -> bool:
    """Update an alert to reference a new metric UUID.

    Args:
        alert: AlertRule to update
        old_metric_id: Old metric UUID (for verification)
        new_metric_id: New metric UUID or name

    Returns:
        True if successfully updated, False otherwise

    Note:
        This deletes and recreates the alert, which may change notification configs.
        Alerts cannot be modified in-place via the SDK.
    """
    # Verify this alert references the old metric
    if alert.metric_id != old_metric_id and alert.metric_id != old_metric_id:
        logger.warning(
            f"Alert '{alert.name}' does not reference metric {old_metric_id}. "
            f"It references: {alert.metric_id}"
        )
        return False

    try:
        # Get the new metric to use its name
        new_metric = fdl.CustomMetric.get(id_=new_metric_id)

        logger.info(
            f"Migrating alert '{alert.name}' from metric {old_metric_id} "
            f"to {new_metric.name} ({new_metric_id})"
        )

        # Delete old alert
        alert_config = {
            'name': alert.name,
            'model_id': alert.model_id,
            'metric_id': new_metric.name,  # Use metric name, not ID
            'bin_size': alert.bin_size,
            'priority': alert.priority,
            'compare_to': alert.compare_to,
            'condition': alert.condition,
            'warning_threshold': alert.warning_threshold,
            'critical_threshold': alert.critical_threshold,
        }

        # Add optional fields if present
        if hasattr(alert, 'compare_period'):
            alert_config['compare_period'] = alert.compare_period
        if hasattr(alert, 'compare_threshold'):
            alert_config['compare_threshold'] = alert.compare_threshold
        if hasattr(alert, 'columns') and alert.columns:
            alert_config['columns'] = alert.columns

        # Delete old alert
        alert.delete()

        # Create new alert with updated metric reference
        new_alert = fdl.AlertRule(**alert_config)
        new_alert.create()

        logger.info(f"Successfully migrated alert '{alert.name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to migrate alert '{alert.name}': {e}")
        return False


def migrate_chart_metric_reference(
    chart: Dict,
    old_metric_id: str,
    new_metric_id: str,
    url: Optional[str] = None,
    token: Optional[str] = None
) -> bool:
    """Update a chart to reference a new metric UUID.

    Note: This uses the unofficial Fiddler Chart API which may change.

    Args:
        chart: Chart dictionary
        old_metric_id: Old metric UUID (for verification)
        new_metric_id: New metric UUID
        url: Fiddler instance URL (optional)
        token: API token (optional)

    Returns:
        True if successfully updated, False otherwise
    """
    from .charts import ChartManager

    # Get the new metric
    try:
        new_metric = fdl.CustomMetric.get(id_=new_metric_id)
    except Exception as e:
        logger.error(f"Failed to get new metric {new_metric_id}: {e}")
        return False

    # Check if chart references the old metric
    data_source = chart.get('data_source', {})
    queries = data_source.get('queries', [])

    found_reference = False
    for query in queries:
        if query.get('metric_type') == 'custom' and query.get('metric') == old_metric_id:
            found_reference = True
            # Update metric reference
            query['metric'] = new_metric_id
            query['metric_name'] = new_metric.name

    if not found_reference:
        logger.warning(
            f"Chart '{chart.get('title', 'Untitled')}' does not reference "
            f"metric {old_metric_id}"
        )
        return False

    # Update chart via API
    try:
        chart_mgr = ChartManager(url=url, token=token)
        chart_id = chart.get('uuid')

        logger.info(
            f"Migrating chart '{chart.get('title', 'Untitled')}' from metric "
            f"{old_metric_id} to {new_metric_id}"
        )

        # Update chart using RequestClient
        client = chart_mgr._get_client()
        response = client.put(f'/v1/charts/{chart_id}', json=chart)

        if response.status_code == 200:
            logger.info(
                f"Successfully migrated chart '{chart.get('title', 'Untitled')}'"
            )
            return True
        else:
            logger.error(
                f"Failed to update chart: HTTP {response.status_code} - "
                f"{response.text}"
            )
            return False

    except Exception as e:
        logger.error(
            f"Failed to migrate chart '{chart.get('title', 'Untitled')}': {e}"
        )
        return False


def safe_update_metric(
    metric: fdl.CustomMetric,
    new_definition: str,
    auto_migrate: bool = True,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    validate: bool = True
) -> Tuple[fdl.CustomMetric, Dict[str, Any]]:
    """Safely update a custom metric's FQL definition with reference migration.

    This is the recommended way to update custom metrics. It handles the entire
    workflow:
    1. Find all references (charts, alerts)
    2. Validate new definition (optional)
    3. Delete old metric
    4. Create new metric with same name (gets new UUID)
    5. Migrate all references to new UUID (if auto_migrate=True)

    Args:
        metric: Existing CustomMetric to update
        new_definition: New FQL definition
        auto_migrate: If True, automatically migrate all charts/alerts
        project_id: Project ID (required if auto_migrate=True)
        url: Fiddler instance URL (optional)
        token: API token (optional)
        validate: If True, validate new definition before updating

    Returns:
        Tuple of (new_metric, migration_report):
        - new_metric: Updated CustomMetric object with new UUID
        - migration_report: Dict with migration statistics

    Example:
        ```python
        # Safe update with automatic reference migration
        metric = fdl.CustomMetric.from_name(name='my_metric', model_id=model.id)

        new_metric, report = safe_update_metric(
            metric=metric,
            new_definition='sum(if(tp(), 1, 0)) / sum(1)',
            auto_migrate=True,
            project_id=project.id,
            url='https://demo.fiddler.ai',
            token='your-api-token'
        )

        print(f"✓ Updated metric {new_metric.name}")
        print(f"  Old ID: {report['old_metric_id']}")
        print(f"  New ID: {new_metric.id}")
        print(f"  Migrated: {report['migrated_count']} references")
        print(f"  Failed: {report['failed_count']} references")
        ```
    """
    old_metric_id = metric.id
    old_name = metric.name

    logger.info(f"Starting safe update for metric '{old_name}' (ID: {old_metric_id})")

    # Step 1: Find all references
    references = None
    if auto_migrate:
        if not project_id:
            raise ValueError(
                "project_id is required when auto_migrate=True to find chart references"
            )

        logger.info("Finding all references to metric...")
        references = find_all_metric_references(old_metric_id, project_id, url, token)

        logger.info(
            f"Found {references['total_count']} references: "
            f"{references['chart_count']} charts, {references['alert_count']} alerts"
        )

    # Step 2: Validate new definition (optional)
    if validate:
        from .metrics import CustomMetricManager

        model = fdl.Model.get(id_=metric.model_id)
        mgr = CustomMetricManager()
        is_valid, error = mgr.validate_metric_definition(new_definition, model)

        if not is_valid:
            from ..exceptions import FQLError
            raise FQLError(
                f"Invalid metric definition: {error}",
                expression=new_definition
            )

        logger.info("✓ New definition validated successfully")

    # Step 3: Delete old metric
    logger.info(f"Deleting old metric '{old_name}' (ID: {old_metric_id})")
    old_description = metric.description if hasattr(metric, 'description') else ''
    metric.delete()

    # Brief pause to ensure deletion is processed
    time.sleep(0.5)

    # Step 4: Create new metric with same name (gets new UUID)
    logger.info(f"Creating new metric '{old_name}' with updated definition")
    new_metric = fdl.CustomMetric(
        model_id=metric.model_id,
        name=old_name,
        description=old_description,
        definition=new_definition,
    )
    new_metric.create()

    new_metric_id = new_metric.id
    logger.info(f"✓ Created new metric with ID: {new_metric_id}")

    # Step 5: Migrate references (if enabled)
    migration_report = {
        'old_metric_id': old_metric_id,
        'new_metric_id': new_metric_id,
        'metric_name': old_name,
        'auto_migrate_enabled': auto_migrate,
        'migrated_count': 0,
        'failed_count': 0,
        'skipped_count': 0,
        'chart_migrations': [],
        'alert_migrations': [],
    }

    if auto_migrate and references and references['has_references']:
        logger.info(f"Migrating {references['total_count']} references...")

        # Migrate charts
        for chart in references['charts']:
            success = migrate_chart_metric_reference(
                chart, old_metric_id, new_metric_id, url, token
            )

            chart_title = chart.get('title', 'Untitled')
            if success:
                migration_report['migrated_count'] += 1
                migration_report['chart_migrations'].append({
                    'title': chart_title,
                    'success': True
                })
            else:
                migration_report['failed_count'] += 1
                migration_report['chart_migrations'].append({
                    'title': chart_title,
                    'success': False
                })

        # Migrate alerts
        for alert in references['alerts']:
            success = migrate_alert_metric_reference(
                alert, old_metric_id, new_metric_id
            )

            if success:
                migration_report['migrated_count'] += 1
                migration_report['alert_migrations'].append({
                    'name': alert.name,
                    'success': True
                })
            else:
                migration_report['failed_count'] += 1
                migration_report['alert_migrations'].append({
                    'name': alert.name,
                    'success': False
                })

        logger.info(
            f"Migration complete: {migration_report['migrated_count']} successful, "
            f"{migration_report['failed_count']} failed"
        )
    elif auto_migrate and references and not references['has_references']:
        logger.info("No references found - no migration needed")
    elif not auto_migrate:
        migration_report['skipped_count'] = (
            len(references['charts']) + len(references['alerts'])
            if references
            else 0
        )
        logger.warning(
            "auto_migrate=False - References were NOT migrated. "
            "Charts and alerts may be broken!"
        )

    return new_metric, migration_report
