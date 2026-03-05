"""Bulk alert creation orchestrator.

This module provides the BulkAlertCreator class for profile-driven,
bulk alert creation across all models in a Fiddler environment. It
composes AlertManager, BaselineManager, iterate_models_safe(), and
SchemaValidator to create alerts with fault-tolerant iteration,
idempotency, and detailed reporting.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from fiddler.constants.alert_rule import AlertThresholdAlgo
from fiddler.constants.model import ModelTask

from .alert_profiles import AlertProfile, AlertSpec, NotificationConfig, ThresholdStrategy
from .assets.alerts import AlertManager
from .assets.baselines import BaselineManager
from .exceptions import BulkAlertCreationError
from .iteration import iterate_models_safe

logger = logging.getLogger(__name__)

MAX_ALERT_NAME_LENGTH = 255


@dataclass
class ModelScopeFilter:
    """Controls which models are processed.

    All filters are additive (AND logic). If a filter is None/empty,
    it does not restrict.
    """

    project_ids: Optional[List[str]] = None
    project_names: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    model_name_pattern: Optional[str] = None
    task_types: Optional[List[ModelTask]] = None
    exclude_model_names: List[str] = field(default_factory=list)
    exclude_model_ids: List[str] = field(default_factory=list)
    exclude_projects: List[str] = field(default_factory=list)
    max_models: Optional[int] = None


# Fields that can be updated via alert.update()
MUTABLE_THRESHOLD_FIELDS = {
    'warning_threshold', 'critical_threshold',
    'evaluation_delay', 'auto_threshold_params',
}

# Fields that are immutable or not patchable — changes require delete+recreate
# NOTE: priority is NOT supported by the SDK's update() method, so it's here
IMMUTABLE_FIELDS = {
    'metric_id', 'bin_size', 'compare_to', 'condition',
    'columns', 'baseline_id', 'priority',
}


@dataclass
class BulkAlertResult:
    """Result of a bulk alert creation or update operation."""

    models_processed: int = 0
    models_skipped: int = 0
    models_failed: int = 0
    alerts_created: int = 0
    alerts_updated: int = 0
    alerts_recreated: int = 0
    alerts_skipped_existing: int = 0
    alerts_skipped_invalid: int = 0
    alerts_failed: int = 0
    baselines_created: int = 0
    notifications_configured: int = 0
    errors: List[Tuple[str, str, str]] = field(default_factory=list)
    model_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def total_alerts_attempted(self) -> int:
        return (
            self.alerts_created
            + self.alerts_updated
            + self.alerts_recreated
            + self.alerts_skipped_existing
            + self.alerts_skipped_invalid
            + self.alerts_failed
        )

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f'Models: {self.models_processed} processed, '
            f'{self.models_skipped} skipped, {self.models_failed} failed',
            f'Alerts: {self.alerts_created} created, '
            f'{self.alerts_updated} updated, '
            f'{self.alerts_recreated} recreated, '
            f'{self.alerts_skipped_existing} skipped (existing), '
            f'{self.alerts_skipped_invalid} skipped (invalid), '
            f'{self.alerts_failed} failed',
            f'Baselines: {self.baselines_created} auto-created',
            f'Notifications: {self.notifications_configured} configured',
        ]
        if self.errors:
            lines.append(f'Errors: {len(self.errors)} total')
            for model, alert, err in self.errors[:5]:
                lines.append(f'  - {model}/{alert}: {err}')
            if len(self.errors) > 5:
                lines.append(f'  ... and {len(self.errors) - 5} more')
        return '\n'.join(lines)


class BulkAlertCreator:
    """Orchestrator for bulk alert creation and update across models.

    Uses alert profiles to declaratively define what alerts should exist,
    then creates or updates them across all matching models with
    fault-tolerant iteration, idempotency, and detailed reporting.

    Supports two modes:
    - **create**: Create new alerts. Skip existing (by name match) unless
      overwrite=True, in which case delete+recreate.
    - **update**: Update existing alerts to match the profile. Mutable fields
      (thresholds, evaluation_delay) are patched in-place. Immutable fields
      (bin_size, metric_id, etc.) trigger delete+recreate. Missing alerts
      are created. Notifications are always synced.

    Example:
        ```python
        from fiddler_utils import get_or_init, BulkAlertCreator
        from fiddler_utils.alert_profiles import get_default_ml_profile, NotificationConfig

        get_or_init(url=URL, token=TOKEN, log_level='ERROR')

        profile = get_default_ml_profile()
        profile.notification = NotificationConfig(emails=['team@company.com'])

        creator = BulkAlertCreator(profile=profile)

        # Create alerts (dry run first)
        result = creator.run(mode='create', dry_run=True)
        creator.print_report(result)

        # Later: update thresholds and notifications
        profile.default_sigma_warning = 2.5
        profile.notification = NotificationConfig(emails=['new-team@company.com'])
        result = creator.run(mode='update', dry_run=False)
        creator.print_report(result)
        ```
    """

    def __init__(
        self,
        profile: AlertProfile,
        scope: Optional[ModelScopeFilter] = None,
        overwrite: bool = False,
        skip_invalid: bool = True,
        on_error: str = 'warn',
    ):
        self.profile = profile
        self.scope = scope or ModelScopeFilter()
        self.overwrite = overwrite
        self.skip_invalid = skip_invalid
        self.on_error = on_error
        self._alert_mgr = AlertManager()
        self._baseline_mgr = BaselineManager()

    def run(
        self,
        mode: str = 'create',
        dry_run: bool = False,
    ) -> BulkAlertResult:
        """Execute bulk alert creation or update.

        Args:
            mode: Operation mode.
                'create' — Create new alerts, skip existing (unless overwrite=True).
                'update' — Update existing alerts to match profile. Creates missing
                    alerts. Patches mutable fields in-place, delete+recreates for
                    immutable field changes.
            dry_run: If True, validate and report without modifying alerts.

        Returns:
            BulkAlertResult with detailed results.
        """
        if mode not in ('create', 'update'):
            raise ValueError(f"mode must be 'create' or 'update', got '{mode}'")

        result = BulkAlertResult()
        run_label = f'{mode.upper()} {"(DRY RUN)" if dry_run else "(LIVE)"}'
        logger.info(f'Starting bulk alert {run_label} with profile: {self.profile.name}')

        # Resolve project filter
        project_ids = self._resolve_project_ids()

        models_count = 0
        for project, model, error in iterate_models_safe(
            project_ids=project_ids,
            fetch_full=True,
            on_error=self.on_error,
        ):
            if error:
                result.models_skipped += 1
                continue

            if not self._should_process_model(project, model):
                result.models_skipped += 1
                continue

            if self.scope.max_models and models_count >= self.scope.max_models:
                break

            models_count += 1
            model_result = self._process_model(project, model, mode, dry_run, result)
            result.model_details[model.name] = model_result

            if model_result.get('failed', False):
                result.models_failed += 1
            else:
                result.models_processed += 1

        logger.info(f'Bulk alert {run_label} complete. {result.summary()}')
        return result

    def _process_model(
        self,
        project: fdl.Project,
        model: fdl.Model,
        mode: str,
        dry_run: bool,
        result: BulkAlertResult,
    ) -> Dict[str, Any]:
        """Process a single model — resolve specs, create or update alerts."""
        model_result: Dict[str, Any] = {
            'project': project.name,
            'model': model.name,
            'task_type': str(getattr(model, 'task', 'unknown')),
            'created': 0,
            'updated': 0,
            'recreated': 0,
            'skipped_existing': 0,
            'skipped_invalid': 0,
            'failed': False,
            'alerts': [],
        }

        task_type = self._resolve_model_task_type(model)
        logger.info(
            f'Processing model: {project.name}/{model.name} (task: {task_type})'
        )

        # Get existing alerts (full objects for update mode, names for create)
        existing_alerts = self._get_existing_alerts(model.id)
        existing_names = {a.name for a in existing_alerts}

        # Ensure baseline if needed
        baseline_id = None
        has_drift_specs = any(
            s.requires_baseline for s in self.profile.filter_for_task_type(task_type)
        )
        if has_drift_specs:
            baseline_id = self._ensure_baseline(model, dry_run, result)

        # Resolve specs into concrete alert definitions
        alert_defs = self._resolve_specs_for_model(model, task_type, baseline_id)

        if not alert_defs:
            logger.info(f'No applicable alert specs for {model.name}')

        for alert_def in alert_defs:
            alert_name = alert_def['name']

            if mode == 'create':
                status = self._handle_create_mode(
                    model, alert_def, alert_name, existing_names,
                    dry_run, result,
                )
            else:  # mode == 'update'
                existing_alert = next(
                    (a for a in existing_alerts if a.name == alert_name), None
                )
                status = self._handle_update_mode(
                    model, alert_def, alert_name, existing_alert,
                    dry_run, result,
                )

            model_result['alerts'].append({
                'name': alert_name,
                'status': status,
                'metric_id': alert_def['metric_id'],
            })
            if status == 'created':
                model_result['created'] += 1
            elif status == 'updated':
                model_result['updated'] += 1
            elif status == 'recreated':
                model_result['recreated'] += 1
            elif status == 'skipped_existing':
                model_result['skipped_existing'] += 1
            elif status == 'skipped_invalid':
                model_result['skipped_invalid'] += 1

        return model_result

    def _handle_create_mode(
        self,
        model: fdl.Model,
        alert_def: Dict[str, Any],
        alert_name: str,
        existing_names: Set[str],
        dry_run: bool,
        result: BulkAlertResult,
    ) -> str:
        """Handle a single alert in create mode."""
        # Idempotency: skip existing unless overwrite
        if alert_name in existing_names and not self.overwrite:
            result.alerts_skipped_existing += 1
            return 'skipped_existing'

        # Overwrite: delete existing first
        if alert_name in existing_names and self.overwrite:
            if not dry_run:
                try:
                    existing_alert = self._alert_mgr.get_alert_by_name(
                        model.id, alert_name
                    )
                    existing_alert.delete()
                    logger.info(f'Deleted existing alert: {alert_name}')
                except Exception as e:
                    logger.warning(f'Failed to delete existing alert {alert_name}: {e}')
                    result.alerts_failed += 1
                    result.errors.append((model.name, alert_name, str(e)))
                    return 'failed'

        return self._create_single_alert(model, alert_def, dry_run, result)

    def _handle_update_mode(
        self,
        model: fdl.Model,
        alert_def: Dict[str, Any],
        alert_name: str,
        existing_alert: Optional[fdl.AlertRule],
        dry_run: bool,
        result: BulkAlertResult,
    ) -> str:
        """Handle a single alert in update mode.

        Logic:
        1. Alert doesn't exist → create it.
        2. Alert exists, immutable fields differ → delete+recreate.
        3. Alert exists, only mutable fields differ → patch in-place.
        4. Alert exists, nothing differs → sync notifications only.
        """
        if existing_alert is None:
            # Alert doesn't exist yet — create it
            return self._create_single_alert(model, alert_def, dry_run, result)

        # Compare existing alert against desired spec
        immutable_changes = self._diff_immutable_fields(existing_alert, alert_def)
        mutable_changes = self._diff_mutable_fields(existing_alert, alert_def)

        if immutable_changes:
            # Immutable fields changed — must delete+recreate
            logger.info(
                f'  Immutable fields changed for {alert_name}: '
                f'{", ".join(immutable_changes)}. Will delete+recreate.'
            )
            return self._recreate_alert(
                model, alert_def, alert_name, existing_alert, dry_run, result
            )

        if mutable_changes:
            # Only mutable fields changed — patch in-place
            return self._update_single_alert(
                existing_alert, alert_def, mutable_changes, dry_run, result
            )

        # Nothing changed — just sync notifications
        if not dry_run and self.profile.notification.has_notifications:
            if self._configure_notification(existing_alert):
                result.notifications_configured += 1

        result.alerts_skipped_existing += 1
        return 'skipped_existing'

    def _diff_immutable_fields(
        self,
        existing: fdl.AlertRule,
        desired: Dict[str, Any],
    ) -> List[str]:
        """Compare immutable fields between existing alert and desired spec.

        Returns list of field names that differ. Empty list = no changes.
        """
        changes = []
        for field_name in IMMUTABLE_FIELDS:
            if field_name not in desired:
                continue
            existing_val = getattr(existing, field_name, None)
            desired_val = desired[field_name]

            # Normalize for comparison (enums vs strings)
            if hasattr(existing_val, 'value'):
                existing_val = existing_val.value
            if hasattr(desired_val, 'value'):
                desired_val = desired_val.value

            # Normalize None vs empty list for columns
            if existing_val is None and desired_val is None:
                continue
            if field_name == 'columns':
                existing_val = sorted(existing_val) if existing_val else []
                desired_val = sorted(desired_val) if desired_val else []

            if str(existing_val) != str(desired_val):
                changes.append(field_name)

        return changes

    def _diff_mutable_fields(
        self,
        existing: fdl.AlertRule,
        desired: Dict[str, Any],
    ) -> List[str]:
        """Compare mutable threshold fields between existing and desired.

        Only checks fields that can be patched via alert.update():
        warning_threshold, critical_threshold, evaluation_delay, auto_threshold_params.

        Returns list of field names that differ.
        """
        changes = []
        for field_name in MUTABLE_THRESHOLD_FIELDS:
            if field_name not in desired:
                continue
            existing_val = getattr(existing, field_name, None)
            desired_val = desired[field_name]

            if field_name == 'auto_threshold_params':
                if existing_val != desired_val:
                    changes.append(field_name)
            else:
                if existing_val is None and desired_val is None:
                    continue
                if existing_val != desired_val:
                    changes.append(field_name)

        return changes

    def _update_single_alert(
        self,
        existing: fdl.AlertRule,
        desired: Dict[str, Any],
        changed_fields: List[str],
        dry_run: bool,
        result: BulkAlertResult,
    ) -> str:
        """Update mutable threshold fields on an existing alert in-place.

        Only called when _diff_mutable_fields detected changes. All fields
        in changed_fields are guaranteed to be patchable via alert.update().
        """
        alert_name = existing.name

        if dry_run:
            logger.info(
                f'  [DRY RUN] Would update {alert_name}: '
                f'{", ".join(changed_fields)}'
            )
            result.alerts_updated += 1
            return 'updated'

        try:
            for f in changed_fields:
                setattr(existing, f, desired.get(f))
            existing.update()
            logger.info(f'  Updated {alert_name}: {", ".join(changed_fields)}')
        except Exception as e:
            logger.warning(f'  Failed to update {alert_name}: {e}')
            result.alerts_failed += 1
            result.errors.append((str(existing.model_id), alert_name, str(e)))
            return 'failed'

        # Sync notifications after successful update
        if self.profile.notification.has_notifications:
            if self._configure_notification(existing):
                result.notifications_configured += 1

        result.alerts_updated += 1
        return 'updated'

    def _recreate_alert(
        self,
        model: fdl.Model,
        alert_def: Dict[str, Any],
        alert_name: str,
        existing_alert: fdl.AlertRule,
        dry_run: bool,
        result: BulkAlertResult,
    ) -> str:
        """Delete an existing alert and create a new one from the spec.

        Used when immutable fields (bin_size, metric_id, priority, etc.)
        have changed and cannot be patched in-place.

        Counts directly into alerts_recreated — never touches alerts_created.
        """
        if dry_run:
            logger.info(f'  [DRY RUN] Would recreate: {alert_name}')
            result.alerts_recreated += 1
            return 'recreated'

        # Delete existing
        try:
            existing_alert.delete()
            logger.info(f'  Deleted for recreate: {alert_name}')
        except Exception as e:
            logger.warning(f'  Failed to delete {alert_name} for recreate: {e}')
            result.alerts_failed += 1
            result.errors.append((model.name, alert_name, str(e)))
            return 'failed'

        # Create new
        try:
            alert = self._alert_mgr._create_asset(
                model_id=str(model.id),
                asset_data=alert_def,
            )
            result.alerts_recreated += 1
            logger.info(f'  Recreated: {alert_name}')

            if self.profile.notification.has_notifications:
                if self._configure_notification(alert):
                    result.notifications_configured += 1

            return 'recreated'
        except Exception as e:
            if self.skip_invalid:
                result.alerts_failed += 1
                result.errors.append((model.name, alert_name, str(e)))
                logger.warning(f'  Failed to recreate {alert_name}: {e}')
                return 'failed'
            else:
                raise BulkAlertCreationError(
                    f'Failed to recreate alert {alert_name}: {e}',
                    partial_result=result,
                )

    def _should_process_model(
        self,
        project: fdl.Project,
        model: fdl.Model,
    ) -> bool:
        """Apply scope filter to determine if a model should be processed."""
        scope = self.scope

        if scope.project_names and project.name not in scope.project_names:
            return False
        if scope.exclude_projects and project.name in scope.exclude_projects:
            return False
        if scope.model_names and model.name not in scope.model_names:
            return False
        if scope.exclude_model_names and model.name in scope.exclude_model_names:
            return False
        if scope.exclude_model_ids and str(model.id) in scope.exclude_model_ids:
            return False
        if scope.model_name_pattern:
            if not re.search(scope.model_name_pattern, model.name):
                return False
        if scope.task_types:
            task_type = self._resolve_model_task_type(model)
            if task_type not in scope.task_types:
                return False

        return True

    def _resolve_model_task_type(self, model: fdl.Model) -> ModelTask:
        """Determine the ModelTask for a given fdl.Model."""
        task = getattr(model, 'task', None)
        if task is None:
            return ModelTask.NOT_SET

        if isinstance(task, ModelTask):
            return task

        # Handle string values
        task_str = str(task).lower().replace('modeltask.', '')
        for mt in ModelTask:
            if mt.value == task_str:
                return mt

        return ModelTask.NOT_SET

    def _get_existing_alerts(self, model_id: str) -> List[fdl.AlertRule]:
        """Get all existing alert rules for a model."""
        try:
            return self._alert_mgr.list_assets(model_id=model_id)
        except Exception as e:
            logger.warning(f'Could not list existing alerts: {e}')
            return []

    def _ensure_baseline(
        self,
        model: fdl.Model,
        dry_run: bool,
        result: BulkAlertResult,
    ) -> Optional[str]:
        """Ensure a baseline exists for drift alerts.

        If auto_create_baseline is True and no baseline exists, creates
        a rolling baseline using the profile's baseline_config.

        Returns:
            baseline_id if available, None otherwise.
        """
        # Check for existing baselines
        try:
            baselines = list(fdl.Baseline.list(model_id=model.id))
            if baselines:
                baseline = baselines[0]
                logger.info(
                    f'Using existing baseline: {baseline.name} (ID: {baseline.id})'
                )
                return str(baseline.id)
        except Exception as e:
            logger.warning(f'Could not list baselines for {model.name}: {e}')

        if not self.profile.auto_create_baseline:
            logger.warning(
                f'No baseline for {model.name} and auto_create_baseline=False. '
                f'Drift alerts will be skipped.'
            )
            return None

        if dry_run:
            logger.info(f'Would auto-create rolling baseline for {model.name}')
            result.baselines_created += 1
            return '__dry_run_baseline_id__'

        # Create rolling baseline
        config = self.profile.baseline_config
        baseline_name = config.get('name_template', '__auto_rolling_{model_name}').format(
            model_name=model.name
        )

        try:
            from fiddler.constants.baseline import WindowBinSize
            from fiddler.constants.dataset import EnvType

            baseline = fdl.Baseline(
                name=baseline_name,
                model_id=model.id,
                environment=EnvType.PRODUCTION,
                type_='ROLLING',
                offset_delta=config.get('offset_delta', 7),
                window_bin_size=config.get('window_bin_size', WindowBinSize.DAY),
            )
            baseline.create()
            result.baselines_created += 1
            logger.info(
                f'Created rolling baseline: {baseline_name} (ID: {baseline.id})'
            )
            return str(baseline.id)
        except Exception as e:
            logger.warning(
                f'Failed to auto-create baseline for {model.name}: {e}. '
                f'Drift alerts will be skipped.'
            )
            return None

    def _resolve_specs_for_model(
        self,
        model: fdl.Model,
        task_type: ModelTask,
        baseline_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Resolve profile specs into concrete alert definitions for a model.

        Handles:
        - Filtering specs by task type
        - Expanding per-column specs into one alert per column
        - Rendering name templates
        - Resolving thresholds (profile-level sigma defaults)
        - Injecting baseline_id for drift alerts
        """
        applicable_specs = self.profile.filter_for_task_type(task_type)
        alert_defs = []

        for spec in applicable_specs:
            # Skip drift alerts if no baseline available
            if spec.requires_baseline and not baseline_id:
                logger.info(
                    f'Skipping {spec.metric_id} for {model.name}: no baseline available'
                )
                continue

            # Resolve columns
            columns_list = self._get_columns_for_spec(model, spec)
            if spec.requires_columns and not columns_list:
                logger.info(
                    f'Skipping {spec.metric_id} for {model.name}: no applicable columns'
                )
                continue

            # Resolve threshold with profile-level defaults
            threshold = self.profile.resolve_threshold(spec)
            threshold_params = threshold.to_sdk_params()

            # Expand per-column or create single alert
            for column in columns_list:
                alert_name = spec.name_template.format(
                    model_name=model.name,
                    column_name=column or '',
                )

                # Truncate name if too long
                if len(alert_name) > MAX_ALERT_NAME_LENGTH:
                    alert_name = alert_name[: MAX_ALERT_NAME_LENGTH - 3] + '...'

                alert_def: Dict[str, Any] = {
                    'name': alert_name,
                    'metric_id': spec.metric_id,
                    'bin_size': spec.bin_size,
                    'compare_to': spec.compare_to,
                    'condition': spec.condition,
                    'priority': spec.priority,
                    'category': spec.category,
                    **threshold_params,
                }

                if column:
                    alert_def['columns'] = [column]

                if spec.requires_baseline and baseline_id:
                    alert_def['baseline_id'] = baseline_id

                if spec.compare_bin_delta is not None:
                    alert_def['compare_bin_delta'] = spec.compare_bin_delta

                alert_defs.append(alert_def)

        return alert_defs

    def _get_columns_for_spec(
        self,
        model: fdl.Model,
        spec: AlertSpec,
    ) -> List[Optional[str]]:
        """Resolve which columns an alert spec should be applied to.

        Returns:
            List of column names. Returns [None] if spec is not per-column.
        """
        if not spec.requires_columns:
            return [None]

        source = spec.columns_source

        if source == 'inputs':
            columns = list(getattr(model.spec, 'inputs', None) or [])
        elif source == 'outputs':
            columns = list(getattr(model.spec, 'outputs', None) or [])
        elif source == 'all':
            inputs = list(getattr(model.spec, 'inputs', None) or [])
            outputs = list(getattr(model.spec, 'outputs', None) or [])
            columns = inputs + outputs
        elif isinstance(source, list):
            columns = source
        else:
            columns = []

        if len(columns) > 100:
            logger.warning(
                f'Model {model.name} has {len(columns)} columns for '
                f'{spec.metric_id} alerts. Consider filtering with columns_source.'
            )

        return columns if columns else []

    def _create_single_alert(
        self,
        model: fdl.Model,
        alert_def: Dict[str, Any],
        dry_run: bool,
        result: BulkAlertResult,
    ) -> str:
        """Create a single alert. Returns status string."""
        alert_name = alert_def['name']

        if dry_run:
            logger.info(f'  [DRY RUN] Would create: {alert_name}')
            result.alerts_created += 1
            return 'created'

        try:
            alert = self._alert_mgr._create_asset(
                model_id=str(model.id),
                asset_data=alert_def,
            )
            result.alerts_created += 1
            logger.info(f'  Created: {alert_name}')

            # Configure notifications
            if self.profile.notification.has_notifications:
                if self._configure_notification(alert):
                    result.notifications_configured += 1

            return 'created'

        except Exception as e:
            if self.skip_invalid:
                result.alerts_skipped_invalid += 1
                result.errors.append((model.name, alert_name, str(e)))
                logger.warning(f'  Failed to create {alert_name}: {e}')
                return 'skipped_invalid'
            else:
                raise BulkAlertCreationError(
                    f'Failed to create alert {alert_name} for model {model.name}: {e}',
                    partial_result=result,
                )

    def _configure_notification(self, alert: fdl.AlertRule) -> bool:
        """Set notification config on a created alert."""
        nc = self.profile.notification
        try:
            kwargs: Dict[str, Any] = {}
            if nc.emails:
                kwargs['emails'] = nc.emails
            if nc.pagerduty_services:
                kwargs['pagerduty_services'] = nc.pagerduty_services
                kwargs['pagerduty_severity'] = nc.pagerduty_severity
            if nc.webhooks:
                kwargs['webhooks'] = nc.webhooks

            alert.set_notification_config(**kwargs)
            return True
        except Exception as e:
            logger.warning(
                f"Failed to configure notifications for '{alert.name}': {e}"
            )
            return False

    def _resolve_project_ids(self) -> Optional[List[str]]:
        """Resolve project name filter to project IDs."""
        if self.scope.project_ids:
            return self.scope.project_ids

        if self.scope.project_names:
            try:
                projects = list(fdl.Project.list())
                return [
                    str(p.id)
                    for p in projects
                    if p.name in self.scope.project_names
                ]
            except Exception as e:
                logger.warning(f'Failed to resolve project names: {e}')
                return None

        return None

    def print_report(
        self,
        result: BulkAlertResult,
        show_per_model: bool = False,
    ) -> None:
        """Print formatted report of bulk creation results."""
        print('\n' + '=' * 60)
        print(f'Bulk Alert Creation Report — Profile: {self.profile.name}')
        print('=' * 60)
        print(result.summary())

        if show_per_model and result.model_details:
            print('\n--- Per-Model Details ---')
            for model_name, details in result.model_details.items():
                created = details.get('created', 0)
                skipped = details.get('skipped_existing', 0)
                invalid = details.get('skipped_invalid', 0)
                print(
                    f'  {details["project"]}/{model_name} '
                    f'({details["task_type"]}): '
                    f'{created} created, {skipped} skipped, {invalid} invalid'
                )

        print('=' * 60 + '\n')

    def export_report_csv(
        self,
        result: BulkAlertResult,
        output_path: str = 'bulk_alert_report.csv',
    ) -> str:
        """Export detailed results to CSV."""
        rows = []
        for model_name, details in result.model_details.items():
            for alert_info in details.get('alerts', []):
                rows.append({
                    'project': details.get('project', ''),
                    'model': model_name,
                    'task_type': details.get('task_type', ''),
                    'alert_name': alert_info.get('name', ''),
                    'metric_id': alert_info.get('metric_id', ''),
                    'status': alert_info.get('status', ''),
                })

        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f'Report exported to {output_path}')
        else:
            logger.info('No alert details to export')

        return output_path
