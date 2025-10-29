"""Project management utilities for Fiddler projects and environment-wide operations.

This module provides ProjectManager for environment discovery, statistical analysis,
and inventory reporting that spans multiple projects and models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import logging
import pandas as pd

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .exceptions import FiddlerUtilsError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a single model.

    Attributes:
        id: Model UUID
        name: Model name
        version: Model version (if versioned)
        features: List of feature names from model spec
        feature_count: Number of features
        created_at: Model creation timestamp
        updated_at: Model last updated timestamp
        error: Error message if model fetch failed
    """
    id: str
    name: str
    version: Optional[str] = None
    features: List[str] = field(default_factory=list)
    feature_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class ProjectInfo:
    """Information about a single project.

    Attributes:
        id: Project UUID
        name: Project name
        models: Dictionary of model_id -> ModelInfo
        model_count: Number of models in project
        feature_count: Total number of features across all models
    """
    id: str
    name: str
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    model_count: int = 0
    feature_count: int = 0


@dataclass
class EnvironmentHierarchy:
    """Complete environment hierarchy.

    Attributes:
        projects: Dictionary of project_id -> ProjectInfo
        total_projects: Total number of projects
        total_models: Total number of models across all projects
        total_features: Total number of features across all models
        collected_at: Timestamp when data was collected
    """
    projects: Dict[str, ProjectInfo] = field(default_factory=dict)
    total_projects: int = 0
    total_models: int = 0
    total_features: int = 0
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EnvironmentStats:
    """Aggregated environment statistics.

    Attributes:
        total_projects: Total number of projects
        total_models: Total number of models
        total_features: Total number of features
        models_per_project_mean: Average models per project
        models_per_project_median: Median models per project
        models_per_project_min: Minimum models in any project
        models_per_project_max: Maximum models in any project
        features_per_model_mean: Average features per model
        features_per_model_median: Median features per model
        features_per_model_min: Minimum features in any model
        features_per_model_max: Maximum features in any model
        top_projects_by_models: Top N projects ranked by model count
        top_models_by_features: Top N models ranked by feature count
    """
    total_projects: int = 0
    total_models: int = 0
    total_features: int = 0
    models_per_project_mean: float = 0.0
    models_per_project_median: float = 0.0
    models_per_project_min: int = 0
    models_per_project_max: int = 0
    features_per_model_mean: float = 0.0
    features_per_model_median: float = 0.0
    features_per_model_min: int = 0
    features_per_model_max: int = 0
    top_projects_by_models: List[Tuple[str, int]] = field(default_factory=list)
    top_models_by_features: List[Tuple[str, str, int]] = field(default_factory=list)  # (project, model, count)


@dataclass
class TimestampAnalysis:
    """Timestamp-based analysis of models.

    Attributes:
        models_with_timestamps: Number of models with creation timestamps
        timestamp_coverage_pct: Percentage of models with timestamps
        earliest_created: Earliest model creation date
        latest_created: Latest model creation date
        most_recent_update: Most recent model update date
        avg_days_between_create_update: Average days between creation and last update
        newest_models: List of newest models (ModelInfo)
        oldest_models: List of oldest models (ModelInfo)
        most_recently_updated_models: List of most recently updated models (ModelInfo)
    """
    models_with_timestamps: int = 0
    timestamp_coverage_pct: float = 0.0
    earliest_created: Optional[datetime] = None
    latest_created: Optional[datetime] = None
    most_recent_update: Optional[datetime] = None
    avg_days_between_create_update: Optional[float] = None
    newest_models: List[ModelInfo] = field(default_factory=list)
    oldest_models: List[ModelInfo] = field(default_factory=list)
    most_recently_updated_models: List[ModelInfo] = field(default_factory=list)


class ProjectManager:
    """Manager for Fiddler projects and environment-wide operations.

    Handles project-level operations, environment discovery, and inventory
    analysis that span multiple projects and models.

    Example:
        ```python
        from fiddler_utils import ProjectManager

        mgr = ProjectManager()

        # Get complete environment hierarchy
        hierarchy = mgr.get_environment_hierarchy(
            include_features=True,
            include_timestamps=True
        )

        # Calculate statistics
        stats = mgr.get_environment_statistics(hierarchy)

        # Export to CSV
        files = mgr.export_environment_to_csv(prefix='env_stats')
        ```
    """

    def __init__(self):
        """Initialize ProjectManager."""
        pass

    def list_projects(
        self,
        names: Optional[List[str]] = None,
        include_stats: bool = False
    ) -> List[fdl.Project]:
        """List all projects, optionally filtered and with statistics.

        Args:
            names: Filter to specific project names
            include_stats: Include model/feature counts (requires additional API calls)

        Returns:
            List of Project objects

        Example:
            ```python
            # List all projects
            projects = mgr.list_projects()

            # List specific projects with stats
            projects = mgr.list_projects(
                names=['project1', 'project2'],
                include_stats=True
            )
            ```
        """
        projects = list(fdl.Project.list())

        if names:
            projects = [p for p in projects if p.name in names]

        logger.info(f'Listed {len(projects)} projects')
        return projects

    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Get statistics for a single project.

        Args:
            project_id: Project UUID

        Returns:
            Dict with keys: model_count, feature_count, etc.

        Example:
            ```python
            stats = mgr.get_project_stats(project.id)
            print(f"Models: {stats['model_count']}")
            ```
        """
        models = list(fdl.Model.list(project_id=project_id))
        model_count = len(models)
        feature_count = 0

        for model in models:
            try:
                # Get full model to access spec
                full_model = fdl.Model.get(id_=model.id)
                inputs = full_model.spec.inputs or []
                feature_count += len(inputs)
            except Exception as e:
                logger.debug(f'Could not get feature count for model {model.id}: {e}')

        return {
            'model_count': model_count,
            'feature_count': feature_count,
        }

    def get_environment_hierarchy(
        self,
        include_features: bool = True,
        include_timestamps: bool = True,
        include_assets: bool = False
    ) -> EnvironmentHierarchy:
        """Traverse entire environment and extract hierarchy.

        Core functionality extracted from env_stats notebook - builds complete
        inventory of projects → models → features.

        Args:
            include_features: Extract feature lists from model specs
            include_timestamps: Include created_at/updated_at timestamps
            include_assets: Include counts of segments, metrics, alerts (slower)

        Returns:
            EnvironmentHierarchy dataclass with nested structure

        Example:
            ```python
            hierarchy = mgr.get_environment_hierarchy(
                include_features=True,
                include_timestamps=True
            )

            print(f"Projects: {hierarchy.total_projects}")
            print(f"Models: {hierarchy.total_models}")
            print(f"Features: {hierarchy.total_features}")
            ```
        """
        logger.info('Starting environment hierarchy traversal')
        hierarchy = EnvironmentHierarchy()

        # List all projects
        projects = list(fdl.Project.list())
        logger.info(f'Found {len(projects)} projects')

        for project in projects:
            project_info = ProjectInfo(
                id=str(project.id),
                name=project.name
            )

            try:
                # List models in project
                models = list(fdl.Model.list(project_id=project.id))
                logger.debug(f'Project {project.name}: {len(models)} models')

                for model_compact in models:
                    # Get full model details
                    try:
                        model = fdl.Model.get(id_=model_compact.id)

                        # Extract features
                        features = []
                        if include_features:
                            features = model.spec.inputs or []

                        # Create ModelInfo
                        model_info = ModelInfo(
                            id=str(model.id),
                            name=model.name,
                            version=model.version,
                            features=features,
                            feature_count=len(features)
                        )

                        # Add timestamps if requested
                        if include_timestamps:
                            model_info.created_at = getattr(model, 'created_at', None)
                            model_info.updated_at = getattr(model, 'updated_at', None)

                        # Add to project
                        project_info.models[str(model.id)] = model_info
                        project_info.feature_count += len(features)

                    except Exception as e:
                        # Record error but continue
                        error_msg = str(e)[:100]
                        logger.warning(f'Failed to fetch model {model_compact.name}: {error_msg}')

                        model_info = ModelInfo(
                            id=str(model_compact.id),
                            name=model_compact.name,
                            version=getattr(model_compact, 'version', None),
                            error=error_msg
                        )
                        project_info.models[str(model_compact.id)] = model_info

                project_info.model_count = len(project_info.models)

            except Exception as e:
                logger.error(f'Failed to list models for project {project.name}: {e}')

            # Add project to hierarchy
            hierarchy.projects[str(project.id)] = project_info
            hierarchy.total_models += project_info.model_count
            hierarchy.total_features += project_info.feature_count

        hierarchy.total_projects = len(hierarchy.projects)

        logger.info(f'Environment traversal complete: {hierarchy.total_projects} projects, '
                   f'{hierarchy.total_models} models, {hierarchy.total_features} features')

        return hierarchy

    def list_all_models(
        self,
        project_ids: Optional[List[str]] = None,
        include_versions: bool = True,
        fetch_full: bool = False
    ) -> List[fdl.Model]:
        """List all models across all (or specified) projects.

        Args:
            project_ids: Filter to specific projects
            include_versions: Include all model versions
            fetch_full: Fetch full Model (not ModelCompact) - slower

        Returns:
            Flat list of all models

        Example:
            ```python
            # List all models (compact)
            models = mgr.list_all_models()

            # List full models for specific projects
            models = mgr.list_all_models(
                project_ids=['proj1', 'proj2'],
                fetch_full=True
            )
            ```
        """
        projects = list(fdl.Project.list())

        if project_ids:
            projects = [p for p in projects if str(p.id) in project_ids]

        all_models = []
        for project in projects:
            try:
                models = list(fdl.Model.list(project_id=project.id))

                if fetch_full:
                    # Fetch full model details
                    models = [fdl.Model.get(id_=m.id) for m in models]

                all_models.extend(models)

            except Exception as e:
                logger.warning(f'Failed to list models for project {project.name}: {e}')

        logger.info(f'Listed {len(all_models)} models across {len(projects)} projects')
        return all_models

    def get_environment_statistics(
        self,
        hierarchy: Optional[EnvironmentHierarchy] = None
    ) -> EnvironmentStats:
        """Calculate comprehensive environment statistics.

        Functionality from env_stats notebook - computes aggregations,
        distributions, and summary metrics.

        Args:
            hierarchy: Pre-computed hierarchy (or will fetch if None)

        Returns:
            EnvironmentStats dataclass with summary metrics

        Example:
            ```python
            stats = mgr.get_environment_statistics()

            print(f"Total models: {stats.total_models}")
            print(f"Avg features per model: {stats.features_per_model_mean:.1f}")
            print(f"Top projects: {stats.top_projects_by_models}")
            ```
        """
        if hierarchy is None:
            hierarchy = self.get_environment_hierarchy()

        stats = EnvironmentStats(
            total_projects=hierarchy.total_projects,
            total_models=hierarchy.total_models,
            total_features=hierarchy.total_features
        )

        # Calculate models per project statistics
        if hierarchy.projects:
            model_counts = [p.model_count for p in hierarchy.projects.values()]
            if model_counts:
                stats.models_per_project_mean = sum(model_counts) / len(model_counts)
                sorted_counts = sorted(model_counts)
                mid = len(sorted_counts) // 2
                stats.models_per_project_median = sorted_counts[mid] if len(sorted_counts) % 2 == 1 else (sorted_counts[mid-1] + sorted_counts[mid]) / 2
                stats.models_per_project_min = min(model_counts)
                stats.models_per_project_max = max(model_counts)

            # Top projects by model count
            top_projects = sorted(
                [(p.name, p.model_count) for p in hierarchy.projects.values()],
                key=lambda x: x[1],
                reverse=True
            )[:15]
            stats.top_projects_by_models = top_projects

        # Calculate features per model statistics
        all_models = []
        for project in hierarchy.projects.values():
            for model in project.models.values():
                if model.error is None:  # Only include successfully fetched models
                    all_models.append(model)

        if all_models:
            feature_counts = [m.feature_count for m in all_models]
            if feature_counts:
                stats.features_per_model_mean = sum(feature_counts) / len(feature_counts)
                sorted_counts = sorted(feature_counts)
                mid = len(sorted_counts) // 2
                stats.features_per_model_median = sorted_counts[mid] if len(sorted_counts) % 2 == 1 else (sorted_counts[mid-1] + sorted_counts[mid]) / 2
                stats.features_per_model_min = min(feature_counts)
                stats.features_per_model_max = max(feature_counts)

            # Top models by feature count
            top_models = []
            for project in hierarchy.projects.values():
                for model in project.models.values():
                    if model.error is None:
                        top_models.append((project.name, model.name, model.feature_count))

            top_models = sorted(top_models, key=lambda x: x[2], reverse=True)[:15]
            stats.top_models_by_features = top_models

        logger.info(f'Calculated statistics: {stats.total_projects} projects, '
                   f'{stats.total_models} models, {stats.total_features} features')

        return stats

    def get_timestamp_analysis(
        self,
        hierarchy: Optional[EnvironmentHierarchy] = None
    ) -> TimestampAnalysis:
        """Analyze model creation/update patterns over time.

        Args:
            hierarchy: Pre-computed hierarchy (or will fetch if None)

        Returns:
            TimestampAnalysis with date ranges, age distributions

        Example:
            ```python
            analysis = mgr.get_timestamp_analysis()

            print(f"Coverage: {analysis.timestamp_coverage_pct:.1f}%")
            print(f"Newest models: {analysis.newest_models}")
            ```
        """
        if hierarchy is None:
            hierarchy = self.get_environment_hierarchy(include_timestamps=True)

        analysis = TimestampAnalysis()

        # Collect models with timestamps
        models_with_ts = []
        total_models = 0

        for project in hierarchy.projects.values():
            for model in project.models.values():
                total_models += 1
                if model.created_at is not None:
                    models_with_ts.append(model)

        analysis.models_with_timestamps = len(models_with_ts)
        if total_models > 0:
            analysis.timestamp_coverage_pct = (len(models_with_ts) / total_models) * 100

        if models_with_ts:
            # Find earliest and latest
            created_dates = [m.created_at for m in models_with_ts if m.created_at]
            if created_dates:
                analysis.earliest_created = min(created_dates)
                analysis.latest_created = max(created_dates)

            # Find most recent update
            updated_dates = [m.updated_at for m in models_with_ts if m.updated_at]
            if updated_dates:
                analysis.most_recent_update = max(updated_dates)

            # Calculate average days between create and update
            deltas = []
            for m in models_with_ts:
                if m.created_at and m.updated_at:
                    delta = (m.updated_at - m.created_at).total_seconds() / 86400  # days
                    if delta >= 0:
                        deltas.append(delta)

            if deltas:
                analysis.avg_days_between_create_update = sum(deltas) / len(deltas)

            # Newest and oldest models (by creation date)
            sorted_by_created = sorted(models_with_ts, key=lambda m: m.created_at or datetime.min, reverse=True)
            analysis.newest_models = sorted_by_created[:10]
            analysis.oldest_models = sorted_by_created[-10:][::-1]

            # Most recently updated models (by update date)
            sorted_by_updated = sorted(models_with_ts, key=lambda m: m.updated_at or datetime.min, reverse=True)
            analysis.most_recently_updated_models = sorted_by_updated[:10]

        logger.info(f'Timestamp analysis: {analysis.models_with_timestamps}/{total_models} '
                   f'models ({analysis.timestamp_coverage_pct:.1f}%) have timestamps')

        return analysis

    def export_environment_to_dataframe(
        self,
        level: str = 'model',
        hierarchy: Optional[EnvironmentHierarchy] = None
    ) -> pd.DataFrame:
        """Export environment data as DataFrame.

        Args:
            level: Granularity level - 'project', 'model', or 'feature'
            hierarchy: Pre-computed hierarchy

        Returns:
            Pandas DataFrame ready for CSV export or analysis

        Example:
            ```python
            # Model-level export
            df = mgr.export_environment_to_dataframe(level='model')

            # Feature-level export
            df = mgr.export_environment_to_dataframe(level='feature')
            ```
        """
        if hierarchy is None:
            hierarchy = self.get_environment_hierarchy(
                include_features=(level == 'feature'),
                include_timestamps=True
            )

        if level == 'project':
            # Project-level DataFrame
            rows = []
            for project in hierarchy.projects.values():
                rows.append({
                    'project_id': project.id,
                    'project_name': project.name,
                    'model_count': project.model_count,
                    'feature_count': project.feature_count,
                })
            return pd.DataFrame(rows)

        elif level == 'model':
            # Model-level DataFrame (matches original env_stats notebook format)
            rows = []
            for project in hierarchy.projects.values():
                for model in project.models.values():
                    rows.append({
                        'project': project.name,
                        'model': model.name,
                        'version_name': model.version or model.name,
                        'created_at': model.created_at,
                        'updated_at': model.updated_at,
                        'feature_count': model.feature_count,
                    })
            return pd.DataFrame(rows)

        elif level == 'feature':
            # Feature-level DataFrame (one row per feature, matches original env_stats notebook format)
            rows = []
            for project in hierarchy.projects.values():
                for model in project.models.values():
                    for feature in model.features:
                        rows.append({
                            'project': project.name,
                            'model': model.name,
                            'version_name': model.version or model.name,
                            'feature': feature,
                            'created_at': model.created_at,
                            'updated_at': model.updated_at,
                        })
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Invalid level '{level}'. Must be 'project', 'model', or 'feature'")

    def export_environment_to_csv(
        self,
        output_dir: str = '.',
        prefix: str = 'env_stats'
    ) -> List[str]:
        """Export complete environment analysis to CSV files.

        Creates multiple CSVs:
        - {prefix}__overview.csv (model-level)
        - {prefix}__flattened_hierarchy.csv (feature-level)

        Args:
            output_dir: Output directory path
            prefix: Filename prefix

        Returns:
            List of created file paths

        Example:
            ```python
            files = mgr.export_environment_to_csv(
                output_dir='exports',
                prefix='fiddler_env'
            )
            print(f"Created: {files}")
            ```
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Get hierarchy once
        hierarchy = self.get_environment_hierarchy(
            include_features=True,
            include_timestamps=True
        )

        created_files = []

        # Model-level export (matches original env_stats__overview.csv)
        model_df = self.export_environment_to_dataframe(level='model', hierarchy=hierarchy)
        model_file = os.path.join(output_dir, f'{prefix}__overview.csv')
        model_df.to_csv(model_file, index=False)
        created_files.append(model_file)
        logger.info(f'Exported model-level data to {model_file}')

        # Feature-level export (matches original env_stats__flattened_hierarchy.csv)
        feature_df = self.export_environment_to_dataframe(level='feature', hierarchy=hierarchy)
        feature_file = os.path.join(output_dir, f'{prefix}__flattened_hierarchy.csv')
        feature_df.to_csv(feature_file, index=False)
        created_files.append(feature_file)
        logger.info(f'Exported feature-level data to {feature_file}')

        return created_files

    def display_environment_summary(
        self,
        hierarchy: Optional[EnvironmentHierarchy] = None,
        show_top_n: int = 10
    ):
        """Print formatted environment summary to console.

        Args:
            hierarchy: Pre-computed hierarchy
            show_top_n: Show top N projects/models in rankings

        Example:
            ```python
            mgr.display_environment_summary(show_top_n=15)
            ```
        """
        if hierarchy is None:
            hierarchy = self.get_environment_hierarchy(
                include_features=True,
                include_timestamps=True
            )

        stats = self.get_environment_statistics(hierarchy)

        print("\n" + "="*60)
        print("ENVIRONMENT SUMMARY")
        print("="*60)

        print(f"\nTotal Projects: {stats.total_projects}")
        print(f"Total Models: {stats.total_models}")
        print(f"Total Features: {stats.total_features}")

        print(f"\nModels per Project:")
        print(f"  Mean: {stats.models_per_project_mean:.1f}")
        print(f"  Median: {stats.models_per_project_median:.1f}")
        print(f"  Range: {stats.models_per_project_min} - {stats.models_per_project_max}")

        print(f"\nFeatures per Model:")
        print(f"  Mean: {stats.features_per_model_mean:.1f}")
        print(f"  Median: {stats.features_per_model_median:.1f}")
        print(f"  Range: {stats.features_per_model_min} - {stats.features_per_model_max}")

        if stats.top_projects_by_models:
            print(f"\nTop {show_top_n} Projects by Model Count:")
            for i, (name, count) in enumerate(stats.top_projects_by_models[:show_top_n], 1):
                print(f"  {i:2d}. {name:40s} {count:4d} models")

        if stats.top_models_by_features:
            print(f"\nTop {show_top_n} Models by Feature Count:")
            for i, (project, model, count) in enumerate(stats.top_models_by_features[:show_top_n], 1):
                print(f"  {i:2d}. {project}/{model:30s} {count:4d} features")

        print("\n" + "="*60 + "\n")
