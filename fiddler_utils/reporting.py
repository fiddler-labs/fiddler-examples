"""Environment reporting utilities for Fiddler.

This module provides EnvironmentReporter, a high-level facade for environment
analysis and reporting that simplifies the env_stats notebook workflow.
"""

from typing import Optional, List
import logging

from .projects import (
    ProjectManager,
    EnvironmentHierarchy,
    EnvironmentStats,
    TimestampAnalysis,
)

logger = logging.getLogger(__name__)


class EnvironmentReporter:
    """High-level facade for environment analysis and reporting.

    Combines ProjectManager discovery with formatted output, export, and
    analysis - essentially the env_stats notebook as a reusable class.

    This class provides a simple interface for common environment analysis tasks:
    1. analyze_environment() - collect data
    2. generate_report() - display formatted output
    3. export_to_csv() - export results

    Example:
        ```python
        from fiddler_utils import get_or_init, EnvironmentReporter

        get_or_init(url=URL, token=TOKEN, log_level='ERROR')

        # Run complete analysis
        reporter = EnvironmentReporter()
        reporter.analyze_environment(
            include_features=True,
            include_timestamps=True
        )

        # Display formatted report
        reporter.generate_report(top_n=15)

        # Export to CSV
        files = reporter.export_to_csv(prefix='env_stats')
        print(f"Exported: {files}")
        ```
    """

    def __init__(self, project_manager: Optional[ProjectManager] = None):
        """Initialize EnvironmentReporter.

        Args:
            project_manager: Optional ProjectManager instance (creates new if None)
        """
        self.project_mgr = project_manager or ProjectManager()
        self._hierarchy: Optional[EnvironmentHierarchy] = None
        self._stats: Optional[EnvironmentStats] = None
        self._timestamp_analysis: Optional[TimestampAnalysis] = None

    def analyze_environment(
        self,
        include_features: bool = True,
        include_timestamps: bool = True,
        include_assets: bool = False
    ):
        """Run complete environment analysis (data collection phase).

        This is the core data collection step that traverses all projects,
        models, and features in the environment.

        Args:
            include_features: Extract feature lists from model specs
            include_timestamps: Include created_at/updated_at timestamps
            include_assets: Include counts of segments, metrics, alerts (slower)

        Example:
            ```python
            reporter = EnvironmentReporter()
            reporter.analyze_environment(
                include_features=True,
                include_timestamps=True
            )
            ```
        """
        logger.info('Starting environment analysis')

        # Collect hierarchy
        self._hierarchy = self.project_mgr.get_environment_hierarchy(
            include_features=include_features,
            include_timestamps=include_timestamps,
            include_assets=include_assets
        )

        # Calculate statistics
        self._stats = self.project_mgr.get_environment_statistics(
            hierarchy=self._hierarchy
        )

        # Analyze timestamps if included
        if include_timestamps:
            self._timestamp_analysis = self.project_mgr.get_timestamp_analysis(
                hierarchy=self._hierarchy
            )

        logger.info(f'Environment analysis complete: {self._stats.total_projects} projects, '
                   f'{self._stats.total_models} models, {self._stats.total_features} features')

    def generate_report(
        self,
        show_projects: bool = True,
        show_models: bool = True,
        show_timestamps: bool = True,
        show_newest_oldest: bool = True,
        top_n: int = 15
    ):
        """Generate and print formatted report to console.

        Displays comprehensive environment analysis including:
        - Overall statistics (projects, models, features)
        - Distribution metrics (mean, median, min, max)
        - Top projects by model count
        - Top models by feature count
        - Timestamp analysis (if available)
        - Newest and oldest models (if available)

        Args:
            show_projects: Show project breakdown
            show_models: Show model analysis
            show_timestamps: Show timestamp analysis
            show_newest_oldest: Show newest/oldest models
            top_n: Number of items to show in top lists

        Raises:
            RuntimeError: If analyze_environment() has not been called yet

        Example:
            ```python
            reporter.generate_report(
                show_projects=True,
                show_models=True,
                show_timestamps=True,
                top_n=10
            )
            ```
        """
        if self._hierarchy is None or self._stats is None:
            raise RuntimeError(
                'analyze_environment() must be called before generate_report(). '
                'Call reporter.analyze_environment() first.'
            )

        # Overall summary
        print("\n" + "="*70)
        print("FIDDLER ENVIRONMENT ANALYSIS")
        print("="*70)
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Projects: {self._stats.total_projects}")
        print(f"   Total Models: {self._stats.total_models}")
        print(f"   Total Features: {self._stats.total_features}")

        # Models per project
        if show_projects and self._stats.total_projects > 0:
            print(f"\n📁 Models per Project:")
            print(f"   Mean: {self._stats.models_per_project_mean:.1f}")
            print(f"   Median: {self._stats.models_per_project_median:.1f}")
            print(f"   Range: {self._stats.models_per_project_min} - {self._stats.models_per_project_max}")

            if self._stats.top_projects_by_models:
                print(f"\n🏆 Top {top_n} Projects by Model Count:")
                for i, (name, count) in enumerate(self._stats.top_projects_by_models[:top_n], 1):
                    bar = "█" * min(count, 50)
                    print(f"   {i:2d}. {name:35s} {count:4d} │{bar}")

        # Features per model
        if show_models and self._stats.total_models > 0:
            print(f"\n🔧 Features per Model:")
            print(f"   Mean: {self._stats.features_per_model_mean:.1f}")
            print(f"   Median: {self._stats.features_per_model_median:.1f}")
            print(f"   Range: {self._stats.features_per_model_min} - {self._stats.features_per_model_max}")

            if self._stats.top_models_by_features:
                print(f"\n🏆 Top {top_n} Models by Feature Count:")
                for i, (proj, model, count) in enumerate(self._stats.top_models_by_features[:top_n], 1):
                    model_display = f"{proj}/{model}"
                    bar = "█" * min(int(count/5), 50)
                    print(f"   {i:2d}. {model_display:45s} {count:4d} │{bar}")

        # Timestamp analysis
        if show_timestamps and self._timestamp_analysis is not None:
            ts = self._timestamp_analysis
            print(f"\n📅 Timestamp Analysis:")
            print(f"   Models with timestamps: {ts.models_with_timestamps} "
                  f"({ts.timestamp_coverage_pct:.1f}% coverage)")

            if ts.earliest_created and ts.latest_created:
                print(f"   Earliest created: {ts.earliest_created.strftime('%Y-%m-%d')}")
                print(f"   Latest created: {ts.latest_created.strftime('%Y-%m-%d')}")

            if ts.most_recent_update:
                print(f"   Most recent update: {ts.most_recent_update.strftime('%Y-%m-%d')}")

            if ts.avg_days_between_create_update is not None:
                print(f"   Avg days between create/update: {ts.avg_days_between_create_update:.1f}")

            # Newest models
            if show_newest_oldest and ts.newest_models:
                print(f"\n🆕 Newest Models (Top {min(10, len(ts.newest_models))}):")
                for i, model in enumerate(ts.newest_models[:10], 1):
                    created = model.created_at.strftime('%Y-%m-%d') if model.created_at else 'N/A'
                    print(f"   {i:2d}. {model.name:40s} {created}")

            # Oldest models
            if show_newest_oldest and ts.oldest_models:
                print(f"\n👴 Oldest Models (Top {min(10, len(ts.oldest_models))}):")
                for i, model in enumerate(ts.oldest_models[:10], 1):
                    created = model.created_at.strftime('%Y-%m-%d') if model.created_at else 'N/A'
                    print(f"   {i:2d}. {model.name:40s} {created}")

            # Most recently updated models
            if show_newest_oldest and ts.most_recently_updated_models:
                print(f"\n🔄 Most Recently Updated Models (Top {min(5, len(ts.most_recently_updated_models))}):")
                for i, model in enumerate(ts.most_recently_updated_models[:5], 1):
                    updated = model.updated_at.strftime('%Y-%m-%d') if model.updated_at else 'N/A'
                    print(f"   {i:2d}. {model.name:40s} {updated}")

        print("\n" + "="*70)
        print(f"✓ Analysis complete")
        print("="*70 + "\n")

    def export_to_csv(
        self,
        output_dir: str = '.',
        prefix: str = 'env_stats'
    ) -> List[str]:
        """Export analysis results to CSV files.

        Creates multiple CSV files:
        - {prefix}__overview.csv (model-level data)
        - {prefix}__flattened_hierarchy.csv (feature-level data)

        Args:
            output_dir: Output directory path
            prefix: Filename prefix

        Returns:
            List of created file paths

        Raises:
            RuntimeError: If analyze_environment() has not been called yet

        Example:
            ```python
            files = reporter.export_to_csv(
                output_dir='exports',
                prefix='fiddler_env'
            )
            print(f"Created {len(files)} files: {files}")
            ```
        """
        if self._hierarchy is None:
            raise RuntimeError(
                'analyze_environment() must be called before export_to_csv(). '
                'Call reporter.analyze_environment() first.'
            )

        files = self.project_mgr.export_environment_to_csv(
            output_dir=output_dir,
            prefix=prefix
        )

        logger.info(f'Exported {len(files)} CSV files')
        return files

    def get_hierarchy(self) -> EnvironmentHierarchy:
        """Get raw hierarchy data for custom analysis.

        Returns:
            EnvironmentHierarchy dataclass

        Raises:
            RuntimeError: If analyze_environment() has not been called yet

        Example:
            ```python
            hierarchy = reporter.get_hierarchy()
            for project_id, project in hierarchy.projects.items():
                print(f"{project.name}: {project.model_count} models")
            ```
        """
        if self._hierarchy is None:
            raise RuntimeError(
                'analyze_environment() must be called first. '
                'Call reporter.analyze_environment() to collect data.'
            )
        return self._hierarchy

    def get_statistics(self) -> EnvironmentStats:
        """Get statistics object for custom analysis.

        Returns:
            EnvironmentStats dataclass

        Raises:
            RuntimeError: If analyze_environment() has not been called yet

        Example:
            ```python
            stats = reporter.get_statistics()
            print(f"Average features per model: {stats.features_per_model_mean}")
            ```
        """
        if self._stats is None:
            raise RuntimeError(
                'analyze_environment() must be called first. '
                'Call reporter.analyze_environment() to collect data.'
            )
        return self._stats

    def get_timestamp_analysis(self) -> Optional[TimestampAnalysis]:
        """Get timestamp analysis object for custom analysis.

        Returns:
            TimestampAnalysis dataclass, or None if timestamps not collected

        Raises:
            RuntimeError: If analyze_environment() has not been called yet

        Example:
            ```python
            ts_analysis = reporter.get_timestamp_analysis()
            if ts_analysis:
                print(f"Coverage: {ts_analysis.timestamp_coverage_pct}%")
            ```
        """
        if self._hierarchy is None:
            raise RuntimeError(
                'analyze_environment() must be called first. '
                'Call reporter.analyze_environment() to collect data.'
            )
        return self._timestamp_analysis
