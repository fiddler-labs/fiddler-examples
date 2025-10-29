"""Dashboard management utilities.

This module provides the DashboardManager class for working with Fiddler dashboards.

Note: Dashboards use an unofficial Fiddler API through RequestClient and lack
the same backwards compatibility guarantees as the official SDK.
"""

from typing import List, Dict, Set, Any, Optional
from uuid import uuid4
import logging
import json

try:
    import fiddler as fdl
    from fiddler.libs.http_client import RequestClient
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .base import BaseAssetManager, AssetType
from ..exceptions import AssetNotFoundError, AssetImportError

logger = logging.getLogger(__name__)


class DashboardManager(BaseAssetManager[Dict]):
    """Manager for Fiddler dashboards.

    Dashboards are collections of charts arranged in a grid layout.
    This manager uses the unofficial Fiddler API through RequestClient.

    IMPORTANT: The dashboard API is not officially supported and may change
    without notice. Use with caution in production environments.

    Example:
        ```python
        from fiddler_utils.assets import DashboardManager
        import fiddler as fdl

        mgr = DashboardManager(url=FIDDLER_URL, token=API_TOKEN)

        # List all dashboards in a project
        dashboards = mgr.list_dashboards(project_id=project.id)
        print(f"Found {len(dashboards)} dashboards")

        # Export dashboard for backup
        exported = mgr.export_dashboard(
            project_id=project.id,
            dashboard_id=dashboard['uuid']
        )

        # Create dashboard from chart list
        mgr.create_dashboard(
            project_id=project.id,
            model_id=model.id,
            title='My Dashboard',
            chart_ids=['chart-id-1', 'chart-id-2']
        )
        ```
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        """Initialize DashboardManager.

        Args:
            url: Fiddler instance URL (e.g., 'https://demo.fiddler.ai')
            token: API token for authentication

        Note: If url/token not provided, they will be read from current
        fiddler client connection when needed.
        """
        super().__init__()
        self._url = url
        self._token = token
        self._client: Optional[RequestClient] = None

    def _get_client(self) -> RequestClient:
        """Get or create RequestClient for API calls."""
        if self._client is None:
            url = self._url
            token = self._token

            if not url or not token:
                raise ValueError(
                    'DashboardManager requires url and token to be passed explicitly. '
                    "Example: DashboardManager(url='https://demo.fiddler.ai', token='abc123')"
                )

            self._client = RequestClient(
                base_url=url,
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                },
            )
            logger.info(f'Initialized RequestClient for {url}')

        return self._client

    def _get_asset_type(self) -> AssetType:
        """Get asset type."""
        return AssetType.DASHBOARD

    def _list_assets(self, model_id: str) -> List[Dict]:
        """List all dashboards for a model.

        Note: This method is not used directly. Use list_dashboards() instead.
        """
        raise NotImplementedError(
            'Use list_dashboards(project_id) instead. '
            'Dashboards are project-level, not model-level.'
        )

    def _get_asset_name(self, asset: Dict) -> str:
        """Get dashboard title."""
        return asset.get('title', 'Untitled Dashboard')

    def _extract_referenced_columns(self, asset: Dict) -> Set[str]:
        """Extract column references from dashboard.

        Dashboards don't directly reference columns.
        """
        return set()

    def _extract_asset_data(self, asset: Dict) -> Dict[str, Any]:
        """Extract dashboard data for export.

        Args:
            asset: Dashboard dictionary from API

        Returns:
            Dictionary with dashboard data
        """
        return {
            'title': asset.get('title'),
            'layouts': asset.get('layouts', []),
            'options': asset.get('options', {}),
            'metadata': {
                'uuid': asset.get('uuid'),
                'project_name': asset.get('project_name'),
                'model_name': asset.get('model_name'),
                'created_at': asset.get('created_at'),
                'updated_at': asset.get('updated_at'),
            },
        }

    def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> Dict:
        """Create a dashboard from data.

        Note: This is a complex operation. Use create_dashboard() instead.
        """
        raise NotImplementedError(
            'Use create_dashboard() method instead. '
            'Dashboard creation requires project and chart resolution.'
        )

    def list_dashboards(
        self, project_id: str, model_id: Optional[str] = None
    ) -> List[Dict]:
        """List all dashboards in a project.

        Args:
            project_id: Project ID
            model_id: Optional model ID to filter by

        Returns:
            List of dashboard dictionaries

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            dashboards = mgr.list_dashboards(project_id=project.id)

            for dashboard in dashboards:
                print(f"Dashboard: {dashboard['title']}")
                print(f"  Charts: {len(dashboard.get('layouts', []))}")
            ```
        """
        client = self._get_client()

        # Get project name for API call
        project = fdl.Project.get(id_=project_id)
        project_name = project.name

        # List dashboards endpoint
        # Note: v2 API for dashboards
        dashboards_url = f'/v2/dashboards?project_name={project_name}'

        try:
            response = client.get(url=dashboards_url)
            data = response if isinstance(response, dict) else response.json()
            dashboards = data.get('data', [])

            if isinstance(dashboards, dict):
                dashboards = [dashboards]

            logger.info(
                f'Retrieved {len(dashboards)} dashboards from project {project_name}'
            )

            # Filter by model if specified
            if model_id:
                model = fdl.Model.get(id_=model_id)
                model_name = model.name
                dashboards = [
                    d for d in dashboards if d.get('model_name') == model_name
                ]
                logger.info(
                    f'Filtered to {len(dashboards)} dashboards for model {model_name}'
                )

            return dashboards

        except Exception as e:
            logger.error(f'Failed to list dashboards: {e}')
            return []

    def get_dashboard_by_title(self, project_id: str, title: str) -> Dict:
        """Get a specific dashboard by title.

        Args:
            project_id: Project ID
            title: Dashboard title

        Returns:
            Dashboard dictionary

        Raises:
            AssetNotFoundError: If dashboard not found

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            dashboard = mgr.get_dashboard_by_title(project.id, 'Performance Dashboard')
            print(f"Dashboard UUID: {dashboard['uuid']}")
            ```
        """
        dashboards = self.list_dashboards(project_id)

        for dashboard in dashboards:
            if dashboard.get('title') == title:
                return dashboard

        raise AssetNotFoundError(
            f"Dashboard '{title}' not found in project {project_id}",
            asset_type='Dashboard',
            asset_id=title,
        )

    def export_dashboard(self, project_id: str, dashboard_id: str) -> Dict:
        """Export a dashboard by UUID.

        Args:
            project_id: Project ID
            dashboard_id: Dashboard UUID

        Returns:
            Dashboard dictionary

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            exported = mgr.export_dashboard(project.id, dashboard['uuid'])
            ```
        """
        dashboards = self.list_dashboards(project_id)

        for dashboard in dashboards:
            if dashboard.get('uuid') == dashboard_id:
                logger.info(f'Exported dashboard: {dashboard.get("title")}')
                return dashboard

        raise AssetNotFoundError(
            f"Dashboard with UUID '{dashboard_id}' not found",
            asset_type='Dashboard',
            asset_id=dashboard_id,
        )

    def create_dashboard(
        self,
        project_id: str,
        model_id: str,
        title: str,
        chart_ids: List[str],
        layout: Optional[List[Dict]] = None,
        set_as_default: bool = False,
    ) -> Dict:
        """Create a new dashboard from a list of charts.

        Args:
            project_id: Target project ID
            model_id: Target model ID
            title: Dashboard title
            chart_ids: List of chart UUIDs to include
            layout: Optional custom layout. If None, creates grid automatically.
            set_as_default: If True, set as default dashboard for model

        Returns:
            Created dashboard dictionary

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)

            # Create dashboard with automatic layout
            dashboard = mgr.create_dashboard(
                project_id=project.id,
                model_id=model.id,
                title='My Dashboard',
                chart_ids=['chart-uuid-1', 'chart-uuid-2'],
                set_as_default=True
            )

            # Create with custom layout
            custom_layout = [
                {
                    'chart_uuid': 'chart-uuid-1',
                    'grid_props': {
                        'position_x': 0,
                        'position_y': 0,
                        'width': 2,
                        'height': 1
                    }
                },
                # ... more layouts
            ]
            dashboard = mgr.create_dashboard(
                project_id=project.id,
                model_id=model.id,
                title='Custom Layout Dashboard',
                chart_ids=['chart-uuid-1'],
                layout=custom_layout
            )
            ```
        """
        client = self._get_client()

        # Get project and model names
        project = fdl.Project.get(id_=project_id)
        model = fdl.Model.get(id_=model_id)

        # Build layout if not provided
        if layout is None:
            layout = self._generate_grid_layout(chart_ids)
        else:
            # Validate that layout matches chart_ids
            layout_chart_ids = {l.get('chart_uuid') for l in layout}
            if not layout_chart_ids.issuperset(set(chart_ids)):
                logger.warning(
                    "Provided layout doesn't include all chart_ids. "
                    'Some charts may be missing from dashboard.'
                )

        # Build dashboard payload
        dashboard_data = {
            'title': title,
            'project_name': project.name,
            'model_name': model.name,
            'organization_name': 'default',  # This may need to be dynamic
            'layouts': layout,
            'options': {'filters': {'time_label': '7d', 'time_zone': 'UTC'}},
        }

        # Create dashboard (v2 API)
        dashboards_url = '/v2/dashboards'

        try:
            response = client.post(url=dashboards_url, data=dashboard_data)
            result = response if isinstance(response, dict) else response.json()
            created_dashboard = result.get('data', {})

            logger.info(
                f'Created dashboard: {title} (UUID: {created_dashboard.get("uuid")})'
            )

            # Set as default if requested
            if set_as_default:
                self.set_default_dashboard(model_id, created_dashboard.get('uuid'))

            return created_dashboard

        except Exception as e:
            logger.error(f"Failed to create dashboard '{title}': {e}")
            raise AssetImportError(
                f'Failed to create dashboard', asset_name=title, reason=str(e)
            )

    def _generate_grid_layout(
        self, chart_ids: List[str], columns: int = 2
    ) -> List[Dict]:
        """Generate a grid layout for charts.

        Args:
            chart_ids: List of chart UUIDs
            columns: Number of columns in grid (default: 2)

        Returns:
            List of layout dictionaries
        """
        layout = []

        for i, chart_id in enumerate(chart_ids):
            row = i // columns
            col = i % columns

            layout.append(
                {
                    'chart_uuid': chart_id,
                    'grid_props': {
                        'position_x': col,
                        'position_y': row,
                        'width': 1,
                        'height': 1,
                    },
                }
            )

        logger.debug(
            f'Generated grid layout for {len(chart_ids)} charts in {columns} columns'
        )
        return layout

    def import_dashboard(
        self,
        target_project_id: str,
        target_model_id: str,
        dashboard_data: Dict,
        chart_title_mapping: Optional[Dict[str, str]] = None,
        set_as_default: bool = False,
    ) -> Dict:
        """Import a dashboard to a target project/model.

        This resolves chart titles to UUIDs in the target environment.

        Args:
            target_project_id: Target project ID
            target_model_id: Target model ID
            dashboard_data: Dashboard dictionary to import
            chart_title_mapping: Optional mapping of old chart titles to new titles
            set_as_default: If True, set as default dashboard for model

        Returns:
            Created dashboard dictionary

        Example:
            ```python
            mgr = DashboardManager(url=TARGET_URL, token=TARGET_TOKEN)

            # Import with automatic chart resolution
            result = mgr.import_dashboard(
                target_project_id=target_project.id,
                target_model_id=target_model.id,
                dashboard_data=exported_dashboard
            )

            # Import with chart title mapping
            mapping = {'Old Chart Name': 'New Chart Name'}
            result = mgr.import_dashboard(
                target_project_id=target_project.id,
                target_model_id=target_model.id,
                dashboard_data=exported_dashboard,
                chart_title_mapping=mapping
            )
            ```
        """
        from .charts import ChartManager

        # Get charts in target project
        chart_mgr = ChartManager(url=self._url, token=self._token)
        target_charts = chart_mgr.list_charts(project_id=target_project_id)

        # Build chart title to UUID mapping
        chart_lookup = {c['title']: c['id'] for c in target_charts}

        # Resolve chart UUIDs in layouts
        new_layouts = []
        missing_charts = []

        for layout in dashboard_data.get('layouts', []):
            # Get chart title (may be stored in different fields depending on export)
            chart_title = layout.get('chart_title')
            if not chart_title:
                # Try to look up by existing UUID
                chart_uuid = layout.get('chart_uuid')
                # Find title from source charts if possible
                # For now, skip if we can't determine title
                logger.warning(f'Could not determine chart title for layout: {layout}')
                continue

            # Apply title mapping if provided
            if chart_title_mapping and chart_title in chart_title_mapping:
                chart_title = chart_title_mapping[chart_title]

            # Look up UUID in target
            if chart_title in chart_lookup:
                new_layout = {
                    'chart_uuid': chart_lookup[chart_title],
                    'grid_props': layout.get(
                        'grid_props',
                        {'position_x': 0, 'position_y': 0, 'width': 1, 'height': 1},
                    ),
                }
                new_layouts.append(new_layout)
            else:
                missing_charts.append(chart_title)
                logger.warning(f"Chart '{chart_title}' not found in target project")

        if missing_charts:
            logger.warning(
                f'Dashboard import incomplete: {len(missing_charts)} charts missing: '
                f'{missing_charts[:5]}'
            )

        if not new_layouts:
            raise AssetImportError(
                'Cannot import dashboard',
                asset_name=dashboard_data.get('title'),
                reason='No charts could be resolved in target project',
            )

        # Extract chart IDs for create_dashboard
        chart_ids = [l['chart_uuid'] for l in new_layouts]

        # Create dashboard in target
        return self.create_dashboard(
            project_id=target_project_id,
            model_id=target_model_id,
            title=dashboard_data.get('title', 'Imported Dashboard'),
            chart_ids=chart_ids,
            layout=new_layouts,
            set_as_default=set_as_default,
        )

    def set_default_dashboard(self, model_id: str, dashboard_uuid: str) -> bool:
        """Set a dashboard as the default for a model.

        Args:
            model_id: Model ID
            dashboard_uuid: Dashboard UUID to set as default

        Returns:
            True if successful

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            mgr.set_default_dashboard(model.id, dashboard['uuid'])
            ```
        """
        client = self._get_client()

        # Set default dashboard endpoint (v3 API)
        default_url = f'/v3/models/{model_id}/default-dashboard'

        payload = {'dashboard_uuid': dashboard_uuid}

        try:
            client.put(url=default_url, data=payload)
            logger.info(
                f'Set dashboard {dashboard_uuid} as default for model {model_id}'
            )
            return True
        except Exception as e:
            logger.error(f'Failed to set default dashboard: {e}')
            raise

    def delete_dashboard(self, dashboard_uuid: str) -> bool:
        """Delete a dashboard.

        Args:
            dashboard_uuid: Dashboard UUID to delete

        Returns:
            True if successful

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            dashboard = mgr.get_dashboard_by_title(project.id, 'Old Dashboard')
            mgr.delete_dashboard(dashboard['uuid'])
            ```
        """
        client = self._get_client()

        # Delete endpoint (v2 API)
        delete_url = f'/v2/dashboards/{dashboard_uuid}'

        try:
            client.delete(url=delete_url)
            logger.info(f'Deleted dashboard: {dashboard_uuid}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete dashboard {dashboard_uuid}: {e}')
            raise

    def analyze_dashboards(self, project_id: str) -> Dict[str, Any]:
        """Analyze dashboards in a project.

        Args:
            project_id: Project ID

        Returns:
            Dictionary with dashboard statistics

        Example:
            ```python
            mgr = DashboardManager(url=URL, token=TOKEN)
            analysis = mgr.analyze_dashboards(project.id)

            print(f"Total dashboards: {analysis['total']}")
            print(f"By model: {analysis['by_model']}")
            print(f"Avg charts per dashboard: {analysis['avg_charts_per_dashboard']}")
            ```
        """
        dashboards = self.list_dashboards(project_id)

        by_model = {}
        total_charts = 0

        for dashboard in dashboards:
            model_name = dashboard.get('model_name', 'unknown')
            by_model[model_name] = by_model.get(model_name, 0) + 1

            charts_count = len(dashboard.get('layouts', []))
            total_charts += charts_count

        analysis = {
            'total': len(dashboards),
            'by_model': by_model,
            'unique_models': len(by_model),
            'total_charts': total_charts,
            'avg_charts_per_dashboard': total_charts / len(dashboards)
            if dashboards
            else 0,
        }

        logger.info(
            f'Dashboard analysis for project {project_id}: '
            f'{analysis["total"]} total, {analysis["unique_models"]} models'
        )

        return analysis
