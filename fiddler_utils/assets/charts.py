"""Chart management utilities.

This module provides the ChartManager class for working with Fiddler charts.

Note: Charts use an unofficial Fiddler API through RequestClient and lack
the same backwards compatibility guarantees as the official SDK.
"""

from typing import List, Dict, Set, Any, Optional
from uuid import uuid4, UUID
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


def _is_uuid(value: Any) -> bool:
    """Return True if value is a string that parses as a valid UUID."""
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


class ChartManager(BaseAssetManager[Dict]):
    """Manager for Fiddler charts.

    Charts are visual representations of model metrics and analytics.
    This manager uses the unofficial Fiddler API through RequestClient.

    IMPORTANT: The chart API is not officially supported and may change
    without notice. Use with caution in production environments.

    Example:
        ```python
        from fiddler_utils.assets import ChartManager
        import fiddler as fdl

        mgr = ChartManager(url=FIDDLER_URL, token=API_TOKEN)

        # List all charts in a project
        charts = mgr.list_charts(project_id=project.id)
        print(f"Found {len(charts)} charts")

        # Export charts for backup
        exported = mgr.export_charts(
            project_id=project.id,
            model_id=model.id
        )

        # Import charts to another model
        result = mgr.import_charts(
            target_project_id=target_project.id,
            target_model_id=target_model.id,
            charts=exported
        )
        ```
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        """Initialize ChartManager.

        Args:
            url: Fiddler instance URL (e.g., 'https://acme.cloud.fiddler.ai')
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
            # Try to get from stored values or current connection
            url = self._url
            token = self._token

            if not url or not token:
                # Try to get from fiddler client state
                try:
                    # This is a bit hacky but necessary since fiddler client
                    # doesn't expose current connection info
                    logger.warning(
                        'URL/token not provided to ChartManager. '
                        'Attempting to use current fiddler client connection. '
                        'For best results, pass url/token explicitly.'
                    )
                    # We'll need the user to pass these explicitly
                    raise ValueError(
                        'ChartManager requires url and token to be passed explicitly. '
                        "Example: ChartManager(url='https://acme.cloud.fiddler.ai', token='abc123')"
                    )
                except Exception as e:
                    raise ValueError(
                        f'Could not initialize RequestClient: {e}. '
                        'Please pass url and token to ChartManager constructor.'
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
        return AssetType.CHART

    def _list_assets(self, model_id: str) -> List[Dict]:
        """List all charts for a model.

        Note: This method is not used directly. Use list_charts() instead.
        """
        raise NotImplementedError(
            'Use list_charts(project_id) instead. '
            'Charts are project-level, not model-level.'
        )

    def _get_asset_name(self, asset: Dict) -> str:
        """Get chart title."""
        return asset.get('title', 'Untitled Chart')

    def _extract_referenced_columns(self, asset: Dict) -> Set[str]:
        """Extract column references from chart definition.

        Args:
            asset: Chart dictionary

        Returns:
            Set of column names referenced in the chart
        """
        columns = set()

        # Extract columns from queries
        data_source = asset.get('data_source', {})

        # For ANALYTICS charts
        if data_source.get('query_type') == 'ANALYTICS':
            # Analytics charts may have column references in data_source
            pass  # TODO: Add ANALYTICS column extraction if needed

        # For MONITORING charts
        queries = data_source.get('queries', [])
        for query in queries:
            # Columns field contains column names
            if 'columns' in query:
                columns.update(query['columns'])

            # Categories might reference column values
            # But these aren't column names themselves

        return columns

    def _extract_asset_data(self, asset: Dict) -> Dict[str, Any]:
        """Extract chart data for export.

        Args:
            asset: Chart dictionary from API

        Returns:
            Dictionary with chart data
        """
        # Return the full chart object
        # We'll clean it up for import
        return {
            'title': asset.get('title'),
            'query_type': asset.get('query_type'),
            'description': asset.get('description', ''),
            'options': asset.get('options', {}),
            'data_source': asset.get('data_source', {}),
            'metadata': {
                'id': asset.get('id'),
                'project_id': asset.get('project', {}).get('id'),
                'created_at': asset.get('created_at'),
                'updated_at': asset.get('updated_at'),
            },
        }

    def _create_asset(self, model_id: str, asset_data: Dict[str, Any]) -> Dict:
        """Create a chart from data.

        Note: This is a complex operation that requires ID resolution.
        Use import_charts() instead for proper handling.
        """
        raise NotImplementedError(
            'Use import_charts() method instead. '
            'Chart creation requires complex ID resolution.'
        )

    def get_charts_from_dashboard(self, dashboard_id: str) -> List[str]:
        """Get list of chart IDs from a dashboard.

        Args:
            dashboard_id: Dashboard UUID

        Returns:
            List of chart IDs (UUIDs as strings)

        Raises:
            Exception: If dashboard fetch fails

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            chart_ids = mgr.get_charts_from_dashboard(dashboard_id='abc-123')
            print(f"Found {len(chart_ids)} charts in dashboard")

            # Fetch full chart definitions
            charts = [mgr.get_chart_by_id(cid) for cid in chart_ids]
            ```
        """
        client = self._get_client()
        dashboard_url = f'/v2/dashboards/{dashboard_id}'

        try:
            response = client.get(url=dashboard_url)

            # Extract JSON data from response object
            if hasattr(response, 'json'):
                dashboard_response = response.json()
            elif hasattr(response, 'data'):
                dashboard_response = response.data
            else:
                dashboard_response = response

            # The API returns the dashboard nested in a 'data' key
            if isinstance(dashboard_response, dict) and 'data' in dashboard_response:
                dashboard_data = dashboard_response['data']
            else:
                dashboard_data = dashboard_response

            # Extract chart IDs from layouts
            chart_ids = []
            layouts = dashboard_data.get('layouts', [])
            for layout in layouts:
                chart_uuid = layout.get('chart_uuid')
                if chart_uuid:
                    chart_ids.append(chart_uuid)

            logger.info(
                f"Retrieved {len(chart_ids)} chart IDs from dashboard {dashboard_id}"
            )
            return chart_ids

        except Exception as e:
            logger.error(f'Failed to fetch dashboard {dashboard_id}: {e}')
            raise Exception(f'Failed to fetch dashboard {dashboard_id}: {str(e)}')

    def get_chart_by_id(self, chart_id: str) -> Dict:
        """Get a chart by its ID.

        Args:
            chart_id: Chart UUID

        Returns:
            Chart dictionary

        Raises:
            AssetNotFoundError: If chart not found

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            chart = mgr.get_chart_by_id(chart_id='abc-123')
            print(f"Chart: {chart['title']}")
            ```
        """
        client = self._get_client()
        charts_url = f'/v3/charts/{chart_id}'

        try:
            response = client.get(url=charts_url)

            # Extract JSON data from response object
            if hasattr(response, 'json'):
                chart_response = response.json()
            elif hasattr(response, 'data'):
                chart_response = response.data
            else:
                chart_response = response

            # The API returns the chart nested in a 'data' key
            if isinstance(chart_response, dict) and 'data' in chart_response:
                chart = chart_response['data']
            else:
                chart = chart_response

            logger.debug(f"Retrieved chart: {chart.get('title', 'Untitled')}")
            return chart

        except Exception as e:
            logger.error(f'Failed to fetch chart {chart_id}: {e}')
            raise AssetNotFoundError(
                f"Chart '{chart_id}' not found", asset_type='Chart', asset_id=chart_id
            )

    def list_charts(
        self, project_id: str, model_id: Optional[str] = None
    ) -> List[Dict]:
        """List all charts in a project.

        Note: This endpoint may not be available on all Fiddler instances.
        Consider using get_charts_from_dashboard() instead.

        Args:
            project_id: Project ID
            model_id: Optional model ID to filter by

        Returns:
            List of chart dictionaries

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            charts = mgr.list_charts(project_id=project.id)

            for chart in charts:
                print(f"Chart: {chart['title']}")
                print(f"  Type: {chart['query_type']}")
            ```
        """
        client = self._get_client()

        # Get all charts for the project
        charts_url = f'/v3/projects/{project_id}/charts'

        try:
            response = client.get(url=charts_url)
            charts = response if isinstance(response, list) else []

            logger.info(f'Retrieved {len(charts)} charts from project {project_id}')

            # Filter by model if specified
            if model_id:
                filtered_charts = []
                for chart in charts:
                    # Check if chart references this model
                    data_source = chart.get('data_source', {})

                    # Check ANALYTICS charts
                    if data_source.get('query_type') == 'ANALYTICS':
                        model_ref = data_source.get('model', {})
                        if model_ref.get('id') == model_id:
                            filtered_charts.append(chart)

                    # Check MONITORING charts
                    else:
                        queries = data_source.get('queries', [])
                        for query in queries:
                            if query.get('model', {}).get('id') == model_id:
                                filtered_charts.append(chart)
                                break

                charts = filtered_charts
                logger.info(f'Filtered to {len(charts)} charts for model {model_id}')

            return charts

        except Exception as e:
            logger.error(f'Failed to list charts: {e}')
            return []

    def export_charts(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        names: Optional[List[str]] = None,
        dashboard_id: Optional[str] = None,
        chart_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Export charts from various sources.

        Priority order:
        1. If dashboard_id provided, fetch charts from dashboard
        2. If chart_ids provided, fetch specific charts by ID
        3. If project_id provided, list charts in project (may not work on all instances)

        Args:
            project_id: Source project ID (optional if using dashboard_id or chart_ids)
            model_id: Optional model ID to filter by
            names: Optional list of chart titles to filter results
            dashboard_id: Dashboard UUID to fetch charts from
            chart_ids: List of chart UUIDs to fetch directly

        Returns:
            List of chart dictionaries

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)

            # Export all charts from a dashboard
            exported = mgr.export_charts(
                dashboard_id='abc-123-dashboard-uuid'
            )

            # Export specific charts by ID
            exported = mgr.export_charts(
                chart_ids=['chart-uuid-1', 'chart-uuid-2']
            )

            # Export all charts for a model (may not work on all instances)
            exported = mgr.export_charts(
                project_id=project.id,
                model_id=model.id
            )

            # Filter exported charts by name
            exported = mgr.export_charts(
                dashboard_id='abc-123',
                names=['Performance Chart', 'Drift Chart']
            )
            ```
        """
        charts = []

        # Option 1: Fetch from dashboard
        if dashboard_id:
            logger.info(f'Fetching charts from dashboard {dashboard_id}')
            chart_id_list = self.get_charts_from_dashboard(dashboard_id)
            charts = [self.get_chart_by_id(cid) for cid in chart_id_list]
            logger.info(f'Retrieved {len(charts)} charts from dashboard')

        # Option 2: Fetch specific chart IDs
        elif chart_ids:
            logger.info(f'Fetching {len(chart_ids)} charts by ID')
            for chart_id in chart_ids:
                try:
                    chart = self.get_chart_by_id(chart_id)
                    charts.append(chart)
                except Exception as e:
                    logger.warning(f'Failed to fetch chart {chart_id}: {e}')
            logger.info(f'Retrieved {len(charts)} charts by ID')

        # Option 3: List charts in project (may not work on all instances)
        elif project_id:
            logger.info(f'Listing charts in project {project_id}')
            charts = self.list_charts(project_id, model_id)
            logger.info(f'Retrieved {len(charts)} charts from project')

        else:
            raise ValueError(
                'Must provide one of: dashboard_id, chart_ids, or project_id'
            )

        # Filter by names if specified
        if names:
            original_count = len(charts)
            charts = [c for c in charts if c.get('title') in names]
            logger.info(
                f'Filtered from {original_count} to {len(charts)} charts matching names: {names}'
            )

        # Enrich charts with asset names for cross-instance import
        enriched_charts = []
        for chart in charts:
            try:
                enriched_chart = self._enrich_chart_with_names(chart)
                enriched_charts.append(enriched_chart)
            except Exception as e:
                logger.warning(f"Failed to enrich chart '{chart.get('title')}': {e}")
                # Keep original chart even if enrichment fails
                enriched_charts.append(chart)

        logger.info(f'Exported {len(enriched_charts)} charts')
        return enriched_charts

    def _enrich_chart_with_names(self, chart: Dict) -> Dict:
        """Enrich chart with asset names for cross-instance import.

        Adds name fields alongside ID fields for segments, custom metrics, etc.
        This allows import_charts() to resolve IDs without needing source connection.

        Args:
            chart: Chart dictionary

        Returns:
            Chart with added name fields
        """
        import copy

        enriched = copy.deepcopy(chart)
        data_source = enriched.get('data_source', {})
        query_type = data_source.get('query_type')

        if query_type == 'ANALYTICS':
            # Enrich segment (data_source.segment)
            segment_ref = data_source.get('segment')
            if segment_ref and isinstance(segment_ref, dict) and segment_ref.get('id'):
                try:
                    segment = fdl.Segment.get(id_=segment_ref['id'])
                    data_source['segment_name'] = segment.name
                    logger.debug(
                        f"Enriched ANALYTICS segment ID {segment_ref['id']} with name '{segment.name}'"
                    )
                except Exception as e:
                    logger.debug(f"Could not enrich segment {segment_ref.get('id')}: {e}")

            # Enrich payload.metrics: baseline_name and metric_name (for custom metrics)
            payload = data_source.get('payload', {})
            for metric in payload.get('metrics', []):
                if metric.get('baseline_id'):
                    try:
                        baseline = fdl.Baseline.get(id_=metric['baseline_id'])
                        metric['baseline_name'] = baseline.name
                        logger.debug(
                            f"Enriched baseline ID {metric['baseline_id']} with name '{baseline.name}'"
                        )
                    except Exception as e:
                        logger.debug(f"Could not enrich baseline {metric.get('baseline_id')}: {e}")

                metric_id = metric.get('id')
                if metric_id is not None and _is_uuid(metric_id):
                    try:
                        custom_metric = fdl.CustomMetric.get(id_=metric_id)
                        metric['metric_name'] = custom_metric.name
                        logger.debug(
                            f"Enriched custom metric ID {metric_id} with name '{custom_metric.name}'"
                        )
                    except Exception as e:
                        logger.debug(f"Could not enrich custom metric {metric_id}: {e}")

        else:
            # MONITORING: enrich each query
            queries = data_source.get('queries', [])

            for query in queries:
                # Enrich segment reference with name
                if 'segment' in query and query['segment']:
                    segment_ref = query['segment']
                    if isinstance(segment_ref, dict) and 'id' in segment_ref:
                        segment_id = segment_ref['id']
                        try:
                            segment = fdl.Segment.get(id_=segment_id)
                            query['segment_name'] = segment.name
                            logger.debug(
                                f"Enriched segment ID {segment_id} with name '{segment.name}'"
                            )
                        except Exception as e:
                            logger.debug(f"Could not enrich segment {segment_id}: {e}")

                # Enrich custom metric reference with name
                if query.get('metric_type') == 'custom' and 'metric' in query:
                    metric_id = query['metric']
                    try:
                        metric = fdl.CustomMetric.get(id_=metric_id)
                        query['metric_name'] = metric.name
                        logger.debug(
                            f"Enriched custom metric ID {metric_id} with name '{metric.name}'"
                        )
                    except Exception as e:
                        logger.debug(f"Could not enrich custom metric {metric_id}: {e}")

        return enriched

    def _process_analytics_chart(self, chart: Dict) -> Dict:
        """Process an analytics chart payload to make it compatible for import."""

        if chart.get('query_type') != 'ANALYTICS':
            return chart

        if 'data_source' in chart and 'segment' in chart['data_source']:
            segment = chart['data_source']['segment']
            if isinstance(segment, dict):
                if 'id' in segment and 'definition' in segment:
                    if segment['id'] is None:
                        del segment['id']
                    elif segment['definition'] is None:
                        del segment['definition']
        
        return chart

        

    def import_charts(
        self,
        target_project_id: str,
        target_model_id: str,
        charts: List[Dict],
        validate: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """Import charts to a target project/model.

        This method resolves all IDs (model, baseline, custom metrics, segments)
        to match the target environment.

        Args:
            target_project_id: Target project ID
            target_model_id: Target model ID
            charts: List of chart dictionaries to import
            validate: If True, validate chart data before import
            dry_run: If True, validate only (don't create charts)

        Returns:
            Dictionary with counts: {'successful': N, 'failed': N}

        Example:
            ```python
            mgr = ChartManager(url=TARGET_URL, token=TARGET_TOKEN)

            result = mgr.import_charts(
                target_project_id=target_project.id,
                target_model_id=target_model.id,
                charts=exported_charts,
                dry_run=True  # Test first
            )

            print(f"Would import {result['successful']} charts")
            ```
        """
        client = self._get_client()
        charts_url = '/v3/charts'

        result = {'successful': 0, 'failed': 0, 'errors': []}

        # Get target model for ID resolution
        target_model = fdl.Model.get(id_=target_model_id)
        target_project = fdl.Project.get(id_=target_project_id)

        imported_charts = []

        for chart in charts:
            chart_title = chart.get('title', 'Untitled')

            try:
                # Create a copy to modify
                new_chart = json.loads(json.dumps(chart))

                # Generate new ID
                new_chart['id'] = str(uuid4())

                # Update project reference
                new_chart['project_id'] = target_project_id
                new_chart['project'] = {
                    'id': target_project_id,
                    'name': target_project.name,
                }

                # Resolve IDs in data_source
                data_source = new_chart.get('data_source', {})
                query_type = data_source.get('query_type')

                if query_type == 'ANALYTICS':
                    # Update model reference
                    data_source['model'] = {
                        'id': target_model_id,
                        'name': target_model.name,
                    }

                    # Handle dataset if present
                    if 'env_id' in data_source:
                        # Would need to resolve dataset ID
                        # For now, remove it to avoid errors
                        logger.warning(
                            f"Chart '{chart_title}' references dataset. "
                            'Dataset ID resolution not yet implemented. Removing reference.'
                        )
                        del data_source['env_id']

                    # Resolve segment (data_source.segment)
                    if data_source.get('segment') and data_source.get('segment_name'):
                        try:
                            segment = fdl.Segment.from_name(
                                name=data_source['segment_name'],
                                model_id=target_model_id,
                            )
                            data_source['segment'] = {'id': segment.id}
                            del data_source['segment_name']
                        except Exception as e:
                            logger.warning(
                                f"Could not resolve segment '{data_source['segment_name']}': {e}"
                            )
                    elif data_source.get('segment'):
                        logger.warning(
                            f"Chart '{chart_title}': Segment has no name field. "
                            "Cross-instance import may fail."
                        )

                    # Resolve baselines and custom metrics in payload.metrics
                    payload = data_source.get('payload', {})
                    for metric in payload.get('metrics', []):
                        # Resolve baseline_id by baseline_name (enriched)
                        if metric.get('baseline_id'):
                            if metric.get('baseline_name'):
                                try:
                                    baseline = fdl.Baseline.from_name(
                                        name=metric['baseline_name'],
                                        model_id=target_model_id,
                                    )
                                    metric['baseline_id'] = baseline.id
                                    del metric['baseline_name']
                                except Exception as e:
                                    logger.warning(
                                        f"Could not resolve baseline '{metric.get('baseline_name')}': {e}"
                                    )
                            else:
                                logger.warning(
                                    f"Chart '{chart_title}': Metric has baseline_id but no baseline_name. "
                                    "Cross-instance import may fail."
                                )

                        # Resolve custom metric by metric_name (id is UUID => custom metric)
                        metric_id = metric.get('id')
                        if metric_id is not None and _is_uuid(metric_id):
                            metric_name = metric.get('metric_name')
                            if not metric_name:
                                logger.warning(
                                    f"Chart '{chart_title}': Custom metric has no metric_name. "
                                    "Cross-instance import may fail."
                                )
                                metric_name = metric_id
                            try:
                                custom_metric = fdl.CustomMetric.from_name(
                                    name=metric_name, model_id=target_model_id
                                )
                                metric['id'] = custom_metric.id
                                if 'metric_name' in metric:
                                    del metric['metric_name']
                            except Exception as e:
                                logger.warning(
                                    f"Could not resolve custom metric '{metric_name}': {e}"
                                )

                elif query_type == 'MONITORING':
                    # Update queries
                    queries = data_source.get('queries', [])
                    for query in queries:
                        # Update model reference
                        query['model'] = {
                            'id': target_model_id,
                            'name': target_model.name,
                        }
                        query['model_name'] = target_model.name

                        # Resolve baseline if present
                        if 'baseline_name' in query:
                            try:
                                baseline = fdl.Baseline.from_name(
                                    name=query['baseline_name'],
                                    model_id=target_model_id,
                                )
                                query['baseline_id'] = baseline.id
                                del query['baseline_name']
                            except Exception as e:
                                logger.warning(
                                    f"Could not resolve baseline '{query.get('baseline_name')}': {e}"
                                )

                        # Resolve custom metric if present
                        if query.get('metric_type') == 'custom' and 'metric' in query:
                            # Use enriched metric_name if available, otherwise try metric ID as name
                            metric_name = query.get('metric_name')
                            if not metric_name:
                                logger.warning(
                                    f"Chart '{chart_title}': Custom metric has no name field. "
                                    "Cross-instance import may fail."
                                )
                                metric_name = query['metric']  # Try using ID as name (likely fails)

                            try:
                                # Look up custom metric by name in target
                                custom_metric = fdl.CustomMetric.from_name(
                                    name=metric_name, model_id=target_model_id
                                )
                                query['metric'] = custom_metric.id
                                # Clean up temporary name field
                                if 'metric_name' in query:
                                    del query['metric_name']
                            except Exception as e:
                                logger.warning(
                                    f"Could not resolve custom metric '{metric_name}': {e}"
                                )

                        # Resolve segment if present
                        if 'segment' in query and query['segment']:
                            segment_ref = query['segment']

                            # Use enriched segment_name if available
                            segment_name = query.get('segment_name')

                            if segment_name:
                                # Have enriched name - use it to look up target segment
                                try:
                                    segment = fdl.Segment.from_name(
                                        name=segment_name,
                                        model_id=target_model_id,
                                    )
                                    query['segment'] = {'id': segment.id}
                                    # Clean up temporary name field
                                    del query['segment_name']
                                except Exception as e:
                                    logger.warning(
                                        f"Could not resolve segment '{segment_name}': {e}"
                                    )
                            elif isinstance(segment_ref, str):
                                # Legacy: segment stored as name string
                                try:
                                    segment = fdl.Segment.from_name(
                                        name=segment_ref,
                                        model_id=target_model_id,
                                    )
                                    query['segment'] = {'id': segment.id}
                                except Exception as e:
                                    logger.warning(
                                        f"Could not resolve segment '{segment_ref}': {e}"
                                    )
                            else:
                                # No name field available - cross-instance import will fail
                                logger.warning(
                                    f"Chart '{chart_title}': Segment has no name field. "
                                    "Cross-instance import may fail. "
                                    f"Segment ref: {segment_ref}"
                                )

                # Remove metadata fields that shouldn't be in POST
                for field in [
                    'created_at',
                    'updated_at',
                    'created_by',
                    'updated_by',
                    'organization',
                ]:
                    if field in new_chart:
                        del new_chart[field]

                if dry_run:
                    logger.info(f'[DRY RUN] Would import chart: {chart_title}')
                    result['successful'] += 1
                else:
                    # Create the chart
                    processed_chart = self._process_analytics_chart(new_chart)
                    response = client.post(url=charts_url, data=processed_chart)
                    logger.info(f'Successfully imported chart: {chart_title}')
                    result['successful'] += 1
                    imported_charts.append(response.json()['data'])

            except Exception as e:
                logger.error(f"Failed to import chart '{chart_title}': {e}")
                result['failed'] += 1
                result['errors'].append((chart_title, str(e)))

        logger.info(
            f'Chart import complete: {result["successful"]} successful, '
            f'{result["failed"]} failed'
        )

        result_dict = {
            'successful': result['successful'],
            'failed': result['failed'],
            'errors': result['errors'],
            'imported_charts': imported_charts,
        }

        return result_dict

    def get_chart_by_title(self, project_id: str, title: str) -> Dict:
        """Get a specific chart by title.

        Args:
            project_id: Project ID
            title: Chart title

        Returns:
            Chart dictionary

        Raises:
            AssetNotFoundError: If chart not found

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            chart = mgr.get_chart_by_title(project.id, 'Performance Chart')
            print(f"Chart ID: {chart['id']}")
            ```
        """
        charts = self.list_charts(project_id)

        for chart in charts:
            if chart.get('title') == title:
                return chart

        raise AssetNotFoundError(
            f"Chart '{title}' not found in project {project_id}",
            asset_type='Chart',
            asset_id=title,
        )

    def delete_chart(self, chart_id: str) -> bool:
        """Delete a chart.

        Args:
            chart_id: Chart ID to delete

        Returns:
            True if successful

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            chart = mgr.get_chart_by_title(project.id, 'Old Chart')
            mgr.delete_chart(chart['id'])
            ```
        """
        client = self._get_client()
        delete_url = f'/v3/charts/{chart_id}'

        try:
            client.delete(url=delete_url)
            logger.info(f'Deleted chart: {chart_id}')
            return True
        except Exception as e:
            logger.error(f'Failed to delete chart {chart_id}: {e}')
            raise

    def analyze_charts(self, project_id: str) -> Dict[str, Any]:
        """Analyze charts in a project.

        Args:
            project_id: Project ID

        Returns:
            Dictionary with chart statistics

        Example:
            ```python
            mgr = ChartManager(url=URL, token=TOKEN)
            analysis = mgr.analyze_charts(project.id)

            print(f"Total charts: {analysis['total']}")
            print(f"By type: {analysis['by_type']}")
            print(f"By model: {analysis['by_model']}")
            ```
        """
        charts = self.list_charts(project_id)

        by_type = {}
        by_model = {}

        for chart in charts:
            # Count by query type
            query_type = chart.get('query_type', 'unknown')
            by_type[query_type] = by_type.get(query_type, 0) + 1

            # Count by model
            data_source = chart.get('data_source', {})

            if data_source.get('query_type') == 'ANALYTICS':
                model_id = data_source.get('model', {}).get('id')
                if model_id:
                    by_model[model_id] = by_model.get(model_id, 0) + 1
            else:
                queries = data_source.get('queries', [])
                for query in queries:
                    model_id = query.get('model', {}).get('id')
                    if model_id:
                        by_model[model_id] = by_model.get(model_id, 0) + 1

        analysis = {
            'total': len(charts),
            'by_type': by_type,
            'by_model': by_model,
            'unique_models': len(by_model),
        }

        logger.info(
            f'Chart analysis for project {project_id}: '
            f'{analysis["total"]} total, {analysis["unique_models"]} models'
        )

        return analysis
