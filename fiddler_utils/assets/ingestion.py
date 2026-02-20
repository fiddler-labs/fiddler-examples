"""OpenTelemetry ingestion utilities for Fiddler.

This module provides helpers for ingesting data into Fiddler via OpenTelemetry
Protocol (OTLP): log_pandas_traces for pandas DataFrames and log_event_traces
for lists of event dicts (e.g. from JSON or APIs, without loading into pandas).
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import pandas as pd

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
except ImportError:
    raise ImportError(
        'OpenTelemetry packages are required. Install with: '
        'pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http'
    )

try:
    from fiddler.libs.http_client import RequestClient
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for OTLP batch processing.
    
    This bundles batch-related parameters for OpenTelemetry span export,
    making it easier to manage batch settings and reuse configurations.
    
    Attributes:
        max_export_batch_size: Maximum number of spans to include in a single
                              export batch. Default: 50. Smaller batches help
                              avoid Python 3.10-3.12 SSL issues with large requests.
        export_timeout_seconds: Timeout in seconds for exporting spans. Default: 30.0.
        schedule_delay_seconds: Delay in seconds between batch exports. Default: 1.0.
        max_queue_size: Maximum number of spans to queue before dropping. Default: 512.
    
    Example:
        ```python
        from fiddler_utils.assets.ingestion import BatchConfig
        
        # Use default settings (small batches for SSL safety)
        config = BatchConfig()
        
        # Custom configuration for larger datasets
        config = BatchConfig(
            max_export_batch_size=100,
            export_timeout_seconds=60.0,
            schedule_delay_seconds=2.0
        )
        
        # Use with ingestion function
        log_pandas_traces(df, client, batch_config=config)
        ```
    """
    max_export_batch_size: int = 50
    export_timeout_seconds: float = 30.0
    schedule_delay_seconds: float = 1.0
    max_queue_size: int = 512
    
    @classmethod
    def default(cls) -> 'BatchConfig':
        """Create default batch config (small batches for SSL safety).
        
        Returns:
            BatchConfig with default settings
        """
        return cls()
    
    @classmethod
    def large_batches(cls) -> 'BatchConfig':
        """Create config optimized for larger batches and better throughput.
        
        Returns:
            BatchConfig with larger batch sizes
        """
        return cls(
            max_export_batch_size=100,
            export_timeout_seconds=60.0,
            schedule_delay_seconds=2.0,
            max_queue_size=1024
        )


def _extract_url_and_token(fiddler_client: RequestClient) -> Tuple[str, str]:
    """Extract Fiddler URL and authorization token from RequestClient.
    
    Args:
        fiddler_client: RequestClient instance
        
    Returns:
        Tuple of (url, token)
        
    Raises:
        ValueError: If URL or token cannot be extracted
    """
    # Try to get base_url from various possible attribute names
    url = None
    if hasattr(fiddler_client, 'base_url'):
        url = fiddler_client.base_url
    elif hasattr(fiddler_client, 'url'):
        url = fiddler_client.url
    elif hasattr(fiddler_client, '_base_url'):
        url = fiddler_client._base_url
    elif hasattr(fiddler_client, '_url'):
        url = fiddler_client._url
    
    if not url:
        raise ValueError(
            'Could not extract URL from fiddler_client. '
            'Expected base_url, url, _base_url, or _url attribute.'
        )
    
    # Extract token from _default_headers
    token = None
    headers = None
    if hasattr(fiddler_client, '_default_headers'):
        headers = fiddler_client._default_headers
    
    if headers:
        auth_header = headers.get('Authorization') or headers.get('authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header
    
    if not token:
        raise ValueError(
            'Could not extract Authorization token from fiddler_client._default_headers. '
            'Expected Authorization header with Bearer token.'
        )
    
    return url, token


def _to_nanoseconds(time_value: Any) -> int:
    """Convert a time value to nanoseconds (int).
    
    Args:
        time_value: Can be pd.Timestamp, datetime, string, or None
        
    Returns:
        Nanoseconds since epoch as integer
    """
    if time_value is None or pd.isna(time_value):
        return int(pd.Timestamp.now().timestamp() * 1e9)
    
    dt = pd.to_datetime(time_value)
    return int(dt.timestamp() * 1e9)


def _serialize_value(value: Any) -> str:
    """Serialize a value to string, handling dict/list as JSON.
    
    Args:
        value: Value to serialize
        
    Returns:
        String representation
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    elif pd.isna(value):
        return ''
    else:
        return str(value)


def _ingest_records_to_otlp(
    fiddler_client: RequestClient,
    records: List[Dict[str, Any]],
    start_ns_list: List[int],
    end_ns_list: List[int],
    column_mapping: Dict[str, str],
    static_attributes: Dict[str, Any],
    batch_config: BatchConfig,
    index_for_errors: Optional[Callable[[int], Any]] = None,
    span_name: str = 'span',
) -> None:
    """Internal: send a list of record dicts to Fiddler via OTLP.
    
    Shared by log_pandas_traces and log_event_traces. Caller must ensure
    len(records) == len(start_ns_list) == len(end_ns_list).
    """
    if index_for_errors is None:
        index_for_errors = lambda i: i  # noqa: E731
    n = len(records)
    fiddler_url, auth_token = _extract_url_and_token(fiddler_client)
    otlp_endpoint = f'{fiddler_url.rstrip("/")}/v1/traces'
    application_id = static_attributes.get('application.id')
    if not application_id and hasattr(fiddler_client, '_default_headers'):
        headers = fiddler_client._default_headers
        if headers:
            application_id = (
                headers.get('Fiddler-Application-Id')
                or headers.get('fiddler-application-id')
            )
    resource_attributes = {}
    if application_id:
        resource_attributes['application.id'] = str(application_id)
    resource = Resource.create(resource_attributes) if resource_attributes else Resource.create({})
    tracer_provider = TracerProvider(resource=resource)
    otlp_headers = {'Authorization': f'Bearer {auth_token}'}
    if application_id:
        otlp_headers['Fiddler-Application-Id'] = str(application_id)
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=otlp_headers,
        timeout=batch_config.export_timeout_seconds,
    )
    otlp_processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=batch_config.max_queue_size,
        max_export_batch_size=batch_config.max_export_batch_size,
        schedule_delay_millis=int(batch_config.schedule_delay_seconds * 1000),
        export_timeout_millis=int(batch_config.export_timeout_seconds * 1000),
    )
    tracer_provider.add_span_processor(otlp_processor)
    tracer = tracer_provider.get_tracer(__name__)
    logger.info(f'OpenTelemetry configured with endpoint: {otlp_endpoint}')
    if application_id:
        logger.info(f'Using Fiddler Application ID: {application_id}')
    logger.info(f'Starting ingestion of {n} rows...')
    static_serialized = {k: _serialize_value(v) for k, v in static_attributes.items()}
    rows_attrs = []
    for record in records:
        attrs = dict(static_serialized)
        for col_name, attr_name in column_mapping.items():
            if col_name in record:
                attrs[attr_name] = _serialize_value(record[col_name])
        if 'gen_ai.agent.name' not in attrs:
            attrs['gen_ai.agent.name'] = 'my_agent'
        rows_attrs.append(attrs)
    successful_rows = 0
    failed_rows = 0
    errors = []
    for i in range(n):
        index = index_for_errors(i)
        start_ns = start_ns_list[i]
        end_ns = end_ns_list[i]
        try:
            span = tracer.start_span(
                name=span_name,
                start_time=start_ns,
                kind=trace.SpanKind.CLIENT,
            )
            try:
                for attr_name, value in rows_attrs[i].items():
                    span.set_attribute(attr_name, value)
                span.set_status(trace.Status(trace.StatusCode.OK))
                successful_rows += 1
            except Exception as e:
                logger.error(f'Error processing row {index}: {str(e)}')
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                failed_rows += 1
                errors.append((index, str(e)))
            finally:
                span.end(end_time=end_ns)
        except Exception as e:
            logger.error(f'Failed to create span for row {index}: {str(e)}')
            failed_rows += 1
            errors.append((index, str(e)))
    try:
        tracer_provider.force_flush(timeout_millis=int(batch_config.export_timeout_seconds * 2000))
        logger.info('Force flushed all spans to Fiddler')
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f'Error during force flush ({error_type}): {error_msg}')
        if 'SSL' in error_type or 'SSLError' in error_type or 'SSL' in error_msg:
            logger.warning(
                'SSL error detected. This may be due to a known Python 3.10-3.12 bug '
                'with large HTTPS requests. Consider:\n'
                '  1. Processing smaller DataFrames (split into chunks)\n'
                '  2. Upgrading to Python 3.13+\n'
                '  3. Some spans may have been successfully sent before the error'
            )
    if successful_rows > 0:
        logger.info(f'✅ Successfully ingested {successful_rows} rows')
    if failed_rows > 0:
        logger.warning(f'⚠️ Failed to ingest {failed_rows} rows')
        for idx, error in errors[:10]:
            logger.warning(f'  Row {idx}: {error}')
        if len(errors) > 10:
            logger.warning(f'  ... and {len(errors) - 10} more errors')
    if successful_rows == 0 and failed_rows > 0:
        raise RuntimeError(
            f'All {failed_rows} rows failed to ingest. '
            f'First error: {errors[0][1] if errors else "Unknown error"}'
        )
    if failed_rows == 0 and successful_rows > 0:
        print(f'✅ Successfully ingested {successful_rows} rows')


def log_pandas_traces(
    df: pd.DataFrame,
    fiddler_client: RequestClient,
    column_mapping: Optional[Dict[str, str]] = None,
    static_attributes: Optional[Dict[str, Any]] = None,
    batch_config: Optional[BatchConfig] = None,
    span_name: str = 'span',
) -> None:
    """Ingest a pandas DataFrame into Fiddler via OpenTelemetry (OTLP).
    
    This function treats each row as a separate trace and sends it to Fiddler
    using the OpenTelemetry Protocol. Each row is converted to a span with
    attributes mapped from DataFrame columns.
    
    Args:
        df: The pandas DataFrame to ingest. Each row becomes one trace.
        fiddler_client: An instance of RequestClient from fiddler.libs.http_client.
                       The function extracts the Fiddler URL and Authorization
                       token from this client.
        column_mapping: Optional dictionary mapping DataFrame column names to
                       Fiddler attribute names. For example:
                       {'question': 'gen_ai.llm.input.user',
                        'response': 'gen_ai.llm.output'}
                       If a column is not in the mapping, it won't be included
                       as an attribute.
        static_attributes: Optional dictionary of attributes to apply to every
                          span. For example:
                          {'application.id': 'my-app-uuid',
                           'version': '1.0.0'}
                          If 'gen_ai.agent.name' is not provided in static_attributes
                          or column_mapping, it defaults to 'my_agent'.
                          Note: If 'application.id' is provided in static_attributes,
                          it will be automatically added as the 'Fiddler-Application-Id'
                          header to the OTLP requests.
        batch_config: Optional BatchConfig object to configure batch processing.
                    If None, uses BatchConfig.default() with safe defaults for
                    Python 3.10-3.12 SSL compatibility.
        span_name: Optional name for the OpenTelemetry span. Default: 'span'.
    
    Raises:
        ValueError: If URL or token cannot be extracted from fiddler_client
        RuntimeError: If all rows fail to ingest
        requests.exceptions.SSLError: If SSL connection fails (known issue in Python 3.10-3.12
                                      with large HTTPS requests - consider using smaller DataFrames
                                      or upgrading to Python 3.13+)
    
    Example:
        ```python
        import pandas as pd
        from fiddler.libs.http_client import RequestClient
        from fiddler_utils.assets.ingestion import log_pandas_traces
        
        # Create sample data
        df = pd.DataFrame([
            {
                'prompt': 'What is the weather?',
                'response': 'I can help with that...',
                'start_time': '2025-01-08 10:00:00',
                'end_time': '2025-01-08 10:00:02',
                'model_name': 'gpt-4',
                'total_tokens': 150
            }
        ])
        
        # Initialize RequestClient
        client = RequestClient(
            base_url='https://acme.cloud.fiddler.ai',
            headers={'Authorization': 'Bearer your-token'}
        )
        
        # Map columns to Fiddler attributes
        column_mapping = {
            'prompt': 'gen_ai.llm.input.user',
            'response': 'gen_ai.llm.output',
            'model_name': 'gen_ai.request.model',
            'total_tokens': 'gen_ai.usage.total_tokens'
        }
        
        # Set static attributes for all spans
        static_attrs = {
            'application.id': 'my-app-123',
            'gen_ai.agent.id': 'agent-001'
        }
        
        # Ingest with default batch settings
        log_pandas_traces(
            df=df,
            fiddler_client=client,
            column_mapping=column_mapping,
            static_attributes=static_attrs
        )
        
        # Or use custom batch configuration
        from fiddler_utils.assets.ingestion import BatchConfig
        
        batch_config = BatchConfig.large_batches()  # Optimized for throughput
        log_pandas_traces(
            df=df,
            fiddler_client=client,
            column_mapping=column_mapping,
            static_attributes=static_attrs,
            batch_config=batch_config
        )
        ```
    """
    if df.empty:
        logger.warning('DataFrame is empty. Nothing to ingest.')
        return

    if batch_config is None:
        batch_config = BatchConfig.default()
    column_mapping = column_mapping or {}
    static_attributes = static_attributes or {}

    # Determine start_time and end_time column names
    start_time_col = None
    end_time_col = None
    for col, attr in column_mapping.items():
        if attr == 'start_time':
            start_time_col = col
        if attr == 'end_time':
            end_time_col = col
    if start_time_col is None and 'start_time' in df.columns:
        start_time_col = 'start_time'
        
    if end_time_col is None and 'end_time' in df.columns:
        end_time_col = 'end_time'

    # Timestamp conversion (per-element for robustness with mixed types/timezones)
    now_ns = int(pd.Timestamp.now().timestamp() * 1e9)
    n = len(df)
    records = df.to_dict('records')
    if start_time_col and start_time_col in df.columns:
        start_ns_list = [
            _to_nanoseconds(rec.get(start_time_col))
            for rec in records
        ]
    else:
        start_ns_list = [now_ns] * n
    if end_time_col and end_time_col in df.columns:
        end_ns_list = []
        for rec, start_ns in zip(records, start_ns_list):
            end_val = rec.get(end_time_col)
            if end_val is None or pd.isna(end_val):
                end_ns_list.append(start_ns)
            else:
                end_ns_list.append(_to_nanoseconds(end_val))
    else:
        end_ns_list = start_ns_list

    _ingest_records_to_otlp(
        fiddler_client=fiddler_client,
        records=records,
        start_ns_list=start_ns_list,
        end_ns_list=end_ns_list,
        column_mapping=column_mapping,
        static_attributes=static_attributes,
        batch_config=batch_config,
        index_for_errors=lambda i: df.index[i],
        span_name=span_name,
    )


def log_event_traces(
    events: List[Dict[str, Any]],
    fiddler_client: RequestClient,
    column_mapping: Optional[Dict[str, str]] = None,
    static_attributes: Optional[Dict[str, Any]] = None,
    batch_config: Optional[BatchConfig] = None,
    span_name: str = 'span',
) -> None:
    """Ingest a list of event dicts into Fiddler via OpenTelemetry (OTLP).

    Use this when events are already in record form (e.g. from JSON, APIs, or
    storage) and you do not need to load into a pandas DataFrame. Each dict
    should have the same shape as one row from ``df.to_dict("records")``.

    Args:
        events: List of dictionaries; each dict becomes one trace/span. Keys
                are field names (e.g. 'prompt', 'response', 'start_time').
        fiddler_client: An instance of RequestClient from fiddler.libs.http_client.
        column_mapping: Optional mapping from event dict keys to Fiddler attribute
                       names. Same semantics as in log_pandas_traces.
        static_attributes: Optional attributes applied to every span. Same as
                          in log_pandas_traces.
        batch_config: Optional BatchConfig. If None, uses BatchConfig.default().
        span_name: Optional name for the OpenTelemetry span. Default: 'span'.

    Raises:
        ValueError: If URL or token cannot be extracted from fiddler_client.
        RuntimeError: If all events fail to ingest.

    Example:
        ```python
        from fiddler.libs.http_client import RequestClient
        from fiddler_utils.assets.ingestion import log_event_traces

        events = [
            {
                'prompt': 'What is the weather?',
                'response': 'I can help with that...',
                'start_time': '2025-01-08 10:00:00',
                'end_time': '2025-01-08 10:00:02',
                'model_name': 'gpt-4',
            }
        ]
        column_mapping = {
            'prompt': 'gen_ai.llm.input.user',
            'response': 'gen_ai.llm.output',
            'model_name': 'gen_ai.request.model',
        }
        log_event_traces(
            events=events,
            fiddler_client=client,
            column_mapping=column_mapping,
            static_attributes={'application.id': 'my-app-123'},
        )
        ```
    """
    if not events:
        logger.warning('events is empty. Nothing to ingest.')
        return

    if batch_config is None:
        batch_config = BatchConfig.default()
    column_mapping = column_mapping or {}
    static_attributes = static_attributes or {}

    # Resolve start_time / end_time keys from mapping or from event keys
    start_time_col = None
    end_time_col = None
    for col, attr in column_mapping.items():
        if attr == 'start_time':
            start_time_col = col
        if attr == 'end_time':
            end_time_col = col
    if start_time_col is None or end_time_col is None:
        all_keys = set()
        for evt in events:
            all_keys.update(evt.keys())
        if start_time_col is None and 'start_time' in all_keys:
            start_time_col = 'start_time'
        if end_time_col is None and 'end_time' in all_keys:
            end_time_col = 'end_time'

    start_ns_list = [
        _to_nanoseconds(evt.get(start_time_col) if start_time_col else None)
        for evt in events
    ]
    end_ns_list = []
    for evt, start_ns in zip(events, start_ns_list):
        end_val = evt.get(end_time_col) if end_time_col else None
        if end_time_col is None or end_val is None or pd.isna(end_val):
            end_ns_list.append(start_ns)
        else:
            end_ns_list.append(_to_nanoseconds(end_val))

    _ingest_records_to_otlp(
        fiddler_client=fiddler_client,
        records=events,
        start_ns_list=start_ns_list,
        end_ns_list=end_ns_list,
        column_mapping=column_mapping,
        static_attributes=static_attributes,
        batch_config=batch_config,
        index_for_errors=lambda i: i,
        span_name=span_name,
    )

