"""OpenTelemetry ingestion utilities for Fiddler.

This module provides helpers for ingesting pandas DataFrames into Fiddler
via OpenTelemetry Protocol (OTLP).
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union, Tuple
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
        ingest_pandas_to_otlp(df, client, batch_config=config)
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


def log_pandas_traces(
    df: pd.DataFrame,
    fiddler_client: RequestClient,
    column_mapping: Optional[Dict[str, str]] = None,
    static_attributes: Optional[Dict[str, Any]] = None,
    batch_config: Optional[BatchConfig] = None,
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
        from fiddler_utils.assets.ingestion import ingest_pandas_to_otlp
        
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
        ingest_pandas_to_otlp(
            df=df,
            fiddler_client=client,
            column_mapping=column_mapping,
            static_attributes=static_attrs
        )
        
        # Or use custom batch configuration
        from fiddler_utils.assets.ingestion import BatchConfig
        
        batch_config = BatchConfig.large_batches()  # Optimized for throughput
        ingest_pandas_to_otlp(
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
    
    # Use provided batch config or default
    if batch_config is None:
        batch_config = BatchConfig.default()
    
    # Extract URL and token from fiddler_client
    fiddler_url, auth_token = _extract_url_and_token(fiddler_client)
    
    # Construct OTLP endpoint
    otlp_endpoint = f'{fiddler_url.rstrip("/")}/v1/traces'
    
    # Normalize column_mapping and static_attributes
    column_mapping = column_mapping or {}
    static_attributes = static_attributes or {}
    
    # Extract application ID from static_attributes for Fiddler-Application-Id header
    # Fall back to fiddler_client._default_headers if not found in static_attributes
    application_id = static_attributes.get('application.id')
    
    # Fallback to _default_headers if not found in static_attributes
    if not application_id and hasattr(fiddler_client, '_default_headers'):
        headers = fiddler_client._default_headers
        if headers:
            application_id = (
                headers.get('Fiddler-Application-Id') or 
                headers.get('fiddler-application-id')
            )
    
    # Create Resource with application.id (required at Resource level)
    resource_attributes = {}
    if application_id:
        resource_attributes['application.id'] = str(application_id)
    
    resource = Resource.create(resource_attributes) if resource_attributes else Resource.create({})
    
    # Initialize local TracerProvider (not global) to avoid singleton issues
    # Pass resource to ensure application.id is set at Resource level
    tracer_provider = TracerProvider(resource=resource)
    
    # Build OTLP headers
    otlp_headers = {
        'Authorization': f'Bearer {auth_token}',
    }
    if application_id:
        otlp_headers['Fiddler-Application-Id'] = str(application_id)
    
    # Configure OTLP exporter with timeout to avoid SSL errors
    # Note: Python 3.10-3.12 have known SSL issues with large HTTPS requests
    # Using smaller batches and timeouts helps mitigate this
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=otlp_headers,
        timeout=batch_config.export_timeout_seconds,
    )
    
    # Add batch span processor with configurable batch sizes
    # This helps mitigate Python 3.10-3.12 SSL EOF errors with large payloads
    otlp_processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=batch_config.max_queue_size,
        max_export_batch_size=batch_config.max_export_batch_size,
        schedule_delay_millis=int(batch_config.schedule_delay_seconds * 1000),
        export_timeout_millis=int(batch_config.export_timeout_seconds * 1000),
    )
    tracer_provider.add_span_processor(otlp_processor)
    
    # Get tracer from the configured provider (must be after adding processors)
    # Use the provider's get_tracer method to ensure spans are associated correctly
    tracer = tracer_provider.get_tracer(__name__)
    
    logger.info(f'OpenTelemetry configured with endpoint: {otlp_endpoint}')
    if application_id:
        logger.info(f'Using Fiddler Application ID: {application_id}')
    logger.info(f'Starting ingestion of {len(df)} rows...')
    
    # Determine start_time and end_time column names
    # Look for 'start_time' and 'end_time' columns (these are not Fiddler attributes,
    # but special columns used for span timestamps)
    # If column_mapping has entries for these, use the mapped column names
    start_time_col = None
    end_time_col = None
    
    # Check if any columns are mapped to 'start_time' or 'end_time' attributes
    # (though these aren't standard Fiddler attributes, allow for flexibility)
    for col, attr in column_mapping.items():
        if attr == 'start_time':
            start_time_col = col
        if attr == 'end_time':
            end_time_col = col
    
    # If not found in mapping, check for default column names
    if start_time_col is None and 'start_time' in df.columns:
        start_time_col = 'start_time'
    if end_time_col is None and 'end_time' in df.columns:
        end_time_col = 'end_time'
    
    # Track success/failure
    successful_rows = 0
    failed_rows = 0
    errors = []
    
    # Iterate through DataFrame rows
    for index, row in df.iterrows():
        try:
            # Prepare timestamps
            start_ns = _to_nanoseconds(
                row.get(start_time_col) if start_time_col else None
            )
            end_ns = _to_nanoseconds(
                row.get(end_time_col) if end_time_col else None
            )
            
            # Create span using start_span (not start_as_current_span)
            # This ensures spans are detached and valid for historical backfilling
            span = tracer.start_span(
                name='backfilled_operation',
                start_time=start_ns,
                kind=trace.SpanKind.CLIENT,
            )
            
            try:
                # Apply static attributes first
                for attr_name, attr_value in static_attributes.items():
                    span.set_attribute(attr_name, _serialize_value(attr_value))
                
                # Map DataFrame columns to Fiddler attributes
                for col_name, attr_name in column_mapping.items():
                    if col_name in row:
                        value = row[col_name]
                        span.set_attribute(
                            attr_name,
                            _serialize_value(value)
                        )
                
                # Default gen_ai.agent.name if not provided in static_attributes or column_mapping
                if 'gen_ai.agent.name' not in static_attributes:
                    # Check if it was set from column_mapping
                    agent_name_set = any(
                        attr == 'gen_ai.agent.name' and col in row
                        for col, attr in column_mapping.items()
                    )
                    
                    if not agent_name_set:
                        span.set_attribute('gen_ai.agent.name', 'my_agent')
                
                # Set status to OK
                span.set_status(trace.Status(trace.StatusCode.OK))
                successful_rows += 1
                
            except Exception as e:
                error_msg = f'Error processing row {index}: {str(e)}'
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                failed_rows += 1
                errors.append((index, str(e)))
            
            finally:
                # End the span with explicit end timestamp
                span.end(end_time=end_ns)
                
        except Exception as e:
            error_msg = f'Failed to create span for row {index}: {str(e)}'
            logger.error(error_msg)
            failed_rows += 1
            errors.append((index, str(e)))
    
    # Force flush to ensure all spans are sent
    try:
        tracer_provider.force_flush(timeout_millis=int(batch_config.export_timeout_seconds * 2000))  # 2x export timeout
        logger.info('Force flushed all spans to Fiddler')
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f'Error during force flush ({error_type}): {error_msg}')
        
        # Provide helpful guidance for known SSL issues
        if 'SSL' in error_type or 'SSLError' in error_type or 'SSL' in error_msg:
            logger.warning(
                'SSL error detected. This may be due to a known Python 3.10-3.12 bug '
                'with large HTTPS requests. Consider:\n'
                '  1. Processing smaller DataFrames (split into chunks)\n'
                '  2. Upgrading to Python 3.13+\n'
                '  3. Some spans may have been successfully sent before the error'
            )
    
    # Print summary
    if successful_rows > 0:
        logger.info(f'✅ Successfully ingested {successful_rows} rows')
    
    if failed_rows > 0:
        logger.warning(f'⚠️ Failed to ingest {failed_rows} rows')
        for idx, error in errors[:10]:  # Show first 10 errors
            logger.warning(f'  Row {idx}: {error}')
        if len(errors) > 10:
            logger.warning(f'  ... and {len(errors) - 10} more errors')
    
    if successful_rows == 0 and failed_rows > 0:
        raise RuntimeError(
            f'All {failed_rows} rows failed to ingest. '
            f'First error: {errors[0][1] if errors else "Unknown error"}'
        )
    
    # Print success message if all rows were ingested successfully
    if failed_rows == 0 and successful_rows > 0:
        print(f'✅ Successfully ingested all {successful_rows} rows to Fiddler!')

