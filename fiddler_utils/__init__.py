"""Fiddler Utilities - Internal helper library for field engineers.

This package provides convenience wrappers and utilities for common
Fiddler administrative tasks, reducing code duplication across utility
scripts and notebooks.

NOT part of the official Fiddler SDK - for internal use only.

Version compatibility: Requires fiddler-client >= 3.10.0

Example:
    ```python
    from fiddler_utils import get_or_init, configure_fiddler_logging
    from fiddler_utils import SchemaValidator, fql

    # Suppress verbose Fiddler client logs (recommended)
    configure_fiddler_logging(level='ERROR')

    # Initialize connection (with optional logging control)
    get_or_init(url='https://demo.fiddler.ai', token='abc123', log_level='ERROR')

    # Extract columns from FQL expression
    columns = fql.extract_columns('"age" > 30 and "status" == \'active\'')

    # Validate schema compatibility
    is_valid, missing = SchemaValidator.validate_columns(
        columns, target_model
    )
    ```
"""

import logging

__version__ = '0.1.0'
__author__ = 'Fiddler AI'

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Allow users to configure logging


# Import public API - Connection utilities
from .connection import (
    get_or_init,
    reset_connection,
    connection_context,
    ConnectionManager,
)

# Import public API - Schema validation
from .schema import (
    SchemaValidator,
    SchemaComparison,
    ColumnInfo,
    ColumnRole,
)

# Import public API - FQL utilities (as module)
from . import fql

# Import public API - Asset managers
from .assets import (
    SegmentManager,
    CustomMetricManager,
    AlertManager,
    ChartManager,
    DashboardManager,
    ModelManager,
    BaselineManager,
    AssetExportData,
    ImportResult,
    ValidationResult,
    ModelExportData,
    ColumnExportData,
)

# Import public API - Project and environment management
from .projects import (
    ProjectManager,
    EnvironmentHierarchy,
    ProjectInfo,
    ModelInfo,
    EnvironmentStats,
    TimestampAnalysis,
)

# Import public API - Environment reporting
from .reporting import EnvironmentReporter

# Import public API - Safe iteration utilities
from .iteration import (
    iterate_projects_safe,
    iterate_models_safe,
    count_models_by_project,
)

# Import public API - Model comparison
from .comparison import (
    ModelComparator,
    ComparisonResult,
    ComparisonConfig,
    ConfigurationComparison,
    SpecComparison,
    AssetComparison,
    ValueDifference,
)

# Import public API - Exceptions
from .exceptions import (
    FiddlerUtilsError,
    ConnectionError,
    ValidationError,
    SchemaValidationError,
    FQLError,
    AssetNotFoundError,
    AssetImportError,
    BulkOperationError,
)

# Define public API
__all__ = [
    # Version
    '__version__',
    # Logging utilities
    'configure_logging',
    'configure_fiddler_logging',
    # Connection utilities
    'get_or_init',
    'reset_connection',
    'connection_context',
    'ConnectionManager',
    # Schema validation
    'SchemaValidator',
    'SchemaComparison',
    'ColumnInfo',
    'ColumnRole',
    # FQL module
    'fql',
    # Asset managers
    'SegmentManager',
    'CustomMetricManager',
    'AlertManager',
    'ChartManager',
    'DashboardManager',
    'ModelManager',
    'BaselineManager',
    'AssetExportData',
    'ImportResult',
    'ValidationResult',
    'ModelExportData',
    'ColumnExportData',
    # Project and environment management
    'ProjectManager',
    'EnvironmentHierarchy',
    'ProjectInfo',
    'ModelInfo',
    'EnvironmentStats',
    'TimestampAnalysis',
    # Environment reporting
    'EnvironmentReporter',
    # Safe iteration utilities
    'iterate_projects_safe',
    'iterate_models_safe',
    'count_models_by_project',
    # Model comparison
    'ModelComparator',
    'ComparisonResult',
    'ComparisonConfig',
    'ConfigurationComparison',
    'SpecComparison',
    'AssetComparison',
    'ValueDifference',
    # Exceptions
    'FiddlerUtilsError',
    'ConnectionError',
    'ValidationError',
    'SchemaValidationError',
    'FQLError',
    'AssetNotFoundError',
    'AssetImportError',
    'BulkOperationError',
]


def configure_logging(
    level: str = 'INFO',
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers: list = None,
):
    """Configure logging for fiddler_utils package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format
        handlers: Optional list of logging handlers

    Example:
        ```python
        from fiddler_utils import configure_logging

        # Enable debug logging
        configure_logging(level='DEBUG')

        # Custom format
        configure_logging(
            level='INFO',
            format='%(levelname)s - %(message)s'
        )
        ```
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    if handlers:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        # Add default console handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)

    logger.info(f'Fiddler Utils logging configured at {level} level')


def configure_fiddler_logging(
    level: str = 'WARNING',
    format: str = None,
):
    """Configure logging for the Fiddler Python client package.

    This function provides package-scoped control over Fiddler client logging
    output, affecting only loggers in the 'fiddler' namespace without changing
    global logging configuration.

    Unlike logging.basicConfig() (which affects all packages), this approach
    only modifies the Fiddler client's log output, following Python logging
    best practices for library configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Default: WARNING (shows warnings and errors only)
        format: Optional custom log message format. If None, only the level
                is set without adding handlers.

    Example:
        ```python
        from fiddler_utils import configure_fiddler_logging
        import fiddler as fdl

        # Initialize first, then configure logging
        fdl.init(url=URL, token=TOKEN)

        # Suppress verbose Fiddler client logs (most common)
        configure_fiddler_logging(level='ERROR')

        # Enable debug logging for troubleshooting
        configure_fiddler_logging(level='DEBUG')

        # Custom format string
        configure_fiddler_logging(
            level='INFO',
            format='%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s: %(message)s'
        )
        ```

    Notes:
        * Best to call AFTER fdl.init() for immediate effect
        * Only affects 'fiddler.*' loggers (package-scoped)
        * Does NOT affect other packages like pandas, numpy, etc.
        * Sets level on both logger and all its handlers
        * For fiddler_utils package logging, use configure_logging() instead
    """
    log_level = getattr(logging, level.upper())

    # Get the fiddler package logger
    fiddler_logger = logging.getLogger('fiddler')
    fiddler_logger.setLevel(log_level)

    # Also set level on all existing handlers (important!)
    for handler in fiddler_logger.handlers:
        handler.setLevel(log_level)

    # Add handler with custom format if specified
    if format:
        # Remove existing handlers to avoid duplicates
        fiddler_logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(format))
        fiddler_logger.addHandler(handler)
