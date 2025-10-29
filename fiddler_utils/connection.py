"""Connection management utilities for Fiddler client.

This module provides helpers to simplify the repeated boilerplate of
initializing the Fiddler client and managing connections across multiple
instances.
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .exceptions import ConnectionError

logger = logging.getLogger(__name__)


# Global state to track if client has been initialized
_initialized = False
_current_url: Optional[str] = None
_current_token: Optional[str] = None


def get_or_init(
    url: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
    log_level: Optional[str] = None,
) -> None:
    """Initialize Fiddler client if not already initialized.

    This function wraps the common pattern of fdl.init() and handles
    the case where the client has already been initialized.

    Args:
        url: Fiddler instance URL (e.g., 'https://customer.fiddler.ai')
        token: API token for authentication
        force: If True, reinitialize even if already initialized
        log_level: Optional logging level for Fiddler client
                   (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   If specified, configures package-scoped logging for
                   the Fiddler client to suppress verbose output.

    Raises:
        ConnectionError: If URL or token are not provided and client not initialized
        ConnectionError: If connection to Fiddler fails

    Example:
        ```python
        from fiddler_utils import get_or_init

        # Initialize with suppressed Fiddler logs (most common)
        get_or_init(
            url='https://demo.fiddler.ai',
            token='abc123',
            log_level='ERROR'
        )

        # Subsequent calls do nothing
        get_or_init()  # No-op
        ```
    """
    global _initialized, _current_url, _current_token

    if _initialized and not force:
        logger.debug(f'Client already initialized for {_current_url}')
        return

    if url is None or token is None:
        if not _initialized:
            raise ConnectionError('url and token are required for first initialization')
        # Use cached values
        url = _current_url
        token = _current_token

    try:
        # Pre-configure logging BEFORE init to prevent Fiddler's auto-attach behavior
        # The Fiddler client checks if ROOT logger is configured, and if not, auto-attaches
        # a stderr handler. We prevent this by ensuring root logger appears configured.
        root_handler_added = False
        if log_level:
            # Only add handler to root logger if user hasn't already configured one
            # This prevents Fiddler's auto-attach behavior which checks root logger
            root_logger = logging.getLogger()
            if not root_logger.handlers:
                root_logger.addHandler(logging.NullHandler())
                root_handler_added = True

            # Configure the fiddler logger to suppress messages during init
            fiddler_logger = logging.getLogger('fiddler')
            fiddler_logger.setLevel(logging.CRITICAL)
            fiddler_logger.propagate = False

        logger.info(f'Initializing Fiddler client for {url}')
        fdl.init(url=url, token=token)
        _initialized = True
        _current_url = url
        _current_token = token

        # Configure logging AFTER init (Fiddler creates handlers during init)
        if log_level:
            target_level = getattr(logging, log_level.upper())
            fiddler_logger = logging.getLogger('fiddler')
            fiddler_logger.setLevel(target_level)

            # Set level on all handlers created during init
            for handler in fiddler_logger.handlers:
                handler.setLevel(target_level)

            # Clean up temporary NullHandler from root logger
            if root_handler_added:
                root_logger = logging.getLogger()
                root_logger.handlers = [
                    h for h in root_logger.handlers if not isinstance(h, logging.NullHandler)
                ]

        logger.info(f'Successfully connected to {url}')
    except Exception as e:
        raise ConnectionError(f'Failed to connect to Fiddler: {str(e)}', url=url)


def reset_connection() -> None:
    """Reset connection state.

    Useful for testing or when switching between different Fiddler instances.
    """
    global _initialized, _current_url, _current_token
    _initialized = False
    _current_url = None
    _current_token = None
    logger.info('Connection state reset')


@contextmanager
def connection_context(url: str, token: str, log_level: Optional[str] = None):
    """Context manager for temporarily switching Fiddler connections.

    This is useful when working with multiple Fiddler instances
    (e.g., copying assets from dev to prod).

    Args:
        url: Fiddler instance URL
        token: API token
        log_level: Optional logging level for Fiddler client

    Example:
        ```python
        from fiddler_utils import connection_context

        # Connect to source instance
        with connection_context(url=SOURCE_URL, token=SOURCE_TOKEN, log_level='ERROR'):
            source_model = fdl.Model.from_name('my_model', project_id=proj.id)
            segments = list(fdl.Segment.list(model_id=source_model.id))

        # Connect to target instance
        with connection_context(url=TARGET_URL, token=TARGET_TOKEN):
            target_model = fdl.Model.from_name('my_model', project_id=proj.id)
            # Import segments to target
        ```
    """
    # Save current state
    global _initialized, _current_url, _current_token
    old_initialized = _initialized
    old_url = _current_url
    old_token = _current_token

    try:
        # Initialize new connection
        reset_connection()
        get_or_init(url=url, token=token, force=True, log_level=log_level)
        logger.info(f'Switched to connection: {url}')
        yield
    finally:
        # Restore previous state
        reset_connection()
        if old_initialized:
            get_or_init(url=old_url, token=old_token, force=True)
            logger.info(f'Restored connection: {old_url}')


class ConnectionManager:
    """Manager for handling multiple Fiddler connections.

    This class makes it easier to work with multiple Fiddler instances
    by maintaining a registry of connections.

    Example:
        ```python
        from fiddler_utils import ConnectionManager

        # Initialize with optional logging configuration
        mgr = ConnectionManager(log_level='ERROR')
        mgr.add('source', url=SOURCE_URL, token=SOURCE_TOKEN)
        mgr.add('target', url=TARGET_URL, token=TARGET_TOKEN)

        # Use specific connection
        with mgr.use('source'):
            source_model = fdl.Model.from_name('my_model', project_id=proj.id)

        with mgr.use('target'):
            target_model = fdl.Model.from_name('my_model', project_id=proj.id)
        ```
    """

    def __init__(self, log_level: Optional[str] = None):
        """Initialize ConnectionManager.

        Args:
            log_level: Optional logging level for Fiddler client
                       (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                       If specified, configures package-scoped logging for
                       the Fiddler client to suppress verbose output.
        """
        self.connections: Dict[str, Dict[str, str]] = {}
        self.current: Optional[str] = None
        self.log_level: Optional[str] = log_level

    def add(self, name: str, url: str, token: str) -> None:
        """Add a named connection.

        Args:
            name: Identifier for this connection (e.g., 'source', 'target', 'prod')
            url: Fiddler instance URL
            token: API token
        """
        self.connections[name] = {'url': url, 'token': token}
        logger.info(f"Added connection '{name}' for {url}")

    def remove(self, name: str) -> None:
        """Remove a named connection.

        Args:
            name: Connection identifier to remove
        """
        if name in self.connections:
            del self.connections[name]
            logger.info(f"Removed connection '{name}'")

    @contextmanager
    def use(self, name: str):
        """Context manager to use a specific named connection.

        Args:
            name: Connection identifier

        Raises:
            ConnectionError: If connection name not found

        Example:
            ```python
            with mgr.use('source'):
                # Work with source instance
                pass
            ```
        """
        if name not in self.connections:
            raise ConnectionError(
                f"Connection '{name}' not found. "
                f'Available: {list(self.connections.keys())}'
            )

        conn = self.connections[name]
        with connection_context(url=conn['url'], token=conn['token'], log_level=self.log_level):
            self.current = name
            yield
            self.current = None

    def get_current(self) -> Optional[str]:
        """Get the name of the currently active connection.

        Returns:
            Connection name or None if no active connection
        """
        return self.current

    def list_connections(self) -> list[str]:
        """List all registered connection names.

        Returns:
            List of connection names
        """
        return list(self.connections.keys())
