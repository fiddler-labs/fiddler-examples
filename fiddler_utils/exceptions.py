"""Custom exceptions for fiddler_utils package.

This module defines a hierarchy of exceptions for better error handling
and more informative error messages.
"""

from typing import List, Optional


class FiddlerUtilsError(Exception):
    """Base exception for all fiddler_utils errors."""

    pass


class ConnectionError(FiddlerUtilsError):
    """Raised when there are issues connecting to Fiddler."""

    def __init__(self, message: str, url: Optional[str] = None):
        self.url = url
        super().__init__(message)


class ValidationError(FiddlerUtilsError):
    """Raised when validation fails (schema, FQL, assets, etc.)."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        field: Optional[str] = None,
    ):
        self.errors = errors or []
        self.field = field
        full_message = message
        if self.errors:
            full_message += f'\nErrors: {", ".join(self.errors)}'
        super().__init__(full_message)


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""

    def __init__(
        self,
        message: str,
        missing_columns: Optional[List[str]] = None,
        incompatible_types: Optional[List[str]] = None,
    ):
        self.missing_columns = missing_columns or []
        self.incompatible_types = incompatible_types or []

        errors = []
        if self.missing_columns:
            errors.append(f'Missing columns: {", ".join(self.missing_columns)}')
        if self.incompatible_types:
            errors.append(f'Incompatible types: {", ".join(self.incompatible_types)}')

        super().__init__(message, errors=errors)


class FQLError(ValidationError):
    """Raised when FQL expression validation or parsing fails."""

    def __init__(
        self,
        message: str,
        expression: Optional[str] = None,
        position: Optional[int] = None,
    ):
        self.expression = expression
        self.position = position
        full_message = message
        if expression:
            full_message += f'\nExpression: {expression}'
        if position is not None:
            full_message += f'\nPosition: {position}'
        super().__init__(full_message)


class AssetNotFoundError(FiddlerUtilsError):
    """Raised when an asset (model, segment, metric, etc.) is not found."""

    def __init__(
        self,
        message: str,
        asset_type: Optional[str] = None,
        asset_id: Optional[str] = None,
    ):
        self.asset_type = asset_type
        self.asset_id = asset_id
        full_message = message
        if asset_type:
            full_message = f'{asset_type} not found: {message}'
        super().__init__(full_message)


class AssetImportError(FiddlerUtilsError):
    """Raised when asset import fails."""

    def __init__(
        self,
        message: str,
        asset_name: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.asset_name = asset_name
        self.reason = reason
        full_message = message
        if asset_name:
            full_message = f"Failed to import '{asset_name}': {message}"
        if reason:
            full_message += f' ({reason})'
        super().__init__(full_message)


class BulkOperationError(FiddlerUtilsError):
    """Raised when bulk operations encounter errors."""

    def __init__(
        self,
        message: str,
        successful_count: int = 0,
        failed_count: int = 0,
        errors: Optional[List[tuple]] = None,
    ):
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.errors = errors or []  # List of (item, error_message) tuples

        full_message = f'{message}\n'
        full_message += f'Successful: {successful_count}, Failed: {failed_count}'
        if self.errors:
            full_message += f'\nFirst few errors:\n'
            for item, error in self.errors[:5]:
                full_message += f'  - {item}: {error}\n'

        super().__init__(full_message)
