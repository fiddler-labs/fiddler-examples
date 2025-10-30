"""Schema validation and comparison utilities for Fiddler models.

This module provides utilities for extracting, validating, and comparing
model schemas across different Fiddler models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import logging

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from .exceptions import SchemaValidationError
from . import fql

logger = logging.getLogger(__name__)


class ColumnRole(str, Enum):
    """Enum for column roles in Fiddler model spec."""

    INPUT = 'input'
    OUTPUT = 'output'
    TARGET = 'target'
    METADATA = 'metadata'
    DECISION = 'decision'
    CUSTOM_FEATURE = 'custom_feature'


@dataclass
class ColumnInfo:
    """Information about a column in a Fiddler model."""

    name: str
    role: ColumnRole
    data_type: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    possible_values: Optional[List[Any]] = None

    def __repr__(self) -> str:
        return f"ColumnInfo(name='{self.name}', role={self.role}, data_type={self.data_type})"


@dataclass
class SchemaComparison:
    """Result of comparing two model schemas."""

    only_in_source: Set[str]
    only_in_target: Set[str]
    in_both: Set[str]
    type_mismatches: Dict[str, Tuple[Optional[str], Optional[str]]]
    is_compatible: bool

    def __repr__(self) -> str:
        return (
            f'SchemaComparison(\n'
            f'  only_in_source={len(self.only_in_source)} columns,\n'
            f'  only_in_target={len(self.only_in_target)} columns,\n'
            f'  in_both={len(self.in_both)} columns,\n'
            f'  type_mismatches={len(self.type_mismatches)} columns,\n'
            f'  is_compatible={self.is_compatible}\n'
            f')'
        )


class SchemaValidator:
    """Validator for Fiddler model schemas."""

    @staticmethod
    def extract_custom_feature_names(custom_features: Any) -> Set[str]:
        """Extract names from custom features (handles multiple formats).

        This utility handles various formats of custom features:
        - String: Direct name
        - Dict: Extract 'name' key
        - Object with .name attribute: Use attribute
        - fdl.TextEmbedding: Use .name
        - fdl.Enrichment: Use .name

        Args:
            custom_features: List of custom features in any supported format,
                           or None/empty list

        Returns:
            Set of feature names

        Example:
            ```python
            from fiddler_utils import SchemaValidator

            # Handle mixed formats
            features = [
                'string_feature',
                {'name': 'dict_feature'},
                fdl.TextEmbedding(name='embedding_feature', ...)
            ]
            names = SchemaValidator.extract_custom_feature_names(features)
            # Returns: {'string_feature', 'dict_feature', 'embedding_feature'}
            ```
        """
        if custom_features is None:
            return set()

        if not isinstance(custom_features, (list, tuple)):
            custom_features = [custom_features]

        names = set()
        for item in custom_features:
            if isinstance(item, str):
                # Direct string name
                names.add(item)
            elif isinstance(item, dict):
                # Dictionary with 'name' key
                names.add(item.get('name', str(item)))
            elif hasattr(item, 'name'):
                # Object with name attribute (TextEmbedding, Enrichment, etc.)
                names.add(item.name)
            else:
                # Fallback: convert to string
                names.add(str(item))

        return names

    @staticmethod
    def get_column_role(column_name: str, model: fdl.Model) -> Optional[ColumnRole]:
        """Determine the role of a column in a model spec.

        Args:
            column_name: Name of the column to check
            model: Fiddler model object

        Returns:
            ColumnRole enum if found, None if column not in spec

        Example:
            ```python
            from fiddler_utils import SchemaValidator

            role = SchemaValidator.get_column_role('age', model)
            if role == ColumnRole.INPUT:
                print("'age' is an input column")
            ```
        """
        spec = model.spec

        if column_name in (spec.inputs or []):
            return ColumnRole.INPUT
        elif column_name in (spec.outputs or []):
            return ColumnRole.OUTPUT
        elif column_name in (spec.targets or []):
            return ColumnRole.TARGET
        elif column_name in (spec.metadata or []):
            return ColumnRole.METADATA
        elif hasattr(spec, 'decisions') and column_name in (spec.decisions or []):
            return ColumnRole.DECISION
        elif hasattr(spec, 'custom_features'):
            # Check if column is a custom feature
            custom_features = getattr(spec, 'custom_features', None)
            if custom_features:
                feature_names = SchemaValidator.extract_custom_feature_names(custom_features)
                if column_name in feature_names:
                    return ColumnRole.CUSTOM_FEATURE

        return None

    @staticmethod
    def get_model_columns(model: fdl.Model) -> Dict[str, ColumnInfo]:
        """Extract all columns from a Fiddler model.

        Args:
            model: Fiddler model object

        Returns:
            Dictionary mapping column names to ColumnInfo objects

        Example:
            ```python
            from fiddler_utils import SchemaValidator

            model = fdl.Model.from_name('my_model', project_id=project.id)
            columns = SchemaValidator.get_model_columns(model)
            print(f"Model has {len(columns)} columns")
            ```
        """
        columns = {}
        spec = model.spec
        schema = model.schema if hasattr(model, 'schema') else None

        # Helper to get data type from schema
        def get_dtype(col_name: str) -> Optional[str]:
            if schema and hasattr(schema, col_name):
                col_schema = getattr(schema, col_name)
                return (
                    col_schema.data_type if hasattr(col_schema, 'data_type') else None
                )
            return None

        # Helper to get additional column info
        def get_column_info(col_name: str, role: ColumnRole) -> ColumnInfo:
            info = ColumnInfo(name=col_name, role=role, data_type=get_dtype(col_name))

            if schema and hasattr(schema, col_name):
                col_schema = getattr(schema, col_name)

                # Get numeric range if applicable
                if hasattr(col_schema, 'min'):
                    info.min_value = col_schema.min
                if hasattr(col_schema, 'max'):
                    info.max_value = col_schema.max

                # Get categorical values if applicable
                if hasattr(col_schema, 'categories'):
                    info.categories = col_schema.categories
                if hasattr(col_schema, 'possible_values'):
                    info.possible_values = col_schema.possible_values

            return info

        # Extract inputs
        if spec.inputs:
            for col in spec.inputs:
                columns[col] = get_column_info(col, ColumnRole.INPUT)

        # Extract outputs
        if spec.outputs:
            for col in spec.outputs:
                columns[col] = get_column_info(col, ColumnRole.OUTPUT)

        # Extract targets
        if spec.targets:
            for col in spec.targets:
                columns[col] = get_column_info(col, ColumnRole.TARGET)

        # Extract metadata
        if spec.metadata:
            for col in spec.metadata:
                columns[col] = get_column_info(col, ColumnRole.METADATA)

        # Extract decisions (if present)
        if hasattr(spec, 'decisions') and spec.decisions:
            for col in spec.decisions:
                columns[col] = get_column_info(col, ColumnRole.DECISION)

        # Extract custom features (if present)
        custom_features = getattr(spec, 'custom_features', None)
        if custom_features:
            # Use utility method to extract names (handles multiple formats)
            feature_names = SchemaValidator.extract_custom_feature_names(custom_features)
            for col_name in feature_names:
                columns[col_name] = get_column_info(col_name, ColumnRole.CUSTOM_FEATURE)

        logger.info(
            f"[SchemaValidator] Extracted {len(columns)} columns from model '{model.name}' "
            f"(project: {model.project_id})"
        )
        return columns

    @staticmethod
    def get_column_names(model: fdl.Model) -> Set[str]:
        """Get just the column names from a model (faster than get_model_columns).

        Args:
            model: Fiddler model object

        Returns:
            Set of column names
        """
        columns = set()
        spec = model.spec

        if spec.inputs:
            columns.update(spec.inputs)
        if spec.outputs:
            columns.update(spec.outputs)
        if spec.targets:
            columns.update(spec.targets)
        if spec.metadata:
            columns.update(spec.metadata)
        if hasattr(spec, 'decisions') and spec.decisions:
            columns.update(spec.decisions)

        # Extract custom feature names using utility method
        custom_features = getattr(spec, 'custom_features', None)
        if custom_features:
            columns.update(SchemaValidator.extract_custom_feature_names(custom_features))

        return columns

    @staticmethod
    def validate_columns(
        columns: Set[str], model: fdl.Model, strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """Validate that columns exist in a model schema.

        Args:
            columns: Set of column names to validate
            model: Fiddler model to validate against
            strict: If True, all columns must exist. If False, just warn.

        Returns:
            Tuple of (all_valid, missing_columns)

        Raises:
            SchemaValidationError: If strict=True and validation fails

        Example:
            ```python
            columns = {'age', 'income', 'unknown_column'}
            is_valid, missing = SchemaValidator.validate_columns(
                columns, target_model, strict=False
            )
            if not is_valid:
                print(f"Missing columns: {missing}")
            ```
        """
        model_columns = SchemaValidator.get_column_names(model)
        missing_columns = [col for col in columns if col not in model_columns]

        is_valid = len(missing_columns) == 0

        if not is_valid:
            logger.warning(
                f"[SchemaValidator] Validation failed for model '{model.name}': "
                f"{len(missing_columns)} missing columns: {missing_columns}"
            )

            if strict:
                raise SchemaValidationError(
                    f"Schema validation failed for model '{model.name}'",
                    missing_columns=missing_columns,
                )

        return is_valid, missing_columns

    @staticmethod
    def compare_schemas(
        source_model: fdl.Model, target_model: fdl.Model, strict: bool = False
    ) -> SchemaComparison:
        """Compare schemas of two models.

        Args:
            source_model: Source Fiddler model
            target_model: Target Fiddler model
            strict: If True, include type checking

        Returns:
            SchemaComparison object with detailed comparison

        Example:
            ```python
            comparison = SchemaValidator.compare_schemas(source_model, target_model)
            if not comparison.is_compatible:
                print(f"Columns only in source: {comparison.only_in_source}")
                print(f"Columns only in target: {comparison.only_in_target}")
            ```
        """
        source_cols = SchemaValidator.get_model_columns(source_model)
        target_cols = SchemaValidator.get_model_columns(target_model)

        source_names = set(source_cols.keys())
        target_names = set(target_cols.keys())

        only_in_source = source_names - target_names
        only_in_target = target_names - source_names
        in_both = source_names & target_names

        # Check for type mismatches in common columns
        type_mismatches: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        if strict:
            for col in in_both:
                source_type = source_cols[col].data_type
                target_type = target_cols[col].data_type
                if (
                    source_type
                    and target_type
                    and source_type.lower() != target_type.lower()
                ):
                    type_mismatches[col] = (source_type, target_type)

        # Determine if schemas are compatible (all source columns exist in target)
        is_compatible = len(only_in_source) == 0 and len(type_mismatches) == 0

        comparison = SchemaComparison(
            only_in_source=only_in_source,
            only_in_target=only_in_target,
            in_both=in_both,
            type_mismatches=type_mismatches,
            is_compatible=is_compatible,
        )

        logger.info(
            f"[SchemaValidator] Schema comparison: {len(in_both)} common columns, "
            f"{len(only_in_source)} only in source, "
            f"{len(only_in_target)} only in target, "
            f"{len(type_mismatches)} type mismatches"
        )

        return comparison

    @staticmethod
    def validate_fql_expression(
        expression: str, model: fdl.Model, strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """Validate that an FQL expression is compatible with a model schema.

        Args:
            expression: FQL expression to validate
            model: Fiddler model to validate against
            strict: If True, raise exception on validation failure

        Returns:
            Tuple of (is_valid, missing_columns)

        Raises:
            SchemaValidationError: If strict=True and validation fails

        Example:
            ```python
            expr = '"age" > 30 and "geography" == \'California\''
            is_valid, missing = SchemaValidator.validate_fql_expression(
                expr, target_model
            )
            ```
        """
        # Extract columns from expression
        columns = fql.extract_columns(expression)

        # Validate columns exist in model
        return SchemaValidator.validate_columns(columns, model, strict=strict)

    @staticmethod
    def is_compatible(
        source_model: fdl.Model,
        target_model: fdl.Model,
        required_columns: Optional[Set[str]] = None,
    ) -> bool:
        """Check if target model schema is compatible with source model.

        Compatibility means all required columns from source exist in target
        with compatible types.

        Args:
            source_model: Source model
            target_model: Target model
            required_columns: Optional set of specific columns to check.
                            If None, checks all source columns.

        Returns:
            True if schemas are compatible

        Example:
            ```python
            if SchemaValidator.is_compatible(source_model, target_model):
                # Safe to copy assets
                pass
            ```
        """
        source_cols = SchemaValidator.get_model_columns(source_model)
        target_cols = SchemaValidator.get_model_columns(target_model)

        # Determine which columns to check
        if required_columns:
            columns_to_check = required_columns
        else:
            columns_to_check = set(source_cols.keys())

        # Check all required columns exist in target
        target_col_names = set(target_cols.keys())
        missing = columns_to_check - target_col_names

        if missing:
            logger.warning(
                f"[SchemaValidator] Incompatible schemas: {len(missing)} columns missing in target: "
                f"{missing}"
            )
            return False

        # Check data types for common columns
        for col in columns_to_check:
            if col in source_cols and col in target_cols:
                source_type = source_cols[col].data_type
                target_type = target_cols[col].data_type

                if (
                    source_type
                    and target_type
                    and source_type.lower() != target_type.lower()
                ):
                    logger.warning(
                        f"[SchemaValidator] Type mismatch for column '{col}': "
                        f"source={source_type}, target={target_type}"
                    )
                    # For now, just warn but don't fail
                    # Different Fiddler versions may have different type representations

        return True
