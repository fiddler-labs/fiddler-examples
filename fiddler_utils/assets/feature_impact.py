"""Feature Impact management utilities for Fiddler models.

This module provides FeatureImpactManager for loading, validating, and uploading
user-defined feature impact (importance) scores to Fiddler models.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Set
import logging
import pandas as pd
import json

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

from ..exceptions import ValidationError, FiddlerUtilsError

logger = logging.getLogger(__name__)


class FeatureImpactManager:
    """Manager for feature impact (importance) scores.

    This class provides utilities to load, validate, and upload user-defined
    global feature impact scores to Fiddler models. Feature impact scores enable
    model explainability without requiring a model artifact.

    Example:
        ```python
        from fiddler_utils import FeatureImpactManager
        import fiddler as fdl

        # Initialize manager
        fi_mgr = FeatureImpactManager()

        # Load from CSV
        impact_dict = fi_mgr.load_from_csv(
            filepath='feature_impact.csv',
            feature_col='feature_name',
            impact_col='impact_score'
        )

        # Validate against model
        fi_mgr.validate_features(
            model=model,
            feature_impact_dict=impact_dict,
            remove_extra=True
        )

        # Upload to Fiddler
        model.upload_feature_impact(
            feature_impact_map=impact_dict,
            update=True
        )
        ```
    """

    @staticmethod
    def load_from_csv(
        filepath: str,
        feature_col: str = 'feature_name',
        impact_col: str = 'feature_impact'
    ) -> Dict[str, float]:
        """Load feature impact scores from CSV file.

        Args:
            filepath: Path to CSV file
            feature_col: Name of column containing feature names
            impact_col: Name of column containing impact scores

        Returns:
            Dictionary mapping feature names to impact scores

        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If required columns missing or values non-numeric

        Example:
            ```python
            # CSV format:
            # feature_name,feature_impact
            # age,0.45
            # income,0.32
            # ...

            impact_dict = fi_mgr.load_from_csv(
                filepath='impacts.csv',
                feature_col='feature_name',
                impact_col='feature_impact'
            )
            ```
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        except Exception as e:
            raise FiddlerUtilsError(f"Failed to read CSV file: {str(e)}")

        # Validate columns exist
        if feature_col not in df.columns:
            raise ValueError(
                f"Feature column '{feature_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        if impact_col not in df.columns:
            raise ValueError(
                f"Impact column '{impact_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Convert to dict
        impact_series = df.set_index(feature_col)[impact_col]

        # Validate numeric values
        try:
            impact_dict = impact_series.astype(float).to_dict()
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Impact values in column '{impact_col}' must be numeric. Error: {str(e)}"
            )

        logger.info(f"Loaded {len(impact_dict)} feature impact scores from {filepath}")
        return impact_dict

    @staticmethod
    def load_from_json(filepath: str) -> Dict[str, float]:
        """Load feature impact scores from JSON file.

        The JSON file should contain a dictionary mapping feature names to scores,
        or a pandas Series representation.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary mapping feature names to impact scores

        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If JSON format invalid or values non-numeric

        Example:
            ```python
            # JSON format (dict):
            # {
            #   "age": 0.45,
            #   "income": 0.32,
            #   ...
            # }

            impact_dict = fi_mgr.load_from_json('impacts.json')
            ```
        """
        try:
            # Try pandas Series format first (used in quickstart example)
            impact_series = pd.read_json(filepath, typ='series')
            impact_dict = impact_series.to_dict()
        except:
            # Fall back to plain JSON dict
            try:
                with open(filepath, 'r') as f:
                    impact_dict = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"JSON file not found: {filepath}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}")

        # Validate it's a dict
        if not isinstance(impact_dict, dict):
            raise ValueError(
                f"JSON must contain a dictionary, got {type(impact_dict)}"
            )

        # Validate numeric values
        for feature, value in impact_dict.items():
            try:
                impact_dict[feature] = float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Impact value for feature '{feature}' must be numeric, got {value}"
                )

        logger.info(f"Loaded {len(impact_dict)} feature impact scores from {filepath}")
        return impact_dict

    @staticmethod
    def load_from_dict(data: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize feature impact dictionary.

        Args:
            data: Dictionary mapping feature names to impact scores

        Returns:
            Validated dictionary with float values

        Raises:
            ValueError: If input is not a dict or values non-numeric

        Example:
            ```python
            raw_data = {'age': 0.45, 'income': '0.32'}
            impact_dict = fi_mgr.load_from_dict(raw_data)
            # Returns: {'age': 0.45, 'income': 0.32}
            ```
        """
        if not isinstance(data, dict):
            raise ValueError(f"Input must be a dictionary, got {type(data)}")

        # Validate and convert to float
        impact_dict = {}
        for feature, value in data.items():
            try:
                impact_dict[feature] = float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Impact value for feature '{feature}' must be numeric, got {value}"
                )

        logger.info(f"Validated {len(impact_dict)} feature impact scores")
        return impact_dict

    @staticmethod
    def validate_features(
        model: fdl.Model,
        feature_impact_dict: Dict[str, float],
        remove_extra: bool = False,
        strict: bool = True
    ) -> Tuple[bool, Set[str], Set[str]]:
        """Validate feature impact dictionary against model spec.

        Ensures all model input features have impact scores. Optionally removes
        extra features not in the model spec.

        Args:
            model: Fiddler Model object
            feature_impact_dict: Dictionary of feature name -> impact score
            remove_extra: If True, remove features not in model.spec.inputs
            strict: If True, raise exception on validation failure

        Returns:
            Tuple of (is_valid, missing_features, extra_features)

        Raises:
            ValidationError: If strict=True and validation fails

        Example:
            ```python
            is_valid, missing, extra = fi_mgr.validate_features(
                model=model,
                feature_impact_dict=impact_dict,
                remove_extra=True  # Auto-clean extra features
            )

            if not is_valid:
                print(f"Missing features: {missing}")
                print(f"Extra features: {extra}")
            ```
        """
        # Get model input features
        model_inputs = set(model.spec.inputs or [])
        impact_features = set(feature_impact_dict.keys())

        # Check for missing features (REQUIRED per user requirements)
        missing_features = model_inputs - impact_features
        extra_features = impact_features - model_inputs

        is_valid = len(missing_features) == 0

        # Log findings
        if missing_features:
            logger.warning(
                f"[FeatureImpactManager] {len(missing_features)} model input features "
                f"missing from impact map: {sorted(missing_features)}"
            )

        if extra_features:
            logger.warning(
                f"[FeatureImpactManager] {len(extra_features)} features in impact map "
                f"not found in model inputs: {sorted(extra_features)}"
            )

        # Remove extra features if requested
        if remove_extra and extra_features:
            for feature in extra_features:
                del feature_impact_dict[feature]
            logger.info(
                f"[FeatureImpactManager] Removed {len(extra_features)} extra features "
                f"from impact map"
            )

        # Raise exception if strict and invalid
        if strict and not is_valid:
            raise ValidationError(
                f"Feature impact validation failed for model '{model.name}': "
                f"{len(missing_features)} required features missing. "
                f"Missing features: {sorted(missing_features)}"
            )

        return is_valid, missing_features, extra_features

    @staticmethod
    def upload_feature_impact(
        model: fdl.Model,
        feature_impact_map: Dict[str, float],
        update: Optional[bool] = None,
        validate: bool = True,
        remove_extra: bool = True
    ) -> Dict[str, float]:
        """Upload feature impact scores to Fiddler model with validation.

        Convenience wrapper around model.upload_feature_impact() that adds
        validation, error handling, and smart create-or-update logic.

        Args:
            model: Fiddler Model object
            feature_impact_map: Dictionary of feature name -> impact score
            update: Upload mode:
                - None (default): Auto-detect - try create first, update if exists
                - True: Force update existing values
                - False: Force create (fail if values already exist)
            validate: If True, validate features against model spec before upload
            remove_extra: If True and validate=True, remove extra features

        Returns:
            Uploaded feature impact dictionary (after any modifications)

        Raises:
            ValidationError: If validation fails
            FiddlerUtilsError: If upload fails

        Example:
            ```python
            # Auto-detect mode (recommended)
            result = fi_mgr.upload_feature_impact(
                model=model,
                feature_impact_map=impact_dict
            )

            # Force update mode
            result = fi_mgr.upload_feature_impact(
                model=model,
                feature_impact_map=impact_dict,
                update=True
            )
            ```
        """
        # Validate if requested
        if validate:
            logger.info(
                f"[FeatureImpactManager] Validating {len(feature_impact_map)} "
                f"feature impact scores for model '{model.name}'"
            )
            is_valid, missing, extra = FeatureImpactManager.validate_features(
                model=model,
                feature_impact_dict=feature_impact_map,
                remove_extra=remove_extra,
                strict=True  # Always strict when validating before upload
            )

        # Auto-detect mode: try create first, fall back to update
        if update is None:
            logger.info(
                f"[FeatureImpactManager] Auto-detect mode: trying create first..."
            )
            try:
                result = model.upload_feature_impact(
                    feature_impact_map=feature_impact_map,
                    update=False
                )
                logger.info(
                    f"[FeatureImpactManager] Created {len(result)} feature impact scores "
                    f"for model '{model.name}'"
                )
                return result
            except Exception as e:
                error_msg = str(e)
                # Check if error is because values already exist
                if "cannot update" in error_msg.lower() or "does not exist" in error_msg.lower():
                    # Values don't exist yet, but we got "cannot update" error
                    # This means we tried update=False and values don't exist - unexpected
                    raise FiddlerUtilsError(
                        f"Failed to create feature impact for model '{model.name}': {error_msg}"
                    )
                elif "already exists" in error_msg.lower() or "cannot create" in error_msg.lower():
                    # Values already exist, try updating
                    logger.info(
                        f"[FeatureImpactManager] Values already exist, updating instead..."
                    )
                    try:
                        result = model.upload_feature_impact(
                            feature_impact_map=feature_impact_map,
                            update=True
                        )
                        logger.info(
                            f"[FeatureImpactManager] Updated {len(result)} feature impact scores "
                            f"for model '{model.name}'"
                        )
                        return result
                    except Exception as update_error:
                        raise FiddlerUtilsError(
                            f"Failed to update feature impact for model '{model.name}': {str(update_error)}"
                        )
                else:
                    # Different error
                    raise FiddlerUtilsError(
                        f"Failed to upload feature impact to model '{model.name}': {error_msg}"
                    )

        # Explicit mode (update=True or update=False)
        else:
            action = "Updating" if update else "Creating"
            logger.info(
                f"[FeatureImpactManager] {action} {len(feature_impact_map)} "
                f"feature impact scores (update={update})"
            )
            try:
                result = model.upload_feature_impact(
                    feature_impact_map=feature_impact_map,
                    update=update
                )
                action_past = "Updated" if update else "Created"
                logger.info(
                    f"[FeatureImpactManager] {action_past} {len(result)} feature impact scores "
                    f"for model '{model.name}'"
                )
                return result
            except Exception as e:
                raise FiddlerUtilsError(
                    f"Failed to upload feature impact to model '{model.name}': {str(e)}"
                )

    @staticmethod
    def create_uniform_impact(
        model: fdl.Model,
        default_value: float = 1.0
    ) -> Dict[str, float]:
        """Create uniform feature impact scores for all model inputs.

        Useful for initializing impact scores or testing.

        Args:
            model: Fiddler Model object
            default_value: Impact score to assign to all features

        Returns:
            Dictionary mapping all input features to default_value

        Example:
            ```python
            # Create uniform scores for all features
            uniform_impact = fi_mgr.create_uniform_impact(
                model=model,
                default_value=1.0
            )
            # Returns: {'age': 1.0, 'income': 1.0, ...}
            ```
        """
        model_inputs = model.spec.inputs or []
        impact_dict = {feature: float(default_value) for feature in model_inputs}

        logger.info(
            f"[FeatureImpactManager] Created uniform impact scores "
            f"({default_value}) for {len(impact_dict)} features"
        )
        return impact_dict
