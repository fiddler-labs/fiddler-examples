"""Tests for schema validation utilities (Phase 3 changes)."""

import pytest
from unittest.mock import Mock
from fiddler_utils.schema import SchemaValidator, ColumnRole


class TestExtractCustomFeatureNames:
    """Tests for SchemaValidator.extract_custom_feature_names() (Phase 3.1)."""

    def test_none_input(self):
        """Test with None input returns empty set."""
        result = SchemaValidator.extract_custom_feature_names(None)
        assert result == set()

    def test_empty_list(self):
        """Test with empty list returns empty set."""
        result = SchemaValidator.extract_custom_feature_names([])
        assert result == set()

    def test_string_names(self):
        """Test with list of string names."""
        features = ['feature1', 'feature2', 'feature3']
        result = SchemaValidator.extract_custom_feature_names(features)
        assert result == {'feature1', 'feature2', 'feature3'}

    def test_dict_with_name_key(self):
        """Test with list of dicts containing 'name' key."""
        features = [
            {'name': 'feature1', 'type': 'embedding'},
            {'name': 'feature2', 'dimension': 128}
        ]
        result = SchemaValidator.extract_custom_feature_names(features)
        assert result == {'feature1', 'feature2'}

    def test_dict_without_name_key(self):
        """Test with dict missing 'name' key falls back to str()."""
        features = [
            {'type': 'embedding', 'dim': 64}
        ]
        result = SchemaValidator.extract_custom_feature_names(features)
        # Should convert dict to string representation
        assert len(result) == 1
        assert isinstance(list(result)[0], str)

    def test_objects_with_name_attribute(self):
        """Test with objects having .name attribute."""
        obj1 = Mock()
        obj1.name = 'text_embedding'

        obj2 = Mock()
        obj2.name = 'enrichment_feature'

        features = [obj1, obj2]
        result = SchemaValidator.extract_custom_feature_names(features)
        assert result == {'text_embedding', 'enrichment_feature'}

    def test_mixed_formats(self):
        """Test with mixed format custom features."""
        obj = Mock()
        obj.name = 'obj_feature'

        features = [
            'string_feature',
            {'name': 'dict_feature'},
            obj
        ]
        result = SchemaValidator.extract_custom_feature_names(features)
        assert result == {'string_feature', 'dict_feature', 'obj_feature'}

    def test_single_feature_not_in_list(self):
        """Test with single feature (not in a list)."""
        result = SchemaValidator.extract_custom_feature_names('single_feature')
        assert result == {'single_feature'}

    def test_object_without_name_falls_back_to_str(self):
        """Test object without name attribute uses str() representation."""
        class NoNameObject:
            def __str__(self):
                return 'fallback_name'

        obj = NoNameObject()
        result = SchemaValidator.extract_custom_feature_names([obj])
        assert 'fallback_name' in result

    def test_deduplication(self):
        """Test that duplicate names are deduplicated."""
        features = ['feature1', 'feature2', 'feature1', 'feature2']
        result = SchemaValidator.extract_custom_feature_names(features)
        assert result == {'feature1', 'feature2'}
        assert len(result) == 2


class TestGetColumnRole:
    """Tests for SchemaValidator.get_column_role() (Phase 3.3)."""

    def test_input_column(self):
        """Test identifying input column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age', 'income']
        model.spec.outputs = ['prediction']
        model.spec.targets = ['label']
        model.spec.metadata = []
        model.spec.decisions = None

        role = SchemaValidator.get_column_role('age', model)
        assert role == ColumnRole.INPUT

    def test_output_column(self):
        """Test identifying output column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction', 'confidence']
        model.spec.targets = []
        model.spec.metadata = []

        role = SchemaValidator.get_column_role('confidence', model)
        assert role == ColumnRole.OUTPUT

    def test_target_column(self):
        """Test identifying target column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = ['actual_label']
        model.spec.metadata = []

        role = SchemaValidator.get_column_role('actual_label', model)
        assert role == ColumnRole.TARGET

    def test_metadata_column(self):
        """Test identifying metadata column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = ['user_id', 'timestamp']

        role = SchemaValidator.get_column_role('user_id', model)
        assert role == ColumnRole.METADATA

    def test_decision_column(self):
        """Test identifying decision column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = []
        model.spec.decisions = ['final_decision']

        role = SchemaValidator.get_column_role('final_decision', model)
        assert role == ColumnRole.DECISION

    def test_custom_feature_column(self):
        """Test identifying custom feature column."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = []
        model.spec.decisions = []
        model.spec.custom_features = [
            {'name': 'text_embedding'},
            {'name': 'sentiment_score'}
        ]

        role = SchemaValidator.get_column_role('text_embedding', model)
        assert role == ColumnRole.CUSTOM_FEATURE

    def test_unknown_column(self):
        """Test column not in any role returns None."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = []
        model.spec.decisions = []

        role = SchemaValidator.get_column_role('unknown_column', model)
        assert role is None

    def test_none_lists_handled(self):
        """Test handling of None lists in spec."""
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = None
        model.spec.outputs = None
        model.spec.targets = None
        model.spec.metadata = None
        model.spec.decisions = None

        role = SchemaValidator.get_column_role('any_column', model)
        assert role is None

    def test_missing_decisions_attribute(self):
        """Test model spec without decisions attribute."""
        model = Mock()
        model.spec = Mock(spec=['inputs', 'outputs', 'targets', 'metadata'])
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = []

        # Should not raise error, just return None for non-existent column
        role = SchemaValidator.get_column_role('unknown', model)
        assert role is None

    def test_missing_custom_features_attribute(self):
        """Test model spec without custom_features attribute."""
        model = Mock()
        model.spec = Mock(spec=['inputs', 'outputs', 'targets', 'metadata'])
        model.spec.inputs = ['age']
        model.spec.outputs = ['prediction']
        model.spec.targets = []
        model.spec.metadata = []

        role = SchemaValidator.get_column_role('unknown', model)
        assert role is None

    def test_priority_order(self):
        """Test that columns are checked in priority order (inputs first)."""
        # This tests that if a column appears in multiple lists,
        # the first match in priority order is returned
        model = Mock()
        model.spec = Mock()
        model.spec.inputs = ['shared_column']
        model.spec.outputs = ['shared_column']  # Also in outputs (shouldn't happen, but test it)
        model.spec.targets = []
        model.spec.metadata = []

        role = SchemaValidator.get_column_role('shared_column', model)
        # Should return INPUT since that's checked first
        assert role == ColumnRole.INPUT


class TestColumnRole:
    """Tests for ColumnRole enum."""

    def test_enum_values(self):
        """Test ColumnRole enum has expected values."""
        assert ColumnRole.INPUT.value == 'input'
        assert ColumnRole.OUTPUT.value == 'output'
        assert ColumnRole.TARGET.value == 'target'
        assert ColumnRole.METADATA.value == 'metadata'
        assert ColumnRole.DECISION.value == 'decision'
        assert ColumnRole.CUSTOM_FEATURE.value == 'custom_feature'

    def test_enum_can_be_compared(self):
        """Test ColumnRole enum values can be compared."""
        role1 = ColumnRole.INPUT
        role2 = ColumnRole.INPUT
        role3 = ColumnRole.OUTPUT

        assert role1 == role2
        assert role1 != role3

    def test_enum_can_be_used_in_conditionals(self):
        """Test ColumnRole can be used in if statements."""
        role = ColumnRole.INPUT

        if role == ColumnRole.INPUT:
            result = 'input'
        else:
            result = 'other'

        assert result == 'input'
