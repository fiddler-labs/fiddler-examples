"""Tests for comparison utilities (Phases 1-5 changes)."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fiddler_utils.comparison import (
    ValueDifference,
    ComparisonConfig,
    ModelComparator,
    ConfigurationComparison,
    SpecComparison,
    AssetComparison,
    ComparisonResult,
)


class TestValueDifference:
    """Tests for ValueDifference dataclass (Phase 2.3)."""

    def test_basic_creation(self):
        """Test creating ValueDifference with basic values."""
        diff = ValueDifference(source_value="foo", target_value="bar")
        assert diff.source_value == "foo"
        assert diff.target_value == "bar"
        assert diff.context is None

    def test_with_context(self):
        """Test creating ValueDifference with context."""
        diff = ValueDifference(
            source_value=10,
            target_value=20,
            context="batch_size"
        )
        assert diff.source_value == 10
        assert diff.target_value == 20
        assert diff.context == "batch_size"

    def test_str_without_context(self):
        """Test string representation without context."""
        diff = ValueDifference(source_value="a", target_value="b")
        assert str(diff) == "a → b"

    def test_str_with_context(self):
        """Test string representation with context."""
        diff = ValueDifference(
            source_value="LLM",
            target_value="BINARY_CLASSIFICATION",
            context="task"
        )
        assert str(diff) == "task: LLM → BINARY_CLASSIFICATION"

    def test_none_values(self):
        """Test ValueDifference handles None values."""
        diff = ValueDifference(source_value=None, target_value="value")
        assert diff.source_value is None
        assert diff.target_value == "value"
        assert str(diff) == "None → value"


class TestComparisonConfig:
    """Tests for ComparisonConfig dataclass (Phase 5.2)."""

    def test_default_all_true(self):
        """Test default config has all comparisons enabled."""
        config = ComparisonConfig()
        assert config.include_configuration is True
        assert config.include_schema is True
        assert config.include_spec is True
        assert config.include_segments is True
        assert config.include_custom_metrics is True
        assert config.include_alerts is True
        assert config.include_baselines is True
        assert config.include_charts is True

    def test_custom_config(self):
        """Test creating custom config."""
        config = ComparisonConfig(
            include_configuration=True,
            include_schema=True,
            include_spec=False,
            include_segments=False,
            include_custom_metrics=False,
            include_alerts=False,
            include_baselines=False,
            include_charts=False
        )
        assert config.include_configuration is True
        assert config.include_schema is True
        assert config.include_spec is False
        assert config.include_segments is False

    def test_all_factory_method(self):
        """Test ComparisonConfig.all() factory method."""
        config = ComparisonConfig.all()
        assert config.include_configuration is True
        assert config.include_schema is True
        assert config.include_spec is True
        assert config.include_segments is True
        assert config.include_custom_metrics is True
        assert config.include_alerts is True
        assert config.include_baselines is True
        assert config.include_charts is True

    def test_schema_only_factory_method(self):
        """Test ComparisonConfig.schema_only() factory method."""
        config = ComparisonConfig.schema_only()
        assert config.include_configuration is False
        assert config.include_schema is True
        assert config.include_spec is False
        assert config.include_segments is False
        assert config.include_custom_metrics is False
        assert config.include_alerts is False
        assert config.include_baselines is False
        assert config.include_charts is False

    def test_no_assets_factory_method(self):
        """Test ComparisonConfig.no_assets() factory method."""
        config = ComparisonConfig.no_assets()
        assert config.include_configuration is True
        assert config.include_schema is True
        assert config.include_spec is True
        assert config.include_segments is False
        assert config.include_custom_metrics is False
        assert config.include_alerts is False
        assert config.include_baselines is False
        assert config.include_charts is False


class TestModelComparatorGetAssetKey:
    """Tests for ModelComparator._get_asset_key() (Phase 4.2)."""

    def test_asset_with_name(self):
        """Test extracting key from asset with name attribute."""
        asset = Mock()
        asset.name = "my_segment"

        key = ModelComparator._get_asset_key(asset)
        assert key == "my_segment"

    def test_asset_with_name_as_number(self):
        """Test extracting key from asset with numeric name."""
        asset = Mock()
        asset.name = 123

        key = ModelComparator._get_asset_key(asset)
        assert key == "123"
        assert isinstance(key, str)

    def test_asset_with_none_name_fallback_to_id(self):
        """Test fallback to id when name is None."""
        asset = Mock()
        asset.name = None
        asset.id = "asset-uuid-123"

        key = ModelComparator._get_asset_key(asset)
        assert key == "asset-uuid-123"

    def test_asset_without_name_fallback_to_id(self):
        """Test fallback to id when name attribute doesn't exist."""
        asset = Mock(spec=['id'])  # Only has 'id', no 'name'
        asset.id = "asset-uuid-456"

        key = ModelComparator._get_asset_key(asset)
        assert key == "asset-uuid-456"

    def test_asset_with_neither_name_nor_id_raises_error(self):
        """Test error when asset has neither name nor id."""
        asset = Mock(spec=[])  # No attributes

        with pytest.raises(ValueError, match="no usable key"):
            ModelComparator._get_asset_key(asset)

    def test_asset_with_both_none_raises_error(self):
        """Test error when both name and id are None."""
        asset = Mock()
        asset.name = None
        asset.id = None

        with pytest.raises(ValueError, match="no usable key"):
            ModelComparator._get_asset_key(asset)


class TestConfigurationComparison:
    """Tests for ConfigurationComparison dataclass."""

    def test_no_differences(self):
        """Test has_differences returns False when no differences."""
        comp = ConfigurationComparison()
        assert comp.has_differences() is False

    def test_with_differences(self):
        """Test has_differences returns True when differences exist."""
        comp = ConfigurationComparison()
        comp.differences['task'] = ValueDifference('LLM', 'BINARY')
        assert comp.has_differences() is True


class TestSpecComparison:
    """Tests for SpecComparison dataclass."""

    def test_no_differences_all_match(self):
        """Test has_differences when all attributes match."""
        comp = SpecComparison(
            inputs_match=True,
            outputs_match=True,
            targets_match=True,
            decisions_match=True,
            metadata_match=True,
            custom_features_match=True
        )
        assert comp.has_differences() is False

    def test_has_differences_when_inputs_dont_match(self):
        """Test has_differences when inputs don't match."""
        comp = SpecComparison(
            inputs_match=False,
            outputs_match=True,
            targets_match=True,
            decisions_match=True,
            metadata_match=True,
            custom_features_match=True
        )
        assert comp.has_differences() is True


class TestAssetComparison:
    """Tests for AssetComparison dataclass."""

    def test_no_differences(self):
        """Test asset comparison with no differences."""
        comp = AssetComparison(asset_type='segments')
        assert comp.has_differences() is False
        assert comp.total_differences == 0

    def test_only_in_source(self):
        """Test asset comparison with items only in source."""
        comp = AssetComparison(
            asset_type='segments',
            only_in_source=['seg1', 'seg2']
        )
        assert comp.has_differences() is True
        assert comp.total_differences == 2

    def test_only_in_target(self):
        """Test asset comparison with items only in target."""
        comp = AssetComparison(
            asset_type='custom_metrics',
            only_in_target=['metric1']
        )
        assert comp.has_differences() is True
        assert comp.total_differences == 1

    def test_definition_differences(self):
        """Test asset comparison with definition differences."""
        comp = AssetComparison(asset_type='segments')
        comp.definition_differences['seg1'] = ValueDifference(
            'age > 30',
            'age > 40'
        )
        assert comp.has_differences() is True
        assert comp.total_differences == 1

    def test_total_differences_combined(self):
        """Test total_differences combines all difference types."""
        comp = AssetComparison(
            asset_type='alerts',
            only_in_source=['alert1', 'alert2'],
            only_in_target=['alert3']
        )
        comp.definition_differences['alert4'] = ValueDifference('a', 'b')

        assert comp.total_differences == 4  # 2 + 1 + 1


class TestComparisonResultMethods:
    """Tests for ComparisonResult methods."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = ComparisonResult(
            model_a_name='model_a',
            model_b_name='model_b'
        )

        data = result.to_dict()
        assert data['model_a_name'] == 'model_a'
        assert data['model_b_name'] == 'model_b'
        assert 'compared_at' in data

    def test_to_json_without_filepath(self):
        """Test to_json returns JSON string without saving."""
        result = ComparisonResult(
            model_a_name='model_a',
            model_b_name='model_b'
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert 'model_a_name' in json_str
        assert 'model_b' in json_str

    def test_to_json_custom_indent(self):
        """Test to_json with custom indentation."""
        result = ComparisonResult(
            model_a_name='test1',
            model_b_name='test2'
        )

        json_str = result.to_json(indent=4)
        assert isinstance(json_str, str)
        # With indent=4, should have more spaces
        assert '    ' in json_str


class TestCompareCharts:
    """Tests for ModelComparator.compare_charts() (Bug fix for fdl.Chart)."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        model_a = Mock()
        model_a.name = 'model_a'
        model_a.id = 'id_a'
        model_a.project_id = 'project_123'

        model_b = Mock()
        model_b.name = 'model_b'
        model_b.id = 'id_b'
        model_b.project_id = 'project_123'  # Same project

        return model_a, model_b

    def test_compare_charts_same_project(self, mock_models):
        """Test compare_charts with models in same project."""
        model_a, model_b = mock_models
        comparator = ModelComparator(model_a, model_b)

        result = comparator.compare_charts()

        # Should return empty AssetComparison (no differences)
        assert isinstance(result, AssetComparison)
        assert result.asset_type == 'charts'
        assert result.has_differences() is False
        assert result.total_differences == 0

    def test_compare_charts_different_projects(self, mock_models):
        """Test compare_charts with models in different projects."""
        model_a, model_b = mock_models
        model_b.project_id = 'project_456'  # Different project

        comparator = ModelComparator(model_a, model_b)

        result = comparator.compare_charts()

        # Should still return empty AssetComparison without crashing
        assert isinstance(result, AssetComparison)
        assert result.asset_type == 'charts'
        assert result.has_differences() is False


class TestModelComparatorCompareAllBackwardCompatibility:
    """Tests for compare_all() backward compatibility (Phase 5.2)."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        model_a = Mock()
        model_a.name = 'model_a'
        model_a.id = 'id_a'

        model_b = Mock()
        model_b.name = 'model_b'
        model_b.id = 'id_b'

        return model_a, model_b

    def test_compare_all_with_config_object(self, mock_models):
        """Test compare_all accepts ComparisonConfig object."""
        from fiddler_utils.schema import SchemaComparison

        model_a, model_b = mock_models
        comparator = ModelComparator(model_a, model_b)

        config = ComparisonConfig.schema_only()

        # Create proper SchemaComparison object
        mock_schema_comparison = SchemaComparison(
            only_in_source=set(),
            only_in_target=set(),
            in_both=set(),
            type_mismatches={},
            is_compatible=True
        )

        # Mock the comparison methods
        with patch.object(comparator, 'compare_schemas', return_value=mock_schema_comparison):
            result = comparator.compare_all(config=config)

            # Should have called compare_schemas
            comparator.compare_schemas.assert_called_once()
            assert result is not None

    def test_compare_all_with_old_boolean_params(self, mock_models):
        """Test compare_all still works with old boolean parameters."""
        model_a, model_b = mock_models
        comparator = ModelComparator(model_a, model_b)

        # Mock the comparison methods
        with patch.object(comparator, 'compare_configuration', return_value=Mock()):
            with patch.object(comparator, 'compare_schemas', return_value=Mock()):
                result = comparator.compare_all(
                    include_configuration=True,
                    include_schema=True,
                    include_spec=False,
                    include_segments=False,
                    include_custom_metrics=False,
                    include_alerts=False,
                    include_baselines=False,
                    include_charts=False
                )

                # Should have called the enabled comparisons
                comparator.compare_configuration.assert_called_once()
                comparator.compare_schemas.assert_called_once()
                assert result is not None

    def test_compare_all_default_behavior(self, mock_models):
        """Test compare_all with no parameters uses defaults."""
        model_a, model_b = mock_models
        comparator = ModelComparator(model_a, model_b)

        # Mock all comparison methods
        for method in ['compare_configuration', 'compare_schemas', 'compare_specs',
                      'compare_segments', 'compare_custom_metrics', 'compare_alerts',
                      'compare_baselines', 'compare_charts']:
            setattr(comparator, method, Mock(return_value=Mock()))

        result = comparator.compare_all()

        # All methods should have been called
        comparator.compare_configuration.assert_called_once()
        comparator.compare_schemas.assert_called_once()
        comparator.compare_specs.assert_called_once()
        assert result is not None
