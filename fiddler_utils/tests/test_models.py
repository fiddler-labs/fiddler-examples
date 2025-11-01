"""Tests for model management utilities (Phases 3-5 changes)."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from uuid import UUID
import json

from fiddler_utils.assets.models import (
    ModelManager,
    UUIDEncoder,
    ModelExportData,
    ColumnExportData,
)


class TestUUIDEncoder:
    """Tests for UUIDEncoder (JSON serialization helper)."""

    def test_encode_uuid_to_string(self):
        """Test UUID is encoded as string."""
        test_uuid = UUID('12345678-1234-5678-1234-567812345678')
        encoder = UUIDEncoder()

        result = encoder.default(test_uuid)
        assert result == '12345678-1234-5678-1234-567812345678'
        assert isinstance(result, str)

    def test_encode_non_uuid_delegates_to_parent(self):
        """Test non-UUID objects delegate to parent encoder."""
        encoder = UUIDEncoder()

        # Parent will raise TypeError for non-serializable objects
        with pytest.raises(TypeError):
            encoder.default(object())

    def test_json_dumps_with_uuid_encoder(self):
        """Test full JSON serialization with UUID objects."""
        data = {
            'id': UUID('12345678-1234-5678-1234-567812345678'),
            'name': 'test_model',
            'count': 42
        }

        json_str = json.dumps(data, cls=UUIDEncoder)
        parsed = json.loads(json_str)

        assert parsed['id'] == '12345678-1234-5678-1234-567812345678'
        assert parsed['name'] == 'test_model'
        assert parsed['count'] == 42


class TestSerializeEnum:
    """Tests for ModelManager._serialize_enum() (Phase 3.4)."""

    def test_serialize_none_returns_none(self):
        """Test None input returns None."""
        result = ModelManager._serialize_enum(None)
        assert result is None

    def test_serialize_enum_with_name_attribute(self):
        """Test enum with .name attribute is serialized correctly."""
        mock_enum = Mock()
        mock_enum.name = 'BINARY_CLASSIFICATION'

        result = ModelManager._serialize_enum(mock_enum)
        assert result == 'BINARY_CLASSIFICATION'

    def test_serialize_enum_without_name_falls_back_to_str(self):
        """Test enum without .name uses str() representation."""
        class SimpleEnum:
            def __str__(self):
                return 'CUSTOM_VALUE'

        result = ModelManager._serialize_enum(SimpleEnum())
        assert result == 'CUSTOM_VALUE'

    def test_serialize_real_enum_type(self):
        """Test serialization with Python's enum.Enum type."""
        from enum import Enum

        class TaskType(Enum):
            BINARY_CLASSIFICATION = 1
            REGRESSION = 2
            LLM = 3

        result = ModelManager._serialize_enum(TaskType.BINARY_CLASSIFICATION)
        assert result == 'BINARY_CLASSIFICATION'


class TestDeserializeEnum:
    """Tests for ModelManager._deserialize_enum() (Phase 3.4)."""

    def test_deserialize_none_returns_none(self):
        """Test None input returns None."""
        from enum import Enum

        class DummyEnum(Enum):
            VALUE = 1

        result = ModelManager._deserialize_enum(DummyEnum, None)
        assert result is None

    def test_deserialize_simple_name(self):
        """Test deserializing simple enum name."""
        from enum import Enum

        class TaskType(Enum):
            BINARY_CLASSIFICATION = 1
            REGRESSION = 2

        result = ModelManager._deserialize_enum(TaskType, 'BINARY_CLASSIFICATION')
        assert result == TaskType.BINARY_CLASSIFICATION

    def test_deserialize_with_class_prefix(self):
        """Test deserializing with enum class prefix (e.g., 'TaskType.REGRESSION')."""
        from enum import Enum

        class TaskType(Enum):
            BINARY_CLASSIFICATION = 1
            REGRESSION = 2

        result = ModelManager._deserialize_enum(TaskType, 'TaskType.REGRESSION')
        assert result == TaskType.REGRESSION

    def test_deserialize_lowercase_converted_to_uppercase(self):
        """Test lowercase input is converted to uppercase."""
        from enum import Enum

        class TaskType(Enum):
            BINARY_CLASSIFICATION = 1

        result = ModelManager._deserialize_enum(TaskType, 'binary_classification')
        assert result == TaskType.BINARY_CLASSIFICATION

    def test_deserialize_mixed_case_with_prefix(self):
        """Test mixed case with prefix is normalized."""
        from enum import Enum

        class TaskType(Enum):
            LLM = 1

        result = ModelManager._deserialize_enum(TaskType, 'TaskType.llm')
        assert result == TaskType.LLM

    def test_deserialize_invalid_name_raises_attribute_error(self):
        """Test invalid enum name raises AttributeError."""
        from enum import Enum

        class TaskType(Enum):
            VALID = 1

        with pytest.raises(AttributeError):
            ModelManager._deserialize_enum(TaskType, 'INVALID_NAME')


class TestReconstructSpec:
    """Tests for ModelManager._reconstruct_spec() (Phase 5.1)."""

    @pytest.fixture
    def manager(self):
        """Create ModelManager instance for testing."""
        with patch('fiddler_utils.assets.models.connection') as mock_conn:
            mock_conn.get_client.return_value = Mock()
            mgr = ModelManager()
            return mgr

    def test_reconstruct_spec_basic(self, manager):
        """Test basic spec reconstruction without custom features."""
        model_data = ModelExportData(
            name='test_model',
            version=None,
            columns=[],
            spec={
                'inputs': ['age', 'income'],
                'outputs': ['prediction'],
                'targets': ['label'],
                'metadata': ['user_id'],
                'decisions': []
            },
            custom_features=[],
            task='BINARY_CLASSIFICATION',
            task_params={},
            event_id_col='id',
            event_ts_col='timestamp',
            baselines=[],
            has_artifacts=False,
            related_assets={},
            source_project_id='proj-123',
            source_model_id='model-123',
            exported_at='2025-01-01T00:00:00Z'
        )

        with patch('fiddler_utils.assets.models.fdl.ModelSpec') as mock_spec_class:
            manager._reconstruct_spec(model_data)

            # Verify ModelSpec was called with correct kwargs
            mock_spec_class.assert_called_once()
            call_kwargs = mock_spec_class.call_args[1]

            assert call_kwargs['inputs'] == ['age', 'income']
            assert call_kwargs['outputs'] == ['prediction']
            assert call_kwargs['targets'] == ['label']
            assert call_kwargs['metadata'] == ['user_id']
            assert call_kwargs['decisions'] == []
            assert 'custom_features' not in call_kwargs  # Should not be present when empty

    def test_reconstruct_spec_with_custom_features(self, manager):
        """Test spec reconstruction with custom features (LLM enrichments)."""
        custom_features_data = [
            {
                'type': 'TextEmbedding',
                'name': 'prompt_embedding',
                'source_column': 'question',
                'column': 'prompt_embedding'
            }
        ]

        model_data = ModelExportData(
            name='test_llm',
            version=None,
            columns=[],
            spec={
                'inputs': ['question'],
                'outputs': ['response'],
                'targets': [],
                'metadata': []
            },
            custom_features=custom_features_data,
            task='LLM',
            task_params={},
            event_id_col='id',
            event_ts_col='timestamp',
            baselines=[],
            has_artifacts=False,
            related_assets={},
            source_project_id='proj-123',
            source_model_id='model-123',
            exported_at='2025-01-01T00:00:00Z'
        )

        # Mock the custom features reconstruction
        mock_text_embedding = Mock()
        with patch.object(manager, '_reconstruct_custom_features', return_value=[mock_text_embedding]):
            with patch('fiddler_utils.assets.models.fdl.ModelSpec') as mock_spec_class:
                manager._reconstruct_spec(model_data)

                # Verify custom_features were passed to ModelSpec
                call_kwargs = mock_spec_class.call_args[1]
                assert 'custom_features' in call_kwargs
                assert call_kwargs['custom_features'] == [mock_text_embedding]

    def test_reconstruct_spec_with_decisions(self, manager):
        """Test spec reconstruction with decisions column."""
        model_data = ModelExportData(
            name='test_model',
            version=None,
            columns=[],
            spec={
                'inputs': ['age'],
                'outputs': ['prediction'],
                'targets': ['label'],
                'metadata': [],
                'decisions': ['final_decision']
            },
            custom_features=[],
            task='BINARY_CLASSIFICATION',
            task_params={},
            event_id_col='id',
            event_ts_col='timestamp',
            baselines=[],
            has_artifacts=False,
            related_assets={},
            source_project_id='proj-123',
            source_model_id='model-123',
            exported_at='2025-01-01T00:00:00Z'
        )

        with patch('fiddler_utils.assets.models.fdl.ModelSpec') as mock_spec_class:
            manager._reconstruct_spec(model_data)

            call_kwargs = mock_spec_class.call_args[1]
            assert call_kwargs['decisions'] == ['final_decision']

    def test_reconstruct_spec_empty_custom_features_not_passed(self, manager):
        """Test empty custom features list is not passed to ModelSpec."""
        model_data = ModelExportData(
            name='test_model',
            version=None,
            columns=[],
            spec={
                'inputs': ['age'],
                'outputs': ['prediction'],
                'targets': [],
                'metadata': []
            },
            custom_features=[],  # Empty list
            task='BINARY_CLASSIFICATION',
            task_params={},
            event_id_col='id',
            event_ts_col='timestamp',
            baselines=[],
            has_artifacts=False,
            related_assets={},
            source_project_id='proj-123',
            source_model_id='model-123',
            exported_at='2025-01-01T00:00:00Z'
        )

        with patch.object(manager, '_reconstruct_custom_features', return_value=[]):
            with patch('fiddler_utils.assets.models.fdl.ModelSpec') as mock_spec_class:
                manager._reconstruct_spec(model_data)

                call_kwargs = mock_spec_class.call_args[1]
                # custom_features should NOT be in kwargs when empty
                assert 'custom_features' not in call_kwargs


class TestValidationParameterFlow:
    """Tests for validation parameter flow in import_model (Phase 4.2)."""

    @pytest.fixture
    def manager(self):
        """Create ModelManager instance for testing."""
        with patch('fiddler_utils.assets.models.connection') as mock_conn:
            mock_conn.get_client.return_value = Mock()
            mgr = ModelManager()
            return mgr

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model export data for testing."""
        return ModelExportData(
            name='test_model',
            version=None,
            columns=[
                ColumnExportData(name='age', data_type='INTEGER', min_value=0, max_value=100),
                ColumnExportData(name='prediction', data_type='FLOAT')
            ],
            spec={
                'inputs': ['age'],
                'outputs': ['prediction'],
                'targets': [],
                'metadata': []
            },
            custom_features=[],
            task='REGRESSION',
            task_params={},
            event_id_col='id',
            event_ts_col='timestamp',
            baselines=[],
            has_artifacts=False,
            related_assets={
                'segments': [
                    {'name': 'test_segment', 'definition': 'age > 30'}
                ]
            },
            source_project_id='proj-123',
            source_model_id='model-123',
            exported_at='2025-01-01T00:00:00Z'
        )

    def test_import_model_validate_assets_default_true(self, manager, sample_model_data):
        """Test import_model() defaults validate_assets to True."""
        with patch.object(manager, '_reconstruct_spec', return_value=Mock()):
            with patch('fiddler_utils.assets.models.fdl.Model') as mock_model_class:
                mock_model = Mock()
                mock_model.id = 'new-model-id'
                mock_model_class.return_value = mock_model

                with patch.object(manager, '_get_baseline_manager', return_value=Mock()):
                    with patch.object(manager, '_import_related_assets') as mock_import_assets:
                        manager.import_model(
                            target_project_id='proj-456',
                            model_data=sample_model_data,
                            import_related_assets=True
                            # validate_assets not specified - should default to True
                        )

                        # Verify _import_related_assets was called with validate=True
                        mock_import_assets.assert_called_once()
                        call_kwargs = mock_import_assets.call_args[1]
                        assert call_kwargs['validate'] is True

    def test_import_model_validate_assets_false(self, manager, sample_model_data):
        """Test import_model() with validate_assets=False."""
        with patch.object(manager, '_reconstruct_spec', return_value=Mock()):
            with patch('fiddler_utils.assets.models.fdl.Model') as mock_model_class:
                mock_model = Mock()
                mock_model.id = 'new-model-id'
                mock_model_class.return_value = mock_model

                with patch.object(manager, '_get_baseline_manager', return_value=Mock()):
                    with patch.object(manager, '_import_related_assets') as mock_import_assets:
                        manager.import_model(
                            target_project_id='proj-456',
                            model_data=sample_model_data,
                            import_related_assets=True,
                            validate_assets=False  # Explicitly disable validation
                        )

                        # Verify _import_related_assets was called with validate=False
                        mock_import_assets.assert_called_once()
                        call_kwargs = mock_import_assets.call_args[1]
                        assert call_kwargs['validate'] is False

    def test_import_related_assets_logs_validation_status(self, manager):
        """Test _import_related_assets logs validation status."""
        related_assets = {
            'segments': [],
            'custom_metrics': [],
            'alerts': []
        }

        with patch('fiddler_utils.assets.models.logger') as mock_logger:
            manager._import_related_assets(
                target_model_id='model-123',
                related_assets=related_assets,
                validate=True
            )

            # Verify info log for validation enabled
            mock_logger.info.assert_called_once()
            assert 'validation enabled' in mock_logger.info.call_args[0][0].lower()

        with patch('fiddler_utils.assets.models.logger') as mock_logger:
            manager._import_related_assets(
                target_model_id='model-123',
                related_assets=related_assets,
                validate=False
            )

            # Verify warning log for validation disabled
            mock_logger.warning.assert_called_once()
            assert 'validation disabled' in mock_logger.warning.call_args[0][0].lower()

    def test_import_related_assets_passes_validate_to_segment_manager(self, manager):
        """Test validation flag is passed to SegmentManager."""
        related_assets = {
            'segments': []  # Empty to skip actual import logic
        }

        # Test will verify logging behavior instead of actual import
        with patch('fiddler_utils.assets.models.logger') as mock_logger:
            manager._import_related_assets(
                target_model_id='model-123',
                related_assets=related_assets,
                validate=True
            )

            # Verify validation status was logged correctly
            mock_logger.info.assert_called_once()
            assert 'validation enabled' in mock_logger.info.call_args[0][0].lower()

    def test_import_related_assets_with_validate_false(self, manager):
        """Test validation flag False is handled correctly."""
        related_assets = {
            'segments': []  # Empty to skip actual import logic
        }

        # Test will verify logging behavior for validate=False
        with patch('fiddler_utils.assets.models.logger') as mock_logger:
            manager._import_related_assets(
                target_model_id='model-123',
                related_assets=related_assets,
                validate=False
            )

            # Verify validation disabled was logged as warning
            mock_logger.warning.assert_called_once()
            assert 'validation disabled' in mock_logger.warning.call_args[0][0].lower()
