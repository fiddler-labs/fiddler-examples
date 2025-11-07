"""Shared pytest fixtures for fiddler_utils tests.

This module provides common fixtures used across multiple test files,
reducing code duplication and ensuring consistent test setup.
"""

from unittest.mock import Mock, MagicMock
import pytest

# Don't collect test functions from source modules
import os
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "..", "testing.py")
]


@pytest.fixture
def mock_fiddler_client():
    """Mock Fiddler client for testing without actual API calls."""
    client = Mock()
    client.list_projects = Mock(return_value=[])
    client.get_project = Mock()
    client.list_models = Mock(return_value=[])
    client.get_model = Mock()
    return client


@pytest.fixture
def mock_project():
    """Mock Fiddler Project object with common attributes."""
    project = Mock()
    project.id = "test-project-id"
    project.name = "test_project"
    project.created_at = "2024-01-01T00:00:00Z"
    project.updated_at = "2024-01-01T00:00:00Z"
    return project


@pytest.fixture
def mock_model():
    """Mock Fiddler Model object with common attributes."""
    model = Mock()
    model.id = "test-model-id"
    model.name = "test_model"
    model.project_id = "test-project-id"
    model.task = "binary_classification"
    model.created_at = "2024-01-01T00:00:00Z"
    model.updated_at = "2024-01-01T00:00:00Z"

    # Mock spec
    spec = Mock()
    spec.inputs = ["feature1", "feature2"]
    spec.outputs = ["prediction"]
    spec.targets = ["target"]
    spec.metadata = ["timestamp"]
    model.spec = spec

    return model


@pytest.fixture
def mock_model_spec():
    """Mock ModelSpec object."""
    spec = Mock()
    spec.inputs = ["feature1", "feature2", "feature3"]
    spec.outputs = ["prediction", "score"]
    spec.targets = ["target"]
    spec.metadata = ["timestamp", "user_id"]
    spec.custom_features = []
    return spec


@pytest.fixture
def mock_segment():
    """Mock Segment object."""
    segment = Mock()
    segment.id = "test-segment-id"
    segment.name = "test_segment"
    segment.model_id = "test-model-id"
    segment.definition = "age > 30"
    segment.description = "Test segment"
    return segment


@pytest.fixture
def mock_custom_metric():
    """Mock CustomMetric object."""
    metric = Mock()
    metric.id = "test-metric-id"
    metric.name = "test_metric"
    metric.model_id = "test-model-id"
    metric.definition = "sum(if(prediction == target, 1, 0)) / count(*)"
    metric.description = "Test metric"
    return metric


@pytest.fixture
def mock_alert_rule():
    """Mock AlertRule object."""
    alert = Mock()
    alert.id = "test-alert-id"
    alert.name = "test_alert"
    alert.model_id = "test-model-id"
    alert.metric_id = "accuracy"
    alert.condition = "LESS_THAN"
    alert.threshold = 0.8
    alert.enabled = True
    return alert


@pytest.fixture
def mock_baseline():
    """Mock Baseline object."""
    baseline = Mock()
    baseline.id = "test-baseline-id"
    baseline.name = "test_baseline"
    baseline.model_id = "test-model-id"
    baseline.type = "static"
    baseline.dataset_id = "test-dataset-id"
    return baseline


@pytest.fixture
def sample_fql_expressions():
    """Collection of sample FQL expressions for testing."""
    return {
        "simple": "age > 30",
        "compound": "age > 30 and city == 'San Francisco'",
        "with_functions": "sum(if(prediction == target, 1, 0)) / count(*)",
        "complex": "avg(case when prediction > 0.5 then 1 else 0 end)",
        "with_quotes": '"feature_name" > 100',
        "invalid": "age >> 30",  # Invalid syntax
    }


@pytest.fixture
def sample_schema_data():
    """Sample schema data for testing schema validation."""
    return {
        "columns": [
            {"name": "feature1", "data_type": "float", "column_type": "input"},
            {"name": "feature2", "data_type": "int", "column_type": "input"},
            {"name": "prediction", "data_type": "float", "column_type": "output"},
            {"name": "target", "data_type": "int", "column_type": "target"},
            {"name": "timestamp", "data_type": "int", "column_type": "metadata"},
        ]
    }


@pytest.fixture(autouse=True)
def reset_connection_state():
    """Reset connection state before each test to ensure test isolation.

    This fixture automatically runs before each test to clear any global
    connection state that might interfere with other tests.
    """
    # Import here to avoid circular imports
    from fiddler_utils.connection import reset_connection

    # Reset before test
    reset_connection()

    # Run the test
    yield

    # Reset after test
    reset_connection()


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file path for testing file I/O."""
    return tmp_path / "test_data.json"


@pytest.fixture
def mock_connection_manager():
    """Mock ConnectionManager for multi-instance testing."""
    from fiddler_utils.connection import ConnectionManager

    manager = ConnectionManager(log_level='WARNING')

    # Add some mock connections
    manager.add_connection(
        name='source',
        url='https://source.fiddler.ai',
        token='source-token'
    )
    manager.add_connection(
        name='target',
        url='https://target.fiddler.ai',
        token='target-token'
    )

    return manager


# Markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring Fiddler connection"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )


def pytest_collection_modifyitems(config, items):
    """Don't collect functions from source modules as tests."""
    # Filter out test_metric_definition - it's a utility function, not a test
    # It gets imported into test_testing.py which causes pytest to collect it as a test
    filtered_items = []
    for item in items:
        # Skip if it's the imported test_metric_definition function
        if (item.name == "test_metric_definition" and
            hasattr(item, 'fspath') and
            'test_testing.py' in str(item.fspath)):
            continue
        filtered_items.append(item)
    items[:] = filtered_items
