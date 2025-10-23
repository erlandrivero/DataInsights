"""
Pytest Configuration and Shared Fixtures for DataInsights Tests

Contains reusable test fixtures and configuration for all test modules.

Author: DataInsights Team
Created: Oct 23, 2025
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== Sample Data Fixtures ====================

@pytest.fixture
def sample_dataframe():
    """Create a simple sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randint(20000, 150000, 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'flag': np.random.choice([True, False], 100)
    })


@pytest.fixture
def sample_classification_data():
    """Create sample data for classification tasks."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples)
    })
    
    y = pd.Series(np.random.choice(['Class_A', 'Class_B', 'Class_C'], n_samples))
    
    return X, y


@pytest.fixture
def sample_regression_data():
    """Create sample data for regression tasks."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    # Create target with some relationship to features
    y = pd.Series(
        2 * X['feature1'] + 3 * X['feature2'] - X['feature3'] + np.random.randn(n_samples) * 0.5
    )
    
    return X, y


@pytest.fixture
def sample_timeseries_data():
    """Create sample time series data."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create trend + seasonality + noise
    trend = np.linspace(100, 200, len(dates))
    seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    noise = np.random.randn(len(dates)) * 5
    
    return pd.DataFrame({
        'date': dates,
        'value': trend + seasonality + noise
    })


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for market basket analysis."""
    transactions = [
        ['bread', 'milk', 'eggs'],
        ['bread', 'butter'],
        ['milk', 'butter', 'cheese'],
        ['bread', 'milk', 'butter'],
        ['eggs', 'cheese'],
        ['bread', 'eggs', 'milk'],
        ['butter', 'cheese'],
        ['bread', 'milk', 'cheese']
    ] * 10  # Repeat to get more transactions
    
    return pd.DataFrame({
        'transaction_id': range(len(transactions)),
        'items': transactions
    })


@pytest.fixture
def sample_rfm_data():
    """Create sample RFM (Recency, Frequency, Monetary) data."""
    np.random.seed(42)
    n_customers = 100
    
    return pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'recency': np.random.randint(1, 365, n_customers),
        'frequency': np.random.randint(1, 50, n_customers),
        'monetary': np.random.uniform(10, 10000, n_customers)
    })


@pytest.fixture
def sample_text_data():
    """Create sample text data for NLP tasks."""
    texts = [
        "This product is amazing! I love it.",
        "Terrible experience. Would not recommend.",
        "Average quality, nothing special.",
        "Excellent service and great value!",
        "Disappointed with the purchase.",
        "Best product I've ever bought!",
        "Not worth the money.",
        "Pretty good, meets expectations."
    ] * 10
    
    return pd.DataFrame({
        'text': texts,
        'rating': np.random.randint(1, 6, len(texts))
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'col3': np.random.choice(['A', 'B', 'C', None], 100),
        'col4': np.random.randn(100)
    })
    
    # Introduce missing values
    df.loc[0:10, 'col1'] = np.nan
    df.loc[20:25, 'col2'] = np.nan
    
    return df


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicate rows."""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 1, 2],
        'value': [10, 20, 30, 40, 50, 10, 20],
        'category': ['A', 'B', 'C', 'D', 'E', 'A', 'B']
    })
    
    return df


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        'choices': [{
            'message': {
                'content': 'This is a mock AI response for testing purposes.'
            }
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 15,
            'total_tokens': 25
        }
    }


# ==================== Configuration Fixtures ====================

@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


# ==================== Utility Functions for Tests ====================

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
    """
    Assert that two DataFrames are equal with better error messages.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        **kwargs: Additional arguments for pd.testing.assert_frame_equal
    """
    pd.testing.assert_frame_equal(df1, df2, **kwargs)


def assert_series_equal(s1: pd.Series, s2: pd.Series, **kwargs):
    """
    Assert that two Series are equal with better error messages.
    
    Args:
        s1: First Series
        s2: Second Series
        **kwargs: Additional arguments for pd.testing.assert_series_equal
    """
    pd.testing.assert_series_equal(s1, s2, **kwargs)


def assert_dict_contains_keys(d: Dict[str, Any], required_keys: list):
    """
    Assert that a dictionary contains all required keys.
    
    Args:
        d: Dictionary to check
        required_keys: List of required keys
    """
    missing_keys = set(required_keys) - set(d.keys())
    assert not missing_keys, f"Dictionary missing required keys: {missing_keys}"


# ==================== Pytest Configuration ====================

def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API access"
    )
