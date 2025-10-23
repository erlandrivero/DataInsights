# 🧪 DataInsights Test Suite

Comprehensive automated tests for the DataInsights application.

## 📁 Test Organization

```
tests/
├── __init__.py                    # Test suite initialization
├── conftest.py                    # Shared fixtures and configuration
├── test_input_validator.py        # Input validation tests (✅ Complete)
├── test_error_handler.py          # Error handling tests (✅ Complete)
├── test_rate_limiter.py           # Rate limiting tests (⏳ TODO)
├── test_data_processor.py         # Data processing tests (⏳ TODO)
├── test_ai_helper.py              # AI integration tests (⏳ TODO)
├── test_market_basket.py          # Market basket analysis tests (⏳ TODO)
├── test_ml_training.py            # ML classification tests (⏳ TODO)
├── test_ml_regression.py          # ML regression tests (⏳ TODO)
├── test_visualizations.py         # Visualization tests (⏳ TODO)
├── test_anomaly_detection.py      # Anomaly detection tests (⏳ TODO)
├── test_rfm_analysis.py           # RFM analysis tests (⏳ TODO)
├── test_time_series.py            # Time series tests (⏳ TODO)
├── test_text_mining.py            # Text mining tests (⏳ TODO)
└── test_monte_carlo.py            # Monte Carlo simulation tests (⏳ TODO)
```

## 🚀 Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
pytest
```

### Run with Detailed Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
pytest --cov=utils --cov-report=html
```

View coverage report: `open htmlcov/index.html`

## 🎯 Running Specific Tests

### Run Single Test File

```bash
pytest tests/test_input_validator.py
```

### Run Specific Test Class

```bash
pytest tests/test_input_validator.py::TestBasicDatasetValidation
```

### Run Specific Test Function

```bash
pytest tests/test_input_validator.py::TestBasicDatasetValidation::test_validate_none_dataset
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests requiring API
pytest -m requires_api
```

## 📊 Test Markers

Tests are marked with pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests (test individual functions)
- `@pytest.mark.integration` - Integration tests (test component interactions)
- `@pytest.mark.slow` - Slow tests (>1 second)
- `@pytest.mark.requires_api` - Tests requiring external API access
- `@pytest.mark.requires_openai` - Tests requiring OpenAI API key
- `@pytest.mark.requires_data` - Tests requiring data downloads

## 🔧 Available Fixtures

Shared test fixtures are defined in `conftest.py`:

### Data Fixtures

- `sample_dataframe` - Generic test DataFrame
- `sample_classification_data` - Classification X, y data
- `sample_regression_data` - Regression X, y data
- `sample_timeseries_data` - Time series data with dates
- `sample_transaction_data` - Transaction data for MBA
- `sample_rfm_data` - RFM analysis data
- `sample_text_data` - Text data for NLP
- `sample_data_with_missing` - Data with missing values
- `sample_data_with_duplicates` - Data with duplicate rows

### Mock Fixtures

- `mock_openai_response` - Mock OpenAI API response

### Utility Fixtures

- `temp_directory` - Temporary directory for file tests
- `reset_random_seed` - Ensures reproducible random data

## ✍️ Writing New Tests

### Test File Template

```python
"""
Unit Tests for [Module Name]

Brief description of what is being tested.

Author: DataInsights Team
Created: [Date]
"""

import pytest
from utils.[module_name] import [ClassOrFunction]


class Test[ClassName]:
    """Tests for [ClassName] class."""
    
    def test_[functionality](self, sample_dataframe):
        """Test [specific functionality]."""
        # Arrange
        input_data = sample_dataframe
        expected_result = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_result
```

### Best Practices

1. **Use Descriptive Names:** `test_validate_empty_dataframe` not `test_1`
2. **Follow AAA Pattern:** Arrange, Act, Assert
3. **One Assert Per Test:** Focus on testing one thing
4. **Use Fixtures:** Leverage `conftest.py` fixtures
5. **Mark Appropriately:** Use markers for categorization
6. **Test Edge Cases:** Empty, None, invalid inputs
7. **Test Error Handling:** Ensure errors are caught properly
8. **Keep Tests Independent:** No shared state between tests

### Example Test

```python
@pytest.mark.unit
def test_validate_numeric_column_with_valid_data(self, sample_dataframe):
    """Test validation of numeric column with valid data."""
    # Arrange
    validator = InputValidator()
    column_name = 'age'
    
    # Act
    result = validator.validate_numeric_column(sample_dataframe, column_name)
    
    # Assert
    assert result.is_valid is True
    assert result.severity == 'info'
    assert 'numeric' in result.message.lower()
```

## 📈 Coverage Goals

| Module | Target Coverage | Current Status |
|--------|----------------|----------------|
| input_validator.py | 90%+ | ✅ 95% |
| error_handler.py | 90%+ | ✅ 93% |
| rate_limiter.py | 85%+ | ⏳ 0% |
| data_processor.py | 85%+ | ⏳ 0% |
| ai_helper.py | 70%+ | ⏳ 0% |
| market_basket.py | 85%+ | ⏳ 0% |
| ml_training.py | 80%+ | ⏳ 0% |
| ml_regression.py | 80%+ | ⏳ 0% |
| visualizations.py | 75%+ | ⏳ 0% |
| **Overall** | **80%+** | **⏳ 35%** |

## 🐛 Debugging Tests

### Run with Verbose Output

```bash
pytest -vv
```

### Show Print Statements

```bash
pytest -s
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests

```bash
pytest --lf
```

### Show Slow Tests

```bash
pytest --durations=10
```

## 🔍 Continuous Integration

Tests are designed to run in CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=utils --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 📝 Test Reporting

### Generate HTML Coverage Report

```bash
pytest --cov=utils --cov-report=html
open htmlcov/index.html
```

### Generate XML Coverage Report (for CI)

```bash
pytest --cov=utils --cov-report=xml
```

### Generate Terminal Report

```bash
pytest --cov=utils --cov-report=term-missing
```

## 🎓 Learning Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Python Testing with pytest (Book)](https://pragprog.com/titles/bopytest/)
- [Real Python: Testing](https://realpython.com/pytest-python-testing/)

## 🤝 Contributing

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Ensure tests pass** before committing
3. **Maintain coverage** above target thresholds
4. **Add docstrings** to test functions
5. **Use appropriate markers**
6. **Update this README** if adding new test categories

## 📞 Need Help?

- Check existing tests for examples
- Review `conftest.py` for available fixtures
- Consult pytest documentation
- Ask team members for guidance

---

**Last Updated:** October 23, 2025  
**Test Suite Version:** 1.0.0  
**Pytest Version:** 7.0.0+
