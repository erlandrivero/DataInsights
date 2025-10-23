"""
Unit Tests for Input Validator

Tests the input validation utilities for data quality checks.

Author: DataInsights Team
Created: Oct 23, 2025
"""

import pytest
import pandas as pd
import numpy as np
from utils.input_validator import (
    InputValidator, ValidationResult, validate_dataset_for_analysis
)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            severity='info',
            message='Test message',
            recommendation='Test recommendation'
        )
        
        assert result.is_valid is True
        assert result.severity == 'info'
        assert result.message == 'Test message'
        assert result.recommendation == 'Test recommendation'
    
    def test_validation_result_without_recommendation(self):
        """Test creating a ValidationResult without recommendation."""
        result = ValidationResult(
            is_valid=False,
            severity='error',
            message='Error message'
        )
        
        assert result.is_valid is False
        assert result.recommendation is None


class TestBasicDatasetValidation:
    """Tests for basic dataset validation."""
    
    def test_validate_none_dataset(self):
        """Test validation of None dataset."""
        validator = InputValidator()
        result = validator.validate_dataset_basic(None)
        
        assert result.is_valid is False
        assert result.severity == 'critical'
        assert 'No dataset loaded' in result.message
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        validator = InputValidator()
        df = pd.DataFrame()
        result = validator.validate_dataset_basic(df)
        
        assert result.is_valid is False
        assert result.severity == 'critical'
        assert 'empty' in result.message.lower()
    
    def test_validate_insufficient_rows(self):
        """Test validation of dataset with too few rows."""
        validator = InputValidator()
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = validator.validate_dataset_basic(df, min_rows=10)
        
        assert result.is_valid is False
        assert result.severity == 'critical'
        assert '3 rows' in result.message
    
    def test_validate_insufficient_columns(self):
        """Test validation of dataset with too few columns."""
        validator = InputValidator()
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = validator.validate_dataset_basic(df, min_cols=2)
        
        assert result.is_valid is False
        assert result.severity == 'critical'
        assert '1 columns' in result.message
    
    def test_validate_valid_dataset(self, sample_dataframe):
        """Test validation of valid dataset."""
        validator = InputValidator()
        result = validator.validate_dataset_basic(sample_dataframe)
        
        assert result.is_valid is True
        assert result.severity == 'info'


class TestColumnValidation:
    """Tests for column existence and type validation."""
    
    def test_validate_existing_column(self, sample_dataframe):
        """Test validation of existing column."""
        validator = InputValidator()
        result = validator.validate_column_exists(sample_dataframe, 'age')
        
        assert result.is_valid is True
    
    def test_validate_nonexistent_column(self, sample_dataframe):
        """Test validation of non-existent column."""
        validator = InputValidator()
        result = validator.validate_column_exists(sample_dataframe, 'nonexistent')
        
        assert result.is_valid is False
        assert result.severity == 'error'
        assert 'not found' in result.message.lower()
    
    def test_validate_numeric_column(self, sample_dataframe):
        """Test validation of numeric column."""
        validator = InputValidator()
        result = validator.validate_numeric_column(sample_dataframe, 'age')
        
        assert result.is_valid is True
    
    def test_validate_non_numeric_column(self, sample_dataframe):
        """Test validation of non-numeric column."""
        validator = InputValidator()
        result = validator.validate_numeric_column(sample_dataframe, 'category')
        
        assert result.is_valid is False
        assert result.severity == 'error'
        assert 'numeric' in result.message.lower()
    
    def test_validate_datetime_column(self):
        """Test validation of datetime column."""
        validator = InputValidator()
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': range(10)
        })
        
        result = validator.validate_datetime_column(df, 'date')
        assert result.is_valid is True
    
    def test_validate_non_datetime_column(self, sample_dataframe):
        """Test validation of non-datetime column."""
        validator = InputValidator()
        result = validator.validate_datetime_column(sample_dataframe, 'age')
        
        assert result.is_valid is False
        assert 'datetime' in result.message.lower()


class TestMissingValuesValidation:
    """Tests for missing values validation."""
    
    def test_validate_no_missing_values(self, sample_dataframe):
        """Test validation of dataset with no missing values."""
        validator = InputValidator()
        result = validator.validate_missing_values(sample_dataframe)
        
        assert result.is_valid is True
    
    def test_validate_acceptable_missing_values(self):
        """Test validation with acceptable missing values."""
        validator = InputValidator()
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
            'b': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        result = validator.validate_missing_values(df, max_missing_pct=50.0)
        assert result.is_valid is True
    
    def test_validate_excessive_missing_values(self):
        """Test validation with excessive missing values."""
        validator = InputValidator()
        df = pd.DataFrame({
            'a': [1, np.nan, np.nan, np.nan, np.nan] * 4,
            'b': [np.nan, np.nan, 3, 4, 5] * 4
        })
        
        result = validator.validate_missing_values(df, max_missing_pct=30.0)
        assert result.is_valid is False
        assert result.severity == 'warning'


class TestClassBalanceValidation:
    """Tests for class balance validation."""
    
    def test_validate_balanced_classes(self):
        """Test validation of balanced classes."""
        validator = InputValidator()
        y = pd.Series(['A', 'B', 'C'] * 20)
        
        result = validator.validate_class_balance(y)
        assert result.is_valid is True
    
    def test_validate_insufficient_samples_per_class(self):
        """Test validation with insufficient samples per class."""
        validator = InputValidator()
        y = pd.Series(['A', 'A', 'A', 'B'])
        
        result = validator.validate_class_balance(y, min_samples_per_class=2)
        assert result.is_valid is False
        assert result.severity == 'critical'
    
    def test_validate_imbalanced_classes(self):
        """Test validation of imbalanced classes."""
        validator = InputValidator()
        y = pd.Series(['A'] * 100 + ['B'] * 5)
        
        result = validator.validate_class_balance(y)
        assert result.is_valid is True
        assert result.severity == 'warning'
        assert 'imbalance' in result.message.lower()
    
    def test_validate_empty_target(self):
        """Test validation of empty target."""
        validator = InputValidator()
        y = pd.Series([])
        
        result = validator.validate_class_balance(y)
        assert result.is_valid is False
        assert result.severity == 'critical'
    
    def test_validate_target_with_nulls(self):
        """Test validation of target with null values."""
        validator = InputValidator()
        y = pd.Series(['A', 'B', None, 'A', 'B'])
        
        result = validator.validate_class_balance(y)
        assert result.is_valid is False
        assert result.severity == 'error'
        assert 'missing' in result.message.lower()


class TestSampleSizeValidation:
    """Tests for sample size validation."""
    
    def test_validate_sufficient_samples(self, sample_dataframe):
        """Test validation with sufficient samples."""
        validator = InputValidator()
        result = validator.validate_sample_size(sample_dataframe, min_samples=30)
        
        assert result.is_valid is True
    
    def test_validate_insufficient_samples(self):
        """Test validation with insufficient samples."""
        validator = InputValidator()
        df = pd.DataFrame({'a': range(10), 'b': range(10)})
        
        result = validator.validate_sample_size(df, min_samples=30)
        assert result.is_valid is False
        assert result.severity == 'critical'
    
    def test_validate_below_recommended_samples(self):
        """Test validation below recommended samples."""
        validator = InputValidator()
        df = pd.DataFrame({'a': range(50), 'b': range(50)})
        
        result = validator.validate_sample_size(df, min_samples=30, recommended_samples=100)
        assert result.is_valid is True
        assert result.severity == 'warning'


class TestMultipleValidations:
    """Tests for multiple validations."""
    
    def test_validate_multiple_all_pass(self, sample_dataframe):
        """Test multiple validations that all pass."""
        validator = InputValidator()
        
        validations = [
            validator.validate_dataset_basic(sample_dataframe),
            validator.validate_column_exists(sample_dataframe, 'age'),
            validator.validate_numeric_column(sample_dataframe, 'age')
        ]
        
        all_valid, failed = validator.validate_multiple(validations)
        assert all_valid is True
        assert len(failed) == 0
    
    def test_validate_multiple_some_fail(self, sample_dataframe):
        """Test multiple validations with some failures."""
        validator = InputValidator()
        
        validations = [
            validator.validate_dataset_basic(sample_dataframe),
            validator.validate_column_exists(sample_dataframe, 'nonexistent'),
            validator.validate_numeric_column(sample_dataframe, 'age')
        ]
        
        all_valid, failed = validator.validate_multiple(validations)
        assert all_valid is False
        assert len(failed) == 1


class TestDataQualityChecks:
    """Tests for comprehensive data quality checks."""
    
    def test_run_quality_checks(self, sample_dataframe):
        """Test running comprehensive quality checks."""
        validator = InputValidator()
        report = validator.run_data_quality_checks(sample_dataframe)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values_pct' in report
        assert 'duplicate_rows' in report
        assert 'numeric_columns' in report
        assert 'categorical_columns' in report
        
        assert report['total_rows'] == 100
        assert report['total_columns'] == 6
        assert report['numeric_columns'] >= 3
    
    def test_quality_checks_with_missing_data(self, sample_data_with_missing):
        """Test quality checks with missing data."""
        validator = InputValidator()
        report = validator.run_data_quality_checks(sample_data_with_missing)
        
        assert report['missing_values_pct'] > 0
        assert len(report['columns_with_nulls']) > 0
    
    def test_quality_checks_identifies_high_cardinality(self):
        """Test quality checks identify high cardinality columns."""
        validator = InputValidator()
        df = pd.DataFrame({
            'id': range(100),
            'high_card': [f'val_{i}' for i in range(100)],
            'low_card': ['A', 'B', 'C'] * 33 + ['A']
        })
        
        report = validator.run_data_quality_checks(df)
        assert len(report['high_cardinality_cols']) > 0
        assert 'high_card' in report['high_cardinality_cols']


@pytest.mark.integration
class TestIntegrationValidation:
    """Integration tests for validation workflows."""
    
    def test_full_validation_workflow(self, sample_classification_data):
        """Test full validation workflow for classification."""
        X, y = sample_classification_data
        df = X.copy()
        df['target'] = y
        
        validator = InputValidator()
        
        # Run all validations
        basic = validator.validate_dataset_basic(df)
        sample_size = validator.validate_sample_size(df, min_samples=50)
        class_balance = validator.validate_class_balance(y)
        missing = validator.validate_missing_values(df)
        
        assert basic.is_valid
        assert sample_size.is_valid
        assert class_balance.is_valid
        assert missing.is_valid
