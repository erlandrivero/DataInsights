"""
Input Validation Utility for DataInsights

Provides centralized validation functions for data quality checks,
preventing crashes and providing user-friendly error messages.

Author: DataInsights Team
Created: Oct 23, 2025
"""

from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st


class ValidationResult:
    """Container for validation results with severity levels."""
    
    def __init__(self, is_valid: bool, severity: str, message: str, 
                 recommendation: Optional[str] = None):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            severity: One of 'info', 'warning', 'error', 'critical'
            message: Validation message to display
            recommendation: Optional recommendation for fixing the issue
        """
        self.is_valid = is_valid
        self.severity = severity
        self.message = message
        self.recommendation = recommendation


class InputValidator:
    """Centralized input validation for DataInsights application."""
    
    @staticmethod
    def validate_dataset_basic(df: Optional[pd.DataFrame], 
                               min_rows: int = 10, 
                               min_cols: int = 2) -> ValidationResult:
        """
        Validate basic dataset requirements.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required
            min_cols: Minimum number of columns required
            
        Returns:
            ValidationResult object with validation status and message
        """
        if df is None:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message="No dataset loaded. Please upload data first.",
                recommendation="Go to 'Data Upload' page and upload a CSV or Excel file."
            )
        
        if df.empty:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message="Dataset is empty. Please upload valid data.",
                recommendation="Ensure your file contains data rows and columns."
            )
        
        if len(df) < min_rows:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message=f"Dataset has only {len(df)} rows. Minimum {min_rows} rows required.",
                recommendation=f"Upload a dataset with at least {min_rows} rows for reliable analysis."
            )
        
        if len(df.columns) < min_cols:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message=f"Dataset has only {len(df.columns)} columns. Minimum {min_cols} columns required.",
                recommendation=f"Upload a dataset with at least {min_cols} columns."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message="Dataset passed basic validation."
        )
    
    @staticmethod
    def validate_column_exists(df: pd.DataFrame, 
                              column_name: str) -> ValidationResult:
        """
        Validate that a specific column exists in the dataset.
        
        Args:
            df: DataFrame to check
            column_name: Name of column to validate
            
        Returns:
            ValidationResult object
        """
        if column_name not in df.columns:
            available = ', '.join(df.columns.tolist()[:5])
            if len(df.columns) > 5:
                available += f'... ({len(df.columns)} total)'
            
            return ValidationResult(
                is_valid=False,
                severity='error',
                message=f"Column '{column_name}' not found in dataset.",
                recommendation=f"Available columns: {available}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Column '{column_name}' found."
        )
    
    @staticmethod
    def validate_numeric_column(df: pd.DataFrame, 
                               column_name: str) -> ValidationResult:
        """
        Validate that a column contains numeric data.
        
        Args:
            df: DataFrame to check
            column_name: Name of column to validate
            
        Returns:
            ValidationResult object
        """
        # First check if column exists
        exists_result = InputValidator.validate_column_exists(df, column_name)
        if not exists_result.is_valid:
            return exists_result
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return ValidationResult(
                is_valid=False,
                severity='error',
                message=f"Column '{column_name}' must contain numeric data.",
                recommendation=f"Current type: {df[column_name].dtype}. Convert to numeric or select a different column."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Column '{column_name}' is numeric."
        )
    
    @staticmethod
    def validate_datetime_column(df: pd.DataFrame, 
                                column_name: str) -> ValidationResult:
        """
        Validate that a column contains datetime data.
        
        Args:
            df: DataFrame to check
            column_name: Name of column to validate
            
        Returns:
            ValidationResult object
        """
        # First check if column exists
        exists_result = InputValidator.validate_column_exists(df, column_name)
        if not exists_result.is_valid:
            return exists_result
        
        if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
            return ValidationResult(
                is_valid=False,
                severity='error',
                message=f"Column '{column_name}' must contain datetime data.",
                recommendation=f"Current type: {df[column_name].dtype}. Convert to datetime using pd.to_datetime()."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Column '{column_name}' is datetime."
        )
    
    @staticmethod
    def validate_missing_values(df: pd.DataFrame, 
                               max_missing_pct: float = 50.0) -> ValidationResult:
        """
        Validate that missing values are within acceptable threshold.
        
        Args:
            df: DataFrame to check
            max_missing_pct: Maximum percentage of missing values allowed
            
        Returns:
            ValidationResult object
        """
        total_missing = df.isnull().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_values) * 100
        
        if missing_pct > max_missing_pct:
            return ValidationResult(
                is_valid=False,
                severity='warning',
                message=f"Dataset has {missing_pct:.1f}% missing values (threshold: {max_missing_pct}%).",
                recommendation="Consider cleaning the data or using imputation techniques."
            )
        
        if missing_pct > 10:
            return ValidationResult(
                is_valid=True,
                severity='warning',
                message=f"Dataset has {missing_pct:.1f}% missing values. Results may be affected.",
                recommendation="Consider handling missing values before analysis."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Missing values: {missing_pct:.1f}% (acceptable)."
        )
    
    @staticmethod
    def validate_class_balance(y: pd.Series, 
                              min_samples_per_class: int = 2) -> ValidationResult:
        """
        Validate class balance for classification tasks.
        
        Args:
            y: Target variable series
            min_samples_per_class: Minimum samples required per class
            
        Returns:
            ValidationResult object
        """
        if y is None or len(y) == 0:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message="Target variable is empty.",
                recommendation="Select a valid target column."
            )
        
        # Check for missing values in target
        if y.isnull().any():
            missing_count = y.isnull().sum()
            return ValidationResult(
                is_valid=False,
                severity='error',
                message=f"Target variable has {missing_count} missing values.",
                recommendation="Remove or impute missing values in the target column."
            )
        
        # Check class distribution
        class_counts = y.value_counts()
        min_class_size = class_counts.min()
        
        if min_class_size < min_samples_per_class:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message=f"Smallest class has only {min_class_size} samples (minimum: {min_samples_per_class}).",
                recommendation="Collect more data or remove underrepresented classes."
            )
        
        # Check for severe imbalance
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if imbalance_ratio > 100:
            return ValidationResult(
                is_valid=True,
                severity='warning',
                message=f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1).",
                recommendation="Consider using sampling techniques (SMOTE, under-sampling) or stratified splitting."
            )
        elif imbalance_ratio > 10:
            return ValidationResult(
                is_valid=True,
                severity='warning',
                message=f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1).",
                recommendation="Results may favor the majority class. Consider class weighting."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Classes are reasonably balanced (ratio: {imbalance_ratio:.1f}:1)."
        )
    
    @staticmethod
    def validate_sample_size(df: pd.DataFrame, 
                            min_samples: int = 30,
                            recommended_samples: int = 100) -> ValidationResult:
        """
        Validate that dataset has sufficient samples for analysis.
        
        Args:
            df: DataFrame to check
            min_samples: Minimum samples required
            recommended_samples: Recommended number of samples
            
        Returns:
            ValidationResult object
        """
        n_samples = len(df)
        
        if n_samples < min_samples:
            return ValidationResult(
                is_valid=False,
                severity='critical',
                message=f"Dataset has only {n_samples} samples (minimum: {min_samples}).",
                recommendation="Collect more data for reliable statistical analysis."
            )
        
        if n_samples < recommended_samples:
            return ValidationResult(
                is_valid=True,
                severity='warning',
                message=f"Dataset has {n_samples} samples (recommended: {recommended_samples}+).",
                recommendation="Results may be more reliable with additional data."
            )
        
        return ValidationResult(
            is_valid=True,
            severity='info',
            message=f"Dataset has {n_samples} samples (sufficient)."
        )
    
    @staticmethod
    def validate_and_display(validation_result: ValidationResult) -> bool:
        """
        Display validation result in Streamlit and return validity status.
        
        Args:
            validation_result: ValidationResult object to display
            
        Returns:
            True if validation passed, False otherwise
        """
        if validation_result.severity == 'critical' or validation_result.severity == 'error':
            st.error(f"❌ {validation_result.message}")
            if validation_result.recommendation:
                st.info(f"💡 **Recommendation:** {validation_result.recommendation}")
        elif validation_result.severity == 'warning':
            st.warning(f"⚠️ {validation_result.message}")
            if validation_result.recommendation:
                st.info(f"💡 **Recommendation:** {validation_result.recommendation}")
        elif validation_result.severity == 'info':
            # Only show info messages in expander to avoid clutter
            with st.expander("✅ Validation Details", expanded=False):
                st.success(validation_result.message)
        
        return validation_result.is_valid
    
    @staticmethod
    def validate_multiple(validations: List[ValidationResult]) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate multiple conditions and return combined result.
        
        Args:
            validations: List of ValidationResult objects
            
        Returns:
            Tuple of (all_valid, failed_validations)
        """
        failed = [v for v in validations if not v.is_valid]
        all_valid = len(failed) == 0
        
        return all_valid, failed
    
    @staticmethod
    def run_data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive data quality checks and return summary.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'high_cardinality_cols': [col for col in df.select_dtypes(include=['object']).columns 
                                     if df[col].nunique() > 50]
        }
        
        return quality_report


# Convenience function for quick validation
def validate_dataset_for_analysis(df: Optional[pd.DataFrame], 
                                  analysis_type: str = "general") -> bool:
    """
    Quick validation for common analysis types.
    
    Args:
        df: DataFrame to validate
        analysis_type: Type of analysis (general, classification, regression, timeseries)
        
    Returns:
        True if validation passes, False otherwise
    """
    validator = InputValidator()
    
    # Basic validation
    basic_result = validator.validate_dataset_basic(df)
    if not validator.validate_and_display(basic_result):
        st.stop()
        return False
    
    # Analysis-specific validation
    if analysis_type == "classification":
        sample_result = validator.validate_sample_size(df, min_samples=50, recommended_samples=200)
        validator.validate_and_display(sample_result)
    
    elif analysis_type == "regression":
        sample_result = validator.validate_sample_size(df, min_samples=30, recommended_samples=100)
        validator.validate_and_display(sample_result)
    
    elif analysis_type == "timeseries":
        sample_result = validator.validate_sample_size(df, min_samples=30, recommended_samples=100)
        validator.validate_and_display(sample_result)
    
    # Missing values check
    missing_result = validator.validate_missing_values(df, max_missing_pct=50.0)
    validator.validate_and_display(missing_result)
    
    return True
