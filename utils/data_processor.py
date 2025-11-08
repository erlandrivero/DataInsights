"""Data processing utilities for DataInsights with caching and type hints.

This module provides comprehensive data loading, validation, and profiling
capabilities with Streamlit caching for improved performance.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional, Union
from io import BytesIO


class DataProcessor:
    """Handles data loading, validation, and profiling with caching.
    
    This class provides static methods for processing uploaded data,
    generating comprehensive profiles, and detecting quality issues.
    All methods use Streamlit caching for optimal performance.
    
    Attributes:
        None - All methods are static
    
    Example:
        >>> # Load data with automatic caching
        >>> df = DataProcessor.load_data(uploaded_file)
        >>> 
        >>> # Generate profile (cached for 30 min)
        >>> profile = DataProcessor.profile_data(df)
        >>> 
        >>> # Detect quality issues
        >>> issues = DataProcessor.detect_data_quality_issues(df)
    """
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(file: Union[BytesIO, Any]) -> pd.DataFrame:
        """Load data from uploaded file with 1-hour caching.
        
        Supports CSV and Excel file formats. Results are cached to improve
        performance on repeated loads of the same file.
        
        Args:
            file: Uploaded file object from st.file_uploader
                 Must have a .name attribute with file extension
        
        Returns:
            Loaded dataset as pandas DataFrame
        
        Raises:
            ValueError: If file format is not supported (not CSV/Excel)
            Exception: If file cannot be read or parsed
            
        Example:
            >>> uploaded = st.file_uploader("Upload CSV", type=['csv'])
            >>> if uploaded:
            >>>     df = DataProcessor.load_data(uploaded)
            >>>     st.write(f"Loaded {len(df)} rows")
        
        Note:
            Cache persists for 1 hour (3600 seconds) or until app restart
        """
        try:
            if file.name.endswith('.csv'):
                # Try to detect delimiter automatically
                # Read first few lines to detect delimiter
                file.seek(0)  # Reset file pointer
                sample = file.read(10000).decode('utf-8', errors='ignore')
                file.seek(0)  # Reset again for actual read
                
                # Check for common delimiters
                delimiters = [',', ';', '\t', '|']
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                
                # If semicolon is much more common than comma, use semicolon
                if delimiter_counts[';'] > delimiter_counts[','] * 2:
                    detected_delimiter = ';'
                
                df = pd.read_csv(file, delimiter=detected_delimiter)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError(
                    "Unsupported file format. Please upload CSV or Excel files."
                )
            
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile with 30-minute caching.
        
        Analyzes dataset structure, missing values, numeric statistics,
        and categorical summaries. Results cached for performance.
        
        Args:
            df: DataFrame to profile
        
        Returns:
            Dictionary containing:
                - basic_info (dict): Row/column counts, memory usage, duplicates
                - column_info (list): Per-column statistics and samples
                - missing_data (dict): Missing value summary
                - numeric_summary (dict): Statistics for numeric columns
                - categorical_summary (dict): Top values for categorical columns
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> print(f"Rows: {profile['basic_info']['rows']}")
            >>> print(f"Missing: {profile['missing_data']['missing_percentage']}")
        
        Note:
            - Limits categorical summary to first 5 columns
            - Cache persists for 30 minutes (1800 seconds)
        """
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'duplicates': df.duplicated().sum()
            },
            'column_info': [],
            'missing_data': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Column information
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'missing_pct': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
                'unique': df[col].nunique(),
                'unique_pct': f"{(df[col].nunique() / len(df)) * 100:.2f}%"
            }
            
            # Add sample values
            if df[col].dtype in ['object', 'category']:
                col_info['sample'] = df[col].value_counts().head(3).to_dict()
            else:
                col_info['sample'] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None
                }
            
            profile['column_info'].append(col_info)
        
        # Missing data summary
        missing_cols = df.columns[df.isnull().any()].tolist()
        profile['missing_data'] = {
            'columns_with_missing': len(missing_cols),
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%"
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            profile['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            profile['categorical_summary'] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:5]  # Limit to first 5 categorical columns
            }
        
        return profile
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential data quality issues with caching.
        
        Analyzes dataset for common quality problems including missing values,
        duplicates, constant columns, and high cardinality categorical variables.
        
        Args:
            df: DataFrame to analyze for quality issues
        
        Returns:
            List of issue dictionaries, each containing:
                - type (str): Issue category
                - column (str): Affected column name
                - severity (str): 'High', 'Medium', or 'Low'
                - description (str): Human-readable description
        
        Example:
            >>> issues = DataProcessor.detect_data_quality_issues(df)
            >>> high_issues = [i for i in issues if i['severity'] == 'High']
            >>> print(f"Found {len(high_issues)} high-severity issues")
        
        Note:
            - Missing value thresholds: >50% = High, >20% = Medium
            - Duplicate threshold: >10% = High, otherwise Medium
            - High cardinality: >90% unique values
            - Cache persists for 30 minutes
        """
        issues: List[Dict[str, Any]] = []
        
        # Check for high missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                issues.append({
                    'type': 'High Missing Values',
                    'column': col,
                    'severity': 'High',
                    'description': f"{missing_pct:.1f}% of values are missing"
                })
            elif missing_pct > 20:
                issues.append({
                    'type': 'Moderate Missing Values',
                    'column': col,
                    'severity': 'Medium',
                    'description': f"{missing_pct:.1f}% of values are missing"
                })
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                'type': 'Duplicate Rows',
                'column': 'All',
                'severity': 'Medium' if dup_count < len(df) * 0.1 else 'High',
                'description': f"{dup_count} duplicate rows found ({(dup_count/len(df)*100):.1f}%)"
            })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append({
                    'type': 'Constant Column',
                    'column': col,
                    'severity': 'Low',
                    'description': 'Column has only one unique value'
                })
        
        # Check for high cardinality in categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct > 90:
                issues.append({
                    'type': 'High Cardinality',
                    'column': col,
                    'severity': 'Low',
                    'description': f"{unique_pct:.1f}% unique values (might need encoding review)"
                })
        
        return issues
