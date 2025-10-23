"""Export utilities for DataInsights with complete type hints.

This module provides comprehensive export capabilities for data and analysis
results in multiple formats (CSV, Excel, JSON).

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Union
import json
from datetime import datetime
import io


class ExportHelper:
    """Handles data and results export in multiple formats.
    
    This class provides static methods for exporting cleaned data,
    analysis summaries, and converting data types for JSON serialization.
    
    Attributes:
        None - All methods are static
    
    Example:
        >>> # Export cleaned data as CSV
        >>> csv_bytes = ExportHelper.export_cleaned_data(df, 'csv')
        >>> st.download_button("Download CSV", csv_bytes, "data.csv")
        >>> 
        >>> # Export analysis summary
        >>> summary_json = ExportHelper.export_analysis_summary(profile, issues)
    """
    
    @staticmethod
    def export_cleaned_data(df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Export cleaned data in specified format.
        
        Converts a DataFrame to bytes in the specified format (CSV, Excel, or JSON)
        ready for download or storage.
        
        Args:
            df: DataFrame to export
            format: Output format ('csv', 'excel', or 'json')
        
        Returns:
            Bytes object containing the exported data
        
        Raises:
            ValueError: If format is not supported
        
        Example:
            >>> # Export as CSV
            >>> csv_data = ExportHelper.export_cleaned_data(df, 'csv')
            >>> 
            >>> # Export as Excel
            >>> excel_data = ExportHelper.export_cleaned_data(df, 'excel')
            >>> 
            >>> # Export as JSON
            >>> json_data = ExportHelper.export_cleaned_data(df, 'json')
        
        Note:
            - CSV: UTF-8 encoded, no index
            - Excel: Uses openpyxl engine, sheet name 'Data'
            - JSON: Records orientation, 2-space indent
        """
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        elif format == 'json':
            return df.to_json(orient='records', indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'.")
    
    @staticmethod
    def export_analysis_summary(
        profile: Dict[str, Any], 
        issues: List[Dict[str, Any]]
    ) -> str:
        """Export analysis summary as formatted JSON.
        
        Combines data profile and quality issues into a JSON string,
        converting non-serializable types (numpy, pandas) to native Python types.
        
        Args:
            profile: Data profile dictionary from DataProcessor.profile_data()
            issues: Quality issues list from DataProcessor.detect_data_quality_issues()
        
        Returns:
            Formatted JSON string with 2-space indentation
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> issues = DataProcessor.detect_data_quality_issues(df)
            >>> json_summary = ExportHelper.export_analysis_summary(profile, issues)
            >>> st.download_button("Download Analysis", json_summary, "analysis.json")
        
        Note:
            Automatically converts:
            - numpy int64/int32 → int
            - numpy float64/float32 → float
            - pandas Series/DataFrame → dict
            - pandas Timestamp → string
        """
        # Helper function to convert numpy/pandas types to native Python types
        def convert_to_serializable(obj: Any) -> Any:
            """Recursively convert non-serializable types to serializable ones.
            
            Args:
                obj: Object to convert (can be dict, list, numpy/pandas types)
            
            Returns:
                Serializable version of the object
            
            Note:
                Handles nested structures recursively
            """
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return convert_to_serializable(obj.to_dict())
            elif pd.api.types.is_integer_dtype(type(obj)) or hasattr(obj, 'item'):
                # Handle numpy int64, int32, etc.
                try:
                    return int(obj)
                except (ValueError, TypeError):
                    return str(obj)
            elif pd.api.types.is_float_dtype(type(obj)) or isinstance(obj, float):
                # Handle numpy float64, float32, etc.
                try:
                    return float(obj)
                except (ValueError, TypeError):
                    return str(obj)
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            else:
                return obj
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'profile': convert_to_serializable(profile),
            'quality_issues': convert_to_serializable(issues)
        }
        
        return json.dumps(summary, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_downloadable_report(
        content: str, 
        filename: str, 
        file_format: str = 'txt'
    ) -> bytes:
        """Create downloadable report file from text content.
        
        Converts string content to bytes with proper encoding for download.
        
        Args:
            content: Text content of the report
            filename: Suggested filename (informational only)
            file_format: Format extension ('txt' or 'md')
        
        Returns:
            UTF-8 encoded bytes ready for download
        
        Example:
            >>> report_text = "# My Report\\n\\nThis is the content..."
            >>> report_bytes = ExportHelper.create_downloadable_report(
            >>>     report_text, 
            >>>     "report.md", 
            >>>     "md"
            >>> )
            >>> st.download_button(
            >>>     "Download Report", 
            >>>     report_bytes, 
            >>>     "report.md",
            >>>     mime="text/markdown"
            >>> )
        
        Note:
            Always uses UTF-8 encoding for proper unicode support
        """
        return content.encode('utf-8')
    
    @staticmethod
    def format_number(
        value: Union[int, float], 
        decimals: int = 2,
        use_thousands_separator: bool = True
    ) -> str:
        """Format number for display in reports.
        
        Formats numeric values with specified decimal places and optional
        thousands separators for better readability.
        
        Args:
            value: Numeric value to format
            decimals: Number of decimal places (default: 2)
            use_thousands_separator: Whether to use comma separators (default: True)
        
        Returns:
            Formatted string representation of the number
        
        Example:
            >>> ExportHelper.format_number(1234567.89)
            '1,234,567.89'
            >>> 
            >>> ExportHelper.format_number(1234567.89, decimals=0)
            '1,234,568'
            >>> 
            >>> ExportHelper.format_number(1234.5, use_thousands_separator=False)
            '1234.50'
        
        Note:
            Automatically rounds to specified decimal places
        """
        if use_thousands_separator:
            return f"{value:,.{decimals}f}"
        else:
            return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format percentage value for display.
        
        Converts decimal value to percentage string with specified precision.
        
        Args:
            value: Decimal value (e.g., 0.75 for 75%)
            decimals: Number of decimal places (default: 1)
        
        Returns:
            Formatted percentage string with % symbol
        
        Example:
            >>> ExportHelper.format_percentage(0.8532)
            '85.3%'
            >>> 
            >>> ExportHelper.format_percentage(0.8532, decimals=2)
            '85.32%'
        
        Note:
            Multiplies value by 100 before formatting
        """
        return f"{value * 100:.{decimals}f}%"
    
    @staticmethod
    def create_data_dictionary(df: pd.DataFrame, profile: Dict[str, Any]) -> bytes:
        """Create comprehensive data dictionary for a DataFrame.
        
        Generates a detailed data dictionary document describing each column's
        characteristics including data types, statistics, missing values, and
        sample data.
        
        Args:
            df (pd.DataFrame): DataFrame to document
            profile (Dict[str, Any]): Data profile from DataProcessor.profile_data()
        
        Returns:
            bytes: UTF-8 encoded text document ready for download
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> dictionary = ExportHelper.create_data_dictionary(df, profile)
            >>> st.download_button("Download", dictionary, "data_dictionary.txt")
        
        Note:
            - Includes column-by-column breakdown
            - Shows sample values for each column
            - Documents data quality issues
            - Formatted as plain text for universal readability
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DATA DICTIONARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Rows: {len(df):,}")
        lines.append(f"Total Columns: {len(df.columns)}")
        lines.append("")
        
        # Dataset Overview
        lines.append("-" * 80)
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 80)
        if profile:
            lines.append(f"Memory Usage: {profile.get('memory_usage', 'N/A')}")
            lines.append(f"Missing Values: {profile.get('missing_percentage', 'N/A')}")
            lines.append(f"Duplicate Rows: {profile.get('duplicate_rows', 'N/A')}")
        lines.append("")
        
        # Column Details
        lines.append("-" * 80)
        lines.append("COLUMN DETAILS")
        lines.append("-" * 80)
        
        for idx, col in enumerate(df.columns, 1):
            lines.append(f"\n{idx}. {col}")
            lines.append("   " + "-" * 70)
            
            # Data type
            lines.append(f"   Data Type: {df[col].dtype}")
            
            # Unique values
            n_unique = df[col].nunique()
            lines.append(f"   Unique Values: {n_unique:,}")
            
            # Missing values
            n_missing = df[col].isnull().sum()
            pct_missing = (n_missing / len(df)) * 100 if len(df) > 0 else 0
            lines.append(f"   Missing Values: {n_missing:,} ({pct_missing:.1f}%)")
            
            # Sample values (first 5 non-null unique values)
            sample_values = df[col].dropna().unique()[:5]
            if len(sample_values) > 0:
                sample_str = ", ".join([str(v)[:50] for v in sample_values])
                lines.append(f"   Sample Values: {sample_str}")
            
            # Statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(f"   Min: {df[col].min()}")
                lines.append(f"   Max: {df[col].max()}")
                lines.append(f"   Mean: {df[col].mean():.2f}")
                lines.append(f"   Median: {df[col].median():.2f}")
                lines.append(f"   Std Dev: {df[col].std():.2f}")
            
            # Value counts for categorical with few unique values
            if n_unique <= 10 and n_unique > 0:
                lines.append(f"   Value Distribution:")
                value_counts = df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    lines.append(f"      {val}: {count:,} ({pct:.1f}%)")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF DATA DICTIONARY")
        lines.append("=" * 80)
        
        return "\n".join(lines).encode('utf-8')
