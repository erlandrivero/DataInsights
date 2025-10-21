"""
Comprehensive data cleaning and preprocessing utilities.
Implements robust cleaning pipeline similar to SuperWrangler.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing class.
    Implements a full cleaning pipeline with detailed reporting.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaner with a DataFrame.
        
        Args:
            df: Input DataFrame to clean
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.cleaning_log = []
        self.stats = {
            'original_shape': df.shape,
            'original_dtypes': df.dtypes.value_counts().to_dict(),
            'original_missing': df.isnull().sum().sum(),
            'original_duplicates': 0
        }
    
    def normalize_column_names(self) -> 'DataCleaner':
        """
        Normalize column names to lowercase with underscores.
        
        Returns:
            Self for method chaining
        """
        old_columns = self.df.columns.tolist()
        new_columns = []
        
        for i, col in enumerate(old_columns):
            # Convert to string and lowercase
            col_str = str(col).lower().strip()
            
            # Replace spaces and special characters with underscore
            col_str = re.sub(r'[^a-z0-9_]', '_', col_str)
            
            # Remove multiple consecutive underscores
            col_str = re.sub(r'_+', '_', col_str)
            
            # Remove leading/trailing underscores
            col_str = col_str.strip('_')
            
            # Handle empty names
            if not col_str or col_str == '':
                col_str = f'unnamed_column_{i+1}'
            
            new_columns.append(col_str)
        
        # Handle duplicate column names
        seen = {}
        final_columns = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                final_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_columns.append(col)
        
        self.df.columns = final_columns
        
        renamed_count = sum(1 for old, new in zip(old_columns, final_columns) if str(old) != new)
        if renamed_count > 0:
            self.cleaning_log.append(f"✅ Normalized {renamed_count} column names")
        
        return self
    
    def convert_to_numeric(self, exclude_columns: List[str] = None) -> 'DataCleaner':
        """
        Convert columns to numeric where possible.
        
        Args:
            exclude_columns: List of column names to exclude from conversion
            
        Returns:
            Self for method chaining
        """
        if exclude_columns is None:
            exclude_columns = []
        
        conversions = 0
        
        for col in self.df.columns:
            if col in exclude_columns:
                continue
            
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert
            try:
                converted = pd.to_numeric(self.df[col], errors='coerce')
                # Only apply if at least 50% of values are valid numbers
                valid_ratio = converted.notna().sum() / len(converted)
                if valid_ratio > 0.5:
                    self.df[col] = converted
                    conversions += 1
            except:
                pass
        
        if conversions > 0:
            self.cleaning_log.append(f"✅ Converted {conversions} columns to numeric")
        
        return self
    
    def remove_duplicates(self) -> 'DataCleaner':
        """
        Remove exact duplicate rows.
        
        Returns:
            Self for method chaining
        """
        before_count = len(self.df)
        self.df = self.df.drop_duplicates()
        after_count = len(self.df)
        
        duplicates_removed = before_count - after_count
        self.stats['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            pct = (duplicates_removed / before_count) * 100
            self.cleaning_log.append(f"✅ Removed {duplicates_removed:,} duplicate rows ({pct:.1f}%)")
        else:
            self.cleaning_log.append("✅ No duplicates found")
        
        return self
    
    def fill_missing_values(self, strategy: str = 'median', 
                           custom_fills: Dict[str, Any] = None) -> 'DataCleaner':
        """
        Fill missing values using specified strategy.
        
        Args:
            strategy: 'median', 'mean', 'mode', or 'drop'
            custom_fills: Dictionary of {column: value} for custom fills
            
        Returns:
            Self for method chaining
        """
        if custom_fills is None:
            custom_fills = {}
        
        total_filled = 0
        filled_by_column = {}
        
        for col in self.df.columns:
            # Use custom fill if specified
            if col in custom_fills:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    self.df[col].fillna(custom_fills[col], inplace=True)
                    total_filled += missing_count
                    filled_by_column[col] = missing_count
                continue
            
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else self.df[col].median()
                else:  # drop
                    continue
                
                self.df[col].fillna(fill_value, inplace=True)
                total_filled += missing_count
                filled_by_column[col] = missing_count
            
            # For categorical columns
            else:
                if strategy in ['median', 'mean', 'mode']:
                    # Use mode for categorical
                    if len(self.df[col].mode()) > 0:
                        fill_value = self.df[col].mode()[0]
                        self.df[col].fillna(fill_value, inplace=True)
                        total_filled += missing_count
                        filled_by_column[col] = missing_count
        
        self.stats['missing_filled'] = total_filled
        self.stats['filled_by_column'] = filled_by_column
        
        if total_filled > 0:
            self.cleaning_log.append(f"✅ Filled {total_filled:,} missing values using {strategy} strategy")
        else:
            self.cleaning_log.append("✅ No missing values to fill")
        
        return self
    
    def drop_missing_rows(self, threshold: float = 0.5) -> 'DataCleaner':
        """
        Drop rows with too many missing values.
        
        Args:
            threshold: Drop rows with more than this fraction of missing values
            
        Returns:
            Self for method chaining
        """
        before_count = len(self.df)
        
        # Calculate missing ratio per row
        missing_ratio = self.df.isnull().sum(axis=1) / len(self.df.columns)
        
        # Drop rows exceeding threshold
        self.df = self.df[missing_ratio <= threshold]
        
        after_count = len(self.df)
        rows_dropped = before_count - after_count
        
        if rows_dropped > 0:
            self.cleaning_log.append(f"✅ Dropped {rows_dropped:,} rows with >{threshold*100:.0f}% missing values")
        
        return self
    
    def drop_missing_columns(self, threshold: float = 0.5) -> 'DataCleaner':
        """
        Drop columns with too many missing values.
        
        Args:
            threshold: Drop columns with more than this fraction of missing values
            
        Returns:
            Self for method chaining
        """
        before_count = len(self.df.columns)
        
        # Calculate missing ratio per column
        missing_ratio = self.df.isnull().sum() / len(self.df)
        
        # Get columns to keep
        columns_to_keep = missing_ratio[missing_ratio <= threshold].index.tolist()
        
        self.df = self.df[columns_to_keep]
        
        after_count = len(self.df.columns)
        cols_dropped = before_count - after_count
        
        if cols_dropped > 0:
            self.cleaning_log.append(f"✅ Dropped {cols_dropped} columns with >{threshold*100:.0f}% missing values")
        
        return self
    
    def analyze_columns_for_encoding(self, threshold: int = 50) -> pd.DataFrame:
        """
        Analyze columns and provide encoding recommendations.
        
        Args:
            threshold: Unique value threshold for recommendations
            
        Returns:
            DataFrame with column analysis and recommendations
        """
        analysis = []
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            dtype = str(self.df[col].dtype)
            
            # Determine recommendation
            if unique_count < threshold:
                recommendation = 'keep'
                encoding = 'One-Hot Encoding' if unique_count < 10 else 'Label Encoding'
            elif unique_count < threshold * 2:
                recommendation = 'review'
                encoding = 'Target Encoding or Drop'
            else:
                recommendation = 'drop'
                encoding = 'Likely ID column - Drop'
            
            analysis.append({
                'column': col,
                'dtype': dtype,
                'unique_values': unique_count,
                'recommendation': recommendation,
                'encoding_suggestion': encoding
            })
        
        return pd.DataFrame(analysis)
    
    def check_balance(self, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Check class balance in target column.
        
        Args:
            target_column: Column to check (auto-detected if None)
            
        Returns:
            Dictionary with balance analysis
        """
        # Auto-detect target column if not specified
        if target_column is None:
            # Look for common target column names
            target_names = ['target', 'label', 'class', 'quality', 'output', 'y']
            for name in target_names:
                if name in self.df.columns:
                    target_column = name
                    break
            
            # If still not found, use column with lowest cardinality
            if target_column is None:
                cardinality = {col: self.df[col].nunique() for col in self.df.columns}
                target_column = min(cardinality, key=cardinality.get)
        
        if target_column not in self.df.columns:
            return {
                'error': f"Column '{target_column}' not found",
                'target_column': target_column
            }
        
        # Calculate distribution
        distribution = self.df[target_column].value_counts().to_dict()
        
        # Calculate imbalance ratio
        if len(distribution) > 1:
            max_count = max(distribution.values())
            min_count = min(distribution.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        # Determine status
        if imbalance_ratio < 1.5:
            status = 'Balanced'
            recommendations = ['Dataset is well-balanced']
        elif imbalance_ratio < 3:
            status = 'Slightly Imbalanced'
            recommendations = ['Consider stratified sampling', 'Monitor model performance on minority class']
        elif imbalance_ratio < 5:
            status = 'Imbalanced'
            recommendations = ['Use SMOTE for oversampling', 'Apply class weights', 'Use stratified k-fold']
        else:
            status = 'Severely Imbalanced'
            recommendations = [
                'Use SMOTE or ADASYN for oversampling',
                'Apply heavy class weights',
                'Consider ensemble methods',
                'Use stratified sampling',
                'Collect more data for minority class'
            ]
        
        return {
            'target_column': target_column,
            'distribution': distribution,
            'imbalance_ratio': imbalance_ratio,
            'status': status,
            'recommendations': recommendations
        }
    
    def calculate_quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Returns:
            Quality score
        """
        # Missing value score (40% weight)
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        missing_score = max(0, 100 - missing_pct) * 0.4
        
        # Duplicate score (30% weight)
        duplicate_pct = (self.stats.get('duplicates_removed', 0) / self.stats['original_shape'][0]) * 100
        duplicate_score = max(0, 100 - duplicate_pct) * 0.3
        
        # Column consistency score (20% weight)
        # Penalize columns with very high or very low cardinality
        consistency_scores = []
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if 0.01 < unique_ratio < 0.95:  # Good range
                consistency_scores.append(100)
            elif unique_ratio <= 0.01:  # Too few unique values
                consistency_scores.append(50)
            else:  # Too many unique values (likely ID)
                consistency_scores.append(30)
        
        consistency_score = np.mean(consistency_scores) * 0.2 if consistency_scores else 0
        
        # Data type appropriateness (10% weight)
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        type_score = (len(numeric_cols) / len(self.df.columns)) * 100 * 0.1
        
        total_score = missing_score + duplicate_score + consistency_score + type_score
        
        return round(total_score, 2)
    
    def clean_pipeline(self, 
                      normalize_cols: bool = True,
                      convert_numeric: bool = True,
                      remove_dups: bool = True,
                      fill_missing: bool = True,
                      missing_strategy: str = 'median',
                      drop_high_missing_cols: bool = False,
                      col_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Execute full cleaning pipeline.
        
        Args:
            normalize_cols: Normalize column names
            convert_numeric: Convert to numeric where possible
            remove_dups: Remove duplicate rows
            fill_missing: Fill missing values
            missing_strategy: Strategy for filling ('median', 'mean', 'mode')
            drop_high_missing_cols: Drop columns with too many missing values
            col_threshold: Threshold for dropping columns
            
        Returns:
            Dictionary with cleaned DataFrame and report
        """
        self.cleaning_log = []
        
        # Step 1: Normalize column names
        if normalize_cols:
            self.normalize_column_names()
        
        # Step 2: Drop high missing columns (if enabled)
        if drop_high_missing_cols:
            self.drop_missing_columns(threshold=col_threshold)
        
        # Step 3: Convert to numeric
        if convert_numeric:
            self.convert_to_numeric()
        
        # Step 4: Remove duplicates
        if remove_dups:
            self.remove_duplicates()
        
        # Step 5: Fill missing values
        if fill_missing:
            self.fill_missing_values(strategy=missing_strategy)
        
        # Calculate final statistics
        final_stats = {
            'cleaned_shape': self.df.shape,
            'cleaned_dtypes': self.df.dtypes.value_counts().to_dict(),
            'cleaned_missing': self.df.isnull().sum().sum(),
            'rows_removed': self.stats['original_shape'][0] - self.df.shape[0],
            'cols_removed': self.stats['original_shape'][1] - self.df.shape[1]
        }
        
        self.stats.update(final_stats)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score()
        
        return {
            'cleaned_df': self.df,
            'original_df': self.original_df,
            'cleaning_log': self.cleaning_log,
            'stats': self.stats,
            'quality_score': quality_score
        }
    
    def get_cleaning_report(self) -> str:
        """
        Generate a markdown cleaning report.
        
        Returns:
            Markdown formatted report
        """
        report = "# Data Cleaning Report\n\n"
        
        report += "## Summary\n\n"
        report += f"- **Original Shape:** {self.stats['original_shape'][0]:,} rows × {self.stats['original_shape'][1]} columns\n"
        report += f"- **Cleaned Shape:** {self.df.shape[0]:,} rows × {self.df.shape[1]} columns\n"
        report += f"- **Rows Removed:** {self.stats.get('rows_removed', 0):,}\n"
        report += f"- **Columns Removed:** {self.stats.get('cols_removed', 0)}\n"
        report += f"- **Quality Score:** {self.calculate_quality_score():.1f}/100\n\n"
        
        report += "## Cleaning Steps Performed\n\n"
        for log in self.cleaning_log:
            report += f"- {log}\n"
        
        report += "\n## Data Quality Metrics\n\n"
        report += f"- **Original Missing Values:** {self.stats['original_missing']:,}\n"
        report += f"- **Cleaned Missing Values:** {self.stats.get('cleaned_missing', 0):,}\n"
        report += f"- **Duplicates Removed:** {self.stats.get('duplicates_removed', 0):,}\n"
        report += f"- **Missing Values Filled:** {self.stats.get('missing_filled', 0):,}\n"
        
        return report
