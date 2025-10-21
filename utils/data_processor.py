import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataProcessor:
    """Handles data loading, validation, and profiling."""
    
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
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
    def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential data quality issues."""
        issues = []
        
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
