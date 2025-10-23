import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
import json
from datetime import datetime

class ExportHelper:
    """Handles data and results export."""
    
    @staticmethod
    def export_cleaned_data(df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Export cleaned data in specified format."""
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format == 'excel':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        elif format == 'json':
            return df.to_json(orient='records', indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def export_analysis_summary(profile: Dict[str, Any], issues: List[Dict[str, Any]]) -> str:
        """Export analysis summary as JSON."""
        # Helper function to convert numpy/pandas types to native Python types
        def convert_to_serializable(obj):
            """Recursively convert non-serializable types to serializable ones."""
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
            elif hasattr(obj, 'tolist'):
                # Handle numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                # Handle custom objects
                return str(obj)
            else:
                return obj
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'basic_info': convert_to_serializable(profile.get('basic_info', {})),
            'column_info': convert_to_serializable(profile.get('column_info', [])),
            'missing_data': convert_to_serializable(profile.get('missing_data', {})),
            'quality_issues': convert_to_serializable(issues)
        }
        return json.dumps(summary, indent=2)
    
    @staticmethod
    def export_visualization(fig: go.Figure, format: str = 'png') -> bytes:
        """Export visualization in specified format."""
        if format == 'png':
            return fig.to_image(format='png')
        elif format == 'html':
            return fig.to_html().encode('utf-8')
        elif format == 'json':
            return fig.to_json().encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def create_data_dictionary(df: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Create a data dictionary document."""
        dictionary = f"""
# Data Dictionary

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

## Dataset Information
- **Total Records:** {profile['basic_info']['rows']:,}
- **Total Variables:** {profile['basic_info']['columns']}
- **Memory Usage:** {profile['basic_info']['memory_usage']}

## Variable Definitions

"""
        for col_info in profile['column_info']:
            dictionary += f"""
### {col_info['name']}
- **Data Type:** {col_info['dtype']}
- **Missing Values:** {col_info['missing']} ({col_info['missing_pct']})
- **Unique Values:** {col_info['unique']} ({col_info['unique_pct']})
- **Sample Values:** {col_info.get('sample', 'N/A')}

"""
        return dictionary
