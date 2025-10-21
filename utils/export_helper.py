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
        summary = {
            'generated_at': datetime.now().isoformat(),
            'basic_info': profile['basic_info'],
            'column_info': profile['column_info'],
            'missing_data': profile['missing_data'],
            'quality_issues': issues
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
