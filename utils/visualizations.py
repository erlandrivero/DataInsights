import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

class Visualizer:
    """Handles data visualization creation."""
    
    @staticmethod
    def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data types."""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'columns': [col],
                'description': f'Shows the distribution of values in {col}'
            })
        
        # Bar charts for categorical columns
        for col in categorical_cols[:3]:
            suggestions.append({
                'type': 'bar',
                'title': f'Count of {col}',
                'columns': [col],
                'description': f'Shows the frequency of each category in {col}'
            })
        
        # Scatter plots for numeric pairs
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': f'{numeric_cols[0]} vs {numeric_cols[1]}',
                'columns': [numeric_cols[0], numeric_cols[1]],
                'description': f'Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}'
            })
        
        # Box plots for numeric columns
        for col in numeric_cols[:2]:
            suggestions.append({
                'type': 'box',
                'title': f'Box Plot of {col}',
                'columns': [col],
                'description': f'Shows distribution and outliers in {col}'
            })
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 3:
            suggestions.append({
                'type': 'correlation',
                'title': 'Correlation Heatmap',
                'columns': numeric_cols,
                'description': 'Shows correlations between numeric columns'
            })
        
        return suggestions
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a histogram."""
        fig = px.histogram(
            df,
            x=column,
            title=title or f'Distribution of {column}',
            labels={column: column},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            showlegend=False,
            height=400
        )
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a bar chart."""
        value_counts = df[column].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title=title or f'Top 10 {column} Values',
            xaxis_title=column,
            yaxis_title='Count',
            height=400
        )
        return fig
    
    @staticmethod
    def create_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str = None) -> go.Figure:
        """Create a scatter plot."""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title or f'{x_col} vs {y_col}',
            labels={x_col: x_col, y_col: y_col},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a box plot."""
        fig = px.box(
            df,
            y=column,
            title=title or f'Box Plot of {column}',
            labels={column: column},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None, title: str = None) -> go.Figure:
        """Create a correlation heatmap."""
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title or 'Correlation Heatmap',
            height=500,
            width=500
        )
        return fig
