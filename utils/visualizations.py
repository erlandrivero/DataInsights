"""Visualization utilities for DataInsights with caching and type hints.

This module provides comprehensive chart creation and suggestion capabilities
with Streamlit caching for improved performance.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional


class Visualizer:
    """Handles data visualization creation with smart suggestions.
    
    This class provides static methods for creating various chart types
    and suggesting appropriate visualizations based on data characteristics.
    
    Attributes:
        None - All methods are static
    
    Example:
        >>> # Get visualization suggestions (cached)
        >>> suggestions = Visualizer.suggest_visualizations(df)
        >>> 
        >>> # Create specific charts
        >>> fig = Visualizer.create_histogram(df, 'age')
        >>> st.plotly_chart(fig)
    """
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data types with 30-min caching.
        
        Analyzes the DataFrame's column types and suggests relevant visualizations
        including histograms, bar charts, scatter plots, box plots, and correlation
        heatmaps. Results are cached to improve performance.
        
        Args:
            df: DataFrame to analyze for visualization suggestions
        
        Returns:
            List of suggestion dictionaries, each containing:
                - type (str): Chart type ('histogram', 'bar', 'scatter', 'box', 'correlation')
                - title (str): Suggested title for the chart
                - columns (list): Column names to use in the visualization
                - description (str): Human-readable description of what the chart shows
        
        Example:
            >>> suggestions = Visualizer.suggest_visualizations(df)
            >>> for suggestion in suggestions:
            >>>     st.write(f"{suggestion['title']}: {suggestion['description']}")
        
        Note:
            - Limits to first 3 numeric and 3 categorical columns to avoid overload
            - Requires at least 2 numeric columns for scatter plots
            - Requires at least 3 numeric columns for correlation heatmap
            - Cache persists for 30 minutes (1800 seconds)
        """
        suggestions: List[Dict[str, Any]] = []
        
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
    def create_histogram(
        df: pd.DataFrame, 
        column: str, 
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive histogram chart.
        
        Generates a Plotly histogram showing the distribution of values
        in a numeric column.
        
        Args:
            df: DataFrame containing the data
            column: Name of the column to visualize
            title: Optional custom title (defaults to "Distribution of {column}")
        
        Returns:
            Plotly Figure object ready to display with st.plotly_chart()
        
        Example:
            >>> fig = Visualizer.create_histogram(df, 'age', 'Age Distribution')
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            Chart height is fixed at 400px
            Uses blue color scheme (#1f77b4)
        """
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
    def create_bar_chart(
        df: pd.DataFrame, 
        column: str, 
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive bar chart for categorical data.
        
        Generates a Plotly bar chart showing the top 10 most frequent
        values in a categorical column.
        
        Args:
            df: DataFrame containing the data
            column: Name of the categorical column to visualize
            title: Optional custom title (defaults to "Top 10 {column} Values")
        
        Returns:
            Plotly Figure object showing value counts
        
        Example:
            >>> fig = Visualizer.create_bar_chart(df, 'category')
            >>> st.plotly_chart(fig)
        
        Note:
            - Automatically limits to top 10 values
            - Chart height is fixed at 400px
            - Uses blue color scheme (#1f77b4)
        """
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
    def create_scatter(
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive scatter plot.
        
        Generates a Plotly scatter plot showing the relationship between
        two numeric columns.
        
        Args:
            df: DataFrame containing the data
            x_col: Name of the column for x-axis
            y_col: Name of the column for y-axis
            title: Optional custom title (defaults to "{x_col} vs {y_col}")
        
        Returns:
            Plotly Figure object showing the scatter plot
        
        Example:
            >>> fig = Visualizer.create_scatter(df, 'age', 'salary')
            >>> st.plotly_chart(fig)
        
        Note:
            Chart height is fixed at 400px
            Uses blue color scheme (#1f77b4)
        """
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
    def create_box_plot(
        df: pd.DataFrame, 
        column: str, 
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive box plot for outlier detection.
        
        Generates a Plotly box plot showing the distribution, quartiles,
        and potential outliers in a numeric column.
        
        Args:
            df: DataFrame containing the data
            column: Name of the numeric column to visualize
            title: Optional custom title (defaults to "Box Plot of {column}")
        
        Returns:
            Plotly Figure object showing the box plot
        
        Example:
            >>> fig = Visualizer.create_box_plot(df, 'price')
            >>> st.plotly_chart(fig)
        
        Note:
            - Box plot automatically shows median, quartiles, and outliers
            - Chart height is fixed at 400px
            - Uses blue color scheme (#1f77b4)
        """
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
    def create_correlation_heatmap(
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None, 
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive correlation heatmap.
        
        Generates a Plotly heatmap showing Pearson correlations between
        numeric columns. Useful for identifying relationships and multicollinearity.
        
        Args:
            df: DataFrame containing the data
            columns: Optional list of column names to include (defaults to all numeric columns)
            title: Optional custom title (defaults to "Correlation Heatmap")
        
        Returns:
            Plotly Figure object showing the correlation matrix
        
        Example:
            >>> fig = Visualizer.create_correlation_heatmap(df)
            >>> st.plotly_chart(fig)
            >>> 
            >>> # Or specify columns
            >>> fig = Visualizer.create_correlation_heatmap(
            >>>     df, 
            >>>     columns=['age', 'income', 'score']
            >>> )
        
        Note:
            - Uses RdBu (Red-Blue) colorscale with white at 0 correlation
            - Shows correlation values as text on heatmap
            - Chart size is 500x500px
            - Values are rounded to 2 decimal places
        """
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
