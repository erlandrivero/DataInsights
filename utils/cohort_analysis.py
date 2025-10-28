"""Cohort Analysis Utilities.

This module provides tools for cohort analysis including retention tracking,
customer lifecycle analysis, and cohort comparison visualizations.

Typical usage example:
    analyzer = CohortAnalyzer()
    cohort_data = analyzer.create_cohorts(df, 'customer_id', 'purchase_date')
    retention = analyzer.calculate_retention(cohort_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class CohortAnalyzer:
    """Handles Cohort Analysis and Retention Tracking.
    
    This class provides tools for creating cohorts, calculating retention rates,
    analyzing customer lifecycle, and generating cohort visualizations.
    
    Attributes:
        cohort_data (Optional[pd.DataFrame]): Cohort assignments.
        retention_matrix (Optional[pd.DataFrame]): Retention rates by cohort.
    
    Examples:
        >>> analyzer = CohortAnalyzer()
        >>> cohorts = analyzer.create_cohorts(df, 'user_id', 'signup_date', 'purchase_date')
        >>> retention = analyzer.calculate_retention(cohorts)
    """
    
    def __init__(self):
        """Initialize the Cohort Analyzer."""
        self.cohort_data = None
        self.retention_matrix = None
    
    @st.cache_data(ttl=1800)
    def create_cohorts(_self, 
                       df: pd.DataFrame,
                       user_col: str,
                       cohort_date_col: str,
                       activity_date_col: str,
                       cohort_period: str = 'M') -> pd.DataFrame:
        """Create cohort assignments for users.
        
        Args:
            df: DataFrame with user activity data.
            user_col: Column name for user identifier.
            cohort_date_col: Column name for cohort assignment date (e.g., signup date).
            activity_date_col: Column name for activity date (e.g., purchase date).
            cohort_period: Cohort period - 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter).
        
        Returns:
            DataFrame with cohort assignments and period numbers.
        
        Examples:
            >>> cohorts = analyzer.create_cohorts(df, 'user_id', 'signup_date', 'purchase_date')
        """
        # Convert to datetime
        df = df.copy()
        df[cohort_date_col] = pd.to_datetime(df[cohort_date_col])
        df[activity_date_col] = pd.to_datetime(df[activity_date_col])
        
        # Get cohort assignment (first date per user)
        df['CohortDate'] = df.groupby(user_col)[cohort_date_col].transform('min')
        
        # Determine cohort period
        if cohort_period == 'M':
            df['CohortPeriod'] = df['CohortDate'].dt.to_period('M')
            df['ActivityPeriod'] = df[activity_date_col].dt.to_period('M')
        elif cohort_period == 'W':
            df['CohortPeriod'] = df['CohortDate'].dt.to_period('W')
            df['ActivityPeriod'] = df[activity_date_col].dt.to_period('W')
        elif cohort_period == 'Q':
            df['CohortPeriod'] = df['CohortDate'].dt.to_period('Q')
            df['ActivityPeriod'] = df[activity_date_col].dt.to_period('Q')
        else:  # Daily
            df['CohortPeriod'] = df['CohortDate'].dt.to_period('D')
            df['ActivityPeriod'] = df[activity_date_col].dt.to_period('D')
        
        # Calculate period number (0 = cohort period, 1 = next period, etc.)
        df['PeriodNumber'] = (df['ActivityPeriod'] - df['CohortPeriod']).apply(lambda x: x.n)
        
        return df
    
    @st.cache_data(ttl=1800)
    def calculate_retention(_self, cohort_data: pd.DataFrame, user_col: str) -> pd.DataFrame:
        """Calculate retention matrix.
        
        Args:
            cohort_data: DataFrame from create_cohorts().
            user_col: Column name for user identifier.
        
        Returns:
            DataFrame with retention rates (rows=cohorts, columns=periods).
        
        Examples:
            >>> retention = analyzer.calculate_retention(cohort_data, 'user_id')
        """
        # Count unique users per cohort and period
        cohort_counts = cohort_data.groupby(['CohortPeriod', 'PeriodNumber'])[user_col].nunique().reset_index()
        cohort_counts.columns = ['CohortPeriod', 'PeriodNumber', 'UserCount']
        
        # Pivot to matrix
        retention_matrix = cohort_counts.pivot(index='CohortPeriod', 
                                               columns='PeriodNumber', 
                                               values='UserCount')
        
        # Calculate retention percentages
        cohort_sizes = retention_matrix[0]
        retention_pct = retention_matrix.divide(cohort_sizes, axis=0) * 100
        
        return retention_pct
    
    @st.cache_data(ttl=1800)
    def calculate_cohort_metrics(_self, cohort_data: pd.DataFrame, user_col: str, 
                                  value_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate cohort metrics.
        
        Args:
            cohort_data: DataFrame from create_cohorts().
            user_col: Column name for user identifier.
            value_col: Optional column for value metrics (e.g., revenue).
        
        Returns:
            DataFrame with cohort metrics.
        
        Examples:
            >>> metrics = analyzer.calculate_cohort_metrics(cohort_data, 'user_id', 'revenue')
        """
        metrics = {}
        
        for cohort in cohort_data['CohortPeriod'].unique():
            cohort_df = cohort_data[cohort_data['CohortPeriod'] == cohort]
            
            metrics[cohort] = {
                'cohort_size': cohort_df[user_col].nunique(),
                'total_users': cohort_df[user_col].nunique(),
                'total_activities': len(cohort_df),
                'avg_activities_per_user': len(cohort_df) / cohort_df[user_col].nunique()
            }
            
            if value_col and value_col in cohort_df.columns:
                metrics[cohort]['total_value'] = cohort_df[value_col].sum()
                metrics[cohort]['avg_value_per_user'] = cohort_df[value_col].sum() / cohort_df[user_col].nunique()
                metrics[cohort]['avg_value_per_activity'] = cohort_df[value_col].mean()
        
        return pd.DataFrame(metrics).T
    
    @staticmethod
    def create_retention_heatmap(retention_matrix: pd.DataFrame) -> go.Figure:
        """Create retention heatmap visualization.
        
        Args:
            retention_matrix: Retention matrix from calculate_retention().
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = CohortAnalyzer.create_retention_heatmap(retention_matrix)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = go.Figure(data=go.Heatmap(
            z=retention_matrix.values,
            x=[f"Period {int(col)}" for col in retention_matrix.columns],
            y=[str(idx) for idx in retention_matrix.index],
            colorscale='RdYlGn',
            text=retention_matrix.values.round(1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Retention %")
        ))
        
        fig.update_layout(
            title='Cohort Retention Heatmap',
            xaxis_title='Periods Since Cohort Start',
            yaxis_title='Cohort',
            height=max(400, len(retention_matrix) * 30)
        )
        
        return fig
    
    @staticmethod
    def create_retention_curves(retention_matrix: pd.DataFrame, top_n: int = 5) -> go.Figure:
        """Create retention curves for cohorts.
        
        Args:
            retention_matrix: Retention matrix from calculate_retention().
            top_n: Number of most recent cohorts to show.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = CohortAnalyzer.create_retention_curves(retention_matrix)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = go.Figure()
        
        # Show only most recent cohorts
        cohorts_to_plot = retention_matrix.tail(top_n)
        
        for cohort in cohorts_to_plot.index:
            values = retention_matrix.loc[cohort].dropna()
            fig.add_trace(go.Scatter(
                x=[f"Period {int(col)}" for col in values.index],
                y=values.values,
                mode='lines+markers',
                name=str(cohort),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'Retention Curves (Top {top_n} Cohorts)',
            xaxis_title='Periods Since Cohort Start',
            yaxis_title='Retention Rate (%)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def calculate_ltv_by_cohort(cohort_data: pd.DataFrame, user_col: str, 
                                 value_col: str, max_periods: int = 12) -> pd.DataFrame:
        """Calculate cumulative LTV by cohort.
        
        Args:
            cohort_data: DataFrame from create_cohorts().
            user_col: Column name for user identifier.
            value_col: Column name for value/revenue.
            max_periods: Maximum number of periods to calculate.
        
        Returns:
            DataFrame with cumulative LTV by cohort and period.
        
        Examples:
            >>> ltv = CohortAnalyzer.calculate_ltv_by_cohort(cohort_data, 'user_id', 'revenue')
        """
        # Filter to max periods
        cohort_data_filtered = cohort_data[cohort_data['PeriodNumber'] <= max_periods].copy()
        
        # Calculate cumulative value per user per cohort
        ltv_data = []
        
        for cohort in cohort_data_filtered['CohortPeriod'].unique():
            cohort_df = cohort_data_filtered[cohort_data_filtered['CohortPeriod'] == cohort]
            cohort_size = cohort_df[user_col].nunique()
            
            for period in range(max_periods + 1):
                period_df = cohort_df[cohort_df['PeriodNumber'] <= period]
                if len(period_df) > 0:
                    total_value = period_df[value_col].sum()
                    avg_ltv = total_value / cohort_size
                    
                    ltv_data.append({
                        'CohortPeriod': cohort,
                        'Period': period,
                        'CumulativeLTV': avg_ltv,
                        'TotalValue': total_value,
                        'CohortSize': cohort_size
                    })
        
        return pd.DataFrame(ltv_data)
    
    @staticmethod
    def create_ltv_curves(ltv_data: pd.DataFrame, top_n: int = 5) -> go.Figure:
        """Create LTV curves by cohort.
        
        Args:
            ltv_data: DataFrame from calculate_ltv_by_cohort().
            top_n: Number of cohorts to show.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = CohortAnalyzer.create_ltv_curves(ltv_data)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = go.Figure()
        
        # Get most recent cohorts
        cohorts = ltv_data['CohortPeriod'].unique()
        cohorts_to_plot = sorted(cohorts)[-top_n:]
        
        for cohort in cohorts_to_plot:
            cohort_ltv = ltv_data[ltv_data['CohortPeriod'] == cohort]
            fig.add_trace(go.Scatter(
                x=cohort_ltv['Period'],
                y=cohort_ltv['CumulativeLTV'],
                mode='lines+markers',
                name=str(cohort),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f'Cumulative LTV by Cohort (Top {top_n})',
            xaxis_title='Periods Since Cohort Start',
            yaxis_title='Cumulative LTV ($)',
            hovermode='x unified',
            height=500
        )
        
        return fig
