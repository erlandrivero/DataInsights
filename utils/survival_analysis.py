"""Survival Analysis Utilities.

This module provides tools for survival analysis including Kaplan-Meier curves,
Cox proportional hazards regression, and survival predictions.

Typical usage example:
    analyzer = SurvivalAnalyzer()
    kmf = analyzer.fit_kaplan_meier(df, 'duration', 'event')
    cox_results = analyzer.fit_cox_model(df, 'duration', 'event', covariates)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


class SurvivalAnalyzer:
    """Handles Survival Analysis and Time-to-Event Modeling.
    
    This class provides tools for survival analysis including Kaplan-Meier estimation,
    Cox proportional hazards regression, and survival curve comparisons.
    
    Attributes:
        kmf (Optional[KaplanMeierFitter]): Kaplan-Meier fitter.
        cox_model (Optional[CoxPHFitter]): Cox proportional hazards model.
    
    Examples:
        >>> analyzer = SurvivalAnalyzer()
        >>> analyzer.fit_kaplan_meier(df, 'duration', 'event')
        >>> survival_prob = analyzer.get_survival_probability(time=12)
    """
    
    def __init__(self):
        """Initialize the Survival Analyzer."""
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines package is required. Install with: pip install lifelines")
        self.kmf = None
        self.cox_model = None
        self.survival_data = None
    
    @st.cache_data(ttl=1800)
    def fit_kaplan_meier(_self, df: pd.DataFrame, duration_col: str, 
                         event_col: str, label: str = 'Overall') -> None:
        """Fit Kaplan-Meier survival curve.
        
        Args:
            df: DataFrame with survival data.
            duration_col: Column name for duration/time.
            event_col: Column name for event indicator (1=event, 0=censored).
            label: Label for the survival curve.
        
        Examples:
            >>> analyzer.fit_kaplan_meier(df, 'months', 'churned', label='All Customers')
        """
        _self.kmf = KaplanMeierFitter()
        _self.kmf.fit(
            durations=df[duration_col],
            event_observed=df[event_col],
            label=label
        )
        _self.survival_data = df[[duration_col, event_col]].copy()
    
    def get_survival_probability(self, time: float) -> float:
        """Get survival probability at specific time point.
        
        Args:
            time: Time point for survival probability.
        
        Returns:
            Survival probability (0-1).
        
        Examples:
            >>> prob = analyzer.get_survival_probability(time=12)
            >>> print(f"12-month survival: {prob:.2%}")
        """
        if self.kmf is None:
            raise ValueError("Must fit Kaplan-Meier model first")
        
        return self.kmf.survival_function_at_times(time).values[0]
    
    def get_median_survival_time(self) -> float:
        """Get median survival time.
        
        Returns:
            Median survival time.
        
        Examples:
            >>> median_time = analyzer.get_median_survival_time()
        """
        if self.kmf is None:
            raise ValueError("Must fit Kaplan-Meier model first")
        
        return self.kmf.median_survival_time_
    
    @st.cache_data(ttl=1800)
    def fit_cox_model(_self, df: pd.DataFrame, duration_col: str, 
                      event_col: str, covariate_cols: List[str]) -> Dict[str, Any]:
        """Fit Cox proportional hazards model.
        
        Args:
            df: DataFrame with survival data and covariates.
            duration_col: Column name for duration/time.
            event_col: Column name for event indicator.
            covariate_cols: List of covariate column names.
        
        Returns:
            Dictionary with model results.
        
        Examples:
            >>> results = analyzer.fit_cox_model(df, 'duration', 'event', ['age', 'treatment'])
        """
        _self.cox_model = CoxPHFitter()
        
        # Prepare data
        model_data = df[[duration_col, event_col] + covariate_cols].copy()
        
        # Fit model
        _self.cox_model.fit(
            model_data,
            duration_col=duration_col,
            event_col=event_col
        )
        
        # Extract results
        results = {
            'summary': _self.cox_model.summary,
            'hazard_ratios': np.exp(_self.cox_model.params_),
            'confidence_intervals': np.exp(_self.cox_model.confidence_intervals_),
            'concordance_index': _self.cox_model.concordance_index_,
            'log_likelihood': _self.cox_model.log_likelihood_
        }
        
        return results
    
    def predict_survival_function(self, individual_data: pd.DataFrame) -> pd.DataFrame:
        """Predict survival function for individuals.
        
        Args:
            individual_data: DataFrame with covariate values for individuals.
        
        Returns:
            DataFrame with survival probabilities over time.
        
        Examples:
            >>> survival_curves = analyzer.predict_survival_function(new_patients)
        """
        if self.cox_model is None:
            raise ValueError("Must fit Cox model first")
        
        return self.cox_model.predict_survival_function(individual_data)
    
    @staticmethod
    def perform_logrank_test(df: pd.DataFrame, duration_col: str, 
                            event_col: str, group_col: str) -> Dict[str, Any]:
        """Perform log-rank test to compare survival curves.
        
        Args:
            df: DataFrame with survival data.
            duration_col: Column name for duration/time.
            event_col: Column name for event indicator.
            group_col: Column name for group labels.
        
        Returns:
            Dictionary with test results.
        
        Examples:
            >>> test_result = SurvivalAnalyzer.perform_logrank_test(df, 'duration', 'event', 'treatment')
        """
        groups = df[group_col].unique()
        
        if len(groups) != 2:
            raise ValueError("Log-rank test requires exactly 2 groups")
        
        group1 = df[df[group_col] == groups[0]]
        group2 = df[df[group_col] == groups[1]]
        
        results = logrank_test(
            group1[duration_col], group2[duration_col],
            group1[event_col], group2[event_col]
        )
        
        return {
            'test_statistic': results.test_statistic,
            'p_value': results.p_value,
            'is_significant': results.p_value < 0.05,
            'group1': groups[0],
            'group2': groups[1]
        }
    
    @staticmethod
    def create_kaplan_meier_plot(df: pd.DataFrame, duration_col: str, 
                                 event_col: str, group_col: Optional[str] = None,
                                 title: str = 'Kaplan-Meier Survival Curve') -> go.Figure:
        """Create Kaplan-Meier survival curve plot.
        
        Args:
            df: DataFrame with survival data.
            duration_col: Column name for duration/time.
            event_col: Column name for event indicator.
            group_col: Optional column for grouping curves.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = SurvivalAnalyzer.create_kaplan_meier_plot(df, 'months', 'churned', group_col='segment')
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = go.Figure()
        
        if group_col:
            groups = df[group_col].unique()
            colors = px.colors.qualitative.Plotly
            
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=group_data[duration_col],
                    event_observed=group_data[event_col],
                    label=str(group)
                )
                
                # Add survival curve
                times = kmf.survival_function_.index
                survival_prob = kmf.survival_function_.values.flatten()
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=survival_prob,
                    mode='lines',
                    name=str(group),
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
                
                # Add confidence intervals
                ci = kmf.confidence_interval_survival_function_
                fig.add_trace(go.Scatter(
                    x=times,
                    y=ci.iloc[:, 0],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=times,
                    y=ci.iloc[:, 1],
                    mode='lines',
                    fill='tonexty',
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.2])}',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df[duration_col],
                event_observed=df[event_col]
            )
            
            times = kmf.survival_function_.index
            survival_prob = kmf.survival_function_.values.flatten()
            
            fig.add_trace(go.Scatter(
                x=times,
                y=survival_prob,
                mode='lines',
                name='Survival Probability',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add confidence intervals
            ci = kmf.confidence_interval_survival_function_
            fig.add_trace(go.Scatter(
                x=times,
                y=ci.iloc[:, 0],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=times,
                y=ci.iloc[:, 1],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Survival Probability',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_hazard_ratio_plot(cox_results: Dict[str, Any], 
                                 title: str = 'Hazard Ratios (95% CI)') -> go.Figure:
        """Create forest plot of hazard ratios.
        
        Args:
            cox_results: Results from fit_cox_model().
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = SurvivalAnalyzer.create_hazard_ratio_plot(cox_results)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        hazard_ratios = cox_results['hazard_ratios']
        ci = cox_results['confidence_intervals']
        
        fig = go.Figure()
        
        covariates = hazard_ratios.index.tolist()
        hr_values = hazard_ratios.values
        ci_lower = ci.iloc[:, 0].values
        ci_upper = ci.iloc[:, 1].values
        
        # Create forest plot
        fig.add_trace(go.Scatter(
            x=hr_values,
            y=covariates,
            mode='markers',
            marker=dict(size=10, color='darkblue'),
            name='Hazard Ratio',
            error_x=dict(
                type='data',
                symmetric=False,
                array=ci_upper - hr_values,
                arrayminus=hr_values - ci_lower
            )
        ))
        
        # Add reference line at HR=1
        fig.add_vline(x=1, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=title,
            xaxis_title='Hazard Ratio',
            yaxis_title='Covariate',
            height=max(300, len(covariates) * 50),
            showlegend=False
        )
        
        return fig
