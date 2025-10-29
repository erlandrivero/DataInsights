"""A/B Testing and Statistical Hypothesis Testing Utilities.

This module provides comprehensive tools for A/B testing, including statistical
significance tests, sample size calculations, power analysis, and visualizations.

Typical usage example:
    analyzer = ABTestAnalyzer()
    result = analyzer.run_proportion_test(control_n, control_conv, treatment_n, treatment_conv)
    sample_size = analyzer.calculate_sample_size_proportion(baseline_rate=0.10)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class ABTestAnalyzer:
    """Handles A/B Testing and Statistical Hypothesis Testing.
    
    This class provides a complete toolkit for A/B testing including proportion tests,
    t-tests, chi-square tests, sample size calculations, power analysis, and visualizations.
    
    Attributes:
        test_results (Optional[Dict]): Latest test results from hypothesis testing.
        alpha (float): Significance level (default 0.05).
        power (float): Statistical power (default 0.80).
    
    Examples:
        >>> analyzer = ABTestAnalyzer()
        >>> result = analyzer.run_proportion_test(500, 50, 500, 65)
        >>> print(f"P-value: {result['p_value']:.4f}")
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """Initialize the A/B Test Analyzer.
        
        Args:
            alpha: Significance level (Type I error rate). Default 0.05.
            power: Statistical power (1 - Type II error rate). Default 0.80.
        """
        self.test_results = None
        self.alpha = alpha
        self.power = power
    
    def run_proportion_test(
        self,
        control_n: int,
        control_conversions: int,
        treatment_n: int,
        treatment_conversions: int,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Run a two-proportion z-test.
        
        Tests if there's a significant difference between two proportions (e.g., conversion rates).
        
        Args:
            control_n: Number of observations in control group.
            control_conversions: Number of successes in control group.
            treatment_n: Number of observations in treatment group.
            treatment_conversions: Number of successes in treatment group.
            alternative: Type of test - 'two-sided', 'greater', or 'less'.
        
        Returns:
            Dictionary containing test results with p-value, lift, confidence interval, etc.
        
        Examples:
            >>> result = analyzer.run_proportion_test(1000, 100, 1000, 120)
            >>> print(f"Relative lift: {result['relative_lift']:.1f}%")
        """
        # Calculate proportions
        control_rate = control_conversions / control_n
        treatment_rate = treatment_conversions / treatment_n
        
        # Pooled proportion
        pooled_p = (control_conversions + treatment_conversions) / (control_n + treatment_n)
        pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_n + 1/treatment_n))
        
        # Z-statistic
        z_stat = (treatment_rate - control_rate) / pooled_se
        
        # P-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
        
        # Confidence interval
        se_diff = np.sqrt(control_rate * (1 - control_rate) / control_n + 
                         treatment_rate * (1 - treatment_rate) / treatment_n)
        ci_lower = (treatment_rate - control_rate) - 1.96 * se_diff
        ci_upper = (treatment_rate - control_rate) + 1.96 * se_diff
        
        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
        
        result = {
            'test_type': 'Proportion Test (Z-test)',
            'control_n': control_n,
            'control_conversions': control_conversions,
            'control_rate': control_rate,
            'treatment_n': treatment_n,
            'treatment_conversions': treatment_conversions,
            'treatment_rate': treatment_rate,
            'absolute_lift': treatment_rate - control_rate,
            'relative_lift': ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': effect_size,
            'alpha': self.alpha
        }
        
        self.test_results = result
        return result
    
    def run_ttest(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        equal_var: bool = False
    ) -> Dict[str, Any]:
        """Run an independent samples t-test.
        
        Tests if there's a significant difference between two group means.
        
        Args:
            control_data: Array of observations from control group.
            treatment_data: Array of observations from treatment group.
            equal_var: Assume equal variances (Welch's t-test if False).
        
        Returns:
            Dictionary containing test results.
        
        Examples:
            >>> control = np.random.normal(100, 15, 1000)
            >>> treatment = np.random.normal(105, 15, 1000)
            >>> result = analyzer.run_ttest(control, treatment)
        """
        # Calculate statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_std = np.std(control_data, ddof=1)
        treatment_std = np.std(treatment_data, ddof=1)
        
        # Run t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=equal_var)
        
        # Confidence interval
        n1, n2 = len(control_data), len(treatment_data)
        se_diff = np.sqrt(control_std**2/n1 + treatment_std**2/n2)
        df = n1 + n2 - 2 if equal_var else min(n1-1, n2-1)
        t_crit = stats.t.ppf(0.975, df)
        diff = treatment_mean - control_mean
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*control_std**2 + (n2-1)*treatment_std**2) / (n1+n2-2))
        effect_size = (treatment_mean - control_mean) / pooled_std
        
        result = {
            'test_type': "Welch's t-test" if not equal_var else "Student's t-test",
            'control_n': n1,
            'control_mean': control_mean,
            'control_std': control_std,
            'treatment_n': n2,
            'treatment_mean': treatment_mean,
            'treatment_std': treatment_std,
            'absolute_diff': diff,
            'relative_diff': (diff / control_mean * 100) if control_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': effect_size,
            'alpha': self.alpha
        }
        
        self.test_results = result
        return result
    
    def run_chi_square_test(
        self,
        contingency_table: np.ndarray
    ) -> Dict[str, Any]:
        """Run a chi-square test of independence.
        
        Tests if there's a significant association between two categorical variables.
        
        Args:
            contingency_table: 2D array with observed frequencies.
        
        Returns:
            Dictionary containing test results.
        
        Examples:
            >>> table = np.array([[50, 450], [65, 435]])
            >>> result = analyzer.run_chi_square_test(table)
        """
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Cramér's V effect size
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
        
        result = {
            'test_type': 'Chi-Square Test of Independence',
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < self.alpha,
            'expected_frequencies': expected,
            'observed_frequencies': contingency_table,
            'effect_size': cramers_v,
            'alpha': self.alpha
        }
        
        self.test_results = result
        return result
    
    def calculate_sample_size_proportion(
        self,
        baseline_rate: float,
        min_detectable_effect: float = 0.1,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Calculate required sample size for proportion test.
        
        Args:
            baseline_rate: Baseline conversion rate (e.g., 0.10 for 10%).
            min_detectable_effect: Minimum detectable relative lift (e.g., 0.10 for 10% lift).
            alternative: Type of test - 'two-sided' or 'one-sided'.
        
        Returns:
            Dictionary with required sample sizes.
        
        Examples:
            >>> result = analyzer.calculate_sample_size_proportion(0.10, 0.20)
            >>> print(f"Need {result['sample_size_per_group']:,} per group")
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + min_detectable_effect)
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha/2) if alternative == 'two-sided' else stats.norm.ppf(1 - self.alpha)
        z_beta = stats.norm.ppf(self.power)
        
        # Pooled proportion
        p_bar = (p1 + p2) / 2
        
        # Sample size formula
        n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (p2 - p1)**2
        
        n = int(np.ceil(n))
        
        return {
            'sample_size_per_group': n,
            'total_sample_size': 2 * n,
            'baseline_rate': p1,
            'treatment_rate': p2,
            'absolute_lift': p2 - p1,
            'relative_lift': min_detectable_effect * 100,
            'power': self.power,
            'alpha': self.alpha,
            'alternative': alternative
        }
    
    def calculate_sample_size_means(
        self,
        mean_diff: float,
        std_dev: float,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Calculate required sample size for t-test.
        
        Args:
            mean_diff: Expected difference in means to detect.
            std_dev: Standard deviation (assumed equal for both groups).
            alternative: Type of test - 'two-sided' or 'one-sided'.
        
        Returns:
            Dictionary with required sample sizes.
        
        Examples:
            >>> result = analyzer.calculate_sample_size_means(5.0, 15.0)
            >>> print(f"Need {result['sample_size_per_group']:,} per group")
        """
        effect_size = mean_diff / std_dev
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - self.alpha/2) if alternative == 'two-sided' else stats.norm.ppf(1 - self.alpha)
        z_beta = stats.norm.ppf(self.power)
        
        # Sample size formula
        n = 2 * ((z_alpha + z_beta) / effect_size)**2
        n = int(np.ceil(n))
        
        return {
            'sample_size_per_group': n,
            'total_sample_size': 2 * n,
            'mean_diff': mean_diff,
            'std_dev': std_dev,
            'effect_size': effect_size,
            'power': self.power,
            'alpha': self.alpha,
            'alternative': alternative
        }
    
    def calculate_test_duration(
        self,
        required_sample_size: int,
        daily_traffic: int
    ) -> Dict[str, int]:
        """Calculate how long to run the test.
        
        Args:
            required_sample_size: Total sample size needed (both groups).
            daily_traffic: Average daily visitors/users.
        
        Returns:
            Dictionary with test duration in days and weeks.
        
        Examples:
            >>> duration = analyzer.calculate_test_duration(10000, 500)
            >>> print(f"Run test for {duration['days']} days")
        """
        days = int(np.ceil(required_sample_size / daily_traffic))
        weeks = int(np.ceil(days / 7))
        
        return {
            'days': days,
            'weeks': weeks,
            'required_sample_size': required_sample_size,
            'daily_traffic': daily_traffic
        }
    
    @staticmethod
    def create_ab_test_visualization(result: Dict[str, Any]) -> go.Figure:
        """Create visualization for A/B test results.
        
        Args:
            result: Test result dictionary.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = ABTestAnalyzer.create_ab_test_visualization(test_result)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        if 'control_rate' in result:  # Proportion test
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Control', 'Treatment'],
                y=[result['control_rate'] * 100, result['treatment_rate'] * 100],
                text=[f"{result['control_rate']*100:.2f}%", f"{result['treatment_rate']*100:.2f}%"],
                textposition='auto',
                marker_color=['#1f77b4', '#2ca02c' if result['is_significant'] else '#ff7f0e'],
                name='Conversion Rate'
            ))
            
            fig.update_layout(
                title='A/B Test Results - Conversion Rates',
                xaxis_title='Group',
                yaxis_title='Conversion Rate (%)',
                showlegend=False,
                height=400
            )
            
        else:  # T-test
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Control', 'Treatment'],
                y=[result['control_mean'], result['treatment_mean']],
                text=[f"{result['control_mean']:.2f}", f"{result['treatment_mean']:.2f}"],
                textposition='auto',
                marker_color=['#1f77b4', '#2ca02c' if result['is_significant'] else '#ff7f0e'],
                error_y=dict(
                    type='data',
                    array=[result['control_std'], result['treatment_std']],
                    visible=True
                ),
                name='Mean Value'
            ))
            
            fig.update_layout(
                title='A/B Test Results - Mean Comparison',
                xaxis_title='Group',
                yaxis_title='Mean Value',
                showlegend=False,
                height=400
            )
        
        return fig
    
    @staticmethod
    def interpret_effect_size(effect_size: float, test_type: str = 'cohens_d') -> str:
        """Interpret the magnitude of the effect size.
        
        Args:
            effect_size: Calculated effect size value.
            test_type: Type of effect size - 'cohens_d', 'cohens_h', or 'cramers_v'.
        
        Returns:
            String interpretation.
        
        Examples:
            >>> interpretation = ABTestAnalyzer.interpret_effect_size(0.5, 'cohens_d')
            >>> print(interpretation)  # "Medium"
        """
        abs_effect = abs(effect_size)
        
        if test_type in ['cohens_d', 'cohens_h']:
            if abs_effect < 0.2:
                return "Negligible"
            elif abs_effect < 0.5:
                return "Small"
            elif abs_effect < 0.8:
                return "Medium"
            else:
                return "Large"
        elif test_type == 'cramers_v':
            if abs_effect < 0.1:
                return "Negligible"
            elif abs_effect < 0.3:
                return "Small"
            elif abs_effect < 0.5:
                return "Medium"
            else:
                return "Large"
        
        return "Unknown"
    
    @staticmethod
    def obrien_fleming_bounds(information_fraction: float, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate O'Brien-Fleming spending function boundaries.
        
        O'Brien-Fleming is a conservative approach that spends alpha slowly initially
        and more rapidly toward the end of the trial.
        
        Args:
            information_fraction: Fraction of total information observed (0-1).
            alpha: Overall significance level.
        
        Returns:
            Tuple of (lower_bound, upper_bound) for Z-statistic.
        
        Examples:
            >>> bounds = ABTestAnalyzer.obrien_fleming_bounds(0.5, 0.05)
            >>> print(f"Bounds at 50% information: {bounds}")
        """
        if information_fraction <= 0 or information_fraction > 1:
            return (0, 0)
        
        # O'Brien-Fleming spending function
        z_alpha = stats.norm.ppf(1 - alpha/2)
        boundary = z_alpha / np.sqrt(information_fraction)
        
        return (-boundary, boundary)
    
    @staticmethod
    def pocock_bounds(num_looks: int, look_number: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate Pocock spending function boundaries.
        
        Pocock uses constant boundaries across all interim analyses.
        
        Args:
            num_looks: Total number of planned interim analyses.
            look_number: Current look number (1-indexed).
            alpha: Overall significance level.
        
        Returns:
            Tuple of (lower_bound, upper_bound) for Z-statistic.
        
        Examples:
            >>> bounds = ABTestAnalyzer.pocock_bounds(5, 3, 0.05)
        """
        # Pocock constant (approximate for two-sided tests)
        if num_looks == 1:
            c = stats.norm.ppf(1 - alpha/2)
        elif num_looks == 2:
            c = 1.977
        elif num_looks == 3:
            c = 2.289
        elif num_looks == 4:
            c = 2.361
        elif num_looks == 5:
            c = 2.413
        else:
            # Approximation for larger number of looks
            c = stats.norm.ppf(1 - alpha/(2*num_looks))
        
        return (-c, c)
    
    def sequential_test_proportion(
        self,
        control_n: int,
        control_conversions: int,
        treatment_n: int,
        treatment_conversions: int,
        information_fraction: float,
        spending_function: str = 'obrien_fleming',
        num_looks: int = 5,
        look_number: int = 1
    ) -> Dict[str, Any]:
        """Run sequential A/B test with alpha spending.
        
        Allows for early stopping based on alpha spending functions,
        reducing required sample size and test duration.
        
        Args:
            control_n: Sample size for control group.
            control_conversions: Number of conversions in control.
            treatment_n: Sample size for treatment group.
            treatment_conversions: Number of conversions in treatment.
            information_fraction: Fraction of planned sample size reached (0-1).
            spending_function: 'obrien_fleming' or 'pocock'.
            num_looks: Total number of planned interim analyses.
            look_number: Current look number (1-indexed).
        
        Returns:
            Dictionary with test results and stopping decision.
        
        Examples:
            >>> result = analyzer.sequential_test_proportion(
            ...     250, 25, 250, 35, information_fraction=0.5,
            ...     spending_function='obrien_fleming'
            ... )
        """
        # Calculate standard proportion test
        base_result = self.run_proportion_test(
            control_n, control_conversions,
            treatment_n, treatment_conversions
        )
        
        # Calculate Z-statistic
        control_rate = control_conversions / control_n if control_n > 0 else 0
        treatment_rate = treatment_conversions / treatment_n if treatment_n > 0 else 0
        pooled_rate = (control_conversions + treatment_conversions) / (control_n + treatment_n)
        
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_n + 1/treatment_n))
        z_stat = (treatment_rate - control_rate) / se if se > 0 else 0
        
        # Get boundaries based on spending function
        if spending_function == 'obrien_fleming':
            lower_bound, upper_bound = self.obrien_fleming_bounds(information_fraction, self.alpha)
        else:  # pocock
            lower_bound, upper_bound = self.pocock_bounds(num_looks, look_number, self.alpha)
        
        # Determine stopping decision
        stop_for_efficacy = z_stat > upper_bound
        stop_for_futility = z_stat < lower_bound
        should_stop = stop_for_efficacy or stop_for_futility
        
        # Calculate remaining sample size needed
        remaining_fraction = 1 - information_fraction
        total_planned = (control_n + treatment_n) / information_fraction if information_fraction > 0 else 0
        remaining_needed = int(total_planned * remaining_fraction)
        
        result = {
            **base_result,
            'z_statistic': z_stat,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'information_fraction': information_fraction,
            'should_stop': should_stop,
            'stop_for_efficacy': stop_for_efficacy,
            'stop_for_futility': stop_for_futility,
            'spending_function': spending_function,
            'look_number': look_number,
            'num_looks': num_looks,
            'samples_collected': control_n + treatment_n,
            'remaining_samples_needed': remaining_needed if not should_stop else 0,
            'potential_savings_pct': remaining_fraction * 100 if should_stop else 0
        }
        
        return result
    
    @staticmethod
    def create_sequential_boundary_plot(
        alpha: float = 0.05,
        spending_function: str = 'obrien_fleming',
        num_looks: int = 5
    ) -> go.Figure:
        """Create visualization of sequential testing boundaries.
        
        Shows how the decision boundaries evolve as information accrues.
        
        Args:
            alpha: Significance level.
            spending_function: 'obrien_fleming' or 'pocock'.
            num_looks: Number of planned interim analyses.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = ABTestAnalyzer.create_sequential_boundary_plot(0.05, 'obrien_fleming', 5)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        information_fractions = np.linspace(0.2, 1.0, 50)
        upper_bounds = []
        lower_bounds = []
        
        for info_frac in information_fractions:
            if spending_function == 'obrien_fleming':
                lower, upper = ABTestAnalyzer.obrien_fleming_bounds(info_frac, alpha)
            else:
                # For Pocock, approximate the look number
                look_num = int(info_frac * num_looks) + 1
                lower, upper = ABTestAnalyzer.pocock_bounds(num_looks, look_num, alpha)
            
            upper_bounds.append(upper)
            lower_bounds.append(lower)
        
        fig = go.Figure()
        
        # Upper boundary (efficacy)
        fig.add_trace(go.Scatter(
            x=information_fractions * 100,
            y=upper_bounds,
            mode='lines',
            name='Stop for Efficacy',
            line=dict(color='green', width=2)
        ))
        
        # Lower boundary (futility)
        fig.add_trace(go.Scatter(
            x=information_fractions * 100,
            y=lower_bounds,
            mode='lines',
            name='Stop for Futility',
            line=dict(color='red', width=2)
        ))
        
        # Continue region
        fig.add_trace(go.Scatter(
            x=information_fractions * 100,
            y=[0] * len(information_fractions),
            mode='lines',
            name='Continue Testing',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        title = f'Sequential Testing Boundaries ({spending_function.replace("_", " ").title()})'
        fig.update_layout(
            title=title,
            xaxis_title='Information Fraction (%)',
            yaxis_title='Z-Statistic',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig
