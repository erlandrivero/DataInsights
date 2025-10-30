# Implementation Guide: Cohort Comparison Feature

**Module:** Cohort Analysis  
**Time Required:** 3 hours  
**Difficulty:** Medium  
**Priority:** 🔥 HIGH  
**Impact:** Essential for cohort analysis

---

## What is Cohort Comparison?

**Cohort Comparison** allows you to statistically compare retention rates between two or more cohorts to determine which performs better and whether the difference is significant.

### Example Use Cases:

1. **Product Changes:** Did the Q1 2024 cohort (after new feature) retain better than Q4 2023 cohort (before feature)?

2. **Marketing Campaigns:** Which acquisition channel brings users with better retention?

3. **Seasonal Effects:** Do summer cohorts perform differently than winter cohorts?

4. **A/B Test Validation:** Did users in the test group have better long-term retention?

### Business Value:

- **Identify successful changes** - Know what improved retention
- **Optimize acquisition** - Focus on channels with best retention
- **Predict LTV** - Better-retaining cohorts have higher lifetime value
- **Resource allocation** - Invest in strategies that work

---

## Implementation Steps

### Step 1: Add Comparison Function to `utils/cohort_analysis.py`

**Location:** Add as a method to the `CohortAnalyzer` class

```python
def compare_cohorts(
    self,
    retention_matrix: pd.DataFrame,
    cohort1: str,
    cohort2: str,
    test_type: str = 'ttest'
) -> Dict[str, Any]:
    """
    Compare two cohorts statistically to determine if retention differs significantly.
    
    Args:
        retention_matrix: Retention matrix from analyze_cohorts()
        cohort1: First cohort identifier (must be in retention_matrix.index)
        cohort2: Second cohort identifier (must be in retention_matrix.index)
        test_type: Statistical test to use ('ttest', 'mannwhitney', or 'wilcoxon')
    
    Returns:
        Dictionary containing:
            - cohort1: First cohort name
            - cohort2: Second cohort name
            - cohort1_avg_retention: Average retention rate for cohort 1
            - cohort2_avg_retention: Average retention rate for cohort 2
            - retention_diff: Difference in retention rates (cohort2 - cohort1)
            - retention_pct_change: Percentage change in retention
            - is_significant: Boolean indicating statistical significance (p < 0.05)
            - p_value: P-value from statistical test
            - test_statistic: Test statistic value
            - test_type: Type of test performed
            - winner: Which cohort has better retention
            - effect_size: Cohen's d effect size
            - confidence_interval: 95% CI for the difference
    
    Examples:
        >>> analyzer = CohortAnalyzer()
        >>> result = analyzer.analyze_cohorts(df, 'user_id', 'signup_date', 'activity_date')
        >>> retention_matrix = result['retention_matrix']
        >>> comparison = analyzer.compare_cohorts(retention_matrix, '2024-01', '2024-02')
        >>> print(f"Winner: {comparison['winner']}, P-value: {comparison['p_value']:.4f}")
    """
    from scipy import stats
    import numpy as np
    
    # Validate inputs
    if cohort1 not in retention_matrix.index:
        raise ValueError(f"Cohort '{cohort1}' not found in retention matrix")
    if cohort2 not in retention_matrix.index:
        raise ValueError(f"Cohort '{cohort2}' not found in retention matrix")
    if cohort1 == cohort2:
        raise ValueError("Cannot compare a cohort to itself")
    
    # Get retention rates for both cohorts
    cohort1_retention = retention_matrix.loc[cohort1].dropna()
    cohort2_retention = retention_matrix.loc[cohort2].dropna()
    
    # Calculate summary statistics
    cohort1_avg = cohort1_retention.mean()
    cohort2_avg = cohort2_retention.mean()
    
    retention_diff = cohort2_avg - cohort1_avg
    retention_pct_change = (retention_diff / cohort1_avg * 100) if cohort1_avg > 0 else 0
    
    # Perform statistical test
    if test_type == 'ttest':
        # Independent samples t-test
        test_stat, p_value = stats.ttest_ind(cohort1_retention, cohort2_retention)
    elif test_type == 'mannwhitney':
        # Mann-Whitney U test (non-parametric)
        test_stat, p_value = stats.mannwhitneyu(cohort1_retention, cohort2_retention, alternative='two-sided')
    elif test_type == 'wilcoxon':
        # Wilcoxon signed-rank test (paired)
        # Pad shorter series with NaN
        max_len = max(len(cohort1_retention), len(cohort2_retention))
        c1_padded = np.pad(cohort1_retention, (0, max_len - len(cohort1_retention)), constant_values=np.nan)
        c2_padded = np.pad(cohort2_retention, (0, max_len - len(cohort2_retention)), constant_values=np.nan)
        
        # Remove pairs where either is NaN
        mask = ~(np.isnan(c1_padded) | np.isnan(c2_padded))
        test_stat, p_value = stats.wilcoxon(c1_padded[mask], c2_padded[mask])
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Determine winner
    if cohort2_avg > cohort1_avg:
        winner = cohort2
        winner_advantage = retention_pct_change
    else:
        winner = cohort1
        winner_advantage = -retention_pct_change
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((cohort1_retention.std()**2 + cohort2_retention.std()**2) / 2)
    cohens_d = retention_diff / pooled_std if pooled_std > 0 else 0
    
    # Calculate 95% confidence interval for the difference
    se_diff = np.sqrt(cohort1_retention.var() / len(cohort1_retention) + 
                      cohort2_retention.var() / len(cohort2_retention))
    ci_lower = retention_diff - 1.96 * se_diff
    ci_upper = retention_diff + 1.96 * se_diff
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        'cohort1': cohort1,
        'cohort2': cohort2,
        'cohort1_avg_retention': cohort1_avg,
        'cohort2_avg_retention': cohort2_avg,
        'cohort1_size': len(cohort1_retention),
        'cohort2_size': len(cohort2_retention),
        'retention_diff': retention_diff,
        'retention_pct_change': retention_pct_change,
        'is_significant': p_value < 0.05,
        'p_value': p_value,
        'test_statistic': test_stat,
        'test_type': test_type,
        'winner': winner,
        'winner_advantage': winner_advantage,
        'effect_size': cohens_d,
        'effect_interpretation': effect_interpretation,
        'confidence_interval': (ci_lower, ci_upper),
        'alpha': 0.05
    }


def compare_all_cohorts(
    self,
    retention_matrix: pd.DataFrame,
    test_type: str = 'ttest'
) -> pd.DataFrame:
    """
    Compare all pairs of cohorts and return a comparison matrix.
    
    Args:
        retention_matrix: Retention matrix from analyze_cohorts()
        test_type: Statistical test to use
    
    Returns:
        DataFrame with pairwise comparison results
    """
    cohorts = retention_matrix.index.tolist()
    comparisons = []
    
    for i, cohort1 in enumerate(cohorts):
        for cohort2 in cohorts[i+1:]:
            result = self.compare_cohorts(retention_matrix, cohort1, cohort2, test_type)
            comparisons.append({
                'Cohort 1': cohort1,
                'Cohort 2': cohort2,
                'Avg Retention 1': f"{result['cohort1_avg_retention']:.1%}",
                'Avg Retention 2': f"{result['cohort2_avg_retention']:.1%}",
                'Difference': f"{result['retention_diff']:.1%}",
                'Change (%)': f"{result['retention_pct_change']:.1f}%",
                'P-Value': f"{result['p_value']:.4f}",
                'Significant': '✅' if result['is_significant'] else '❌',
                'Winner': result['winner'],
                'Effect Size': f"{result['effect_size']:.2f} ({result['effect_interpretation']})"
            })
    
    return pd.DataFrame(comparisons)
```

---

### Step 2: Add Comparison UI to `app.py`

**Location:** In `show_cohort_analysis()` function, after retention matrix is displayed

```python
# After retention matrix visualization
st.divider()
st.subheader("🔄 Cohort Comparison")

with st.expander("ℹ️ What is Cohort Comparison?"):
    st.markdown("""
    **Cohort Comparison** allows you to statistically test whether two cohorts have 
    significantly different retention rates.
    
    ### Use Cases:
    - **Product Changes:** Did retention improve after a new feature?
    - **Marketing Campaigns:** Which acquisition channel retains better?
    - **Seasonal Effects:** Do summer cohorts differ from winter cohorts?
    - **A/B Test Validation:** Did the test group retain better long-term?
    
    ### Statistical Tests:
    - **T-Test:** Assumes normal distribution (most common)
    - **Mann-Whitney U:** Non-parametric (no distribution assumption)
    - **Wilcoxon:** Paired comparison (same time periods)
    
    ### Interpretation:
    - **P-value < 0.05:** Statistically significant difference
    - **Effect Size:** Magnitude of the difference (small/medium/large)
    """)

# Get cohorts from retention matrix
if 'cohort_retention' in st.session_state:
    retention_matrix = st.session_state.cohort_retention
    cohorts = retention_matrix.index.tolist()
    
    if len(cohorts) >= 2:
        # Comparison mode selection
        comparison_mode = st.radio(
            "Comparison Mode:",
            ["Compare Two Cohorts", "Compare All Cohorts"],
            horizontal=True
        )
        
        if comparison_mode == "Compare Two Cohorts":
            # Two-cohort comparison
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                cohort1 = st.selectbox(
                    "Select first cohort:",
                    cohorts,
                    key="cohort_compare_1"
                )
            
            with col2:
                # Filter out cohort1 from second selection
                cohort2_options = [c for c in cohorts if c != cohort1]
                cohort2 = st.selectbox(
                    "Select second cohort:",
                    cohort2_options,
                    index=0 if len(cohort2_options) > 0 else None,
                    key="cohort_compare_2"
                )
            
            with col3:
                test_type = st.selectbox(
                    "Test Type:",
                    ["ttest", "mannwhitney", "wilcoxon"],
                    format_func=lambda x: {
                        'ttest': 'T-Test',
                        'mannwhitney': 'Mann-Whitney',
                        'wilcoxon': 'Wilcoxon'
                    }[x]
                )
            
            if st.button("🔄 Compare Cohorts", use_container_width=True, type="primary"):
                try:
                    from utils.cohort_analysis import CohortAnalyzer
                    analyzer = CohortAnalyzer()
                    
                    with st.spinner("Comparing cohorts..."):
                        comparison = analyzer.compare_cohorts(
                            retention_matrix,
                            cohort1,
                            cohort2,
                            test_type
                        )
                    
                    # Store in session state
                    st.session_state.cohort_comparison = comparison
                    
                    # Display results
                    st.markdown(f"### Comparison: {cohort1} vs {cohort2}")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            f"{cohort1} Avg Retention",
                            f"{comparison['cohort1_avg_retention']:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            f"{cohort2} Avg Retention",
                            f"{comparison['cohort2_avg_retention']:.1%}",
                            delta=f"{comparison['retention_pct_change']:.1f}%"
                        )
                    
                    with col3:
                        if comparison['is_significant']:
                            st.metric(
                                "Statistical Significance",
                                "✅ Significant",
                                delta=f"p={comparison['p_value']:.4f}"
                            )
                        else:
                            st.metric(
                                "Statistical Significance",
                                "❌ Not Significant",
                                delta=f"p={comparison['p_value']:.4f}"
                            )
                    
                    with col4:
                        st.metric(
                            "Winner",
                            comparison['winner'],
                            delta=f"+{comparison['winner_advantage']:.1f}%"
                        )
                    
                    # Detailed results
                    st.markdown("#### Detailed Results")
                    
                    results_df = pd.DataFrame({
                        'Metric': [
                            'Cohort 1 Average Retention',
                            'Cohort 2 Average Retention',
                            'Absolute Difference',
                            'Relative Change (%)',
                            'P-Value',
                            'Statistically Significant',
                            'Test Type',
                            'Test Statistic',
                            'Effect Size (Cohen\'s d)',
                            'Effect Interpretation',
                            '95% Confidence Interval',
                            'Winner',
                            'Winner Advantage'
                        ],
                        'Value': [
                            f"{comparison['cohort1_avg_retention']:.2%}",
                            f"{comparison['cohort2_avg_retention']:.2%}",
                            f"{comparison['retention_diff']:.2%}",
                            f"{comparison['retention_pct_change']:.1f}%",
                            f"{comparison['p_value']:.6f}",
                            'Yes' if comparison['is_significant'] else 'No',
                            comparison['test_type'].upper(),
                            f"{comparison['test_statistic']:.4f}",
                            f"{comparison['effect_size']:.3f}",
                            comparison['effect_interpretation'].title(),
                            f"[{comparison['confidence_interval'][0]:.2%}, {comparison['confidence_interval'][1]:.2%}]",
                            comparison['winner'],
                            f"{comparison['winner_advantage']:.1f}%"
                        ]
                    })
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    st.markdown("#### Interpretation")
                    
                    if comparison['is_significant']:
                        st.success(f"""
                        **Statistically Significant Difference Found!**
                        
                        {comparison['winner']} has **{comparison['winner_advantage']:.1f}% better** 
                        average retention than the other cohort. This difference is statistically 
                        significant (p = {comparison['p_value']:.4f} < 0.05), meaning it's unlikely 
                        to be due to random chance.
                        
                        **Effect Size:** {comparison['effect_interpretation'].title()} 
                        (Cohen's d = {comparison['effect_size']:.2f})
                        
                        **Recommendation:** Focus on understanding what made {comparison['winner']} 
                        more successful and replicate those factors in future cohorts.
                        """)
                    else:
                        st.info(f"""
                        **No Statistically Significant Difference**
                        
                        While {comparison['winner']} has slightly better retention 
                        ({comparison['winner_advantage']:.1f}% advantage), this difference is 
                        **not statistically significant** (p = {comparison['p_value']:.4f} > 0.05).
                        
                        This could mean:
                        - The difference is due to random variation
                        - Sample size is too small to detect a real difference
                        - The cohorts truly perform similarly
                        
                        **Recommendation:** Continue monitoring. If the pattern persists over time, 
                        it may become significant with more data.
                        """)
                    
                    # Visualization: Side-by-side retention curves
                    st.markdown("#### Retention Curves Comparison")
                    
                    fig = go.Figure()
                    
                    periods = retention_matrix.columns
                    
                    # Cohort 1
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=retention_matrix.loc[cohort1],
                        name=cohort1,
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Cohort 2
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=retention_matrix.loc[cohort2],
                        name=cohort2,
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add shaded region between curves
                    fig.add_trace(go.Scatter(
                        x=list(periods) + list(periods)[::-1],
                        y=list(retention_matrix.loc[cohort1]) + list(retention_matrix.loc[cohort2])[::-1],
                        fill='toself',
                        fillcolor='rgba(128,128,128,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name='Difference'
                    ))
                    
                    fig.update_layout(
                        title=f"Retention Comparison: {cohort1} vs {cohort2}",
                        xaxis_title="Period",
                        yaxis_title="Retention Rate",
                        yaxis_tickformat='.0%',
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Period-by-period comparison table
                    st.markdown("#### Period-by-Period Comparison")
                    
                    period_comparison = pd.DataFrame({
                        'Period': periods,
                        f'{cohort1} Retention': [f"{val:.1%}" for val in retention_matrix.loc[cohort1]],
                        f'{cohort2} Retention': [f"{val:.1%}" for val in retention_matrix.loc[cohort2]],
                        'Difference': [f"{(retention_matrix.loc[cohort2].iloc[i] - retention_matrix.loc[cohort1].iloc[i]):.1%}" 
                                      for i in range(len(periods))],
                        'Winner': [cohort2 if retention_matrix.loc[cohort2].iloc[i] > retention_matrix.loc[cohort1].iloc[i] 
                                  else cohort1 for i in range(len(periods))]
                    })
                    
                    st.dataframe(period_comparison, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error comparing cohorts: {str(e)}")
                    st.exception(e)
        
        else:  # Compare All Cohorts
            st.markdown("Compare all pairs of cohorts to identify best and worst performers.")
            
            test_type = st.selectbox(
                "Test Type:",
                ["ttest", "mannwhitney"],
                format_func=lambda x: {
                    'ttest': 'T-Test',
                    'mannwhitney': 'Mann-Whitney'
                }[x],
                key="all_cohorts_test_type"
            )
            
            if st.button("🔄 Compare All Cohorts", use_container_width=True, type="primary"):
                try:
                    from utils.cohort_analysis import CohortAnalyzer
                    analyzer = CohortAnalyzer()
                    
                    with st.spinner("Comparing all cohort pairs..."):
                        all_comparisons = analyzer.compare_all_cohorts(retention_matrix, test_type)
                    
                    st.session_state.all_cohort_comparisons = all_comparisons
                    
                    st.markdown("### All Pairwise Comparisons")
                    st.dataframe(all_comparisons, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("### Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        n_significant = len(all_comparisons[all_comparisons['Significant'] == '✅'])
                        st.metric(
                            "Significant Differences",
                            f"{n_significant} / {len(all_comparisons)}"
                        )
                    
                    with col2:
                        # Best performing cohort (wins most comparisons)
                        winner_counts = all_comparisons['Winner'].value_counts()
                        best_cohort = winner_counts.index[0] if len(winner_counts) > 0 else "N/A"
                        st.metric(
                            "Best Performing Cohort",
                            best_cohort
                        )
                    
                    with col3:
                        # Worst performing cohort (wins fewest comparisons)
                        worst_cohort = winner_counts.index[-1] if len(winner_counts) > 0 else "N/A"
                        st.metric(
                            "Worst Performing Cohort",
                            worst_cohort
                        )
                    
                    # Heatmap of pairwise comparisons
                    st.markdown("### Pairwise Comparison Heatmap")
                    
                    # Create matrix of p-values
                    n_cohorts = len(cohorts)
                    p_value_matrix = np.ones((n_cohorts, n_cohorts))
                    
                    for _, row in all_comparisons.iterrows():
                        i = cohorts.index(row['Cohort 1'])
                        j = cohorts.index(row['Cohort 2'])
                        p_val = float(row['P-Value'])
                        p_value_matrix[i, j] = p_val
                        p_value_matrix[j, i] = p_val
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=p_value_matrix,
                        x=cohorts,
                        y=cohorts,
                        colorscale='RdYlGn_r',
                        text=[[f"{val:.4f}" for val in row] for row in p_value_matrix],
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="P-Value")
                    ))
                    
                    fig.update_layout(
                        title="P-Values for All Pairwise Comparisons",
                        xaxis_title="Cohort",
                        yaxis_title="Cohort",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error comparing cohorts: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("Need at least 2 cohorts for comparison. Run cohort analysis first.")
else:
    st.info("Run cohort analysis first to enable comparison.")
```

---

### Step 3: Add Comparison Results to Export

**Location:** In the export section, add comparison results

```python
# Add to markdown report
if 'cohort_comparison' in st.session_state:
    comp = st.session_state.cohort_comparison
    
    report += f"""
## Cohort Comparison: {comp['cohort1']} vs {comp['cohort2']}

### Results:
- **{comp['cohort1']} Average Retention:** {comp['cohort1_avg_retention']:.2%}
- **{comp['cohort2']} Average Retention:** {comp['cohort2_avg_retention']:.2%}
- **Difference:** {comp['retention_diff']:.2%} ({comp['retention_pct_change']:.1f}%)
- **Winner:** {comp['winner']} (+{comp['winner_advantage']:.1f}%)

### Statistical Test:
- **Test Type:** {comp['test_type'].upper()}
- **P-Value:** {comp['p_value']:.6f}
- **Statistically Significant:** {'Yes' if comp['is_significant'] else 'No'}
- **Effect Size (Cohen's d):** {comp['effect_size']:.3f} ({comp['effect_interpretation']})
- **95% Confidence Interval:** [{comp['confidence_interval'][0]:.2%}, {comp['confidence_interval'][1]:.2%}]

### Interpretation:
"""
    
    if comp['is_significant']:
        report += f"{comp['winner']} has significantly better retention. "
    else:
        report += "No statistically significant difference found. "
    
    report += f"The effect size is {comp['effect_interpretation']}.\n\n"

# Add CSV export for comparison results
if 'cohort_comparison' in st.session_state:
    comp = st.session_state.cohort_comparison
    
    comparison_csv = pd.DataFrame({
        'Metric': ['Cohort 1', 'Cohort 2', 'Cohort 1 Avg Retention', 'Cohort 2 Avg Retention',
                   'Difference', 'Relative Change (%)', 'P-Value', 'Significant', 'Winner', 
                   'Effect Size', 'Effect Interpretation'],
        'Value': [comp['cohort1'], comp['cohort2'], comp['cohort1_avg_retention'],
                 comp['cohort2_avg_retention'], comp['retention_diff'], comp['retention_pct_change'],
                 comp['p_value'], comp['is_significant'], comp['winner'], comp['effect_size'],
                 comp['effect_interpretation']]
    }).to_csv(index=False)
    
    st.download_button(
        label="📥 Download Comparison Results (CSV)",
        data=comparison_csv,
        file_name=f"cohort_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
```

---

## Testing Guide

### Test Case 1: Significant Difference
```python
# Cohort 1: [0.9, 0.7, 0.5, 0.3, 0.2]
# Cohort 2: [0.95, 0.85, 0.75, 0.65, 0.55]
# Expected: Cohort 2 wins, p < 0.05
```

### Test Case 2: No Significant Difference
```python
# Cohort 1: [0.9, 0.7, 0.5, 0.3, 0.2]
# Cohort 2: [0.91, 0.71, 0.51, 0.31, 0.21]
# Expected: No significant difference, p > 0.05
```

### Test Case 3: Large Effect Size
```python
# Cohort 1: [0.5, 0.3, 0.2, 0.1, 0.05]
# Cohort 2: [0.95, 0.9, 0.85, 0.8, 0.75]
# Expected: Cohort 2 wins, large effect size
```

---

## Success Criteria

✅ Comparison function added to `utils/cohort_analysis.py`  
✅ Comparison UI added to Cohort Analysis page  
✅ Two-cohort comparison works  
✅ All-cohorts comparison works  
✅ Statistical tests (t-test, Mann-Whitney, Wilcoxon) work  
✅ Visualizations display correctly  
✅ Results included in exports  
✅ Error handling for edge cases  
✅ User-friendly interpretation messages  

---

**Implementation Guide Complete**  
**Estimated Time:** 3 hours  
**Ready to implement!** 🚀

