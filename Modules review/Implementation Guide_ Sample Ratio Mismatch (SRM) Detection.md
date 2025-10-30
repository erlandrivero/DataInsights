# Implementation Guide: Sample Ratio Mismatch (SRM) Detection

**Module:** A/B Testing  
**Time Required:** 2 hours  
**Difficulty:** Medium  
**Priority:** 🔥 HIGH  
**Impact:** Prevents invalid A/B test results

---

## What is Sample Ratio Mismatch (SRM)?

**Sample Ratio Mismatch** occurs when the observed traffic split in an A/B test differs significantly from the expected split. This indicates potential data quality issues that can invalidate test results.

### Example:
- **Expected:** 50/50 split (5,000 control, 5,000 treatment)
- **Observed:** 5,200 control, 4,800 treatment
- **SRM Test:** Chi-square test shows p-value < 0.01
- **Conclusion:** ⚠️ SRM detected - results may not be trustworthy

### Why It Matters:

SRM is one of the **most common causes of invalid A/B tests**. It can be caused by:
- **Randomization failures** - Bug in traffic allocation
- **Bot traffic** - Bots hitting one variant more than the other
- **Data collection errors** - Logging failures in one variant
- **Implementation bugs** - Redirect issues, caching problems

**Professional A/B testing tools** (Optimizely, VWO, Google Optimize) **always check for SRM**.

---

## Implementation Steps

### Step 1: Add SRM Check Function to `utils/ab_testing.py`

**Location:** Add after the `ABTestAnalyzer` class initialization

```python
def check_sample_ratio_mismatch(
    self,
    control_n: int,
    treatment_n: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01
) -> Dict[str, Any]:
    """
    Check for Sample Ratio Mismatch (SRM).
    
    SRM occurs when the observed traffic split differs significantly
    from the expected split, indicating potential data quality issues.
    
    Args:
        control_n: Number of observations in control group
        treatment_n: Number of observations in treatment group
        expected_ratio: Expected proportion in control (default 0.5 for 50/50 split)
        alpha: Significance level for SRM test (default 0.01, stricter than usual)
    
    Returns:
        Dictionary containing:
            - has_srm: Boolean indicating if SRM was detected
            - p_value: Chi-square test p-value
            - chi2_statistic: Chi-square test statistic
            - expected_ratio: Expected control proportion
            - actual_ratio: Observed control proportion
            - control_n: Control group size
            - treatment_n: Treatment group size
            - severity: 'OK', 'WARNING', or 'CRITICAL'
            - recommendation: Action to take
    
    Examples:
        >>> analyzer = ABTestAnalyzer()
        >>> result = analyzer.check_sample_ratio_mismatch(5200, 4800, expected_ratio=0.5)
        >>> if result['has_srm']:
        >>>     print(f"SRM detected! P-value: {result['p_value']:.6f}")
    
    References:
        - https://www.lukasvermeer.nl/srm/
        - https://dl.acm.org/doi/10.1145/3292500.3330722
    """
    # Total sample size
    total_n = control_n + treatment_n
    
    # Expected counts
    expected_control = total_n * expected_ratio
    expected_treatment = total_n * (1 - expected_ratio)
    
    # Chi-square test for goodness of fit
    # H0: Observed split matches expected split
    # H1: Observed split differs from expected split
    observed = [control_n, treatment_n]
    expected = [expected_control, expected_treatment]
    
    chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected))
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    # Actual ratio
    actual_ratio = control_n / total_n
    
    # Determine severity
    if p_value < 0.001:
        severity = 'CRITICAL'
        recommendation = "🚫 DO NOT TRUST RESULTS - Investigate data quality issues immediately"
    elif p_value < alpha:
        severity = 'WARNING'
        recommendation = "⚠️ PROCEED WITH CAUTION - Review data collection and randomization"
    else:
        severity = 'OK'
        recommendation = "✅ Traffic split looks good - Safe to proceed with analysis"
    
    # Calculate absolute and relative differences
    absolute_diff = abs(control_n - expected_control)
    relative_diff = abs(actual_ratio - expected_ratio) / expected_ratio * 100
    
    return {
        'has_srm': p_value < alpha,
        'p_value': p_value,
        'chi2_statistic': chi2_stat,
        'expected_ratio': expected_ratio,
        'actual_ratio': actual_ratio,
        'expected_control': expected_control,
        'expected_treatment': expected_treatment,
        'control_n': control_n,
        'treatment_n': treatment_n,
        'absolute_diff': absolute_diff,
        'relative_diff': relative_diff,
        'severity': severity,
        'recommendation': recommendation,
        'alpha': alpha
    }
```

---

### Step 2: Add SRM Check to UI in `app.py`

**Location:** In `show_ab_testing()` function, after data is loaded but before running the test

```python
# After data source selection and before test configuration
# Add SRM check section

st.divider()
st.subheader("🔍 Data Quality Check: Sample Ratio Mismatch (SRM)")

with st.expander("ℹ️ What is SRM and why does it matter?"):
    st.markdown("""
    **Sample Ratio Mismatch (SRM)** occurs when the observed traffic split differs 
    significantly from the expected split. This is a red flag indicating potential 
    data quality issues.
    
    ### Common Causes:
    - **Randomization failures** - Bug in traffic allocation code
    - **Bot traffic** - Automated traffic hitting one variant more
    - **Data collection errors** - Logging failures in one variant
    - **Implementation bugs** - Redirect issues, caching problems
    - **Browser/device issues** - Compatibility problems with one variant
    
    ### Why It Matters:
    If SRM is detected, your A/B test results **cannot be trusted**, even if they show 
    statistical significance. The test is fundamentally flawed and must be fixed before 
    making any decisions.
    
    ### Industry Standard:
    Professional A/B testing platforms (Optimizely, VWO, Google Optimize) **always** 
    check for SRM before showing results.
    
    ### Learn More:
    - [SRM Checker Tool](https://www.lukasvermeer.nl/srm/)
    - [Research Paper](https://dl.acm.org/doi/10.1145/3292500.3330722)
    """)

# Determine sample sizes based on data source
if data_source == "Use Loaded Dataset":
    # Get group sizes from the loaded data
    control_n = len(df[df[group_col] == control_group])
    treatment_n = len(df[df[group_col] == treatment_group])
elif data_source == "Sample A/B Test Data":
    # Get from sample data
    control_n = len(df[df[group_col] == control_group])
    treatment_n = len(df[df[group_col] == treatment_group])
elif data_source == "Upload Custom Data":
    # Get from uploaded data
    control_n = len(df[df[group_col] == control_group])
    treatment_n = len(df[df[group_col] == treatment_group])
else:  # Manual Calculator
    # Get from manual inputs
    control_n = control_n_input
    treatment_n = treatment_n_input

# Expected ratio input
col1, col2 = st.columns([2, 1])

with col1:
    expected_ratio = st.slider(
        "Expected Control Proportion:",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="What traffic split did you expect? (e.g., 0.5 for 50/50 split)"
    )

with col2:
    st.metric(
        "Expected Split",
        f"{expected_ratio:.0%} / {1-expected_ratio:.0%}"
    )

# Run SRM check
if control_n > 0 and treatment_n > 0:
    srm_result = analyzer.check_sample_ratio_mismatch(
        control_n=control_n,
        treatment_n=treatment_n,
        expected_ratio=expected_ratio,
        alpha=0.01  # Stricter threshold for SRM
    )
    
    # Display results based on severity
    if srm_result['severity'] == 'CRITICAL':
        st.error(f"""
        ### 🚫 CRITICAL: Sample Ratio Mismatch Detected!
        
        **Your A/B test has a severe data quality issue. DO NOT trust the results.**
        
        #### Observed vs Expected:
        - **Expected split:** {srm_result['expected_ratio']:.1%} / {1-srm_result['expected_ratio']:.1%}
        - **Actual split:** {srm_result['actual_ratio']:.1%} / {1-srm_result['actual_ratio']:.1%}
        - **Difference:** {srm_result['relative_diff']:.1f}% deviation
        
        #### Statistical Test:
        - **Chi-square statistic:** {srm_result['chi2_statistic']:.2f}
        - **P-value:** {srm_result['p_value']:.6f} (threshold: {srm_result['alpha']})
        - **Severity:** {srm_result['severity']}
        
        #### Sample Sizes:
        - **Control:** {srm_result['control_n']:,} (expected: {srm_result['expected_control']:.0f})
        - **Treatment:** {srm_result['treatment_n']:,} (expected: {srm_result['expected_treatment']:.0f})
        - **Absolute difference:** {srm_result['absolute_diff']:.0f} observations
        
        #### What to Do:
        1. **STOP** - Do not make any decisions based on this test
        2. **Investigate** - Check randomization code, logging, bot traffic
        3. **Fix** - Resolve the data quality issue
        4. **Re-run** - Start a new test after fixing the issue
        
        #### Common Causes to Check:
        - Randomization algorithm implementation
        - Data collection/logging code
        - Bot detection and filtering
        - Browser/device compatibility
        - Caching or redirect issues
        """)
        
        # Show diagnostic chart
        fig = go.Figure(data=[
            go.Bar(
                name='Expected',
                x=['Control', 'Treatment'],
                y=[srm_result['expected_control'], srm_result['expected_treatment']],
                marker_color='lightblue'
            ),
            go.Bar(
                name='Observed',
                x=['Control', 'Treatment'],
                y=[srm_result['control_n'], srm_result['treatment_n']],
                marker_color='red'
            )
        ])
        
        fig.update_layout(
            title='Sample Ratio Mismatch: Expected vs Observed',
            xaxis_title='Group',
            yaxis_title='Sample Size',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif srm_result['severity'] == 'WARNING':
        st.warning(f"""
        ### ⚠️ WARNING: Potential Sample Ratio Mismatch
        
        **Your test shows a minor traffic split imbalance. Proceed with caution.**
        
        #### Observed vs Expected:
        - **Expected split:** {srm_result['expected_ratio']:.1%} / {1-srm_result['expected_ratio']:.1%}
        - **Actual split:** {srm_result['actual_ratio']:.1%} / {1-srm_result['actual_ratio']:.1%}
        - **Difference:** {srm_result['relative_diff']:.1f}% deviation
        
        #### Statistical Test:
        - **P-value:** {srm_result['p_value']:.4f} (threshold: {srm_result['alpha']})
        - **Severity:** {srm_result['severity']}
        
        #### Recommendation:
        Review your data collection and randomization process. While not critical, 
        this imbalance warrants investigation before making major decisions.
        """)
    
    else:  # OK
        st.success(f"""
        ### ✅ No Sample Ratio Mismatch Detected
        
        **Traffic split looks good! Safe to proceed with analysis.**
        
        #### Observed vs Expected:
        - **Expected split:** {srm_result['expected_ratio']:.1%} / {1-srm_result['expected_ratio']:.1%}
        - **Actual split:** {srm_result['actual_ratio']:.1%} / {1-srm_result['actual_ratio']:.1%}
        - **Difference:** {srm_result['relative_diff']:.1f}% deviation
        
        #### Statistical Test:
        - **P-value:** {srm_result['p_value']:.4f} (threshold: {srm_result['alpha']})
        - **Severity:** {srm_result['severity']}
        
        #### Sample Sizes:
        - **Control:** {srm_result['control_n']:,}
        - **Treatment:** {srm_result['treatment_n']:,}
        - **Total:** {srm_result['control_n'] + srm_result['treatment_n']:,}
        
        The observed traffic split is within expected random variation. 
        You can confidently proceed with your A/B test analysis.
        """)
    
    # Store SRM result in session state for export
    st.session_state.srm_result = srm_result
    
else:
    st.info("Load data to check for Sample Ratio Mismatch")

st.divider()
```

---

### Step 3: Add SRM Result to Report Export

**Location:** In the report generation section, add SRM check results

```python
# In the report generation code (around line 9500)
# Add SRM section to the markdown report

if 'srm_result' in st.session_state:
    srm = st.session_state.srm_result
    
    report += f"""
## Sample Ratio Mismatch (SRM) Check

**Status:** {srm['severity']}

### Traffic Split Analysis:
- **Expected split:** {srm['expected_ratio']:.1%} / {1-srm['expected_ratio']:.1%}
- **Actual split:** {srm['actual_ratio']:.1%} / {1-srm['actual_ratio']:.1%}
- **Deviation:** {srm['relative_diff']:.1f}%

### Sample Sizes:
- **Control:** {srm['control_n']:,} (expected: {srm['expected_control']:.0f})
- **Treatment:** {srm['treatment_n']:,} (expected: {srm['expected_treatment']:.0f})
- **Absolute difference:** {srm['absolute_diff']:.0f} observations

### Statistical Test:
- **Chi-square statistic:** {srm['chi2_statistic']:.4f}
- **P-value:** {srm['p_value']:.6f}
- **Significance level:** {srm['alpha']}
- **SRM detected:** {'Yes' if srm['has_srm'] else 'No'}

### Recommendation:
{srm['recommendation']}

"""
```

---

### Step 4: Add SRM to CSV Export

**Location:** In the CSV export section, include SRM results

```python
# Add to the results CSV export
if 'srm_result' in st.session_state:
    srm = st.session_state.srm_result
    
    # Add SRM rows to results_data dictionary
    results_data['Metric'].extend([
        '--- SRM Check ---',
        'SRM Detected',
        'SRM P-Value',
        'SRM Severity',
        'Expected Control Proportion',
        'Actual Control Proportion',
        'Traffic Split Deviation (%)'
    ])
    
    results_data['Value'].extend([
        '',
        'Yes' if srm['has_srm'] else 'No',
        f"{srm['p_value']:.6f}",
        srm['severity'],
        f"{srm['expected_ratio']:.4f}",
        f"{srm['actual_ratio']:.4f}",
        f"{srm['relative_diff']:.2f}"
    ])
```

---

## Testing Guide

### Test Case 1: No SRM (Normal Test)
```python
# Expected: 50/50 split
# Observed: 5,000 control, 5,000 treatment
# Result: ✅ No SRM detected
```

**Expected Output:**
- P-value: ~1.0
- Severity: OK
- Green success message

---

### Test Case 2: Minor SRM (Warning)
```python
# Expected: 50/50 split
# Observed: 5,100 control, 4,900 treatment
# Result: ⚠️ Warning (p-value between 0.01 and 0.05)
```

**Expected Output:**
- P-value: ~0.02
- Severity: WARNING
- Yellow warning message

---

### Test Case 3: Severe SRM (Critical)
```python
# Expected: 50/50 split
# Observed: 5,500 control, 4,500 treatment
# Result: 🚫 Critical SRM detected
```

**Expected Output:**
- P-value: < 0.001
- Severity: CRITICAL
- Red error message with diagnostic chart

---

### Test Case 4: Custom Expected Ratio
```python
# Expected: 60/40 split
# Observed: 6,000 control, 4,000 treatment
# Result: ✅ No SRM detected
```

**Expected Output:**
- P-value: ~1.0
- Severity: OK
- Matches custom expected ratio

---

## Common Issues & Solutions

### Issue 1: Chi-square test fails with small samples

**Problem:** Chi-square test requires expected counts ≥ 5

**Solution:** Add validation:
```python
if expected_control < 5 or expected_treatment < 5:
    return {
        'has_srm': False,
        'error': 'Sample size too small for SRM test (need at least 10 observations)',
        'severity': 'UNKNOWN'
    }
```

---

### Issue 2: P-value is exactly 0 or 1

**Problem:** Floating point precision issues

**Solution:** Use `scipy.stats.chi2.sf()` instead of `1 - cdf()`:
```python
p_value = stats.chi2.sf(chi2_stat, df=1)
```

---

### Issue 3: SRM check runs before data is loaded

**Problem:** `control_n` and `treatment_n` are undefined

**Solution:** Add conditional check:
```python
if 'data' in st.session_state and st.session_state.data is not None:
    # Run SRM check
else:
    st.info("Load data first to check for SRM")
```

---

## Advanced Features (Optional)

### 1. Historical SRM Tracking

Track SRM over time to detect patterns:

```python
# Store SRM results in a history
if 'srm_history' not in st.session_state:
    st.session_state.srm_history = []

st.session_state.srm_history.append({
    'timestamp': pd.Timestamp.now(),
    'p_value': srm_result['p_value'],
    'severity': srm_result['severity']
})

# Show trend
if len(st.session_state.srm_history) > 1:
    st.line_chart(pd.DataFrame(st.session_state.srm_history).set_index('timestamp')['p_value'])
```

---

### 2. Multiple Group SRM Check

Extend to A/B/C tests:

```python
def check_multi_group_srm(self, group_sizes: List[int], expected_ratios: List[float]) -> Dict:
    """Check SRM for multiple groups (A/B/C/D... tests)"""
    observed = group_sizes
    expected = [sum(group_sizes) * ratio for ratio in expected_ratios]
    
    chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected))
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(group_sizes)-1)
    
    return {
        'has_srm': p_value < 0.01,
        'p_value': p_value,
        'chi2_statistic': chi2_stat
    }
```

---

### 3. Sequential SRM Monitoring

Check SRM at multiple time points during the test:

```python
# Check SRM daily
srm_checks = []
for day in range(1, test_duration_days + 1):
    day_data = df[df['day'] <= day]
    srm_result = analyzer.check_sample_ratio_mismatch(
        control_n=len(day_data[day_data['group'] == 'control']),
        treatment_n=len(day_data[day_data['group'] == 'treatment'])
    )
    srm_checks.append(srm_result)

# Alert if SRM appears during the test
if any(check['has_srm'] for check in srm_checks):
    st.warning("SRM detected during test - investigate immediately")
```

---

## Success Criteria

✅ SRM check function added to `utils/ab_testing.py`  
✅ SRM check UI added to A/B Testing page  
✅ SRM results included in markdown report  
✅ SRM results included in CSV export  
✅ All test cases pass  
✅ Error handling for edge cases  
✅ User-friendly messages for all severity levels  
✅ Diagnostic visualization for critical SRM  
✅ Documentation updated  

---

## References

1. **SRM Checker Tool:** https://www.lukasvermeer.nl/srm/
2. **Research Paper:** "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments" (KDD 2019)
3. **Blog Post:** https://www.exp-platform.com/Documents/2019-FirstPracticalOnlineCEBook.pdf
4. **Optimizely Docs:** https://docs.developers.optimizely.com/experimentation/docs/sample-ratio-mismatch

---

**Implementation Guide Complete**  
**Estimated Time:** 2 hours  
**Ready to implement!** 🚀

