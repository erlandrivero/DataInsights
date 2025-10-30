# Windsurf Prompts: A/B Testing Framework Module

## Overview

Add a comprehensive **A/B Testing Framework** module to DataInsights for statistical significance testing, experiment design, and results analysis.

**Total Time:** ~3 hours  
**Prompts:** 4 detailed prompts  
**Result:** Production-ready A/B testing toolkit with statistical rigor

---

## PROMPT 1: Create A/B Testing Utility Module (1 hour)

### Instructions for Windsurf:

```
Create a new file `utils/ab_testing.py` for DataInsights with comprehensive A/B testing functionality.

Requirements:

1. **ABTestAnalyzer Class** with methods:
   - `prepare_test_data()` - Prepare A/B test data
   - `calculate_sample_size()` - Determine required sample size
   - `run_statistical_test()` - Perform appropriate statistical test
   - `calculate_confidence_interval()` - CI for metrics
   - `check_statistical_power()` - Power analysis
   - `sequential_testing()` - Early stopping analysis

2. **Statistical Tests to Implement:**
   - **For proportions (conversion rates):**
     * Two-proportion z-test
     * Chi-square test
     * Fisher's exact test (small samples)
   
   - **For continuous metrics (revenue, time):**
     * Independent t-test
     * Welch's t-test (unequal variances)
     * Mann-Whitney U test (non-parametric)
   
   - **For multiple variants:**
     * ANOVA
     * Kruskal-Wallis test

3. **Metrics to Calculate:**
   - Conversion rate (%)
   - Average order value
   - Revenue per user
   - Click-through rate
   - Engagement metrics
   - Lift (% improvement)
   - Statistical significance (p-value)
   - Confidence intervals (95%, 99%)
   - Effect size (Cohen's d)
   - Statistical power

4. **Sample Size Calculation:**
   - Input: baseline rate, minimum detectable effect, alpha, power
   - Output: required sample size per variant
   - Consider multiple testing correction (Bonferroni)

5. **Experiment Design:**
   - Validate randomization
   - Check for sample ratio mismatch (SRM)
   - Detect novelty effects
   - Identify confounding variables

6. **Sequential Testing:**
   - Calculate alpha spending
   - Determine stopping boundaries
   - Estimate time to significance
   - Prevent peeking problem

7. **Data Validation:**
   - Check for required columns (variant, metric, user_id)
   - Validate variant labels
   - Handle missing values
   - Check data quality

8. **Error Handling:**
   - Try-except for all operations
   - Informative error messages
   - Handle edge cases (zero conversions, etc.)

9. **Code Quality:**
   - Type hints
   - Comprehensive docstrings
   - Clean code
   - Follow DataInsights patterns

Example usage:
```python
analyzer = ABTestAnalyzer(df, variant_col='variant', metric_col='converted', user_col='user_id')
# Calculate sample size
sample_size = analyzer.calculate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
# Run test
results = analyzer.run_statistical_test(test_type='proportion')
# Get confidence interval
ci = analyzer.calculate_confidence_interval(confidence=0.95)
```

Use scipy.stats for statistical tests, numpy for calculations. Match style of existing utils.

Test with sample A/B test data: variant (A/B), user_id, converted (0/1), revenue.
```

### Expected Output:
- `utils/ab_testing.py` file created
- ABTestAnalyzer class with 6+ methods
- Multiple statistical tests
- Sample size calculator
- Comprehensive metrics

### Testing Checklist:
- [ ] File created in utils folder
- [ ] ABTestAnalyzer instantiates correctly
- [ ] Sample size calculation works
- [ ] Statistical tests run correctly
- [ ] P-values are accurate
- [ ] Confidence intervals calculated
- [ ] Handles edge cases

---

## PROMPT 2: Add A/B Testing Page to App (1 hour)

### Instructions for Windsurf:

```
Add a new "A/B Testing" page to the DataInsights Streamlit app (app.py).

Requirements:

1. **Add to navigation:**
   - Add "A/B Testing" option in sidebar
   - Place after "Monte Carlo Simulation"
   - Use 🧪 emoji icon

2. **Page structure:**
   - Title: "🧪 A/B Testing Framework"
   - Subtitle: "Design experiments and analyze results with statistical rigor"
   - Four main sections:
     a) Experiment Design (Sample Size Calculator)
     b) Data Upload & Configuration
     c) Test Results & Analysis
     d) Recommendations & Insights

3. **Experiment Design section:**
   - Sample size calculator:
     * Baseline conversion rate (%)
     * Minimum detectable effect (MDE) (%)
     * Significance level (alpha) - default 0.05
     * Statistical power - default 0.80
     * Number of variants (2-5)
     * "Calculate Sample Size" button
   
   - Results display:
     * Required sample size per variant
     * Total sample size needed
     * Estimated test duration (if traffic provided)
     * Minimum detectable lift

4. **Data Configuration section:**
   - Column selection:
     * Variant column (A/B/C labels)
     * Metric column (conversion, revenue, etc.)
     * User ID column (optional)
     * Timestamp column (optional)
   
   - Metric type selection:
     * Binary (conversion, click)
     * Continuous (revenue, time)
     * Count (page views, purchases)
   
   - Test parameters:
     * Confidence level (90%, 95%, 99%)
     * Test type (two-tailed, one-tailed)
   
   - Sample data option:
     * E-commerce A/B test
     * Landing page test
     * Pricing test

5. **Results section:**
   - Summary statistics table:
     * Variant | Sample Size | Conversion Rate | Avg Value | Lift
     * A (Control) | 1000 | 10.5% | $25.30 | -
     * B (Treatment) | 1000 | 12.3% | $27.50 | +17.1%
   
   - Statistical significance:
     * P-value
     * Confidence interval
     * Statistical power achieved
     * Winner declaration (if significant)
   
   - Visualizations (see PROMPT 3)

6. **UI elements:**
   - Use st.columns() for layout
   - st.metric() for key results
   - st.dataframe() for detailed stats
   - st.success() for significant results
   - st.warning() for inconclusive results
   - st.expander() for help text

7. **Help text:**
   - Explanation of A/B testing
   - How to interpret p-values
   - When to stop a test
   - Common mistakes to avoid

8. **Error handling:**
   - Check if data is uploaded
   - Validate variant labels
   - Ensure sufficient sample size
   - Show helpful error messages

Import ABTestAnalyzer from utils.ab_testing. Follow existing page patterns.

Add "Run A/B Test Analysis" button that triggers the statistical analysis.
```

### Expected Output:
- New "A/B Testing" page
- Sample size calculator
- Data configuration interface
- Results display
- Help documentation

### Testing Checklist:
- [ ] Page appears in navigation
- [ ] Sample size calculator works
- [ ] Can select variant/metric columns
- [ ] Test parameters adjust correctly
- [ ] Analysis runs successfully
- [ ] Results display clearly
- [ ] Sample data loads

---

## PROMPT 3: Create A/B Testing Visualizations (45 min)

### Instructions for Windsurf:

```
Add comprehensive visualizations to the A/B Testing page in DataInsights.

Requirements:

1. **Conversion Rate Comparison:**
   - Bar chart comparing variants
   - X-axis: Variants (A, B, C...)
   - Y-axis: Conversion rate (%)
   - Error bars: Confidence intervals
   - Color: Green for winner, blue for others
   - Show lift percentage above bars

2. **Distribution Comparison:**
   - For continuous metrics (revenue, time)
   - Overlapping histograms or violin plots
   - Different color per variant
   - Show mean and median lines
   - Interactive hover

3. **Cumulative Results Over Time:**
   - Line chart showing conversion rate over time
   - X-axis: Date/time
   - Y-axis: Cumulative conversion rate
   - Multiple lines for each variant
   - Shows when significance was reached
   - Requires timestamp column

4. **Confidence Interval Plot:**
   - Forest plot style
   - Horizontal bars showing CI for each variant
   - Point estimate in center
   - Vertical line at control mean
   - Easy to see if CIs overlap

5. **Statistical Power Curve:**
   - Shows power vs sample size
   - Current sample size marked
   - Minimum required marked
   - Helps determine if test should continue

6. **Sample Ratio Mismatch (SRM) Check:**
   - Pie chart or bar chart
   - Expected vs actual sample distribution
   - Highlight if SRM detected
   - Warning if imbalanced

7. **Effect Size Visualization:**
   - Cohen's d or other effect size
   - Visual representation of practical significance
   - Separate from statistical significance

8. **Layout:**
   - Use tabs:
     * Tab 1: Results Overview
     * Tab 2: Distribution Analysis
     * Tab 3: Time Series (if timestamps available)
     * Tab 4: Diagnostics (SRM, power)

Use Plotly for all visualizations. Add interpretation text below each chart.

Reference existing visualization patterns from other modules.
```

### Expected Output:
- 6-7 interactive visualizations
- Conversion rate comparison
- Distribution plots
- Confidence intervals
- Time series analysis
- Diagnostic charts

### Testing Checklist:
- [ ] Bar chart displays correctly
- [ ] Confidence intervals show
- [ ] Distribution plots work
- [ ] Time series appears (if data has timestamps)
- [ ] SRM check runs
- [ ] Charts are interactive
- [ ] Export functionality works

---

## PROMPT 4: Add AI Insights & Export (45 min)

### Instructions for Windsurf:

```
Add AI-powered insights and comprehensive export functionality to the A/B Testing module.

Requirements:

1. **AI-Powered Insights:**
   - After test analysis, generate AI insights using GPT-4
   - Use existing ai_helper.py module

2. **Prompt template for GPT-4:**
```
You are an A/B testing expert. Analyze the following experiment results:

Experiment: {experiment_name}
Variants: {variants}
Metric: {metric_name}
Results:
{results_summary}

Statistical Analysis:
- P-value: {p_value}
- Confidence Interval: {ci}
- Statistical Power: {power}
- Effect Size: {effect_size}

Please provide:
1. Clear interpretation of results (winner, loser, or inconclusive)
2. Statistical significance explanation
3. Practical significance assessment
4. Recommendations (ship, iterate, or kill)
5. Potential risks or caveats
6. Next steps

Be concise, actionable, and business-focused.
```

3. **Display insights:**
   - Show in expandable section
   - Categorize:
     * 🏆 Test Results
     * 📊 Statistical Analysis
     * ⚠️ Caveats & Risks
     * 💡 Recommendations
     * 🚀 Next Steps

4. **Export options:**
   - Download test results (CSV)
   - Download statistical summary (CSV)
   - Download A/B test report (Markdown/PDF)
   - Download visualizations (PNG)

5. **A/B Test Report structure:**
   ```markdown
   # A/B Test Report
   Generated: {date}
   
   ## Experiment Overview
   - Test name: {name}
   - Metric: {metric}
   - Variants: {variants}
   - Duration: {duration}
   
   ## Results Summary
   {results_table}
   
   ## Statistical Analysis
   - P-value: {p_value}
   - Confidence Level: {confidence}%
   - Winner: {winner}
   - Lift: {lift}%
   
   ## Confidence Intervals
   {ci_table}
   
   ## AI Insights
   {ai_insights}
   
   ## Recommendations
   {recommendations}
   
   ## Methodology
   - Statistical test: {test_type}
   - Sample size: {n}
   - Significance level: {alpha}
   ```

6. **Documentation:**
   - Add "📚 A/B Testing Guide" expander
   - Explain:
     * What is A/B testing
     * How to design good experiments
     * Interpreting p-values and confidence intervals
     * Common pitfalls (peeking, multiple testing, etc.)
     * When to stop a test
   - Business applications:
     * Website optimization
     * Pricing experiments
     * Feature testing
     * Marketing campaigns
   - Best practices

7. **Export buttons:**
   - Place in sidebar
   - "Download Test Results"
   - "Download Full Report"
   - "Download Visualizations"

8. **Integration:**
   - Add to main Reports page
   - Include A/B test analysis in comprehensive reports

Use export_helper.py and advanced_report_exporter.py. Follow patterns from other modules.

Add "Regenerate Insights" button for fresh AI analysis.
```

### Expected Output:
- AI-generated insights
- Multiple export formats
- Comprehensive report
- Documentation
- Integration with Reports

### Testing Checklist:
- [ ] AI insights generate
- [ ] Insights are relevant and actionable
- [ ] All exports work
- [ ] Report includes all sections
- [ ] Documentation is clear
- [ ] Reports page integration works
- [ ] Regenerate button works

---

## 🎯 Implementation Order

1. **PROMPT 1** - Create A/B testing utility (foundation)
2. **PROMPT 2** - Add page and UI (interface)
3. **PROMPT 3** - Add visualizations (insights)
4. **PROMPT 4** - Add AI insights & exports (polish)

---

## 📊 Expected Final Result

After completing all 4 prompts:

✅ **Sample size calculator**
✅ **Multiple statistical tests**
✅ **Comprehensive visualizations**
✅ **AI-powered insights**
✅ **Professional exports**
✅ **Complete documentation**

**Total time:** ~3 hours  
**Value:** High - demonstrates statistical rigor

---

## 🧪 Final Testing Checklist

- [ ] Calculate sample size for experiment
- [ ] Upload sample A/B test data
- [ ] Select variant and metric columns
- [ ] Run statistical analysis
- [ ] View conversion rate comparison
- [ ] Check confidence intervals
- [ ] Review AI insights
- [ ] Download A/B test report
- [ ] Verify p-values are correct
- [ ] Test with multiple variants (A/B/C)

---

## 💡 Pro Tips

1. **Test with known data** - Verify p-values match manual calculations
2. **Check edge cases** - Zero conversions, equal rates, etc.
3. **Validate SRM detection** - Test with imbalanced samples
4. **Review AI insights** - Ensure they're statistically sound
5. **Test sample size calculator** - Compare with online calculators

---

## 📚 Additional Features (Optional)

If you have extra time, consider adding:

- **Bayesian A/B testing** - Alternative to frequentist approach
- **Multi-armed bandit** - Dynamic allocation
- **Segmentation analysis** - Results by user segment
- **Heterogeneous treatment effects** - Who benefits most

These can be added later as enhancements.

---

**Ready to implement!** Copy prompts 1-4 into Windsurf in order. 🚀

