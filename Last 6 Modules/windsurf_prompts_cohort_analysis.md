# Windsurf Prompts: Cohort Analysis Module

## Overview

Add a comprehensive **Cohort Analysis** module to DataInsights for tracking user behavior over time, retention analysis, and churn prediction.

**Total Time:** ~4 hours  
**Prompts:** 5 detailed prompts  
**Result:** Production-ready cohort analysis with visualizations

---

## PROMPT 1: Create Cohort Analysis Utility Module (1 hour)

### Instructions for Windsurf:

```
Create a new file `utils/cohort_analysis.py` for DataInsights with comprehensive cohort analysis functionality.

Requirements:

1. **CohortAnalyzer Class** with methods:
   - `prepare_cohort_data()` - Prepare data for cohort analysis
   - `create_retention_cohorts()` - Build retention cohorts
   - `calculate_cohort_metrics()` - Calculate retention rates, churn, LTV
   - `generate_cohort_matrix()` - Create cohort retention matrix
   - `analyze_cohort_trends()` - Identify trends across cohorts
   - `predict_churn()` - Simple churn prediction

2. **Features to implement:**
   - Automatic date column detection
   - User/customer ID detection
   - Cohort period selection (daily, weekly, monthly, quarterly)
   - Retention rate calculation
   - Churn rate calculation
   - Cohort size tracking
   - Average order value per cohort
   - Customer lifetime value estimation

3. **Metrics to calculate:**
   - Retention Rate (% of users returning)
   - Churn Rate (% of users lost)
   - Cohort Size (number of users in each cohort)
   - Average Revenue per Cohort
   - Lifetime Value (LTV) estimation
   - Time to Churn
   - Reactivation Rate

4. **Data validation:**
   - Check for required columns (date, user_id, value)
   - Handle missing values
   - Validate date formats
   - Check for sufficient data

5. **Error handling:**
   - Try-except blocks for all operations
   - Informative error messages
   - Fallback options

6. **Code quality:**
   - Type hints for all functions
   - Comprehensive docstrings
   - Clean, readable code
   - Follow existing DataInsights patterns

Example usage pattern:
```python
analyzer = CohortAnalyzer(df, date_col='purchase_date', user_col='customer_id', value_col='revenue')
cohorts = analyzer.create_retention_cohorts(period='monthly')
metrics = analyzer.calculate_cohort_metrics(cohorts)
matrix = analyzer.generate_cohort_matrix(metrics)
```

Use pandas for data manipulation, numpy for calculations. Match the coding style of existing utils files (data_processor.py, rfm_analysis.py).

Test with sample e-commerce data: user_id, purchase_date, revenue.
```

### Expected Output:
- `utils/cohort_analysis.py` file created
- CohortAnalyzer class with 6+ methods
- Comprehensive cohort metrics calculation
- Error handling and validation

### Testing Checklist:
- [ ] File created in utils folder
- [ ] CohortAnalyzer class instantiates correctly
- [ ] Can detect date and user ID columns
- [ ] Creates cohorts by month/week/day
- [ ] Calculates retention rates correctly
- [ ] Handles missing data gracefully
- [ ] Returns proper data structures

---

## PROMPT 2: Add Cohort Analysis Page to App (1 hour)

### Instructions for Windsurf:

```
Add a new "Cohort Analysis" page to the DataInsights Streamlit app (app.py).

Requirements:

1. **Add to navigation:**
   - Add "Cohort Analysis" option in sidebar navigation
   - Place after "RFM Analysis" (logical grouping)
   - Use 📊 emoji icon

2. **Page structure:**
   - Title: "📊 Cohort Analysis"
   - Subtitle: "Track user behavior and retention over time"
   - Three main sections:
     a) Data Configuration
     b) Cohort Analysis Results
     c) Visualizations & Insights

3. **Data Configuration section:**
   - Column selection:
     * Date column (auto-detect, allow manual selection)
     * User/Customer ID column (auto-detect)
     * Value column (revenue, optional)
     * Event column (optional, for event-based cohorts)
   
   - Cohort settings:
     * Period selection: Daily, Weekly, Monthly, Quarterly
     * Analysis type: Retention, Revenue, Engagement
     * Date range filter
   
   - "Run Cohort Analysis" button

4. **Results section (after analysis):**
   - Cohort summary statistics:
     * Total cohorts analyzed
     * Average cohort size
     * Overall retention rate
     * Average time to churn
   
   - Cohort metrics table:
     * Cohort period
     * Initial size
     * Retention rates (Week 1, 2, 3, 4, etc.)
     * Churn rate
     * LTV estimate
   
   - Downloadable results (CSV)

5. **UI elements:**
   - Use st.columns() for layout
   - st.expander() for help text
   - st.metric() for key statistics
   - st.dataframe() for cohort matrix
   - Color-coded retention rates (green=high, red=low)

6. **Help text:**
   - Explanation of cohort analysis
   - How to interpret results
   - Business applications
   - Example use cases

7. **Error handling:**
   - Check if data is uploaded
   - Validate column selections
   - Show helpful error messages
   - Provide sample data option

Import the CohortAnalyzer from utils.cohort_analysis. Follow the existing page patterns from RFM Analysis and Market Basket Analysis pages.

Add sample dataset option: E-commerce transactions with customer_id, purchase_date, revenue.
```

### Expected Output:
- New "Cohort Analysis" page in app.py
- Column selection interface
- Cohort configuration options
- Results display section
- Help documentation

### Testing Checklist:
- [ ] Page appears in navigation
- [ ] Can select date/user/value columns
- [ ] Cohort period selection works
- [ ] "Run Analysis" button triggers analysis
- [ ] Results display correctly
- [ ] Sample data loads successfully
- [ ] Help text is clear and useful

---

## PROMPT 3: Create Cohort Visualizations (1 hour)

### Instructions for Windsurf:

```
Add comprehensive visualizations to the Cohort Analysis page in DataInsights.

Requirements:

1. **Cohort Retention Heatmap:**
   - X-axis: Time periods since cohort start (Week 0, 1, 2, 3...)
   - Y-axis: Cohort start date
   - Color: Retention rate (0-100%)
   - Use Plotly heatmap
   - Color scale: Red (low) → Yellow → Green (high)
   - Show percentages in cells
   - Hover info: Cohort, Period, Retention %, Absolute numbers

2. **Retention Curve:**
   - Line chart showing retention over time
   - Multiple lines for different cohorts
   - X-axis: Periods since start
   - Y-axis: Retention rate (%)
   - Legend: Cohort names
   - Highlight average retention
   - Interactive hover

3. **Cohort Size Over Time:**
   - Bar chart or area chart
   - X-axis: Time periods
   - Y-axis: Number of active users
   - Stacked by cohort
   - Shows growth/decline

4. **Churn Analysis:**
   - Bar chart of churn rates by cohort
   - Sorted by churn rate
   - Color-coded (red for high churn)
   - Shows cohort with highest/lowest churn

5. **Revenue by Cohort (if value column provided):**
   - Line chart of cumulative revenue
   - By cohort
   - Shows which cohorts are most valuable

6. **Cohort Comparison:**
   - Side-by-side metrics for selected cohorts
   - Retention rate comparison
   - Size comparison
   - Revenue comparison (if applicable)

7. **Visualization controls:**
   - Toggle between different chart types
   - Select specific cohorts to display
   - Adjust time range
   - Export charts as PNG

8. **Layout:**
   - Use tabs for different visualization categories:
     * Tab 1: Retention Analysis
     * Tab 2: Churn Analysis
     * Tab 3: Revenue Analysis (if applicable)
     * Tab 4: Cohort Comparison

Use Plotly for all visualizations (matching existing DataInsights style). Add interpretation text below each chart explaining what to look for.

Reference existing visualization patterns from utils/visualizations.py and other analysis pages.
```

### Expected Output:
- 5-6 interactive Plotly visualizations
- Cohort retention heatmap
- Retention curves
- Churn analysis charts
- Tabbed layout
- Export functionality

### Testing Checklist:
- [ ] Heatmap displays correctly
- [ ] Retention curves show multiple cohorts
- [ ] Charts are interactive
- [ ] Colors are meaningful
- [ ] Hover info is informative
- [ ] Charts export successfully
- [ ] Tabs work properly

---

## PROMPT 4: Add AI Insights for Cohorts (45 min)

### Instructions for Windsurf:

```
Add AI-powered insights to the Cohort Analysis page using OpenAI GPT-4.

Requirements:

1. **Generate cohort insights:**
   - After cohort analysis runs, automatically generate AI insights
   - Use the existing ai_helper.py module
   - Pass cohort metrics to GPT-4

2. **Insights to generate:**
   - Retention trends analysis
   - Best and worst performing cohorts
   - Churn patterns identification
   - Seasonality detection
   - Actionable recommendations

3. **Prompt template for GPT-4:**
```
You are a data analyst expert in cohort analysis. Analyze the following cohort data and provide insights:

Cohort Metrics:
{cohort_summary}

Retention Matrix:
{retention_matrix}

Churn Rates:
{churn_rates}

Please provide:
1. Key findings about retention trends
2. Which cohorts perform best/worst and why
3. Patterns in user behavior
4. Churn risk factors
5. Specific recommendations to improve retention

Be concise, actionable, and business-focused.
```

4. **Display insights:**
   - Show in expandable section
   - Use st.info() or st.success() for positive findings
   - Use st.warning() for concerning trends
   - Format with markdown for readability

5. **Insight categories:**
   - 🎯 Key Findings
   - 📈 Retention Trends
   - ⚠️ Churn Warnings
   - 💡 Recommendations
   - 📊 Cohort Comparison

6. **Error handling:**
   - Handle API failures gracefully
   - Show fallback message if AI unavailable
   - Cache insights to avoid repeated API calls

Use the existing ai_helper.generate_insights() function. Follow patterns from other modules that use AI (Insights page, RFM Analysis).

Add a "Regenerate Insights" button to get fresh analysis.
```

### Expected Output:
- AI-generated insights section
- Categorized findings
- Actionable recommendations
- Professional formatting
- Error handling

### Testing Checklist:
- [ ] AI insights generate automatically
- [ ] Insights are relevant and actionable
- [ ] Formatted clearly with categories
- [ ] Regenerate button works
- [ ] Handles API errors gracefully
- [ ] Insights are cached

---

## PROMPT 5: Add Export & Documentation (45 min)

### Instructions for Windsurf:

```
Add comprehensive export functionality and documentation to the Cohort Analysis module.

Requirements:

1. **Export options:**
   - Download cohort retention matrix (CSV)
   - Download cohort metrics summary (CSV)
   - Download full cohort report (Markdown/PDF)
   - Download all visualizations (PNG bundle)

2. **Cohort report structure:**
   ```markdown
   # Cohort Analysis Report
   Generated: {date}
   
   ## Executive Summary
   - Total cohorts analyzed: {n}
   - Analysis period: {start} to {end}
   - Average retention rate: {rate}%
   - Key finding: {insight}
   
   ## Cohort Metrics
   {metrics_table}
   
   ## Retention Analysis
   {retention_matrix}
   
   ## Key Insights
   {ai_insights}
   
   ## Recommendations
   {recommendations}
   
   ## Methodology
   - Cohort period: {period}
   - Metrics calculated: {metrics}
   ```

3. **Export buttons:**
   - Place in sidebar under "Quick Export"
   - "Download Retention Matrix"
   - "Download Cohort Metrics"
   - "Download Full Report"
   - "Download Visualizations"

4. **Documentation:**
   - Add "📚 Cohort Analysis Guide" expander
   - Explain what cohort analysis is
   - How to interpret retention rates
   - Business applications:
     * SaaS: User retention
     * E-commerce: Customer lifetime value
     * Mobile apps: User engagement
     * Subscriptions: Churn prediction
   - Example use cases
   - Best practices

5. **Help tooltips:**
   - Add (?) icons with explanations for:
     * Retention rate
     * Churn rate
     * Cohort period
     * Lifetime value
   - Use st.help() or custom tooltips

6. **Sample data:**
   - Provide downloadable sample dataset
   - E-commerce transactions format
   - Include README explaining columns

Use existing export_helper.py and advanced_report_exporter.py modules. Follow patterns from Market Basket Analysis and RFM Analysis export functionality.

Add to the main Reports page as well - include cohort analysis in comprehensive reports.
```

### Expected Output:
- Multiple export formats
- Comprehensive cohort report
- Documentation and help text
- Sample data
- Integration with Reports page

### Testing Checklist:
- [ ] All export buttons work
- [ ] CSV downloads correctly
- [ ] Report includes all sections
- [ ] Visualizations export properly
- [ ] Documentation is clear
- [ ] Sample data loads
- [ ] Help tooltips are informative

---

## 🎯 Implementation Order

1. **PROMPT 1** - Create utility module (foundation)
2. **PROMPT 2** - Add page and UI (user interface)
3. **PROMPT 3** - Add visualizations (insights)
4. **PROMPT 4** - Add AI insights (intelligence)
5. **PROMPT 5** - Add exports & docs (polish)

---

## 📊 Expected Final Result

After completing all 5 prompts, you'll have:

✅ **Comprehensive cohort analysis module**
✅ **Multiple visualization types**
✅ **AI-powered insights**
✅ **Professional exports**
✅ **Complete documentation**
✅ **Sample data for testing**

**Total time:** ~4 hours  
**Value:** High - fills major gap in platform

---

## 🧪 Final Testing Checklist

After all prompts:

- [ ] Upload sample e-commerce data
- [ ] Select date, user_id, revenue columns
- [ ] Choose monthly cohorts
- [ ] Run analysis
- [ ] View retention heatmap
- [ ] Check retention curves
- [ ] Read AI insights
- [ ] Download cohort report
- [ ] Export visualizations
- [ ] Verify all metrics are accurate

---

## 💡 Pro Tips

1. **Test with real-looking data** - Use realistic e-commerce patterns
2. **Check edge cases** - Single cohort, missing data, etc.
3. **Verify calculations** - Manually check a few retention rates
4. **Review AI insights** - Make sure they're actionable
5. **Test exports** - Ensure all formats work

---

**Ready to implement!** Copy prompts 1-5 into Windsurf in order. 🚀

