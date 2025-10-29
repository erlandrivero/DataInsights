# 🧪 Testing Checklist - New Features (Oct 29, 2025)

**Purpose:** Verify all 8 new features implemented today work correctly  
**Tester:** _____________  
**Date:** _____________

---

## 🎯 Quick Start Guide

### Prerequisites
1. ✅ App running: `streamlit run app.py`
2. ✅ Browser open at localhost
3. ✅ Test data ready (use Sample Data buttons)

---

## 📥 WEEK 1: CSV Export Functionality

### Test 1: A/B Testing CSV Export
- [ ] Navigate to "A/B Testing" module
- [ ] Load sample A/B test data
- [ ] Run statistical test
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: test stats, rates, p-value, effect size
- [ ] ✅ Data is accurate (spot check values)

**Issues:** _______________

---

### Test 2: Cohort Analysis CSV Export
- [ ] Navigate to "Cohort Analysis" module
- [ ] Load sample cohort data
- [ ] Calculate retention
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: retention matrix (cohorts × periods)
- [ ] ✅ Percentages are correct

**Issues:** _______________

---

### Test 3: Recommendation Systems CSV Export
- [ ] Navigate to "Recommendation Systems" module
- [ ] Load sample ratings data
- [ ] Build recommendation model
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: user/item, predictions, scores
- [ ] ✅ Top recommendations included

**Issues:** _______________

---

### Test 4: Geospatial Analysis CSV Export
- [ ] Navigate to "Geospatial Analysis" module
- [ ] Load sample location data
- [ ] Run clustering
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: lat/lon, cluster assignments
- [ ] ✅ Cluster IDs are correct

**Issues:** _______________

---

### Test 5: Survival Analysis CSV Export
- [ ] Navigate to "Survival Analysis" module
- [ ] Load sample survival data
- [ ] Fit Kaplan-Meier curve
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: duration, event, group columns
- [ ] ✅ Data matches source

**Issues:** _______________

---

### Test 6: Network Analysis CSV Export
- [ ] Navigate to "Network Analysis" module
- [ ] Load sample network data
- [ ] Calculate centrality measures
- [ ] Click "📥 Download Results (CSV)"
- [ ] ✅ CSV downloads successfully
- [ ] ✅ Contains: nodes, degree/betweenness/closeness
- [ ] ✅ Top nodes ranked correctly

**Issues:** _______________

---

## 🚨 WEEK 1: SRM Detection in A/B Testing

### Test 7: SRM Detection - Balanced Split
- [ ] Navigate to "A/B Testing" module
- [ ] Load sample data (should be 50/50)
- [ ] ✅ "No SRM Detected" message appears (green)
- [ ] ✅ Shows chi-square and p-value
- [ ] ✅ P-value is > 0.01
- [ ] ✅ Sample sizes displayed correctly

**Expected:** Green success message  
**Issues:** _______________

---

### Test 8: SRM Detection - Imbalanced Split
- [ ] Upload custom CSV with 70/30 split
- [ ] Or create imbalanced data manually
- [ ] ✅ "SRM Detected!" warning appears (red/orange)
- [ ] ✅ Shows percentages (e.g., 70% vs 30%)
- [ ] ✅ P-value is < 0.01
- [ ] ✅ Recommendations displayed

**Expected:** Warning message with recommendations  
**Issues:** _______________

---

### Test 9: SRM Help Section
- [ ] Expand "ℹ️ What is A/B Testing?" section
- [ ] ✅ SRM explanation included
- [ ] ✅ Clear and understandable
- [ ] ✅ Explains business impact

**Issues:** _______________

---

## 📊 WEEK 2: Cohort Comparison Feature

### Test 10: Compare Two Cohorts
- [ ] Navigate to "Cohort Analysis" module
- [ ] Load/calculate retention data
- [ ] Select 2 different cohorts in dropdowns
- [ ] Click "📊 Compare Selected Cohorts"
- [ ] ✅ Comparison results appear
- [ ] ✅ T-statistic and p-value shown
- [ ] ✅ Effect size (Cohen's d) displayed
- [ ] ✅ Confidence intervals shown
- [ ] ✅ Significance indicator correct

**Issues:** _______________

---

### Test 11: Side-by-Side Visualization
- [ ] After comparison (Test 10)
- [ ] Scroll to "Side-by-Side Comparison"
- [ ] ✅ Chart displays correctly
- [ ] ✅ Two lines shown (different colors)
- [ ] ✅ Interactive hover works
- [ ] ✅ Legend labels correct
- [ ] ✅ No Plotly errors

**Issues:** _______________

---

## 🆕 WEEK 2: Cold Start Solutions

### Test 12: Cold Start Metrics Dashboard
- [ ] Navigate to "Recommendation Systems" module
- [ ] Load sample ratings data
- [ ] Build model
- [ ] Scroll to "🆕 Cold Start Analysis"
- [ ] ✅ 4 metrics displayed
- [ ] ✅ Cold Start Users count shown
- [ ] ✅ Cold Start Items count shown
- [ ] ✅ Percentages calculated correctly

**Issues:** _______________

---

### Test 13: Popular Items Fallback
- [ ] Continue in Cold Start section
- [ ] ✅ "⭐ Popular Items" table displays
- [ ] ✅ Shows item_id, avg_rating, rating_count
- [ ] ✅ Sorted by popularity
- [ ] ✅ No errors (fixed ef87660)

**Issues:** _______________

---

### Test 14: New User Simulation
- [ ] Scroll to "🆕 Simulating New User"
- [ ] ✅ Shows recommendations for fake user
- [ ] ✅ "Popular items" strategy indicated
- [ ] ✅ Recommendations displayed
- [ ] ✅ No crashes or errors

**Issues:** _______________

---

## 🏥 WEEK 3: Cox Proportional Hazards Model

### Test 15: Cox Model - Select Covariates
- [ ] Navigate to "Survival Analysis" module
- [ ] Load sample data with groups
- [ ] Run survival analysis
- [ ] Scroll to "🔬 Cox Proportional Hazards Model"
- [ ] ✅ Covariate multiselect appears
- [ ] ✅ Group_indicator option available
- [ ] Select covariates
- [ ] Click "📊 Run Cox Regression"

**Issues:** _______________

---

### Test 16: Cox Model - Results Display
- [ ] After running Cox model (Test 15)
- [ ] ✅ Hazard Ratios table shown
- [ ] ✅ Confidence intervals displayed
- [ ] ✅ P-values shown
- [ ] ✅ Significance indicators (✅/❌)
- [ ] ✅ Interpretation column present
- [ ] ✅ Color-coded (🔴🟡🟢)

**Issues:** _______________

---

### Test 17: Cox Model - Forest Plot
- [ ] Continue from Test 16
- [ ] Scroll to "🌲 Forest Plot"
- [ ] ✅ Chart displays correctly
- [ ] ✅ Hazard ratios plotted
- [ ] ✅ Error bars (95% CI) shown
- [ ] ✅ Reference line at HR=1.0
- [ ] ✅ Interactive hover works

**Issues:** _______________

---

### Test 18: Cox Model - Performance Metrics
- [ ] Check "📈 Model Performance" section
- [ ] ✅ Concordance Index displayed
- [ ] ✅ Log Likelihood shown
- [ ] ✅ Performance interpretation appears
- [ ] ✅ Green/yellow/red indicator correct

**Issues:** _______________

---

## 📊 WEEK 3: Diversity Metrics for Recommendations

### Test 19: Calculate Diversity Metrics
- [ ] Navigate to "Recommendation Systems" module
- [ ] Build recommendation model
- [ ] Scroll to "📊 Diversity & Quality Metrics"
- [ ] Click "🔬 Calculate Diversity Metrics"
- [ ] ✅ Progress indicator shows
- [ ] ✅ "Calculating for sample users..." message
- [ ] ✅ Completes without errors
- [ ] ✅ Success message appears

**Issues:** _______________

---

### Test 20: Diversity Metrics Dashboard
- [ ] After calculation (Test 19)
- [ ] ✅ 4-column metrics display
- [ ] ✅ Diversity score (0-1) shown
- [ ] ✅ Coverage percentage shown
- [ ] ✅ Novelty score shown
- [ ] ✅ Personalization score shown
- [ ] ✅ Color indicators (🟢🟡🔴) correct
- [ ] ✅ Help tooltips work

**Issues:** _______________

---

### Test 21: Improvement Recommendations
- [ ] Scroll to "🎯 Improvement Recommendations"
- [ ] ✅ Section appears
- [ ] If metrics low: ✅ Suggestions displayed
- [ ] If metrics good: ✅ Success message shown
- [ ] ✅ Actionable recommendations provided

**Issues:** _______________

---

## 📢 WEEK 3: Influence Propagation for Network Analysis

### Test 22: Find Optimal Influencers
- [ ] Navigate to "Network Analysis" module
- [ ] Load sample network data
- [ ] Run analysis
- [ ] Scroll to "📢 Influence Propagation"
- [ ] Select propagation model (IC or LT)
- [ ] Set propagation probability
- [ ] Click "🎯 Find Optimal Influencers"
- [ ] ✅ Status shows progress
- [ ] ✅ Completes without errors
- [ ] ✅ Top 5 influencers displayed

**Issues:** _______________

---

### Test 23: Influencer Results & Strategy
- [ ] After finding influencers (Test 22)
- [ ] ✅ Table shows: rank, node, avg_spread, reach%
- [ ] ✅ Metrics display: top reach, avg reach, total nodes
- [ ] ✅ Campaign strategy recommendations appear
- [ ] ✅ Recommendations match reach levels:
  - >50% = Excellent (green)
  - 30-50% = Good (blue)
  - <30% = Moderate (orange)

**Issues:** _______________

---

### Test 24: Custom Spread Simulation
- [ ] Scroll to "🧪 Custom Spread Simulation"
- [ ] Select 3 seed nodes from multiselect
- [ ] Click "▶️ Run Simulation"
- [ ] ✅ Simulation runs
- [ ] ✅ 4 metrics display: nodes, reach%, iterations, amplification
- [ ] ✅ "📈 Spread Over Time" chart appears
- [ ] ✅ Line chart shows growth
- [ ] ✅ Interpretation message shown

**Issues:** _______________

---

### Test 25: Independent Cascade vs Linear Threshold
- [ ] Test with Independent Cascade model
- [ ] Note results
- [ ] Change to Linear Threshold model
- [ ] Run again with same seeds
- [ ] ✅ Results differ (models work differently)
- [ ] ✅ Both complete without errors
- [ ] ✅ Charts update correctly

**Issues:** _______________

---

## 🔄 REGRESSION TESTS (Existing Features)

### Test 26: Basic Upload Still Works
- [ ] Navigate to "Data Upload" page
- [ ] Upload CSV file
- [ ] ✅ File uploads successfully
- [ ] ✅ Preview displays
- [ ] ✅ No errors or crashes

**Issues:** _______________

---

### Test 27: AI Insights Still Work
- [ ] Choose any module with AI insights
- [ ] Generate AI insights
- [ ] ✅ Insights generate successfully
- [ ] ✅ No API errors
- [ ] ✅ Content is relevant

**Issues:** _______________

---

### Test 28: Original Exports Still Work
- [ ] Test Markdown exports in any module
- [ ] ✅ Download Reports work
- [ ] ✅ Markdown format correct
- [ ] ✅ No conflicts with CSV exports

**Issues:** _______________

---

## ⚡ PERFORMANCE TESTS

### Test 29: Large Dataset - CSV Export
- [ ] Load dataset with 10,000+ rows
- [ ] Generate analysis
- [ ] Export CSV
- [ ] ✅ Completes in reasonable time (<30s)
- [ ] ✅ No memory errors
- [ ] ✅ File size appropriate

**Issues:** _______________

---

### Test 30: Multiple Simulations
- [ ] Run influence spread 10 times
- [ ] ✅ Performance acceptable
- [ ] ✅ Results consistent
- [ ] ✅ No slowdowns

**Issues:** _______________

---

## 🐛 EDGE CASE TESTS

### Test 31: Empty Data
- [ ] Try to export CSV with no results
- [ ] ✅ Graceful error message OR
- [ ] ✅ Button disabled OR
- [ ] ✅ Empty CSV with headers

**Issues:** _______________

---

### Test 32: Single Cohort
- [ ] Try cohort comparison with only 1 cohort
- [ ] ✅ Helpful message displayed
- [ ] ✅ No crashes

**Issues:** _______________

---

### Test 33: No Covariates for Cox
- [ ] Survival data with no numeric columns
- [ ] ✅ Helpful message shown
- [ ] ✅ No crashes

**Issues:** _______________

---

## ✅ FINAL CHECKLIST

### Documentation
- [ ] README updated with new features
- [ ] IMPLEMENTATION_COMPLETE.md reviewed
- [ ] All commits have clear messages

### Code Quality
- [ ] No console errors
- [ ] No Python warnings
- [ ] Proper error handling observed
- [ ] Loading states visible

### User Experience
- [ ] All new features intuitive
- [ ] Help text is clear
- [ ] Error messages are helpful
- [ ] Performance is acceptable

---

## 📊 TESTING SUMMARY

**Total Tests:** 33  
**Passed:** _____ / 33  
**Failed:** _____  
**Blocked:** _____

### Critical Issues Found:
1. _______________
2. _______________
3. _______________

### Minor Issues Found:
1. _______________
2. _______________
3. _______________

### Recommendations:
- [ ] ✅ **READY FOR PRODUCTION**
- [ ] ⚠️ Minor fixes needed (document below)
- [ ] 🚫 Major fixes required (document below)

### Notes:
_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

## 🎯 Quick Test (15 minutes)

If short on time, test these critical items:

1. [ ] Test 1: A/B Testing CSV export
2. [ ] Test 8: SRM detection with imbalanced data
3. [ ] Test 11: Cohort comparison visualization
4. [ ] Test 13: Popular items fallback
5. [ ] Test 17: Cox model forest plot
6. [ ] Test 20: Diversity metrics dashboard
7. [ ] Test 24: Influence spread simulation

**Quick Test Result:** _____ / 7 passed

---

**Testing Completed By:** _____________  
**Date:** _____________  
**Time Spent:** _____ hours  
**Environment:** Local / Production

**Sign-off:** _______________
