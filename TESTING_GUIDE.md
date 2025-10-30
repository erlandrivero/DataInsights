# Crash Prevention Testing Guide

**Purpose:** Verify all crash prevention and memory optimization features work correctly  
**Estimated Time:** 30-45 minutes  
**Prerequisites:** App deployed to Streamlit Cloud

---

## 🧪 Test Suite

### Test 1: Sequential ML Classification Training
**Objective:** Verify models train one at a time with memory management

**Steps:**
1. Navigate to **ML Classification** page
2. Use **OpenML Dataset**: Select `bank-marketing` (45K rows)
3. Set target column: `y`
4. Select **ALL 15 models** for training
5. Set CV folds: 3
6. Click **"Train Models"**

**Expected Results:**
- ✅ Navigation locks (radio buttons disabled)
- ✅ Warning message appears: "Do not navigate away"
- ✅ Progress bar shows "Training [Model Name]... (X/15)"
- ✅ Memory before training displayed (e.g., "450MB (25%)")
- ✅ Models train ONE AT A TIME (not all simultaneously)
- ✅ Each model shows: "✅ ModelName - Accuracy: X.XXXX"
- ✅ Memory after training displayed (e.g., "520MB (30%)")
- ✅ Navigation unlocks after completion
- ✅ No crashes or errors
- ✅ All 15 models complete successfully

**Acceptable Performance:**
- Total time: 2-4 minutes
- Peak memory: <800MB
- Memory increase: <200MB

---

### Test 2: Sequential ML Regression Training
**Objective:** Verify regression models use sequential training

**Steps:**
1. Navigate to **ML Regression** page
2. Upload a regression dataset OR use loaded data
3. Select target column (numeric)
4. Select **ALL regression models** (5-7 models)
5. Set CV folds: 3
6. Click **"Train Models"**

**Expected Results:**
- ✅ Memory before/after displayed
- ✅ Progress shows each model separately
- ✅ Models train sequentially
- ✅ Results display correctly (R², RMSE, MAE)
- ✅ No simultaneous training
- ✅ Navigation locks during training

**Acceptable Performance:**
- Total time: 1-2 minutes
- Peak memory: <600MB

---

### Test 3: Large Dataset Handling
**Objective:** Verify automatic sampling prevents crashes

**Steps:**
1. Navigate to **Data Upload** page
2. Upload OR generate a large CSV (>100K rows)
   - Option A: Use OpenML dataset with >100K rows
   - Option B: Create test file with pandas:
     ```python
     import pandas as pd
     import numpy as np
     df = pd.DataFrame({
         'col1': np.random.rand(200000),
         'col2': np.random.rand(200000),
         'col3': np.random.choice(['A','B','C'], 200000)
     })
     df.to_csv('large_test.csv', index=False)
     ```

**Expected Results:**
- ✅ Status: "Loading and analyzing your data..."
- ✅ Status: "Optimizing data..."
- ✅ Message: "Optimized! Saved XXmb (XX%)"
- ✅ Warning: "⚠️ Large Dataset Detected"
- ✅ Message: "Your dataset has 200,000 rows"
- ✅ Sampling slider appears
- ✅ Default value: 50,000 rows
- ✅ Range: 10K to 200K
- ✅ Info: "Using 50,000 sampled rows (from 200,000 total)"
- ✅ Analysis runs on sampled data
- ✅ No crash or timeout

**Memory Check:**
- ✅ Original data saved in session_state.data_full
- ✅ Sampled data in session_state.data
- ✅ Both optimized (category dtype, downcasted)

---

### Test 4: Memory Monitor & Cleanup
**Objective:** Verify memory monitoring and cleanup work correctly

**Steps:**
1. Check **sidebar** for Memory Monitor widget
2. Note initial memory usage
3. Run multiple analyses:
   - Market Basket Analysis
   - RFM Analysis  
   - ML Classification (5+ models)
4. Watch memory usage increase
5. If memory >80%, click **"Clean Up Memory"** button
6. Observe memory after cleanup

**Expected Results:**
- ✅ Memory Monitor shows:
  - Current MB (e.g., "450MB")
  - Percentage (e.g., "25%")
- ✅ Memory increases after each analysis
- ✅ If >80%: Warning "⚠️ High memory usage!"
- ✅ If >80%: "Clean Up Memory" button appears
- ✅ After cleanup: Memory decreases (20-40% reduction)
- ✅ After cleanup: Info message shows what was removed
- ✅ App remains functional after cleanup

**Manual Cleanup Test:**
- Wait 5+ minutes
- Memory should auto-cleanup
- Check if message appears: "🧹 Cleaned up: ..."

---

### Test 5: Navigation Lock During Processing
**Objective:** Verify navigation is prevented during long operations

**Steps:**
1. Navigate to **ML Classification**
2. Start training with **10+ models**
3. While training is running:
   - Try clicking different pages in sidebar
   - Try clicking Home, Data Upload, etc.
4. Observe behavior
5. Wait for training to complete
6. Try navigation again

**Expected Results:**
- ✅ During training:
  - Radio buttons are **DISABLED** (grayed out)
  - Cannot change page selection
  - Warning visible: "⚠️ Process Running"
- ✅ After training completes:
  - Message: "✅ Navigation unlocked"
  - Radio buttons re-enabled
  - Can navigate freely
- ✅ Training completes successfully
- ✅ No data loss from attempted navigation

---

### Test 6: Data Optimization on Upload
**Objective:** Verify DataFrame optimization works correctly

**Steps:**
1. Create test CSV with mixed data types:
   ```python
   import pandas as pd
   df = pd.DataFrame({
       'big_int': [1000000, 2000000, 3000000] * 1000,
       'small_int': [1, 2, 3] * 1000,
       'big_float': [1.234567890123456] * 3000,
       'category': ['A', 'B', 'C'] * 1000,
       'high_cardinality': range(3000)
   })
   df.to_csv('optimization_test.csv', index=False)
   ```
2. Upload the file
3. Observe optimization messages

**Expected Results:**
- ✅ Status: "⚡ Optimizing data..."
- ✅ Message: "✅ Optimized! Saved XXmb (XX%)"
- ✅ Memory savings: 30-50% typical
- ✅ Metrics shown:
  - Before: XXmb
  - After: XXmb  
  - Saved: XXmb (-XX%)
- ✅ Data types optimized:
  - Integers downcasted (int64 → int32/int16/int8)
  - Floats downcasted (float64 → float32)
  - Low-cardinality objects → category

**Verification:**
```python
# Check dtypes in Analysis page
st.session_state.data.dtypes
# Should show optimized types
```

---

### Test 7: Error Recovery
**Objective:** Verify partial results saved on errors

**Steps:**
1. Navigate to ML Classification
2. Select a problematic dataset OR model
3. Start training 10 models
4. IF error occurs during training
5. Check if partial results saved

**Expected Results:**
- ✅ Error message displayed clearly
- ✅ Message: "💾 Partial results saved"
- ✅ Can review completed models
- ✅ Checkpoint saved with error details
- ✅ Option to retry or continue
- ✅ Navigation unlocked after error

**Note:** This test may not trigger naturally. Consider it a passive test.

---

## 📊 Performance Benchmarks

### Target Metrics (Streamlit Cloud Free Tier)

| Metric | Target | Pass Criteria |
|--------|--------|---------------|
| **Peak Memory** | <800MB | ✅ if <850MB |
| **ML Classification (10 models)** | 2-4 min | ✅ if <5 min |
| **ML Regression (5 models)** | 1-2 min | ✅ if <3 min |
| **Large Dataset Load** | <1 min | ✅ if <90s |
| **Memory Cleanup** | 20-40% reduction | ✅ if >15% |
| **Optimization Savings** | 30-50% | ✅ if >20% |

### Failure Criteria
- ❌ App crashes or becomes unresponsive
- ❌ Memory exceeds 950MB
- ❌ Training timeout (>10 minutes)
- ❌ Navigation never unlocks
- ❌ Data loss or corruption
- ❌ No memory optimization detected

---

## 🐛 Known Issues & Workarounds

### Issue: Memory Monitor Shows 0MB
**Cause:** psutil not installed or error in ProcessManager  
**Fix:** Check requirements.txt has `psutil>=5.9.0`, restart app

### Issue: Sampling Slider Doesn't Appear
**Cause:** Dataset <100K rows  
**Expected:** Sampling only for large datasets

### Issue: Training Seems Slow
**Cause:** Sequential training is slower than parallel  
**Expected:** This is normal - stability over speed

### Issue: Cleanup Removes Needed Data
**Cause:** Data stored in non-protected session state key  
**Fix:** Use keys: `data`, `profile`, or `data_full`

---

## ✅ Test Results Template

```markdown
## Test Session: [Date]

**Tester:** [Name]  
**Environment:** Streamlit Cloud / Local  
**App Version:** [Commit Hash]

### Test Results:

| Test # | Test Name | Status | Notes |
|--------|-----------|--------|-------|
| 1 | ML Classification Sequential | ✅ PASS | 10 models, 3.2 min, peak 720MB |
| 2 | ML Regression Sequential | ✅ PASS | 5 models, 1.8 min, peak 550MB |
| 3 | Large Dataset Handling | ✅ PASS | 200K rows, sampled to 50K |
| 4 | Memory Monitor & Cleanup | ✅ PASS | Cleanup reduced from 820MB to 520MB |
| 5 | Navigation Lock | ✅ PASS | Navigation disabled during training |
| 6 | Data Optimization | ✅ PASS | Saved 180MB (42%) |
| 7 | Error Recovery | N/A | No errors triggered |

### Overall Assessment:
- ✅ All critical features working
- ✅ Performance within acceptable range
- ✅ No crashes observed
- ⚠️ Minor issues: [List any]

### Recommendations:
[Any suggestions for improvement]
```

---

## 🎯 Success Criteria

**Minimum Passing Grade (70%):**
- At least 5/7 tests pass
- No critical crashes
- Memory stays under 900MB

**Excellent Performance (90%+):**
- 6/7 or 7/7 tests pass
- Memory consistently under 700MB
- All features work smoothly
- Performance better than targets

**Production Ready:**
- All 7 tests pass
- Performance meets or exceeds targets
- No blocking issues
- Professional user experience

---

## 📞 Reporting Issues

If tests fail, provide:
1. Test number and name
2. Steps to reproduce
3. Expected vs actual behavior
4. Screenshots (especially memory widget)
5. Error messages (full traceback)
6. Streamlit Cloud logs (if applicable)

---

**Happy Testing!** 🧪✨
