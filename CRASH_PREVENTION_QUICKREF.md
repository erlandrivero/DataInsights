# Crash Prevention Quick Reference

**One-page guide for developers working on DataInsights**

---

## 🚀 Quick Start

### When Adding a New Long Operation

```python
from utils.process_manager import ProcessManager

# Pattern to follow
pm = ProcessManager("Operation_Name")

# Show warning
st.warning("⚠️ Do not navigate away during processing!")

# Lock navigation
pm.lock()

try:
    # Check memory before
    memory_stats = ProcessManager.get_memory_stats()
    st.info(f"💾 Memory: {memory_stats['rss_mb']:.1f}MB")
    
    # Your operation here
    result = your_heavy_operation()
    
    # Clean up memory
    import gc
    gc.collect()
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    
finally:
    # Always unlock
    pm.unlock()
    st.info("✅ Navigation unlocked")
```

---

## 📊 Memory Optimization Patterns

### 1. Optimize DataFrame on Load
```python
from utils.data_optimizer import DataOptimizer

df = pd.read_csv('file.csv')
df = DataOptimizer.optimize_dataframe(df)
# Saves 30-50% memory automatically
```

### 2. Sample Large Datasets
```python
if DataOptimizer.should_sample_data(df):  # >100K rows
    df = DataOptimizer.sample_data(df, sample_size=50000)
```

### 3. Sequential Processing
```python
# DON'T: Train all models simultaneously
results = trainer.train_all_models()  # ❌ Memory spike!

# DO: Train one at a time
results = trainer.train_models_sequentially()  # ✅ Stable
```

### 4. Clean Up Between Operations
```python
for item in heavy_items:
    process(item)
    del item  # Free memory immediately
    gc.collect()  # Force cleanup
```

---

## 🛡️ Navigation Lock Pattern

### Sidebar Integration
```python
from utils.process_manager import NavigationGuard

guard = NavigationGuard()
is_processing = guard.is_any_process_running()

selection = st.radio(
    "Navigation",
    options=["Home", "Analysis", "ML"],
    disabled=is_processing  # Lock during operations
)
```

---

## 💾 Memory Monitoring

### Check Current Memory
```python
from utils.process_manager import ProcessManager

stats = ProcessManager.get_memory_stats()
print(f"Memory: {stats['rss_mb']:.1f}MB ({stats['percent']:.1f}%)")
```

### Trigger Cleanup
```python
ProcessManager.cleanup_large_session_state_items()
# Removes items >50MB except: data, profile, data_full
```

### Display Memory Widget
```python
# Already in sidebar (lines 127-144)
# Automatically shows:
# - Current memory usage
# - Cleanup button if >80%
```

---

## 🔧 Common Patterns

### ML Training (Sequential)
```python
# 1. Get available models
all_models = trainer.get_all_models()

# 2. Progress callback
def update_progress(current, total, model_name):
    st.write(f"Training {model_name}... ({current}/{total})")

# 3. Train sequentially
results = trainer.train_models_sequentially(
    model_names=['Model1', 'Model2'],
    cv_folds=3,
    progress_callback=update_progress
)

# 4. Convert to list format if needed
results_list = [
    {**result, 'model_name': name, 'success': True}
    for name, result in results.items()
    if 'error' not in result
]
```

### Data Upload with Optimization
```python
# 1. Load file
df = pd.read_csv(uploaded_file)

# 2. Check size
original_memory = DataOptimizer.get_memory_usage(df)
st.info(f"Original: {original_memory['total_mb']:.1f}MB")

# 3. Optimize
df = DataOptimizer.optimize_dataframe(df)

# 4. Sample if large
if DataOptimizer.should_sample_data(df):
    st.session_state.data_full = df  # Keep original
    df = DataOptimizer.sample_data(df, sample_size=50000)

# 5. Store
st.session_state.data = df
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T
```python
# 1. Store full model objects in session state
st.session_state.model = trained_model  # ❌ Memory leak!

# 2. Train all models at once
for model in models:
    results.append(train(model))  # ❌ All in memory!

# 3. Forget to unlock navigation
pm.lock()
do_something()
# pm.unlock()  # ❌ Forgot! Navigation stuck!

# 4. No cleanup between operations
for i in range(100):
    big_result = heavy_operation()
    results.append(big_result)  # ❌ 100 items in memory!
```

### ✅ DO
```python
# 1. Store only essential results
st.session_state.results = {
    'accuracy': 0.95,
    'metrics': {...}
}  # ✅ Small footprint

# 2. Train sequentially with cleanup
for model in models:
    result = train(model)
    results.append(result)
    del model
    gc.collect()  # ✅ Clean after each

# 3. Always unlock in finally block
pm.lock()
try:
    do_something()
finally:
    pm.unlock()  # ✅ Always unlocks

# 4. Clean up in loop
for i in range(100):
    result = heavy_operation()
    results.append(result)
    gc.collect()  # ✅ Periodic cleanup
```

---

## 📝 Session State Best Practices

### Protected Keys (Never Auto-Cleaned)
- `data` - Current dataset
- `data_full` - Full dataset before sampling
- `profile` - Data profiling results

### Cleanup Candidates (May Be Removed)
- `*_results` - Analysis results
- `*_model` - Trained models
- `*_cached` - Cached data
- Any key with items >50MB

### Naming Convention
```python
# Good names (descriptive, follows pattern)
st.session_state.ml_results = [...]
st.session_state.rfm_segmented = df
st.session_state.mba_ai_insights = "..."

# Avoid
st.session_state.temp = [...]  # Not descriptive
st.session_state.x = df  # Too short
```

---

## 🎯 Memory Budget (Streamlit Cloud Free)

| Component | Budget | Notes |
|-----------|--------|-------|
| **Total RAM** | 1024MB | Hard limit |
| **Base App** | ~150MB | Streamlit + imports |
| **Available** | ~870MB | For operations |
| **Safe Peak** | <800MB | Leave 200MB buffer |
| **Warning Zone** | 800-900MB | Trigger cleanup |
| **Critical** | >900MB | Risk of crash |

### Typical Usage
- Small dataset (10K rows): +50MB
- Medium dataset (100K rows): +200MB
- Large dataset (500K rows): +500MB → **Sample to 50K**
- ML Training (10 models sequential): +150-250MB peak
- ML Training (10 models parallel): +600MB → **Too high!**

---

## 🔍 Debugging Memory Issues

### Check Memory in Real-Time
```python
import streamlit as st
from utils.process_manager import ProcessManager

# Add temporary memory logger
if st.checkbox("Show Memory Debug"):
    stats = ProcessManager.get_memory_stats()
    st.metric("Current Memory", f"{stats['rss_mb']:.0f}MB", 
              delta=f"{stats['percent']:.1f}%")
```

### Find Large Session State Items
```python
import sys

for key, value in st.session_state.items():
    size_mb = sys.getsizeof(value) / 1024 / 1024
    if size_mb > 10:  # >10MB
        st.warning(f"{key}: {size_mb:.1f}MB")
```

### Profile DataFrame Memory
```python
from utils.data_optimizer import DataOptimizer

mem_info = DataOptimizer.get_memory_usage(df)
print(f"Total: {mem_info['total_mb']:.1f}MB")
print(f"Rows: {mem_info['rows']:,}")
print(f"Columns: {mem_info['columns']}")

# Per-column breakdown
for col, mb in mem_info['per_column'].items():
    if mb > 1:  # >1MB
        print(f"  {col}: {mb:.1f}MB")
```

---

## 📦 Required Imports

```python
# Core crash prevention
from utils.process_manager import ProcessManager, NavigationGuard

# Memory optimization
from utils.data_optimizer import DataOptimizer

# Lazy loading (advanced)
from utils.lazy_loader import LazyModuleLoader, SequentialExecutor

# Always needed for cleanup
import gc
import streamlit as st
```

---

## 🎓 Additional Resources

- **Full Implementation:** `CRASH_PREVENTION_IMPLEMENTATION.md`
- **Testing Guide:** `TESTING_GUIDE.md`
- **Original Guide:** `DataInsights_ Crash Prevention & Memory Optimization Guide.md`
- **ProcessManager Code:** `utils/process_manager.py`
- **Data Optimizer Code:** `utils/data_optimizer.py`

---

## 🚨 Emergency Recovery

### If App Crashes
1. Check Streamlit Cloud logs for error
2. Look for "MemoryError" or "Killed"
3. Identify which operation crashed
4. Add ProcessManager if missing
5. Add sequential processing if parallel
6. Reduce sample size if data operation

### If Memory >90%
```python
# Manual emergency cleanup
ProcessManager.cleanup_large_session_state_items()
gc.collect()

# Clear specific large items
if 'large_results' in st.session_state:
    del st.session_state.large_results
    gc.collect()
```

### If Navigation Stuck
```python
# Force unlock (emergency only)
st.session_state.global_process_running = False

# Better: Add to finally block
try:
    operation()
finally:
    pm.unlock()  # Always unlocks
```

---

**Last Updated:** October 29, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅
