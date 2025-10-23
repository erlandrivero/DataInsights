# 🔧 Integration Guide for New Utilities

**Date:** October 23, 2025  
**Purpose:** Guide for integrating new validation, error handling, and rate limiting utilities into existing code

---

## 📚 Table of Contents

1. [Input Validator Integration](#input-validator-integration)
2. [Error Handler Integration](#error-handler-integration)
3. [Rate Limiter Integration](#rate-limiter-integration)
4. [Quick Reference Examples](#quick-reference-examples)
5. [Migration Checklist](#migration-checklist)

---

## 1️⃣ Input Validator Integration

### Where to Add Validation

Add validation at the **start of every analysis function** before processing data.

### Example: ML Classification Module

**Before:**
```python
def show_ml_classification():
    st.header("ML Classification")
    
    if st.session_state.data is None:
        st.warning("Please upload data first")
        return
    
    # Continue with analysis...
```

**After:**
```python
from utils.input_validator import validate_dataset_for_analysis, InputValidator

def show_ml_classification():
    st.header("ML Classification")
    
    # Quick validation with auto-stop
    if not validate_dataset_for_analysis(st.session_state.data, analysis_type="classification"):
        return
    
    df = st.session_state.data
    
    # Continue with analysis...
```

### Example: Custom Validation

```python
from utils.input_validator import InputValidator

def show_time_series_forecasting():
    st.header("Time Series Forecasting")
    
    validator = InputValidator()
    
    # Validate dataset exists
    basic_result = validator.validate_dataset_basic(st.session_state.data, min_rows=30)
    if not validator.validate_and_display(basic_result):
        st.stop()
    
    df = st.session_state.data
    
    # Get date column selection
    date_col = st.selectbox("Select date column", df.columns)
    
    # Validate it's a datetime column
    datetime_result = validator.validate_datetime_column(df, date_col)
    if not validator.validate_and_display(datetime_result):
        st.info("💡 Try converting the column to datetime format")
        st.stop()
    
    # Continue with analysis...
```

---

## 2️⃣ Error Handler Integration

### Where to Add Error Handling

Add error handlers around:
- ML model training
- API calls (OpenAI)
- File operations
- Complex computations
- Data processing operations

### Example: ML Model Training

**Before:**
```python
def train_models(X, y):
    results = []
    for model_name in models:
        model = get_model(model_name)
        model.fit(X, y)
        score = model.score(X, y)
        results.append({'model': model_name, 'score': score})
    return results
```

**After:**
```python
from utils.error_handler import ErrorHandler, ErrorCategory, error_boundary

@error_boundary(ErrorCategory.MODEL_ERROR, "Failed to train ML models")
def train_models(X, y):
    results = []
    for model_name in models:
        try:
            model = get_model(model_name)
            model.fit(X, y)
            score = model.score(X, y)
            results.append({'model': model_name, 'score': score})
        except Exception as e:
            ErrorHandler.handle_error(
                e, 
                ErrorCategory.MODEL_ERROR,
                user_message=f"Failed to train {model_name}. Skipping this model.",
                show_details=True
            )
            continue
    return results
```

### Example: OpenAI API Calls

**Before:**
```python
def get_ai_insights(data_summary):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

**After:**
```python
from utils.error_handler import ErrorHandler, ErrorCategory

def get_ai_insights(data_summary):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except openai.error.RateLimitError as e:
        ErrorHandler.handle_api_error(
            e,
            service_name="OpenAI",
            fallback_message="AI insights are temporarily unavailable. Please try again in a few moments."
        )
        return "AI insights temporarily unavailable."
    except Exception as e:
        ErrorHandler.handle_error(e, ErrorCategory.API_ERROR)
        return None
```

---

## 3️⃣ Rate Limiter Integration

### Where to Add Rate Limiting

Add rate limiting to:
- OpenAI API calls
- External API calls
- Resource-intensive operations

### Example: OpenAI API Integration

**Step 1: Add API Key Validation at App Startup**

In `app.py`, add near the top after imports:

```python
from utils.rate_limiter import validate_api_key

# Validate API key on startup
if not validate_api_key("OPENAI_API_KEY"):
    st.stop()
```

**Step 2: Add Rate Limiting to AI Helper**

In `utils/ai_helper.py`:

```python
from utils.rate_limiter import rate_limited, InputSanitizer, APIUsageTracker

class AIHelper:
    def __init__(self):
        self.sanitizer = InputSanitizer()
    
    @rate_limited(max_calls=20, period_seconds=60)
    def get_insights(self, data_summary: str, user_question: str) -> str:
        """Get AI insights with rate limiting."""
        
        # Sanitize inputs
        try:
            clean_question = self.sanitizer.sanitize_text_query(user_question, max_length=1000)
        except ValueError as e:
            st.error(f"Invalid question: {str(e)}")
            return None
        
        # Make API call
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst assistant."},
                    {"role": "user", "content": f"Data: {data_summary}\n\nQuestion: {clean_question}"}
                ]
            )
            
            # Track usage
            APIUsageTracker.track_usage(
                model='gpt-4',
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
```

**Step 3: Display Usage Stats**

Add to sidebar or insights page:

```python
from utils.rate_limiter import APIUsageTracker

# Display API usage
APIUsageTracker.display_usage_stats()
```

---

## 4️⃣ Quick Reference Examples

### Validate DataFrame Before Processing

```python
from utils.input_validator import validate_dataset_for_analysis

def analyze_data():
    if not validate_dataset_for_analysis(df, "general"):
        return
    # Proceed with analysis
```

### Validate Specific Column Types

```python
from utils.input_validator import InputValidator

validator = InputValidator()

# Check numeric column
result = validator.validate_numeric_column(df, 'price')
if not validator.validate_and_display(result):
    st.stop()

# Check datetime column
result = validator.validate_datetime_column(df, 'date')
if not validator.validate_and_display(result):
    st.stop()
```

### Validate Class Balance

```python
from utils.input_validator import InputValidator

validator = InputValidator()
result = validator.validate_class_balance(df['target'], min_samples_per_class=5)

if not result.is_valid:
    st.error(result.message)
    if result.recommendation:
        st.info(f"💡 {result.recommendation}")
    st.stop()
```

### Safe Mathematical Operations

```python
from utils.error_handler import SafeOperations

# Safe division
avg_value = SafeOperations.safe_divide(total, count, default=0.0)

# Safe percentage
pct_str = SafeOperations.safe_percentage(part, total, decimals=2)

# Safe mean
mean_val = SafeOperations.safe_mean(values_list, default=0.0)
```

### Wrap Function with Error Boundary

```python
from utils.error_handler import error_boundary, ErrorCategory

@error_boundary(ErrorCategory.COMPUTATION_ERROR, "Calculation failed")
def complex_calculation(data):
    # Your code here
    return result
```

### Rate Limit API Calls

```python
from utils.rate_limiter import rate_limited

@rate_limited(max_calls=10, period_seconds=60)
def call_external_api(params):
    # API call code
    return response
```

---

## 5️⃣ Migration Checklist

### For Each Analysis Module:

- [ ] Add `validate_dataset_for_analysis()` at function start
- [ ] Add specific column validation where needed
- [ ] Wrap ML training in error handlers
- [ ] Wrap API calls in error handlers
- [ ] Add rate limiting to API calls
- [ ] Use `SafeOperations` for calculations
- [ ] Display helpful error messages
- [ ] Test with invalid inputs
- [ ] Test with missing data
- [ ] Test with edge cases

### Priority Order:

1. **High Priority (Do First):**
   - ML Classification (model training errors)
   - ML Regression (model training errors)
   - Insights page (OpenAI API calls)
   - Data Upload (file reading errors)

2. **Medium Priority:**
   - Market Basket Analysis (data format validation)
   - RFM Analysis (data format validation)
   - Time Series (datetime validation)
   - Anomaly Detection (numeric validation)

3. **Lower Priority:**
   - Text Mining (text data validation)
   - Monte Carlo (API calls validation)
   - Reports (file generation errors)

---

## 🎯 Testing After Integration

### Test Scenarios:

1. **No Data Loaded:**
   - Navigate to each page without uploading data
   - Should see friendly validation message, not crash

2. **Invalid Data:**
   - Upload CSV with wrong columns
   - Try to run analysis
   - Should see specific validation errors

3. **Missing Values:**
   - Upload data with many NaN values
   - Should see warning about missing values

4. **Small Dataset:**
   - Upload data with <10 rows
   - Should see error about insufficient data

5. **API Rate Limit:**
   - Make multiple AI requests quickly
   - Should see rate limit warning/error

6. **Invalid Column Selection:**
   - Select non-numeric column for numeric analysis
   - Should see clear type error

7. **Model Training Error:**
   - Try to train with insufficient data
   - Should see helpful error, not crash

---

## 📝 Code Review Checklist

Before committing integrated code:

- [ ] All validation added at function starts
- [ ] Error handlers catch all major exceptions
- [ ] User messages are clear and helpful
- [ ] Technical details available in expanders
- [ ] Rate limiting applied to API calls
- [ ] No hardcoded API keys
- [ ] Input sanitization applied
- [ ] Safe operations used for calculations
- [ ] Tests updated/added for new code
- [ ] No regressions in existing functionality

---

## 🚀 Deployment Notes

### Environment Variables Required:

```bash
OPENAI_API_KEY=your_key_here
```

### Streamlit Secrets (for Cloud):

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your_key_here"
```

### Requirements Already Updated:

- ✅ pytest>=7.0.0
- ✅ pytest-cov>=4.0.0
- ✅ pytest-mock>=3.10.0

---

## 💡 Best Practices

1. **Validate Early:** Check inputs before processing
2. **Fail Gracefully:** Never crash, always show helpful messages
3. **Be Specific:** "Price column must be numeric" > "Invalid data"
4. **Offer Solutions:** Always include recommendations
5. **Log Everything:** Use error handler logging
6. **Track Usage:** Monitor API costs
7. **Test Thoroughly:** Test with bad inputs
8. **Document Changes:** Update docstrings

---

## 📞 Support

If you encounter issues during integration:

1. Check this guide first
2. Review utility module docstrings
3. Look at test files for examples
4. Check error handler logs
5. Ask team for help

---

**Last Updated:** October 23, 2025  
**Next Review:** After Phase 2 completion
