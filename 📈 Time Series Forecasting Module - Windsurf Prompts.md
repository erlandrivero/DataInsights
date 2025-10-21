---

## 📈 Time Series Forecasting Module - Windsurf Prompts

**Total Time:** 4-5 hours

**Features:**
- Time series data upload and parsing
- Trend and seasonality decomposition
- Multiple forecasting models (ARIMA, Prophet)
- Interactive forecast visualizations
- Model comparison and evaluation
- AI-powered insights and recommendations

---

### **PROMPT 1: Setup & Data Handling** (45 min)

**Goal:** Install dependencies and create the data handling and analysis utility for time series.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. First, add the following libraries to `requirements.txt`: `statsmodels`, `pmdarima`, and `prophet`. 

Next, create a new utility file at `utils/time_series.py`. In this file, create a class called `TimeSeriesAnalyzer`. This class will handle time series data processing and analysis. 

The `__init__` method should take a pandas DataFrame as input. 

Implement the following methods in the class:

1.  `set_time_column(self, time_col, value_col)`: Sets the time and value columns for analysis. It should convert the time column to datetime objects.
2.  `decompose_time_series(self)`: Performs seasonal decomposition (additive and multiplicative) and returns the trend, seasonal, and residual components. Handle errors for series that are too short.
3.  `get_stationarity_test(self)`: Performs the Augmented Dickey-Fuller test to check for stationarity and returns the test statistic, p-value, and a conclusion.
4.  `get_autocorrelation(self)`: Calculates and returns ACF and PACF plots.

Ensure all methods have proper error handling and docstrings. The class should store intermediate results to avoid re-computation.
```

**Testing Checklist:**
- [ ] Verify `requirements.txt` is updated.
- [ ] Check `utils/time_series.py` exists.
- [ ] Test `set_time_column` with valid and invalid column names.
- [ ] Test `decompose_time_series` and check the output.
- [ ] Verify `get_stationarity_test` returns correct results.
- [ ] Check that `get_autocorrelation` generates plots.

---

### **PROMPT 2: Add Time Series Page to App** (45 min)

**Goal:** Create the main UI for the Time Series Forecasting module in the Streamlit app.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. In `app.py`, add a new page to the main navigation called "Time Series Forecasting". 

On this new page, create the following UI layout:

1.  **Title:** "Time Series Forecasting & Analysis"
2.  **Data Upload:** Use the existing data uploader component. Add a section to select the `Time Column` (datetime) and `Value Column` (numeric) from the uploaded data.
3.  **Analysis Section:** Create a tabbed interface with the following tabs:
    - **Decomposition:** Display the trend, seasonal, and residual plots from the `TimeSeriesAnalyzer`.
    - **Stationarity:** Show the results of the Dickey-Fuller test and an explanation.
    - **Autocorrelation:** Display the ACF and PACF plots.

Instantiate the `TimeSeriesAnalyzer` from `utils/time_series.py` once the user selects the columns. Add loading spinners for long processes. Ensure the UI is clean and professional.
```

**Testing Checklist:**
- [ ] Verify "Time Series Forecasting" appears in the navigation.
- [ ] Test data upload and column selection.
- [ ] Check that the Decomposition tab shows the correct plots.
- [ ] Verify the Stationarity tab displays test results.
- [ ] Ensure the Autocorrelation tab shows ACF/PACF plots.
- [ ] Test with different datasets.

---

### **PROMPT 3: Implement ARIMA Forecasting Model** (1 hour)

**Goal:** Add the auto-ARIMA model for forecasting and display the results.

**Windsurf Prompt:**

```
Upgrade the `utils/time_series.py` file. Add a new method to the `TimeSeriesAnalyzer` class called `run_auto_arima(self, forecast_periods)`. This method will:

1.  Use `pmdarima.auto_arima` to automatically find the best ARIMA model.
2.  Fit the model to the data.
3.  Generate a forecast for the specified number of `forecast_periods`.
4.  Return the forecast, confidence intervals, and the model summary.

In `app.py`, add a new section under the analysis tabs called "Forecasting Models". Create a sub-section for "ARIMA Forecast". Add a number input for the user to specify the forecast horizon (e.g., 30, 90, 365 days). 

When the user clicks a "Run ARIMA Forecast" button, call the `run_auto_arima` method. Display:
- The model summary (order, AIC, etc.).
- A Plotly chart showing the historical data, the forecast, and the confidence intervals.
- A table with the forecasted values.
```

**Testing Checklist:**
- [ ] Check for the new ARIMA section in the UI.
- [ ] Test the forecast horizon input.
- [ ] Verify the ARIMA model runs and displays a summary.
- [ ] Ensure the forecast plot is interactive and clear.
- [ ] Check the forecasted values table.
- [ ] Test with different forecast periods.

---

### **PROMPT 4: Implement Prophet Forecasting Model** (1 hour)

**Goal:** Add the Prophet model for forecasting and comparison.

**Windsurf Prompt:**

```
Upgrade the `utils/time_series.py` file. Add another method to the `TimeSeriesAnalyzer` class called `run_prophet(self, forecast_periods)`. This method will:

1.  Initialize a Prophet model.
2.  Fit the model to the data (handle column naming requirements for Prophet).
3.  Generate a future dataframe and make a forecast.
4.  Return the forecast dataframe, and plots for components (trend, weekly, yearly seasonality).

In `app.py`, add another sub-section under "Forecasting Models" for "Prophet Forecast". Add a button "Run Prophet Forecast".

When clicked, call the `run_prophet` method. Display:
- The forecast plot showing historical data, forecast, and uncertainty intervals.
- The component plots (trend and seasonality).
- A table with the forecasted values.
```

**Testing Checklist:**
- [ ] Check for the new Prophet section in the UI.
- [ ] Verify the Prophet model runs on button click.
- [ ] Ensure the main forecast plot is displayed correctly.
- [ ] Check that component plots are visible.
- [ ] Verify the forecast table is accurate.

---

### **PROMPT 5: Model Comparison & AI Insights** (45 min)

**Goal:** Compare the models and add AI-powered insights.

**Windsurf Prompt:**

```
Upgrade the `app.py` file in the "Forecasting Models" section. After both ARIMA and Prophet models have been run, add a "Model Comparison" section. 

In this section, display:
1.  A side-by-side table comparing key metrics (e.g., MAE, RMSE) for both models (calculate these in the `TimeSeriesAnalyzer` class).
2.  A single chart overlaying the historical data with both the ARIMA and Prophet forecasts for visual comparison.

Next, add an "AI-Powered Insights" button. When clicked, send the analysis results (decomposition, stationarity, model performance, forecasts) to the OpenAI API (using the existing integration). 

Ask the AI to provide:
- An interpretation of the time series characteristics.
- A recommendation on which model is better and why.
- Business insights based on the forecast (e.g., "Expect a 15% increase in sales next quarter, consider increasing inventory.").

Display the AI's response in a formatted markdown container.
```

**Testing Checklist:**
- [ ] Verify the Model Comparison section appears after both models are run.
- [ ] Check the metrics table for accuracy.
- [ ] Ensure the comparison plot is clear.
- [ ] Test the AI-Powered Insights button.
- [ ] Review the quality and relevance of the AI-generated insights.

---

### **PROMPT 6: Final Polish & Export** (30 min)

**Goal:** Add help text, error handling, and export functionality.

**Windsurf Prompt:**

```
Finalize the Time Series Forecasting module in `app.py`.

1.  **Add Help Text:** Create a help section with expanders explaining Time Series Analysis, Decomposition, Stationarity, ARIMA, and Prophet for non-technical users.
2.  **Error Handling:** Ensure there are user-friendly error messages if the data is not suitable for time series analysis (e.g., not enough data points, wrong data types).
3.  **Export Features:** Add buttons to download:
    - The full forecast data (ARIMA and Prophet) as a single CSV.
    - The complete analysis report (including AI insights) as a Markdown file.
    - The forecast comparison chart as a PNG image.

Finally, create a new documentation file `guides/TIME_SERIES_GUIDE.md` explaining how to use the module, what the results mean, and common use cases. Update the main README to link to this new guide.
```

**Testing Checklist:**
- [ ] Check that all help text is clear and informative.
- [ ] Test error handling with unsuitable data.
- [ ] Verify all download buttons work correctly.
- [ ] Check the content and formatting of downloaded files.
- [ ] Ensure `TIME_SERIES_GUIDE.md` is created and linked.
- [ ] Perform a full end-to-end test of the module.

---

