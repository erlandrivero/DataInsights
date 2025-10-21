---

## 🔬 Anomaly Detection Module - Windsurf Prompts

**Total Time:** 3-4 hours

**Features:**
- Multiple anomaly detection algorithms (Isolation Forest, LOF, One-Class SVM)
- Interactive visualization of anomalies
- Anomaly scoring and threshold adjustment
- Feature importance for anomalies
- AI-powered explanation of outliers

---

### **PROMPT 1: Setup & Anomaly Detection Utility** (45 min)

**Goal:** Install dependencies and create the backend utility for running anomaly detection algorithms.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. First, ensure `scikit-learn` is in `requirements.txt`. 

Next, create a new utility file at `utils/anomaly_detection.py`. In this file, create a class called `AnomalyDetector`. This class will encapsulate the logic for various anomaly detection algorithms.

The `__init__` method should take a pandas DataFrame as input.

Implement the following methods in the class:

1.  `set_features(self, feature_cols)`: Sets the numeric feature columns to be used for detection. It should also scale the features using `StandardScaler`.
2.  `run_isolation_forest(self, contamination)`: Runs the Isolation Forest algorithm, predicts anomalies, and returns the dataframe with an 'anomaly_score' and 'is_anomaly' (True/False) column.
3.  `run_local_outlier_factor(self, contamination)`: Runs the Local Outlier Factor (LOF) algorithm for novelty detection and returns the dataframe with anomaly predictions.
4.  `run_one_class_svm(self, nu)`: Runs the One-Class SVM algorithm and returns the dataframe with anomaly predictions.

Ensure all methods include docstrings and handle potential errors, such as non-numeric data.
```

**Testing Checklist:**
- [ ] Verify `scikit-learn` is in `requirements.txt`.
- [ ] Check `utils/anomaly_detection.py` is created.
- [ ] Test `set_features` with valid and invalid column selections.
- [ ] Test each `run_` method and verify the output dataframe contains the correct new columns.
- [ ] Check that feature scaling is applied correctly.

---

### **PROMPT 2: Add Anomaly Detection Page to App** (45 min)

**Goal:** Create the main UI for the Anomaly Detection module.

**Windsurf Prompt:**

```
Upgrade the DataInsights app. In `app.py`, add a new page to the main navigation called "Anomaly Detection".

On this new page, design the following UI:

1.  **Title:** "Anomaly & Outlier Detection"
2.  **Data Upload:** Use the existing data uploader component.
3.  **Feature Selection:** Allow the user to select multiple numeric columns to use for anomaly detection.
4.  **Algorithm Selection:** Create a selectbox for the user to choose an algorithm: "Isolation Forest", "Local Outlier Factor", "One-Class SVM".
5.  **Parameter Controls:** Based on the selected algorithm, show relevant sliders:
    - For Isolation Forest & LOF: "Contamination" (0.01 to 0.5).
    - For One-Class SVM: "Nu" (0.01 to 0.5).
6.  **Execution Button:** A button labeled "Detect Anomalies".

Instantiate the `AnomalyDetector` class from `utils/anomaly_detection.py` after data is uploaded. The execution button should trigger the selected algorithm.
```

**Testing Checklist:**
- [ ] Verify "Anomaly Detection" appears in the navigation.
- [ ] Test data upload and feature selection.
- [ ] Check that parameter sliders change based on algorithm selection.
- [ ] Ensure the "Detect Anomalies" button is present.
- [ ] Test the flow up to the point of running the analysis.

---

### **PROMPT 3: Display Anomaly Results & Visualization** (1 hour)

**Goal:** Show the results of the anomaly detection, including a summary, a data table, and an interactive visualization.

**Windsurf Prompt:**

```
Upgrade the Anomaly Detection page in `app.py`. After the user clicks "Detect Anomalies", display the results in a new section below.

1.  **Summary Metrics:** Show a summary card with:
    - Total number of records analyzed.
    - Number of anomalies detected.
    - Percentage of anomalies.

2.  **Results Table:** Display the original dataframe with the `anomaly_score` and `is_anomaly` columns. Allow the user to filter the table to show only the detected anomalies.

3.  **Interactive Visualization:** 
    - If 2 features are selected, create a 2D scatter plot using Plotly. Color the points based on whether they are an anomaly or not. Add hover-over text to show data values.
    - If more than 2 features are selected, first perform PCA to reduce the data to 2 components, then create the 2D scatter plot. Explain that the plot is a 2D representation of the multi-dimensional space.

Make the visualization large and clear.
```

**Testing Checklist:**
- [ ] Verify the summary metrics are accurate.
- [ ] Check that the results table displays correctly and that the filter works.
- [ ] Test the 2D scatter plot with 2 features.
- [ ] Test the PCA-based scatter plot with 3+ features and ensure the explanation is present.
- [ ] Check that plot colors and hover text are correct.

---

### **PROMPT 4: Anomaly Explanation & AI Insights** (45 min)

**Goal:** Help the user understand *why* certain points are considered anomalies.

**Windsurf Prompt:**

```
Upgrade the Anomaly Detection results section in `app.py`. Add a new tabbed interface for deeper analysis of the detected anomalies.

**Tab 1: Anomaly Profiles**
- For each anomalous point, show a table or a radar chart comparing its feature values to the average values of the normal points. This helps explain what makes it an outlier.

**Tab 2: Feature Importance**
- If using Isolation Forest, calculate and display a bar chart of the feature importances that contribute most to identifying anomalies. (Hint: This can be derived from the tree structures).

**Tab 3: AI-Powered Explanation**
- Add an "Explain Anomalies with AI" button. When clicked, send the summary of normal data and the details of the top 5 anomalies to the OpenAI API.
- Ask the AI to provide a narrative explaining what these anomalies represent in a business context and what might be causing them (e.g., "Anomaly 3 shows unusually high spending and frequency, which could indicate either a high-value VIP customer or potential fraudulent activity.").
- Display the AI response in a formatted container.
```

**Testing Checklist:**
- [ ] Verify the Anomaly Profiles tab shows a clear comparison.
- [ ] Check that the Feature Importance bar chart is displayed for Isolation Forest.
- [ ] Test the AI Explanation button.
- [ ] Review the quality and business relevance of the AI-generated explanations.

---

### **PROMPT 5: Final Polish & Export** (30 min)

**Goal:** Add help text, error handling, and export functionality to complete the module.

**Windsurf Prompt:**

```
Finalize the Anomaly Detection module in `app.py`.

1.  **Add Help Text:** Create a help section with expanders explaining Anomaly Detection, Isolation Forest, LOF, and One-Class SVM for non-technical users.
2.  **Error Handling:** Implement robust error handling for cases like: no numeric columns, too few data points, or algorithms failing to converge.
3.  **Export Features:** Add buttons to download:
    - The full dataset with anomaly flags as a CSV.
    - A CSV containing only the detected anomalies.
    - The complete analysis report (including AI explanations) as a Markdown file.

Finally, create a new documentation file `guides/ANOMALY_DETECTION_GUIDE.md` explaining how to use the module and interpret the results. Update the main README to link to this new guide.
```

**Testing Checklist:**
- [ ] Check that all help text is clear and informative.
- [ ] Test error handling with various invalid inputs.
- [ ] Verify all download buttons work correctly.
- [ ] Check the content and formatting of the downloaded files.
- [ ] Ensure `ANOMALY_DETECTION_GUIDE.md` is created and linked.
- [ ] Perform a full end-to-end test of the module with different datasets and algorithms.

---

