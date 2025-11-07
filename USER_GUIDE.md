# DataInsights - Complete User Guide 📚

**Version 1.0** | **Last Updated: November 2024**

Welcome to the comprehensive user guide for DataInsights - your AI-powered data mining and business intelligence platform.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Modules](#core-modules)
3. [Data Mining Modules](#data-mining-modules)
4. [Advanced Analytics](#advanced-analytics)
5. [AI Features](#ai-features)
6. [Export & Reporting](#export--reporting)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Time Setup

1. **Access the Application**
   - Local: `http://localhost:8501`
   - Cloud: Visit your deployed Streamlit URL

2. **Upload Your Data**
   - Navigate to **📤 Data Upload**
   - Supported formats: CSV, Excel (.xlsx, .xls)
   - Or try sample datasets to explore features

3. **Explore the Sidebar**
   - Module navigation
   - Quick export options
   - Session information

### Understanding the Interface

**Navigation**
- Use the sidebar to switch between modules
- Each module is self-contained with its own workflow
- Data uploaded in one session is available across all modules

**Session State**
- Your data persists across modules during your session
- Analysis results are cached for quick access
- Use "Clear Session" to start fresh

---

## Core Modules

### 1. 📤 Data Upload

**Purpose:** Import and profile your datasets

**Features:**
- **File Upload:** Drag & drop or browse for CSV/Excel files
- **Sample Datasets:** Pre-loaded datasets for testing
  - Iris (Classification)
  - Boston Housing (Regression)
  - Groceries (Market Basket Analysis)
- **Automatic Profiling:**
  - Dataset dimensions
  - Column types
  - Missing values
  - Memory usage
  - Data quality issues

**How to Use:**
1. Click "Browse files" or drag & drop your file
2. Review the automatic data profile
3. Check for data quality issues
4. Proceed to analysis modules

**Supported Data Types:**
- Numerical (int, float)
- Categorical (object, category)
- Datetime
- Boolean

---

### 2. 📊 Data Analysis & Cleaning

**Purpose:** Explore, clean, and understand your data

#### Statistical Analysis Tab

**Features:**
- **Descriptive Statistics:** Mean, median, std dev, quartiles
- **Distribution Analysis:** Skewness, kurtosis
- **Missing Value Report:** Identify and quantify gaps
- **Duplicate Detection:** Find exact duplicate rows
- **Correlation Analysis:** Discover relationships between variables

**How to Use:**
1. Upload data first
2. Navigate to "Statistical Analysis" tab
3. Review summary statistics
4. Identify data quality issues
5. Use AI insights for recommendations

#### Data Cleaning Tab

**Features:**
- **Handle Missing Values:**
  - Drop rows/columns
  - Fill with mean/median/mode
  - Forward/backward fill
  - Custom value fill
  
- **Remove Duplicates:**
  - Keep first/last occurrence
  - Remove all duplicates
  
- **Outlier Detection:**
  - IQR method
  - Z-score method
  - Visual identification
  
- **Column Operations:**
  - Drop columns
  - Rename columns
  - Change data types
  - Create derived columns

**How to Use:**
1. Select cleaning operation
2. Configure parameters
3. Preview changes
4. Apply transformation
5. Download cleaned data

#### Visualizations Tab

**Available Charts:**
- **Histograms:** Distribution of numerical variables
- **Bar Charts:** Frequency of categorical variables
- **Scatter Plots:** Relationships between two numerical variables
- **Box Plots:** Outlier detection and distribution
- **Correlation Heatmap:** Variable relationships
- **Pair Plots:** Multiple variable relationships

**How to Use:**
1. Select chart type
2. Choose variables (X, Y, color, size)
3. Customize appearance
4. Download as PNG/HTML

---

### 3. 🤖 AI Insights & Natural Language Querying

**Purpose:** Ask questions about your data in plain English

**Features:**
- **Natural Language Interface:** Ask questions conversationally
- **Code Generation:** Get executable Python code
- **Interactive Execution:** Run code and see results
- **Chat History:** Review previous questions and answers
- **Context-Aware:** AI understands your dataset structure

**Example Questions:**
```
"What are the top 5 products by sales?"
"Show me the correlation between age and income"
"Create a scatter plot of price vs quantity"
"What percentage of customers are from California?"
"Find outliers in the revenue column"
```

**How to Use:**
1. Type your question in natural language
2. Click "Ask AI"
3. Review the generated code
4. Click "Execute Code" to run
5. View results and visualizations

**Tips:**
- Be specific about what you want to see
- Reference column names accurately
- Ask for visualizations when appropriate
- Use follow-up questions to refine results

---

### 4. 📄 Professional Reports

**Purpose:** Generate comprehensive business-ready reports

**Report Sections:**
1. **Executive Summary:** High-level overview
2. **Data Profile:** Dataset characteristics
3. **Quality Assessment:** Data quality metrics
4. **Statistical Analysis:** Key findings
5. **Visualizations:** Charts and graphs
6. **Recommendations:** Actionable insights

**How to Use:**
1. Navigate to Reports module
2. Select report sections to include
3. Customize report title and description
4. Click "Generate Report"
5. Download as Markdown or Text

**Use Cases:**
- Stakeholder presentations
- Documentation
- Project submissions
- Data quality audits

---

## Data Mining Modules

### 5. 🧺 Market Basket Analysis

**Purpose:** Discover purchasing patterns and product associations

**Algorithm:** Apriori

**Key Metrics:**
- **Support:** How often items appear together
- **Confidence:** Likelihood of B given A
- **Lift:** Strength of association (>1 = positive correlation)

**Features:**
- Interactive threshold controls
- Association rule mining
- Network visualization
- Top items analysis
- Rule filtering and search
- AI-generated business insights

**How to Use:**
1. Load transactional data (Item, Transaction ID format)
2. Or use sample groceries dataset
3. Adjust thresholds:
   - Min Support: 0.001 - 0.1
   - Min Confidence: 0.1 - 0.9
   - Min Lift: 1.0 - 5.0
4. Click "Run Market Basket Analysis"
5. Explore visualizations:
   - Support-Confidence scatter plot
   - Network graph of associations
   - Top items by frequency
6. Filter rules by search
7. Export rules to CSV

**Business Applications:**
- Product placement optimization
- Cross-selling strategies
- Bundle recommendations
- Inventory management

**See:** [MBA_GUIDE.md](MBA_GUIDE.md) for detailed instructions

---

### 6. 👥 RFM Analysis & Customer Segmentation

**Purpose:** Segment customers based on purchasing behavior

**RFM Metrics:**
- **Recency:** Days since last purchase
- **Frequency:** Number of purchases
- **Monetary:** Total spend

**Features:**
- Automatic RFM scoring (1-5 scale)
- K-Means clustering
- Customer segment profiles
- Interactive visualizations
- Segment-specific recommendations
- Export segmented customer lists

**How to Use:**
1. Load customer transaction data with:
   - Customer ID
   - Transaction Date
   - Transaction Amount
2. Or use sample data
3. Click "Run RFM Analysis"
4. Review RFM scores and segments
5. Analyze segment characteristics
6. Export results

**Customer Segments:**
- **Champions:** Best customers (High R, F, M)
- **Loyal Customers:** Frequent buyers
- **At Risk:** Haven't purchased recently
- **Lost:** Inactive customers
- **New Customers:** Recent first-time buyers

**Business Applications:**
- Targeted marketing campaigns
- Customer retention strategies
- Loyalty program design
- Churn prevention

---

### 7. 🎲 Monte Carlo Simulation

**Purpose:** Financial forecasting and risk analysis

**Features:**
- Revenue/profit forecasting
- Risk assessment
- Scenario analysis
- Confidence intervals
- Probability distributions
- Interactive visualizations

**Parameters:**
- Initial value
- Expected return (%)
- Volatility (%)
- Time horizon (days/months/years)
- Number of simulations

**How to Use:**
1. Enter initial investment/revenue
2. Set expected return rate
3. Define volatility (risk)
4. Choose time horizon
5. Set number of simulations (1,000 - 10,000)
6. Click "Run Simulation"
7. Analyze results:
   - Distribution of outcomes
   - Confidence intervals
   - Risk metrics
   - Percentile analysis

**Outputs:**
- Mean expected value
- Standard deviation
- 5th, 50th, 95th percentiles
- Probability of profit/loss
- Value at Risk (VaR)

**Business Applications:**
- Investment planning
- Budget forecasting
- Risk management
- Strategic planning

---

### 8. 🤖 ML Classification

**Purpose:** Predict categorical outcomes with machine learning

**15 Algorithms Available:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM
7. CatBoost
8. Support Vector Machine (SVM)
9. K-Nearest Neighbors (KNN)
10. Naive Bayes
11. AdaBoost
12. Extra Trees
13. Bagging Classifier
14. Stacking Classifier
15. Voting Classifier

**Features:**
- **Smart Model Selection:** AI recommends best models for your data
- **Automated Training:** Train multiple models simultaneously
- **Comprehensive Evaluation:**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curves
  - Confusion matrices
  - Classification reports
- **SHAP Interpretability:** Understand feature importance
- **Cross-Validation:** 5-fold CV for robust evaluation
- **Hyperparameter Tuning:** GridSearchCV for optimization

**Built-in Templates:**
- Lead Scoring
- Churn Prediction
- Credit Risk Assessment
- Customer Segmentation
- Fraud Detection

**How to Use:**
1. Load classification dataset
2. Select target variable (what to predict)
3. Choose features (predictors)
4. Select models to train (or use AI recommendations)
5. Configure training options:
   - Test split ratio
   - Cross-validation folds
   - Handle imbalanced data
6. Click "Train Models"
7. Review model comparison
8. Analyze best model:
   - Performance metrics
   - Feature importance
   - SHAP values
   - Confusion matrix
9. Generate AI insights
10. Export results

**SHAP Visualizations:**
- Summary Plot: Feature impact across all samples
- Feature Importance: Global feature rankings
- Dependence Plots: Feature interactions
- Waterfall Plot: Individual prediction explanation

**Export Options:**
- Model comparison CSV
- Best model metrics
- Feature importance
- Predictions
- SHAP values

---

### 9. 📈 ML Regression

**Purpose:** Predict continuous numerical outcomes

**15 Algorithms Available:**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. XGBoost Regressor
9. LightGBM Regressor
10. CatBoost Regressor
11. Support Vector Regressor (SVR)
12. K-Nearest Neighbors Regressor
13. AdaBoost Regressor
14. Extra Trees Regressor
15. Bagging Regressor

**Features:**
- Same comprehensive features as ML Classification
- Regression-specific metrics:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- Residual analysis
- Prediction vs Actual plots
- SHAP interpretability

**Built-in Templates:**
- House Price Prediction
- Sales Forecasting
- Demand Prediction
- Revenue Estimation

**How to Use:**
Similar to ML Classification, but for continuous targets

**Business Applications:**
- Price optimization
- Demand forecasting
- Resource planning
- Financial modeling

---

### 10. 🔍 Anomaly & Outlier Detection

**Purpose:** Identify unusual patterns and outliers in data

**3 Algorithms:**
1. **Isolation Forest:** Tree-based anomaly detection
2. **Local Outlier Factor (LOF):** Density-based detection
3. **One-Class SVM:** Boundary-based detection

**Features:**
- Multiple algorithm comparison
- Contamination rate adjustment
- Interactive visualizations
- Anomaly scoring
- Export detected anomalies

**How to Use:**
1. Load dataset
2. Select features for analysis
3. Choose algorithm(s)
4. Set contamination rate (expected % of anomalies)
5. Click "Detect Anomalies"
6. Review results:
   - Number of anomalies detected
   - Anomaly scores
   - Visualizations (PCA, t-SNE)
7. Export anomalies for investigation

**Business Applications:**
- Fraud detection
- Quality control
- Network security
- Equipment monitoring
- Transaction monitoring

---

### 11. 📈 Time Series Forecasting

**Purpose:** Predict future values based on historical time series data

**2 Algorithms:**
1. **ARIMA:** Statistical forecasting
2. **Prophet:** Facebook's forecasting tool

**Features:**
- Automatic seasonality detection
- Trend analysis
- Confidence intervals
- Multiple forecast horizons
- Interactive visualizations
- Model comparison

**How to Use:**
1. Load time series data with:
   - Date/timestamp column
   - Value column
2. Select date and value columns
3. Choose forecasting algorithm
4. Set forecast horizon (days/months)
5. Click "Generate Forecast"
6. Review:
   - Historical vs predicted
   - Trend components
   - Seasonality patterns
   - Forecast with confidence intervals
7. Export forecast

**Business Applications:**
- Sales forecasting
- Demand planning
- Inventory optimization
- Resource allocation
- Budget planning

---

### 12. 💬 Text Mining & Sentiment Analysis

**Purpose:** Extract insights from text data

**Features:**
- **Sentiment Analysis:** Positive, Negative, Neutral
- **Topic Modeling:** LDA for theme extraction
- **Named Entity Recognition (NER):** Extract people, places, organizations
- **Word Clouds:** Visual text representation
- **N-gram Analysis:** Common phrases
- **Text Statistics:** Word count, unique words, etc.

**How to Use:**
1. Load dataset with text column
2. Select text column
3. Choose analysis type:
   - Sentiment Analysis
   - Topic Modeling
   - NER
   - Word Cloud
4. Configure parameters
5. Click "Analyze Text"
6. Review results and visualizations
7. Export insights

**Business Applications:**
- Customer feedback analysis
- Social media monitoring
- Product review analysis
- Brand sentiment tracking
- Content analysis

---

## Advanced Analytics

### 13. 🧪 A/B Testing

**Purpose:** Statistical testing for experiment analysis

**Features:**
- Two-sample t-tests
- Chi-square tests
- Proportion tests
- Statistical significance testing
- Effect size calculation
- Sample size recommendations
- SRM (Sample Ratio Mismatch) detection

**How to Use:**
1. Load experiment data with:
   - Group (A/B)
   - Metric values
2. Or use sample data
3. Select test type
4. Choose metric to compare
5. Set significance level (α)
6. Click "Run A/B Test"
7. Review:
   - Statistical significance
   - P-value
   - Effect size
   - Confidence intervals
   - Recommendations
8. Export results

**Business Applications:**
- Website optimization
- Marketing campaign testing
- Product feature testing
- Pricing experiments
- UX improvements

---

### 14. 👥 Cohort Analysis

**Purpose:** Analyze user behavior over time by cohort

**Features:**
- Cohort retention analysis
- Cohort comparison
- Heatmap visualizations
- Trend analysis
- Export cohort data

**Cohort Types:**
- Time-based (monthly, weekly)
- Acquisition channel
- Customer segment
- Product category

**How to Use:**
1. Load data with:
   - User/Customer ID
   - Cohort date (signup, first purchase)
   - Event date
2. Select cohort period (monthly/weekly)
3. Choose metric (retention, revenue, etc.)
4. Click "Analyze Cohorts"
5. Review retention heatmap
6. Compare cohorts
7. Export analysis

**Business Applications:**
- User retention analysis
- Product adoption tracking
- Marketing effectiveness
- Customer lifetime value
- Churn prediction

---

### 15. 🎯 Recommendation Systems

**Purpose:** Build personalized recommendation engines

**3 Approaches:**
1. **Collaborative Filtering:** User-based recommendations
2. **Content-Based:** Item similarity recommendations
3. **Hybrid:** Combination of both

**Features:**
- Multiple algorithm support
- Similarity metrics
- Top-N recommendations
- Evaluation metrics
- Export recommendations

**How to Use:**
1. Load interaction data:
   - User ID
   - Item ID
   - Rating/Interaction
2. Or use sample data (MovieLens)
3. Choose recommendation approach
4. Set parameters (N recommendations, similarity metric)
5. Click "Build Recommendations"
6. Test with specific users
7. Review recommendations
8. Export for deployment

**Business Applications:**
- E-commerce product recommendations
- Content recommendations (movies, articles)
- Personalized marketing
- Cross-selling
- Customer engagement

---

### 16. 🗺️ Geospatial Analysis

**Purpose:** Analyze and visualize location-based data

**Features:**
- Interactive maps
- Heatmaps
- Choropleth maps
- Clustering analysis
- Distance calculations
- Route optimization

**How to Use:**
1. Load data with location information:
   - Latitude/Longitude
   - Or Address (geocoding available)
2. Select map type
3. Choose variables to visualize
4. Configure map settings
5. Click "Generate Map"
6. Interact with map
7. Export map as HTML

**Business Applications:**
- Store location analysis
- Delivery route optimization
- Market penetration analysis
- Customer distribution
- Territory planning

---

### 17. ⏱️ Survival Analysis

**Purpose:** Analyze time-to-event data

**Features:**
- Kaplan-Meier survival curves
- Cox Proportional Hazards model
- Hazard ratios
- Survival probabilities
- Cohort comparison

**How to Use:**
1. Load data with:
   - Time duration
   - Event indicator (0/1)
   - Covariates
2. Select time and event columns
3. Choose analysis type
4. Click "Run Survival Analysis"
5. Review survival curves
6. Analyze hazard ratios
7. Export results

**Business Applications:**
- Customer churn analysis
- Equipment failure prediction
- Employee retention
- Product lifecycle analysis
- Medical studies

---

### 18. 🕸️ Network Analysis

**Purpose:** Analyze relationships and connections

**Features:**
- Network visualization
- Centrality measures
- Community detection
- Path analysis
- Network metrics

**How to Use:**
1. Load edge list data:
   - Source node
   - Target node
   - Weight (optional)
2. Click "Analyze Network"
3. Review network visualization
4. Analyze metrics:
   - Degree centrality
   - Betweenness centrality
   - Closeness centrality
   - PageRank
5. Detect communities
6. Export network data

**Business Applications:**
- Social network analysis
- Supply chain optimization
- Fraud detection networks
- Organizational analysis
- Influence mapping

---

### 19. 🔄 Churn Prediction

**Purpose:** Predict customer churn with specialized models

**Features:**
- Specialized churn models
- Feature engineering for churn
- Churn risk scoring
- Retention recommendations
- SHAP interpretability

**How to Use:**
1. Load customer data with:
   - Customer features
   - Churn indicator (0/1)
2. Or use sample data
3. Select features
4. Choose model
5. Click "Train Churn Model"
6. Review:
   - Churn predictions
   - Risk scores
   - Feature importance
   - Retention strategies
7. Export at-risk customers

**Business Applications:**
- Customer retention
- Proactive interventions
- Loyalty program targeting
- Resource allocation
- Revenue protection

---

## AI Features

### AI-Powered Insights

**Available in:**
- Market Basket Analysis
- RFM Analysis
- ML Classification
- ML Regression
- A/B Testing
- Cohort Analysis

**What AI Provides:**
- Business interpretation of results
- Actionable recommendations
- Pattern identification
- Strategic suggestions
- Risk assessments

**How to Use:**
1. Complete analysis in any module
2. Click "🤖 Generate AI Insights"
3. Wait for analysis (10-30 seconds)
4. Review AI-generated insights
5. Insights are saved in session

**Tips:**
- AI insights are context-aware
- Based on your actual data
- Provides business language explanations
- Suggests next steps

---

## Export & Reporting

### Quick Export (Sidebar)

**Always Available:**
- **CSV:** Raw data export
- **Excel:** Formatted spreadsheet
- **JSON:** Structured data format
- **Data Dictionary:** Column descriptions
- **Analysis Summary:** Key statistics

### Module-Specific Exports

**Market Basket Analysis:**
- Association rules CSV
- Network graph HTML
- Comprehensive report

**RFM Analysis:**
- Customer segments CSV
- RFM scores
- Segment profiles

**ML Classification/Regression:**
- Model comparison CSV
- Best model metrics
- Feature importance
- Predictions
- SHAP values

**A/B Testing:**
- Test results CSV
- Statistical summary

**Time Series:**
- Forecast CSV
- Confidence intervals

### Professional Reports

**Generate From:**
- Reports module
- Individual analysis modules

**Formats:**
- Markdown (.md)
- Plain Text (.txt)
- HTML (some visualizations)

**Includes:**
- Executive summary
- Methodology
- Results
- Visualizations
- Recommendations

---

## Tips & Best Practices

### Data Preparation

1. **Clean Your Data First**
   - Handle missing values
   - Remove duplicates
   - Fix data types
   - Normalize formats

2. **Feature Engineering**
   - Create meaningful derived features
   - Encode categorical variables
   - Scale numerical features
   - Handle outliers appropriately

3. **Data Quality**
   - Check for inconsistencies
   - Validate ranges
   - Verify relationships
   - Document assumptions

### Model Selection

1. **Start Simple**
   - Try basic models first
   - Use AI recommendations
   - Compare multiple approaches

2. **Evaluate Properly**
   - Use cross-validation
   - Check multiple metrics
   - Validate on holdout data
   - Consider business context

3. **Interpret Results**
   - Use SHAP for explainability
   - Understand feature importance
   - Validate predictions
   - Document findings

### Performance Optimization

1. **Large Datasets**
   - Sample for exploration
   - Use efficient algorithms
   - Consider data reduction
   - Monitor memory usage

2. **Model Training**
   - Start with fewer models
   - Use appropriate sample sizes
   - Leverage parallel processing
   - Cache results when possible

3. **Visualization**
   - Limit data points in plots
   - Use aggregation when needed
   - Choose appropriate chart types
   - Export for external tools if needed

### Workflow Best Practices

1. **Iterative Analysis**
   - Start with exploration
   - Form hypotheses
   - Test systematically
   - Refine based on results

2. **Documentation**
   - Export results regularly
   - Save insights
   - Document decisions
   - Create reports

3. **Collaboration**
   - Share exports
   - Generate reports
   - Document methodology
   - Communicate findings

---

## Troubleshooting

### Common Issues

**1. Data Upload Fails**
- **Problem:** File won't upload
- **Solutions:**
  - Check file format (CSV, Excel only)
  - Verify file size (<200MB recommended)
  - Ensure proper encoding (UTF-8)
  - Check for corrupted file

**2. Analysis Takes Too Long**
- **Problem:** Module is slow or unresponsive
- **Solutions:**
  - Reduce dataset size
  - Sample large datasets
  - Train fewer models
  - Close other browser tabs
  - Refresh page if stuck

**3. AI Insights Error**
- **Problem:** "Error generating insights"
- **Solutions:**
  - Check API key configuration
  - Verify internet connection
  - Try again (rate limits)
  - Simplify analysis first

**4. Visualization Not Showing**
- **Problem:** Charts don't display
- **Solutions:**
  - Refresh page
  - Check data format
  - Verify column selection
  - Try different chart type

**5. Export Fails**
- **Problem:** Can't download results
- **Solutions:**
  - Check browser download settings
  - Disable popup blockers
  - Try different format
  - Reduce export size

**6. Model Training Fails**
- **Problem:** "Error training models"
- **Solutions:**
  - Check for missing values
  - Verify target variable
  - Ensure sufficient data
  - Try fewer models
  - Check feature types

**7. Memory Issues**
- **Problem:** "Out of memory" or slow performance
- **Solutions:**
  - Reduce dataset size
  - Clear session and restart
  - Close other applications
  - Use sampling
  - Upgrade resources

### Error Messages

**"No data uploaded"**
- Upload data in Data Upload module first

**"Invalid column selection"**
- Verify column names match your data
- Check for typos
- Ensure column exists

**"Insufficient data"**
- Need minimum rows for analysis
- Check data quality
- Remove excessive missing values

**"API key not configured"**
- Contact administrator
- Check environment variables
- Verify API key validity

### Getting Help

**Resources:**
- README.md - Quick start guide
- MBA_GUIDE.md - Market Basket Analysis details
- DEPLOYMENT_GUIDE.md - Deployment instructions
- TESTING_CHECKLIST.md - Feature verification

**Support:**
- Check documentation first
- Review error messages carefully
- Try sample datasets
- Contact support with:
  - Error message
  - Steps to reproduce
  - Dataset characteristics
  - Browser/system info

---

## Appendix

### Keyboard Shortcuts

- **Ctrl + K:** Focus search
- **Ctrl + R:** Refresh page
- **Ctrl + S:** Save/Export (in some modules)

### Data Format Requirements

**Market Basket Analysis:**
```
Transaction,Item
1,Bread
1,Milk
2,Bread
2,Butter
```

**RFM Analysis:**
```
CustomerID,Date,Amount
C001,2024-01-15,100.00
C001,2024-02-20,150.00
```

**Time Series:**
```
Date,Value
2024-01-01,100
2024-01-02,105
```

**Classification/Regression:**
- Features as columns
- Target variable as column
- One row per observation

### Glossary

**Accuracy:** Percentage of correct predictions
**AUC-ROC:** Area Under Receiver Operating Characteristic curve
**Confidence:** Probability of consequent given antecedent
**Cross-Validation:** Model validation technique
**F1-Score:** Harmonic mean of precision and recall
**Lift:** Strength of association rule
**MAE:** Mean Absolute Error
**MSE:** Mean Squared Error
**Precision:** True positives / (True positives + False positives)
**R²:** Coefficient of determination
**Recall:** True positives / (True positives + False negatives)
**RMSE:** Root Mean Squared Error
**SHAP:** SHapley Additive exPlanations
**Support:** Frequency of itemset in transactions

---

**End of User Guide**

For additional support or questions, please refer to the README.md or contact support.

**Made with ❤️ for Data Mining Capstone Project**
