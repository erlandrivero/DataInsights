"""
AI-Powered Smart Detection for DataInsights
Uses GPT-4 to intelligently detect target columns and recommend settings
"""

import pandas as pd
import json
import streamlit as st
from typing import Dict, List, Optional, Tuple
import os


class AISmartDetection:
    """AI-powered intelligent column detection and configuration recommendations."""
    
    @staticmethod
    def analyze_dataset_for_ml(df: pd.DataFrame, task_type: str = 'classification') -> Dict:
        """
        Use GPT-4 to analyze dataset and provide intelligent recommendations.
        
        Args:
            df: Input DataFrame
            task_type: 'classification', 'regression', 'anomaly_detection', or 'data_cleaning'
            
        Returns:
            Dictionary with recommendations:
            For ML tasks:
            {
                'target_column': str,
                'reasoning': str,
                'confidence': str (High/Medium/Low),
                'features_to_use': List[str],
                'features_to_exclude': List[str],
                'recommended_cv_folds': int,
                'recommended_test_size': int,
                'warnings': List[str],
                'data_quality': str
            }
            
            For anomaly detection:
            {
                'performance_risk': str (Low/Medium/High),
                'performance_warnings': List[str],
                'optimization_suggestions': List[str],
                'features_to_exclude': List[dict],
                'recommended_algorithm': str,
                'recommended_contamination': float
            }
            
            For data cleaning:
            {
                'performance_risk': str (Low/Medium/High),
                'performance_warnings': List[str],
                'optimization_suggestions': List[str],
                'data_quality_issues': List[dict],
                'cleaning_priorities': List[str],
                'columns_to_clean': List[dict],
                'overall_data_quality': str,
                'cleaning_complexity': str
            }
        """
        try:
            # Check if OpenAI API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return AISmartDetection._fallback_detection(df, task_type)
            
            # Prepare dataset summary for GPT-4
            column_info = []
            for col in df.columns:
                info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'unique_values': int(df[col].nunique()),
                    'missing_count': int(df[col].isna().sum()),
                    'missing_pct': round((df[col].isna().sum() / len(df) * 100), 2)
                }
                
                # Add sample values (first 3 non-null)
                sample_values = df[col].dropna().head(3).tolist()
                # Convert to JSON-serializable format
                info['sample_values'] = [str(v) for v in sample_values]
                
                # Add statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    info['mean'] = round(float(df[col].mean()), 2) if not df[col].isna().all() else None
                    info['std'] = round(float(df[col].std()), 2) if not df[col].isna().all() else None
                    info['min'] = round(float(df[col].min()), 2) if not df[col].isna().all() else None
                    info['max'] = round(float(df[col].max()), 2) if not df[col].isna().all() else None
                
                column_info.append(info)
            
            # Create prompt for GPT-4
            if task_type == 'data_cleaning':
                prompt = f"""You are an expert data scientist analyzing a dataset for general data cleaning recommendations.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: DATA CLEANING (General Purpose)

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide general data cleaning recommendations in the following JSON format:
{{
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific cleaning suggestions to improve data quality"],
    "data_quality_issues": [
        {{"column": "column_name", "issue": "missing_values/duplicates/inconsistent_format/wrong_type", "severity": "High/Medium/Low", "recommendation": "Specific cleaning action"}}
    ],
    "cleaning_priorities": ["List of most important cleaning steps in order"],
    "columns_to_clean": [
        {{"column": "column_name", "reason": "Why this column needs cleaning", "suggested_action": "Specific cleaning method"}}
    ],
    "overall_data_quality": "Excellent/Good/Fair/Poor",
    "cleaning_complexity": "Simple/Moderate/Complex"
}}

Guidelines for Data Cleaning Analysis:
1. PERFORMANCE RISK: Assess based on dataset size and memory usage
   - Low: <10K rows, <50 columns, minimal missing data
   - Medium: 10K-100K rows, 50-200 columns, moderate issues
   - High: >100K rows, >200 columns, extensive cleaning needed
2. DATA QUALITY ISSUES: Identify specific problems
   - Missing values: High severity if >20%, Medium if 5-20%, Low if <5%
   - Duplicates: Check for exact and near-duplicate rows
   - Data types: Identify columns with wrong types (dates as strings, etc.)
   - Inconsistent formats: Mixed case, different date formats, etc.
3. CLEANING PRIORITIES: Order by impact on analysis quality
   - Critical: Issues that break analysis (wrong types, excessive missing data)
   - Important: Issues that reduce accuracy (duplicates, inconsistent formats)
   - Optional: Nice-to-have improvements (standardization, normalization)
4. PERFORMANCE OPTIMIZATION: Focus on Streamlit Cloud constraints
   - Memory usage reduction techniques
   - Processing speed improvements
   - Data size reduction methods
5. UNIVERSAL RECOMMENDATIONS: Cleaning that benefits ANY analysis type
   - Don't assume specific ML tasks or target columns
   - Focus on general data quality improvements
   - Consider multiple potential use cases

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'anomaly_detection':
                prompt = f"""You are an expert data scientist analyzing a dataset for anomaly detection.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: ANOMALY DETECTION

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide anomaly detection recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "features_to_exclude": [
        {{"column": "column_name", "reason": "Specific reason for exclusion"}}
    ],
    "recommended_algorithm": "Isolation Forest/Local Outlier Factor/One-Class SVM",
    "recommended_contamination": 0.05,
    "algorithm_reasoning": "Why this algorithm is best for this dataset",
    "contamination_reasoning": "Why this contamination level is appropriate"
}}

Guidelines for Anomaly Detection:
1. DATA SUITABILITY: Assess if dataset is appropriate for anomaly detection
   - Excellent: Multiple numeric features, sufficient data, clear patterns
   - Good: Adequate numeric features, reasonable data quality
   - Fair: Limited features but workable, some data quality issues
   - Poor: No numeric features, too small (<20 rows), or fundamentally unsuitable
2. PERFORMANCE RISK: Assess based on dataset size and complexity
   - Low: <1K rows, <20 columns
   - Medium: 1K-10K rows, 20-50 columns  
   - High: >10K rows, >50 columns
3. ALGORITHM SELECTION:
   - Isolation Forest: Best for high-dimensional data, large datasets, general purpose
   - Local Outlier Factor: Good for local anomalies, medium datasets with clusters
   - One-Class SVM: Best for well-separated normal data, smaller datasets
4. CONTAMINATION: Expected proportion of anomalies
   - Small datasets (<500): 0.05 (5%)
   - Large datasets (>10K): 0.02 (2%) for precision
   - High performance risk: 0.1 (10%) for speed
5. EXCLUDE: ID columns, timestamps, categorical with >50 categories, constant columns
6. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
7. MEMORY OPTIMIZATION: Recommend excluding high-cardinality categorical columns
8. Be specific about performance warnings and optimization suggestions

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'classification':
                prompt = f"""You are an expert data scientist analyzing a dataset for ML Classification.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: ML CLASSIFICATION

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide ML Classification recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for classification",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "target_column": "column_name",
    "target_reasoning": "Why this column is recommended as target",
    "confidence": "High/Medium/Low",
    "data_quality": "Excellent/Good/Fair/Poor",
    "features_to_use": ["list", "of", "feature", "columns"],
    "features_to_exclude": [
        {{"column": "column_name", "reason": "Specific reason for exclusion"}}
    ],
    "recommended_cv_folds": 5,
    "recommended_test_size": 20,
    "class_imbalance_detected": true/false,
    "imbalance_severity": "None/Mild/Moderate/Severe",
    "recommend_smote": true/false,
    "smote_reasoning": "Explanation for SMOTE recommendation",
    "warnings": ["List of warnings"]
}}

CRITICAL Guidelines for ML Classification:
1. DATA SUITABILITY: STRICT assessment if dataset is appropriate for classification
   - Excellent: Clear categorical target (2-20 classes), balanced classes, sufficient samples per class (50+)
   - Good: Categorical target with minor imbalance, adequate samples (20+ per class)
   - Fair: High class imbalance (>10:1) OR many classes (>50) OR small samples (<20 per class)
   - Poor: **NUMERIC/CONTINUOUS target**, only 1 class, >200 classes, <10 total samples per class

2. TARGET COLUMN VALIDATION - CRITICAL:
   **NUMERIC/CONTINUOUS targets are POOR for classification:**
   - If target has >50 unique values AND values look continuous (prices, quantities, measurements)
     → Mark as "Poor" and suggest "Use ML Regression instead"
   - Examples of POOR classification targets: quantity, price, amount, revenue, age, temperature
   - Examples of GOOD classification targets: country, category, status, type, species, label
   
3. TARGET COLUMN SELECTION:
   - PREFER: Categorical columns with meaningful classes (country, category, status, species)
   - Look for columns with: 2-50 unique values, clear categorical nature
   - **CHECK IF NUMERIC**: If column has all numeric values → likely continuous → POOR for classification
   - **AVOID**: ID columns, timestamps, continuous numeric columns, unique identifiers
   
4. CLASS DISTRIBUTION ANALYSIS:
   - Count samples per class
   - Severe imbalance (>100:1): Mark as Fair, recommend SMOTE or downsampling
   - Too many rare classes: Suggest filtering or grouping classes
   
5. PERFORMANCE RISK: Assess based on dataset size and complexity
   - Low: <10K rows, <20 features, 2-10 classes
   - Medium: 10K-100K rows, 20-100 features, 10-50 classes
   - High: >100K rows, >100 features, >50 classes
   
6. FEATURES TO EXCLUDE:
   - ID columns (unique per row)
   - Timestamps (unless creating time-based features)
   - High cardinality categoricals (>100 unique values)
   - Constant columns (all same value)
   - Target leakage columns (directly reveal answer)
   
7. CV FOLDS RECOMMENDATION:
   - Small datasets (<500): 3-fold
   - Medium datasets (500-5K): 5-fold
   - Large datasets (>5K): 3-fold (for speed)
   - Many classes (>20): 3-fold (for stratification)
   
8. SMOTE RECOMMENDATION:
   - Recommend for Moderate/Severe imbalance (>10:1 ratio)
   - Not recommended if: <100 total samples, classes naturally imbalanced
   
9. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
   - >100K rows: Recommend sampling to 50K
   - >100 features: Recommend feature selection or PCA
   - >50 classes: Recommend grouping rare classes

10. BE SPECIFIC: If data is Poor, clearly explain WHY and what would make it suitable

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'regression':
                prompt = f"""You are an expert data scientist analyzing a dataset for ML Regression.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: ML REGRESSION

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide ML Regression recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for regression",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "target_column": "column_name",
    "target_reasoning": "Why this column is recommended as target",
    "confidence": "High/Medium/Low",
    "data_quality": "Excellent/Good/Fair/Poor",
    "features_to_use": ["list", "of", "feature", "columns"],
    "features_to_exclude": [
        {{"column": "column_name", "reason": "Specific reason for exclusion"}}
    ],
    "recommended_cv_folds": 5,
    "recommended_test_size": 20,
    "target_distribution": "Normal/Skewed/Uniform",
    "outliers_detected": true/false,
    "recommend_log_transform": true/false,
    "transform_reasoning": "Explanation for transformation recommendation",
    "warnings": ["List of warnings"]
}}

CRITICAL Guidelines for ML Regression:
1. DATA SUITABILITY: STRICT assessment if dataset is appropriate for regression
   - Excellent: Clear continuous numeric target, good feature correlation, sufficient samples (>100)
   - Good: Numeric target with reasonable distribution, adequate samples (>50)
   - Fair: Highly skewed target OR limited samples (<50) OR weak feature correlation
   - Poor: **CATEGORICAL target**, only 1 unique value, <20 total samples, all features categorical

2. TARGET COLUMN VALIDATION - CRITICAL:
   **CATEGORICAL targets are POOR for regression:**
   - If target has <50 unique values AND values are categorical (countries, statuses, types)
     → Mark as "Poor" and suggest "Use ML Classification instead"
   - Examples of GOOD regression targets: price, quantity, amount, revenue, temperature, salary
   - Examples of POOR regression targets: country, category, status, species, type, label
   
3. TARGET COLUMN SELECTION:
   - REQUIRE: Continuous numeric column (int or float)
   - PREFER: Wide range of values, reasonable distribution
   - **CHECK IF CATEGORICAL**: If column has text values or <20 unique values → likely categorical → POOR for regression
   - **AVOID**: ID columns, timestamps, binary columns, categorical columns
   
4. TARGET DISTRIBUTION ANALYSIS:
   - Check for skewness (mean vs median)
   - Highly skewed (>2 std ratio): Recommend log transform
   - Outliers present: Suggest outlier handling
   
5. FEATURE REQUIREMENTS:
   - Need at least 2-3 numeric features for regression
   - Categorical features are OK if not high-cardinality
   - Check for multicollinearity (mention if detected)
   
6. PERFORMANCE RISK: Assess based on dataset size and complexity
   - Low: <10K rows, <20 features
   - Medium: 10K-100K rows, 20-100 features
   - High: >100K rows, >100 features
   
7. FEATURES TO EXCLUDE:
   - ID columns (unique per row)
   - Timestamps (unless creating time-based features)
   - High cardinality categoricals (>100 unique values)
   - Constant columns (all same value)
   - Perfect correlations with target (data leakage)
   
8. CV FOLDS RECOMMENDATION:
   - Small datasets (<500): 3-fold
   - Medium datasets (500-5K): 5-fold
   - Large datasets (>5K): 3-fold (for speed)
   
9. TRANSFORMATION RECOMMENDATION:
   - Recommend log transform if: target is heavily right-skewed (prices, revenues)
   - Not recommended if: target has zeros or negative values
   
10. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
    - >100K rows: Recommend sampling to 50K
    - >100 features: Recommend feature selection or PCA

11. BE SPECIFIC: If data is Poor, clearly explain WHY and what would make it suitable

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'market_basket_analysis':
                prompt = f"""You are an expert data scientist analyzing a dataset for Market Basket Analysis (MBA).

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: MARKET BASKET ANALYSIS

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide MBA recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for MBA",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "recommended_transaction_column": "column_name",
    "recommended_item_column": "column_name",
    "column_reasoning": "Why these columns are recommended for MBA",
    "recommended_min_support": 0.01,
    "recommended_min_confidence": 0.5,
    "thresholds_reasoning": "Why these thresholds are appropriate"
}}

Guidelines for Market Basket Analysis:
1. DATA SUITABILITY: Assess if dataset is appropriate for MBA
   - Excellent: Clear transaction/item structure, many transactions, good variety
   - Good: Adequate transaction data, reasonable patterns
   - Fair: Limited transactions but workable, some structural issues
   - Poor: No clear transaction structure, too few transactions (<50), or fundamentally unsuitable
2. TRANSACTION COLUMN: Should have repeated values (customer/order IDs, invoice numbers)
3. ITEM COLUMN: Prefer human-readable item names/descriptions over codes
   - PRIORITY: 'description', 'product_name', 'item_name' columns
   - AVOID: 'stockcode', 'sku', 'product_id' (codes are not meaningful for patterns)
4. PERFORMANCE RISK: CRITICAL - Assess based on unique item count (combinatorial explosion risk)
   - Low: <1K transactions, <100 unique items
   - Medium: 1K-10K transactions, 100-1K unique items  
   - High: >10K transactions, >1K unique items
   - CRITICAL: >2K unique items = Extreme memory risk (billions of combinations)
5. DATA REDUCTION: For large datasets, RECOMMEND aggressive filtering BEFORE analysis
   - >10K transactions: Recommend sampling to 5K-8K transactions max
   - >1K items: Recommend removing items appearing <5 times (low-frequency filter)
   - >2K items: MANDATORY - recommend sampling to 3K-5K transactions AND item filtering
   - Suggest time-based filtering (e.g., last 6 months) for transaction data
6. THRESHOLDS: MUST be aggressive for large item catalogs
   - <100 items: 0.01 support (1%)
   - 100-500 items: 0.02 support (2%)
   - 500-1K items: 0.03 support (3%)
   - 1K-2K items: 0.05 support (5%)
   - >2K items: 0.10 support (10%) - MANDATORY for Streamlit Cloud survival
7. EXCLUDE: Columns with unique values per row (likely not transactional)
8. MEMORY PROTECTION: System will limit to 2-item combinations for >2K items automatically
9. PERFORMANCE CONSTRAINTS: Streamlit Cloud has 1GB RAM limit - MBA with large catalogs WILL crash without aggressive thresholds AND data reduction
10. CRITICAL: If >2K items, explicitly state "Data MUST be reduced before analysis" in optimization_suggestions
11. Be specific about unique item count impact and recommend specific reduction strategies

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'rfm_analysis':
                prompt = f"""You are an expert data scientist analyzing a dataset for RFM (Recency, Frequency, Monetary) Analysis.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: RFM ANALYSIS

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide RFM recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for RFM",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "recommended_customer_column": "column_name",
    "recommended_date_column": "column_name",
    "recommended_amount_column": "column_name",
    "column_reasoning": "Why these columns are recommended for RFM",
    "date_format_detected": "YYYY-MM-DD or description",
    "requires_date_parsing": true/false
}}

Guidelines for RFM Analysis:
1. DATA SUITABILITY: Assess if dataset is appropriate for RFM
   - Excellent: Clear customer IDs, dates, amounts; good transaction history
   - Good: Has required columns, reasonable data quality
   - Fair: Limited transaction history or data quality issues
   - Poor: Missing required columns, too few transactions (<50), or fundamentally unsuitable
2. CUSTOMER COLUMN: Should identify unique customers (IDs, emails, names)
   - Look for columns with repeated values (same customer multiple transactions)
   - Avoid columns where every row is unique (likely transaction IDs, not customer IDs)
3. DATE COLUMN: Should contain transaction timestamps
   - Check for date/datetime columns or parseable date strings
   - Sample values should look like dates (YYYY-MM-DD, timestamps, etc.)
4. AMOUNT COLUMN: Should be numeric monetary values
   - Prefer columns with names like 'amount', 'total', 'price', 'revenue'
   - Should have positive numeric values representing money
5. PERFORMANCE RISK: Assess based on unique customers and date range
   - Low: <1K customers, <10K transactions
   - Medium: 1K-10K customers, 10K-100K transactions
   - High: >10K customers, >100K transactions
6. DATA REQUIREMENTS:
   - Minimum 50 transactions for meaningful RFM
   - Need at least 10 unique customers
   - Date range should span multiple time periods
7. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
8. Be specific about column selection reasoning and data quality

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'time_series_forecasting':
                prompt = f"""You are an expert data scientist analyzing a dataset for Time Series Forecasting.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: TIME SERIES FORECASTING

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide Time Series forecasting recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for time series",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "recommended_date_column": "column_name",
    "recommended_value_column": "column_name",
    "column_reasoning": "Why these columns are recommended for time series",
    "recommended_model": "ARIMA/Prophet",
    "model_reasoning": "Why this model is appropriate",
    "frequency_detected": "Daily/Weekly/Monthly/Yearly/Unknown",
    "seasonality_detected": true/false,
    "trend_detected": true/false,
    "forecast_horizon_recommendation": 30,
    "data_preprocessing_needed": ["List of preprocessing steps needed"]
}}

Guidelines for Time Series Forecasting:
1. DATA SUITABILITY: Assess if dataset is appropriate for time series forecasting
   - Excellent: Clear temporal ordering, consistent frequency, sufficient history (>100 points)
   - Good: Has date column and numeric values, reasonable temporal coverage
   - Fair: Limited history (<50 points) or irregular frequency
   - Poor: No date column, too few data points (<20), or fundamentally unsuitable
2. DATE COLUMN: Should contain timestamps or dates
   - Look for datetime columns or parseable date strings
   - Check if dates are sequential and cover a meaningful time range
   - Prefer columns with consistent frequency (daily, weekly, monthly)
3. VALUE COLUMN: Should be numeric column to forecast
   - Prefer columns with clear meaning (sales, revenue, temperature, etc.)
   - Should have continuous numeric values
   - Check for missing values and outliers
4. FREQUENCY DETECTION: Identify time intervals between observations
   - Daily: observations every day
   - Weekly: observations every 7 days
   - Monthly: observations monthly
   - Irregular: inconsistent intervals (may need resampling)
5. MODEL SELECTION:
   - ARIMA: Good for stationary data, univariate, <10K points
   - Prophet: Good for business data with seasonality, holidays, missing data
   - Consider data size, seasonality, and trend patterns
6. PERFORMANCE RISK: Assess based on data size and complexity
   - Low: <1K data points, simple patterns
   - Medium: 1K-10K points, seasonal patterns
   - High: >10K points, complex seasonality, high-dimensional
7. FORECAST HORIZON: Recommend reasonable prediction length
   - Short: 10-20% of historical data length
   - Medium: 20-30% of historical data length
   - Don't forecast beyond data support
8. PREPROCESSING NEEDS: Identify necessary data transformations
   - Handle missing values (interpolation, forward fill)
   - Outlier detection and treatment
   - Differencing for stationarity
   - Seasonal decomposition if needed
9. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
10. Be specific about temporal patterns and data quality

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'text_mining':
                prompt = f"""You are an expert NLP data scientist analyzing a dataset for Text Mining and Sentiment Analysis.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: TEXT MINING & SENTIMENT ANALYSIS

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide Text Mining recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for text mining",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "recommended_text_column": "column_name",
    "column_reasoning": "Why this column is recommended for text analysis",
    "text_quality_assessment": "Excellent/Good/Fair/Poor",
    "text_length_summary": "Brief summary of text length characteristics",
    "recommended_analyses": ["sentiment_analysis", "word_frequency", "topic_modeling"],
    "analysis_reasoning": "Why these analyses are appropriate for this text data",
    "preprocessing_needed": ["List of preprocessing steps recommended"],
    "estimated_processing_time": "Quick/Moderate/Long"
}}

Guidelines for Text Mining:
1. DATA SUITABILITY: Assess if dataset is appropriate for text mining
   - Excellent: Clear text column with substantive content, good variety, >100 texts
   - Good: Has text data with reasonable length and variety
   - Fair: Limited text data (<50 texts) or very short texts
   - Poor: No text column, too few texts (<10), or fundamentally unsuitable
2. TEXT COLUMN: Should contain unstructured text data
   - Look for columns with string/object dtype containing sentences or paragraphs
   - Prefer columns with substantial text (not just single words or codes)
   - Common names: 'text', 'review', 'comment', 'feedback', 'description', 'content'
   - Avoid: Numeric codes, IDs, single-word categories
3. TEXT QUALITY ASSESSMENT: Evaluate text characteristics
   - Check average text length (prefer 20+ words for meaningful analysis)
   - Assess variety (not just repetitive template text)
   - Look for natural language (not just keywords or tags)
4. RECOMMENDED ANALYSES:
   - Sentiment Analysis: Good for opinions, reviews, feedback (subjective text)
   - Word Frequency: Always useful for understanding vocabulary
   - Topic Modeling (LDA): Good for >50 texts with diverse content
   - Skip topic modeling if too few texts (<30) or very similar content
5. PERFORMANCE RISK: Assess based on text volume and length
   - Low: <500 texts, moderate length
   - Medium: 500-5000 texts
   - High: >5000 texts, very long documents (may timeout)
6. PREPROCESSING NEEDS: Identify necessary text cleaning
   - Remove special characters/HTML if present
   - Lowercasing for consistency
   - Remove stopwords for analysis
   - Handle missing values
   - Check for non-English text (VADER is English-optimized)
7. TEXT LENGTH: Analyze typical text length
   - Very short (<10 words): May not be suitable for sentiment/topics
   - Short (10-50 words): Good for sentiment, limited for topics
   - Medium (50-200 words): Ideal for all analyses
   - Long (>200 words): Good for topics, may need truncation
8. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
9. ESTIMATED PROCESSING TIME:
   - Quick: <500 texts, sentiment only
   - Moderate: 500-2000 texts, multiple analyses
   - Long: >2000 texts or topic modeling on large corpus
10. Be specific about text quality and analysis appropriateness

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'ab_testing':
                prompt = f"""You are an expert statistician analyzing a dataset for A/B Testing and Hypothesis Testing.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: A/B TESTING & STATISTICAL HYPOTHESIS TESTING

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide A/B Testing recommendations in the following JSON format:
{{
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Detailed explanation of why this rating was given for A/B testing",
    "alternative_suggestions": ["List of suggestions if data is Poor"],
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve test design"],
    "recommended_group_column": "column_name",
    "recommended_metric_column": "column_name",
    "column_reasoning": "Why these columns are recommended for A/B testing",
    "test_type_recommendation": "proportion_test/t_test/chi_square",
    "test_reasoning": "Why this test type is appropriate for the data",
    "recommended_control_group": "value_in_group_column",
    "recommended_treatment_group": "value_in_group_column",
    "sample_size_assessment": "Excellent/Good/Fair/Poor",
    "sample_size_reasoning": "Assessment of whether sample size is adequate",
    "expected_test_power": "High/Medium/Low",
    "minimum_detectable_effect": "Estimated MDE percentage or range",
    "statistical_significance_level": "Recommended alpha (typically 0.05)",
    "data_quality_checks": ["List of data quality considerations"],
    "recommended_segmentation_columns": ["column1", "column2"],
    "segmentation_reasoning": "Why these columns are good for segmentation analysis"
}}

Guidelines for A/B Testing:
1. DATA SUITABILITY: Assess if dataset is appropriate for A/B testing
   - Excellent: 2 clear groups, good sample sizes (>100 per group), clean metric
   - Good: 2 groups with adequate samples (>50 per group), workable metric
   - Fair: Groups exist but small samples (<50 per group) or unclear metric
   - Poor: <2 groups, too small (<20 total), or fundamentally unsuitable
2. GROUP COLUMN: Must have exactly 2 distinct groups (control vs treatment)
   - Look for binary columns with names like: 'group', 'variant', 'version', 'test_group'
   - Common values: Control/Treatment, A/B, 0/1, Variant_A/Variant_B
   - Must have: exactly 2 unique values (not 1, not 3+)
   - Sample size: Ideally >100 per group, minimum 30 per group
3. METRIC COLUMN: Depends on test type
   - Proportion Test: Binary outcome (0/1, True/False, converted/not_converted)
     - Examples: 'converted', 'clicked', 'purchased', 'signup'
   - T-Test: Continuous numeric metric (revenue, time_on_site, order_value)
     - Examples: 'revenue', 'spend', 'duration', 'value'
   - Chi-Square: Categorical variable with 2+ categories
4. TEST TYPE RECOMMENDATION: Choose based on metric type
   - Binary metric (0/1) → Proportion Test (Z-test)
   - Continuous metric (amounts, times) → T-Test
   - Categorical metric (3+ categories) → Chi-Square Test
5. SAMPLE SIZE ASSESSMENT: Evaluate statistical power
   - Excellent: >1000 per group (high power, detect small effects)
   - Good: 200-1000 per group (good power for medium effects)
   - Fair: 50-200 per group (can detect large effects only)
   - Poor: <50 per group (underpowered, unreliable results)
6. EXPECTED TEST POWER: Estimate ability to detect true effects
   - High: Large samples (>500 per group), can detect small effects (5-10%)
   - Medium: Moderate samples (100-500), detects medium effects (10-20%)
   - Low: Small samples (<100), only detects large effects (>20%)
7. MINIMUM DETECTABLE EFFECT (MDE): Smallest effect size detectable
   - Large samples: 5-10% relative improvement
   - Medium samples: 10-20% relative improvement
   - Small samples: >20% relative improvement
8. DATA QUALITY CHECKS: Identify issues
   - Sample Ratio Mismatch (SRM): Groups not split 50/50 as expected
   - Missing values in group or metric columns
   - Outliers in metric column (for T-test)
   - Zero/low variance in metric (makes test invalid)
9. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
10. SEGMENTATION COLUMNS: Identify categorical columns for post-hoc heterogeneous treatment effect analysis
   - Look for categorical columns with 2-20 unique values (not the group or metric columns)
   - Prefer demographic/contextual columns: age_group, region, device_type, user_type, segment
   - Exclude: IDs, dates, the group column, the metric column, high-cardinality columns
   - Good segmentation reveals which subgroups benefit most from treatment
   - Recommend up to 3 best segmentation columns in priority order
11. Be specific about why groups/metrics are appropriate and expected statistical power

Provide ONLY the JSON response, no additional text."""
            else:
                prompt = f"""You are an expert data scientist analyzing a dataset for machine learning {task_type}.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: {task_type.upper()}

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide recommendations in the following JSON format:
{{
    "target_column": "column_name",
    "reasoning": "Detailed explanation of why this column is the best target (2-3 sentences)",
    "confidence": "High/Medium/Low",
    "features_to_use": ["list", "of", "recommended", "feature", "columns"],
    "features_to_exclude": [
        {{"column": "column_name", "reason": "Specific reason for exclusion"}}
    ],
    "class_imbalance_detected": true/false,
    "imbalance_severity": "None/Mild/Moderate/Severe",
    "recommend_smote": true/false,
    "smote_reasoning": "Why SMOTE is/isn't recommended for this dataset",
    "recommended_cv_folds": 5,
    "recommended_test_size": 20,
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve performance"],
    "warnings": ["List of any data quality concerns or warnings"],
    "data_quality": "Excellent/Good/Fair/Poor"
}}

Guidelines:
1. For CLASSIFICATION: Target should be categorical with 2-50 classes
2. For REGRESSION: Target should be continuous with many unique values
3. Exclude ID columns, timestamps, or highly correlated features
4. ANALYZE CLASS BALANCE: Check if classes are imbalanced (>3:1 ratio = imbalanced)
5. RECOMMEND SMOTE: If severe imbalance detected, recommend SMOTE
6. Recommend appropriate CV folds based on dataset size
7. Flag any data quality issues
8. Be specific about why columns should be excluded
9. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
10. DATASET SIZE LIMITS: Flag datasets >50K rows or >100 columns as performance risks
11. MEMORY OPTIMIZATION: Recommend excluding high-cardinality categorical columns
12. CV FOLDS: Limit to 3-5 folds for large datasets to prevent timeouts

Provide ONLY the JSON response, no additional text."""

            # Call Google Gemini API
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Using Gemini 2.5 Flash for best price-performance, low-latency, and high volume
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Combine system and user prompts for Gemini
            full_prompt = f"""You are an expert data scientist providing ML configuration recommendations. Always respond with valid JSON only.

{prompt}"""
            
            # Configure safety settings to allow data analysis content
            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                ),
                safety_settings=safety_settings
            )
            
            # Parse response - check if valid text exists
            try:
                ai_response = response.text.strip()
            except ValueError:
                # response.text raises ValueError if no valid Part exists
                # Fallback to rule-based detection
                return AISmartDetection._fallback_detection(df, task_type)
            
            # Remove markdown code blocks if present
            if ai_response.startswith('```'):
                ai_response = ai_response.split('```')[1]
                if ai_response.startswith('json'):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            recommendations = json.loads(ai_response)
            
            # Validate target column exists (skip for modules without target columns)
            if task_type not in ['anomaly_detection', 'data_cleaning', 'text_mining', 'time_series', 'ab_testing'] and recommendations.get('target_column') not in df.columns:
                return AISmartDetection._fallback_detection(df, task_type)
            
            return recommendations
            
        except Exception as e:
            # Fallback to rule-based detection on any error
            print(f"AI detection error: {str(e)}")
            return AISmartDetection._fallback_detection(df, task_type)
    
    @staticmethod
    def _fallback_detection(df: pd.DataFrame, task_type: str) -> Dict:
        """Fallback to rule-based detection if AI fails or API key missing."""
        if task_type == 'data_cleaning':
            # Rule-based data cleaning recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 100000 or n_features > 200:
                performance_risk = 'High'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} rows, {n_features} columns) may require significant cleaning time',
                    'Consider processing in smaller batches for better performance'
                ]
            elif n_samples > 10000 or n_features > 50:
                performance_risk = 'Medium'
                performance_warnings = ['Medium-sized dataset - monitor cleaning performance']
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Basic data quality assessment
            missing_data = df.isnull().sum().sum()
            total_cells = len(df) * len(df.columns)
            missing_pct = (missing_data / total_cells) * 100 if total_cells > 0 else 0
            
            duplicates = df.duplicated().sum()
            
            # Generate basic recommendations
            data_quality_issues = []
            columns_to_clean = []
            cleaning_priorities = []
            
            if missing_pct > 20:
                data_quality_issues.append({
                    'column': 'Multiple columns',
                    'issue': 'missing_values',
                    'severity': 'High',
                    'recommendation': f'Address {missing_pct:.1f}% missing data across dataset'
                })
                cleaning_priorities.append('Handle missing values (high priority)')
            elif missing_pct > 5:
                data_quality_issues.append({
                    'column': 'Multiple columns', 
                    'issue': 'missing_values',
                    'severity': 'Medium',
                    'recommendation': f'Address {missing_pct:.1f}% missing data'
                })
                cleaning_priorities.append('Handle missing values (medium priority)')
            
            if duplicates > 0:
                data_quality_issues.append({
                    'column': 'Entire dataset',
                    'issue': 'duplicates',
                    'severity': 'Medium' if duplicates > len(df) * 0.05 else 'Low',
                    'recommendation': f'Remove {duplicates} duplicate rows'
                })
                cleaning_priorities.append('Remove duplicate rows')
            
            # Check for potential data type issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if it might be a date
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                        columns_to_clean.append({
                            'column': col,
                            'reason': 'Potential date column stored as text',
                            'suggested_action': 'Convert to datetime format'
                        })
                    # Check if it might be numeric
                    elif df[col].str.replace('.', '').str.replace('-', '').str.isdigit().any():
                        columns_to_clean.append({
                            'column': col,
                            'reason': 'Potential numeric data stored as text',
                            'suggested_action': 'Convert to numeric format'
                        })
            
            if not cleaning_priorities:
                cleaning_priorities = ['Basic data standardization', 'Column name normalization']
            
            return {
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Remove unnecessary columns to reduce memory usage',
                    'Handle missing values before analysis',
                    'Standardize column names and formats'
                ],
                'data_quality_issues': data_quality_issues,
                'cleaning_priorities': cleaning_priorities,
                'columns_to_clean': columns_to_clean,
                'overall_data_quality': 'Good' if missing_pct < 5 and duplicates == 0 else 'Fair' if missing_pct < 20 else 'Poor',
                'cleaning_complexity': 'Simple' if len(data_quality_issues) <= 1 else 'Moderate' if len(data_quality_issues) <= 3 else 'Complex'
            }
        elif task_type == 'anomaly_detection':
            # Rule-based anomaly detection recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 10000 or n_features > 50:
                performance_risk = 'High'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} rows, {n_features} columns) may cause performance issues',
                    'Consider sampling data or reducing features for better performance'
                ]
                contamination = 0.1  # Higher for speed
            elif n_samples > 1000 or n_features > 20:
                performance_risk = 'Medium'
                performance_warnings = ['Medium-sized dataset - monitor performance']
                contamination = 0.05
            else:
                performance_risk = 'Low'
                performance_warnings = []
                contamination = 0.05
            
            # Algorithm selection
            if n_features > 10 or n_samples > 5000:
                algorithm = 'Isolation Forest'
                algorithm_reason = 'Isolation Forest recommended for high-dimensional or large datasets'
            else:
                algorithm = 'Local Outlier Factor'
                algorithm_reason = 'Local Outlier Factor good for medium-sized datasets'
            
            # Data suitability assessment
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) == 0:
                data_suitability = 'Poor'
                suitability_reasoning = 'No numeric columns found - anomaly detection requires numeric features'
                alternative_suggestions = [
                    'Use Sample Data (built-in credit card transactions)',
                    'Upload a different dataset with numeric features',
                    'Convert categorical columns to numeric if appropriate'
                ]
            elif len(df) < 20:
                data_suitability = 'Poor'
                suitability_reasoning = f'Dataset too small ({len(df)} rows) - need at least 20 rows for reliable anomaly detection'
                alternative_suggestions = [
                    'Use Sample Data (built-in credit card transactions)',
                    'Collect more data points',
                    'Combine with additional datasets'
                ]
            elif len(numeric_cols) < 2:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited numeric features ({len(numeric_cols)} column) - anomaly detection works better with multiple features'
                alternative_suggestions = []
            elif len(df) < 100:
                data_suitability = 'Fair'
                suitability_reasoning = f'Small dataset ({len(df)} rows) - results may be less reliable'
                alternative_suggestions = []
            else:
                data_suitability = 'Good'
                suitability_reasoning = f'Dataset suitable for anomaly detection - {len(numeric_cols)} numeric features, {len(df)} rows'
                alternative_suggestions = []
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Consider using only numeric features for anomaly detection',
                    'Remove ID columns and timestamps before analysis'
                ],
                'features_to_exclude': [],
                'recommended_algorithm': algorithm,
                'recommended_contamination': contamination,
                'algorithm_reasoning': algorithm_reason,
                'contamination_reasoning': f'Standard {contamination*100:.0f}% contamination for rule-based detection'
            }
        elif task_type == 'market_basket_analysis':
            # Rule-based market basket analysis recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 50000:
                performance_risk = 'High'
                performance_warnings = [
                    f'Very large dataset ({n_samples:,} transactions) may cause performance issues',
                    'Consider sampling or filtering transactions for initial analysis'
                ]
            elif n_samples > 10000:
                performance_risk = 'Medium'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} transactions) - monitor performance',
                    'Apriori algorithm may take time with many unique items'
                ]
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Try to identify transaction and item columns
            transaction_col = None
            item_col = None
            
            # Look for transaction ID patterns
            for col in df.columns:
                col_lower = col.lower()
                unique_ratio = df[col].nunique() / len(df)
                if any(pattern in col_lower for pattern in ['transaction', 'invoice', 'order', 'receipt', 'id']):
                    if 0.01 < unique_ratio < 0.9:  # Has repeated values (not all unique)
                        transaction_col = col
                        break
            
            # Look for item/product patterns - prioritize descriptions over codes
            # First pass: Look for description/name columns (most meaningful)
            for col in df.columns:
                if col == transaction_col:
                    continue
                col_lower = col.lower()
                unique_ratio = df[col].nunique() / len(df)
                if any(pattern in col_lower for pattern in ['description', 'product_name', 'item_name', 'product_desc']):
                    if unique_ratio > 0.01:  # Has variety
                        item_col = col
                        break
            
            # Second pass: If no description found, look for other item indicators
            if not item_col:
                for col in df.columns:
                    if col == transaction_col:
                        continue
                    col_lower = col.lower()
                    unique_ratio = df[col].nunique() / len(df)
                    if any(pattern in col_lower for pattern in ['item', 'product']):
                        if unique_ratio > 0.01:  # Has variety
                            item_col = col
                            break
            
            # Fallback: use first two columns, but avoid code-like columns for items
            if not transaction_col:
                transaction_col = df.columns[0]
            if not item_col and len(df.columns) > 1:
                # Try to find any column that's NOT a code (avoid stock, sku, id patterns)
                for col in df.columns:
                    if col == transaction_col:
                        continue
                    col_lower = col.lower()
                    # Skip columns that look like codes
                    if not any(pattern in col_lower for pattern in ['stockcode', 'sku', 'product_id', 'item_id', '_id', 'code']):
                        item_col = col
                        break
                # Ultimate fallback if everything looks like codes
                if not item_col:
                    item_col = df.columns[1] if df.columns[1] != transaction_col else df.columns[0]
            
            # Data suitability assessment
            unique_transactions = df[transaction_col].nunique() if transaction_col else 0
            unique_items = df[item_col].nunique() if item_col else 0
            
            if unique_transactions < 50:
                data_suitability = 'Poor'
                suitability_reasoning = f'Too few transactions ({unique_transactions}) - need at least 50 for meaningful patterns'
                alternative_suggestions = [
                    'Use Sample Data (built-in groceries dataset)',
                    'Collect more transaction data',
                    'Verify correct transaction column selection'
                ]
            elif unique_items < 10:
                data_suitability = 'Poor'
                suitability_reasoning = f'Too few unique items ({unique_items}) - need variety for association rules'
                alternative_suggestions = [
                    'Use Sample Data (built-in groceries dataset)',
                    'Verify correct item column selection',
                    'Ensure dataset has multiple products/items'
                ]
            elif unique_transactions < 200:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited transactions ({unique_transactions}) - patterns may be less reliable'
                alternative_suggestions = []
            else:
                data_suitability = 'Good'
                suitability_reasoning = f'Dataset suitable for MBA - {unique_transactions} transactions, {unique_items} unique items'
                alternative_suggestions = []
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Use appropriate support and confidence thresholds',
                    'Consider filtering low-frequency items for performance'
                ],
                'recommended_transaction_column': transaction_col,
                'recommended_item_column': item_col,
                'column_reasoning': f'Rule-based detection: {transaction_col} for transactions (repeated IDs), {item_col} for items (human-readable names preferred over codes)',
                'recommended_min_support': 0.03 if unique_items > 2000 else 0.02 if unique_items > 1000 else 0.015 if unique_items > 500 else 0.01,
                'recommended_min_confidence': 0.5,
                'thresholds_reasoning': f'Thresholds for {unique_items} unique items after aggressive item filtering (2% of transactions threshold) - system automatically reduces catalog size'
            }
        elif task_type == 'rfm_analysis':
            # Rule-based RFM analysis recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 100000:
                performance_risk = 'High'
                performance_warnings = [
                    f'Very large dataset ({n_samples:,} transactions) may cause performance issues',
                    'Consider filtering to recent transactions or sampling'
                ]
            elif n_samples > 10000:
                performance_risk = 'Medium'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} transactions) - RFM calculation may take time'
                ]
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Try to identify customer, date, and amount columns
            customer_col = None
            date_col = None
            amount_col = None
            
            # Look for customer ID patterns (columns with repeated values, not all unique)
            for col in df.columns:
                col_lower = col.lower()
                unique_ratio = df[col].nunique() / len(df)
                if any(pattern in col_lower for pattern in ['customer', 'client', 'user', 'member']):
                    if 0.01 < unique_ratio < 0.9:  # Has repeated values
                        customer_col = col
                        break
            
            # Look for date columns
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            if date_cols:
                date_col = date_cols[0]
            else:
                # Try to find date patterns in column names
                for col in df.columns:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in ['date', 'time', 'day', 'invoice']):
                        date_col = col
                        break
            
            # Look for amount/monetary columns (numeric)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['amount', 'total', 'price', 'revenue', 'value', 'sales']):
                    amount_col = col
                    break
            
            # Fallback: use first numeric column for amount
            if not amount_col and numeric_cols:
                amount_col = numeric_cols[0]
            
            # Fallback: use first column for customer if not found
            if not customer_col:
                customer_col = df.columns[0]
            
            # Fallback: use second column for date if not found
            if not date_col and len(df.columns) > 1:
                date_col = df.columns[1]
            
            # Data suitability assessment
            unique_customers = df[customer_col].nunique() if customer_col else 0
            has_amount = amount_col is not None
            has_date = date_col is not None
            
            if unique_customers < 10:
                data_suitability = 'Poor'
                suitability_reasoning = f'Too few unique customers ({unique_customers}) - need at least 10 for meaningful RFM'
                alternative_suggestions = [
                    'Use Sample Data (built-in RFM dataset)',
                    'Upload a dataset with more customer transactions',
                    'Verify correct customer ID column selection'
                ]
            elif not has_amount:
                data_suitability = 'Poor'
                suitability_reasoning = 'No numeric amount column found - RFM requires monetary values'
                alternative_suggestions = [
                    'Use Sample Data (built-in RFM dataset)',
                    'Ensure dataset has a numeric amount/total column',
                    'Add monetary values to your transaction data'
                ]
            elif not has_date:
                data_suitability = 'Poor'
                suitability_reasoning = 'No date column found - RFM requires transaction dates for recency'
                alternative_suggestions = [
                    'Use Sample Data (built-in RFM dataset)',
                    'Ensure dataset has a date/datetime column',
                    'Add transaction dates to your data'
                ]
            elif n_samples < 50:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited transactions ({n_samples}) - RFM works better with more data'
                alternative_suggestions = []
            elif unique_customers < 50:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited customers ({unique_customers}) - segments may be less meaningful'
                alternative_suggestions = []
            else:
                data_suitability = 'Good'
                suitability_reasoning = f'Dataset suitable for RFM - {unique_customers} customers, {n_samples} transactions'
                alternative_suggestions = []
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Consider analyzing recent time periods only',
                    'Filter to active customers for focused insights'
                ],
                'recommended_customer_column': customer_col,
                'recommended_date_column': date_col,
                'recommended_amount_column': amount_col,
                'column_reasoning': f'Rule-based detection: {customer_col} for customers (repeated IDs), {date_col} for dates, {amount_col} for monetary amounts',
                'date_format_detected': 'Will be parsed automatically',
                'requires_date_parsing': True
            }
        elif task_type == 'time_series_forecasting':
            # Rule-based Time Series forecasting recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 10000:
                performance_risk = 'High'
                performance_warnings = [
                    f'Very large dataset ({n_samples:,} data points) may cause performance issues',
                    'Consider aggregating to lower frequency (e.g., weekly instead of daily)',
                    'Large forecasts may timeout on Streamlit Cloud'
                ]
            elif n_samples > 1000:
                performance_risk = 'Medium'
                performance_warnings = [
                    f'Medium dataset ({n_samples:,} data points) - forecasting may take time'
                ]
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Try to identify date and value columns
            date_col = None
            value_col = None
            
            # Look for date/datetime columns
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            if date_cols:
                date_col = date_cols[0]
            else:
                # Try to find date patterns in column names
                for col in df.columns:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in ['date', 'time', 'day', 'month', 'year']):
                        date_col = col
                        break
            
            # Look for value columns (numeric)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['sales', 'revenue', 'value', 'price', 'count', 'volume']):
                    value_col = col
                    break
            
            # Fallback: use first numeric column for value
            if not value_col and numeric_cols:
                value_col = numeric_cols[0]
            
            # Fallback: use first column for date if not found
            if not date_col:
                date_col = df.columns[0]
            
            # Fallback: use second column for value if not found
            if not value_col and len(df.columns) > 1:
                value_col = df.columns[1]
            
            # Data suitability assessment
            has_date = date_col is not None
            has_value = value_col is not None
            
            if n_samples < 20:
                data_suitability = 'Poor'
                suitability_reasoning = f'Too few data points ({n_samples}) - need at least 20 for time series forecasting, 100+ recommended'
                alternative_suggestions = [
                    'Use Sample Data (built-in time series dataset)',
                    'Upload a dataset with more historical data points',
                    'Collect more data over time before forecasting'
                ]
            elif not has_date or not has_value:
                data_suitability = 'Poor'
                suitability_reasoning = 'Missing date or value column - time series requires temporal data with numeric values'
                alternative_suggestions = [
                    'Use Sample Data (built-in time series dataset)',
                    'Ensure dataset has a date/time column and numeric value column',
                    'Check column names and data types'
                ]
            elif n_samples < 50:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited history ({n_samples} points) - forecasting possible but results may be less reliable'
                alternative_suggestions = [
                    'Collect more historical data if possible',
                    'Use shorter forecast horizons',
                    'Consider simpler models (moving average)'
                ]
            elif n_samples < 100:
                data_suitability = 'Good'
                suitability_reasoning = f'Reasonable history ({n_samples} points) - suitable for time series forecasting'
                alternative_suggestions = []
            else:
                data_suitability = 'Excellent'
                suitability_reasoning = f'Strong historical data ({n_samples} points) - excellent for time series forecasting'
                alternative_suggestions = []
            
            # Model recommendation based on data size
            if n_samples > 1000:
                recommended_model = 'Prophet'
                model_reasoning = 'Prophet recommended for larger datasets - handles seasonality and missing data well'
            else:
                recommended_model = 'ARIMA'
                model_reasoning = 'ARIMA recommended for smaller datasets - good for univariate time series'
            
            # Forecast horizon recommendation (20% of data length, max 365)
            forecast_horizon = min(max(int(n_samples * 0.2), 10), 365)
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Consider aggregating to lower frequency for large datasets',
                    'Use cross-validation for model evaluation',
                    'Monitor forecast accuracy over time'
                ],
                'recommended_date_column': date_col,
                'recommended_value_column': value_col,
                'column_reasoning': f'Rule-based detection: {date_col} for temporal ordering, {value_col} for values to forecast',
                'recommended_model': recommended_model,
                'model_reasoning': model_reasoning,
                'frequency_detected': 'Unknown - will be auto-detected',
                'seasonality_detected': False,
                'trend_detected': False,
                'forecast_horizon_recommendation': forecast_horizon,
                'data_preprocessing_needed': ['Check for missing values', 'Verify date ordering', 'Handle outliers if present']
            }
        elif task_type == 'text_mining':
            # Rule-based Text Mining recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Try to identify text column
            text_col = None
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Look for text column by name
            for col in text_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['text', 'review', 'comment', 'feedback', 'description', 'content', 'message']):
                    text_col = col
                    break
            
            # Fallback: use first string column
            if not text_col and text_cols:
                text_col = text_cols[0]
            
            # Fallback: use first column
            if not text_col:
                text_col = df.columns[0]
            
            # Assess text quality
            text_quality = 'Unknown'
            text_length_summary = 'Could not assess text length'
            avg_length = 0
            
            if text_col and text_col in df.columns:
                try:
                    # Calculate average text length
                    text_lengths = df[text_col].astype(str).str.split().str.len()
                    avg_length = text_lengths.mean()
                    
                    # Check if avg_length is valid
                    if pd.notna(avg_length) and avg_length > 0:
                        if avg_length < 5:
                            text_quality = 'Poor'
                            text_length_summary = f'Very short texts (avg {avg_length:.1f} words) - may not be suitable for analysis'
                        elif avg_length < 20:
                            text_quality = 'Fair'
                            text_length_summary = f'Short texts (avg {avg_length:.1f} words) - suitable for sentiment, limited for topics'
                        elif avg_length < 100:
                            text_quality = 'Good'
                            text_length_summary = f'Medium texts (avg {avg_length:.1f} words) - good for all analyses'
                        else:
                            text_quality = 'Excellent'
                            text_length_summary = f'Long texts (avg {avg_length:.1f} words) - ideal for comprehensive analysis'
                    else:
                        text_quality = 'Unknown'
                        text_length_summary = f'Column "{text_col}" does not appear to contain text data'
                except Exception as e:
                    text_quality = 'Unknown'
                    text_length_summary = f'Error assessing text: {str(e)}'
            
            # Performance risk assessment
            if n_samples > 5000:
                performance_risk = 'High'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} texts) may cause performance issues',
                    'Topic modeling on large corpus may timeout',
                    'Consider analyzing a sample for initial exploration'
                ]
                estimated_time = 'Long'
            elif n_samples > 500:
                performance_risk = 'Medium'
                performance_warnings = [
                    f'Medium dataset ({n_samples:,} texts) - processing may take time'
                ]
                estimated_time = 'Moderate'
            else:
                performance_risk = 'Low'
                performance_warnings = []
                estimated_time = 'Quick'
            
            # Data suitability assessment
            if n_samples < 10:
                data_suitability = 'Poor'
                suitability_reasoning = f'Too few texts ({n_samples}) - need at least 10 for text mining, 50+ recommended'
                alternative_suggestions = [
                    'Use Sample Data (built-in product reviews)',
                    'Upload a dataset with more text samples',
                    'Collect more text data before analysis'
                ]
            elif text_quality == 'Poor':
                data_suitability = 'Poor'
                suitability_reasoning = f'Text quality insufficient - very short texts (avg {avg_length:.1f} words) not suitable for meaningful analysis'
                alternative_suggestions = [
                    'Use Sample Data (built-in product reviews)',
                    'Ensure text column contains full sentences or paragraphs',
                    'Check if correct column was detected'
                ]
            elif n_samples < 30:
                data_suitability = 'Fair'
                suitability_reasoning = f'Limited text data ({n_samples} texts) - analysis possible but results may be less robust'
                alternative_suggestions = [
                    'Sentiment analysis still useful',
                    'Skip topic modeling (need more texts)',
                    'Collect more text samples if possible'
                ]
            elif n_samples < 100:
                data_suitability = 'Good'
                suitability_reasoning = f'Reasonable text corpus ({n_samples} texts) - suitable for text mining'
                alternative_suggestions = []
            else:
                data_suitability = 'Excellent'
                suitability_reasoning = f'Strong text corpus ({n_samples} texts) - excellent for comprehensive text mining'
                alternative_suggestions = []
            
            # Recommended analyses based on data
            recommended_analyses = ['word_frequency']  # Always useful
            if n_samples >= 10:
                recommended_analyses.insert(0, 'sentiment_analysis')  # Good for opinions
            if n_samples >= 30 and text_quality in ['Good', 'Excellent']:
                recommended_analyses.append('topic_modeling')  # Need sufficient texts
            
            analysis_reasoning = f'Recommending {len(recommended_analyses)} analyses based on {n_samples} texts with {text_quality.lower()} quality'
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Start with sentiment analysis before topic modeling',
                    'Review word frequency to understand vocabulary',
                    'Consider sampling large datasets for initial exploration'
                ],
                'recommended_text_column': text_col,
                'column_reasoning': f'Rule-based detection: {text_col} identified as text column based on data type and content',
                'text_quality_assessment': text_quality,
                'text_length_summary': text_length_summary,
                'recommended_analyses': recommended_analyses,
                'analysis_reasoning': analysis_reasoning,
                'preprocessing_needed': ['Remove special characters if present', 'Handle missing values', 'Lowercasing recommended'],
                'estimated_processing_time': estimated_time
            }
        elif task_type == 'ab_testing':
            # Rule-based A/B Testing recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Try to identify group column (must have exactly 2 unique values)
            group_col = None
            binary_cols = []
            for col in df.columns:
                n_unique = df[col].nunique()
                if n_unique == 2:
                    binary_cols.append(col)
                    col_lower = col.lower()
                    # Prioritize columns with test-related names
                    if any(pattern in col_lower for pattern in ['group', 'variant', 'version', 'test', 'arm', 'condition']):
                        group_col = col
                        break
            
            # Fallback to first binary column
            if not group_col and binary_cols:
                group_col = binary_cols[0]
            
            # Try to identify metric column
            metric_col = None
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                # Look for common metric names
                for col in numeric_cols:
                    col_lower = col.lower()
                    if any(pattern in col_lower for pattern in ['converted', 'conversion', 'clicked', 'click', 'revenue', 'value', 'spend', 'purchase']):
                        metric_col = col
                        break
                
                # Fallback to first numeric column
                if not metric_col:
                    metric_col = numeric_cols[0]
            
            # Determine test type based on metric
            test_type = 'proportion_test'
            test_reasoning = 'Rule-based: Default to proportion test'
            if metric_col and metric_col in df.columns:
                unique_values = df[metric_col].nunique()
                if unique_values == 2:
                    test_type = 'proportion_test'
                    test_reasoning = 'Binary metric detected - suitable for proportion test (Z-test)'
                elif unique_values > 10:
                    test_type = 't_test'
                    test_reasoning = 'Continuous metric detected - suitable for t-test'
            
            # Assess sample sizes
            sample_size_assessment = 'Unknown'
            sample_size_reasoning = 'Could not assess - group column not identified'
            expected_power = 'Unknown'
            mde = 'Unknown'
            
            if group_col and group_col in df.columns:
                groups = df[group_col].unique()
                if len(groups) == 2:
                    control_size = (df[group_col] == groups[0]).sum()
                    treatment_size = (df[group_col] == groups[1]).sum()
                    min_size = min(control_size, treatment_size)
                    
                    if min_size >= 1000:
                        sample_size_assessment = 'Excellent'
                        sample_size_reasoning = f'Large samples ({control_size:,} vs {treatment_size:,}) - high statistical power'
                        expected_power = 'High'
                        mde = '5-10% relative improvement'
                    elif min_size >= 200:
                        sample_size_assessment = 'Good'
                        sample_size_reasoning = f'Good samples ({control_size:,} vs {treatment_size:,}) - adequate power for medium effects'
                        expected_power = 'Medium'
                        mde = '10-20% relative improvement'
                    elif min_size >= 50:
                        sample_size_assessment = 'Fair'
                        sample_size_reasoning = f'Moderate samples ({control_size:,} vs {treatment_size:,}) - can detect large effects'
                        expected_power = 'Medium'
                        mde = '20-30% relative improvement'
                    else:
                        sample_size_assessment = 'Poor'
                        sample_size_reasoning = f'Small samples ({control_size:,} vs {treatment_size:,}) - underpowered, unreliable'
                        expected_power = 'Low'
                        mde = '>30% relative improvement'
            
            # Performance risk (A/B testing is usually fast)
            if n_samples > 100000:
                performance_risk = 'Medium'
                performance_warnings = [f'Large dataset ({n_samples:,} rows) may take time to process']
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Data suitability
            if not group_col:
                data_suitability = 'Poor'
                suitability_reasoning = 'No column with exactly 2 groups found - A/B testing requires 2 groups (control/treatment)'
                alternative_suggestions = [
                    'Use Sample A/B Test Data (built-in dataset)',
                    'Ensure dataset has a column with exactly 2 distinct values representing groups',
                    'Check if groups are in separate columns instead'
                ]
            elif not metric_col:
                data_suitability = 'Poor'
                suitability_reasoning = 'No numeric metric column found - need outcome to measure (conversions, revenue, etc.)'
                alternative_suggestions = [
                    'Use Sample A/B Test Data (built-in dataset)',
                    'Add a numeric metric column (0/1 for conversions, or continuous for revenue)',
                    'Check if metric is encoded as text instead of numbers'
                ]
            elif sample_size_assessment == 'Poor':
                data_suitability = 'Fair'
                suitability_reasoning = f'Groups and metric found but sample size is small - results may be unreliable'
                alternative_suggestions = [
                    'Collect more data before running test',
                    'Only expect to detect very large effects (>30%)',
                    'Consider using Sample Data for practice'
                ]
            elif sample_size_assessment == 'Fair':
                data_suitability = 'Good'
                suitability_reasoning = 'Suitable for A/B testing - can detect large to medium effects'
                alternative_suggestions = []
            else:
                data_suitability = 'Excellent'
                suitability_reasoning = 'Well-suited for A/B testing - good sample sizes and clear structure'
                alternative_suggestions = []
            
            # Get group values for recommendations
            control_group = 'Unknown'
            treatment_group = 'Unknown'
            if group_col and group_col in df.columns:
                groups = df[group_col].unique()
                if len(groups) == 2:
                    control_group = str(groups[0])
                    treatment_group = str(groups[1])
            
            # Detect segmentation columns (categorical with 2-20 values, excluding group/metric)
            segmentation_cols = []
            excluded_for_seg = {group_col, metric_col}
            for col in df.columns:
                if col not in excluded_for_seg:
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 20:
                        # Prioritize columns with segmentation-related names
                        col_lower = col.lower()
                        if any(pattern in col_lower for pattern in ['age', 'region', 'segment', 'type', 'category', 'group', 'device', 'platform']):
                            segmentation_cols.insert(0, col)  # Add to front
                        else:
                            segmentation_cols.append(col)
            
            # Limit to top 3 segmentation columns
            segmentation_cols = segmentation_cols[:3]
            
            segmentation_reasoning = f'Rule-based: Found {len(segmentation_cols)} categorical columns with 2-20 values suitable for segmentation'
            if not segmentation_cols:
                segmentation_reasoning = 'No suitable segmentation columns found (need categorical columns with 2-20 unique values)'
            
            return {
                'data_suitability': data_suitability,
                'suitability_reasoning': suitability_reasoning,
                'alternative_suggestions': alternative_suggestions,
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Ensure groups are randomized properly',
                    'Check for Sample Ratio Mismatch (SRM)',
                    'Consider running power analysis before test'
                ],
                'recommended_group_column': group_col or 'N/A',
                'recommended_metric_column': metric_col or 'N/A',
                'column_reasoning': f'Rule-based detection: {group_col} has 2 groups, {metric_col} is numeric metric',
                'test_type_recommendation': test_type,
                'test_reasoning': test_reasoning,
                'recommended_control_group': control_group,
                'recommended_treatment_group': treatment_group,
                'sample_size_assessment': sample_size_assessment,
                'sample_size_reasoning': sample_size_reasoning,
                'expected_test_power': expected_power,
                'minimum_detectable_effect': mde,
                'statistical_significance_level': '0.05 (standard)',
                'data_quality_checks': ['Check for missing values', 'Verify group randomization', 'Look for outliers in metric'],
                'recommended_segmentation_columns': segmentation_cols,
                'segmentation_reasoning': segmentation_reasoning
            }
        elif task_type == 'classification':
            # Rule-based classification target detection with suitability check
            target = None
            data_suitability = 'Unknown'
            suitability_reasoning = 'Rule-based fallback - limited analysis'
            alternative_suggestions = []
            
            # Look for common target patterns
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['target', 'label', 'class', 'category', 'species', 'country', 'status']):
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 50:
                        target = col
                        break
            
            # Fallback: find categorical column with 2-50 unique values
            if not target:
                for col in df.columns:
                    n_unique = df[col].nunique()
                    # Avoid numeric columns with many unique values (likely continuous)
                    if 2 <= n_unique <= 50:
                        # Check if it's not purely numeric/continuous
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # If numeric with >50 unique values, likely continuous
                            if n_unique > 50:
                                continue
                        target = col
                        break
            
            # Ultimate fallback: last column
            if not target:
                target = df.columns[-1]
            
            # Assess suitability for classification
            if target and target in df.columns:
                n_unique = df[target].nunique()
                n_samples = len(df)
                
                # Check if target is numeric/continuous (BAD for classification)
                if pd.api.types.is_numeric_dtype(df[target]) and n_unique > 50:
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" appears to be continuous/numeric ({n_unique} unique values). Classification requires categorical targets.'
                    alternative_suggestions = [
                        'Use ML Regression instead for continuous numeric targets',
                        'Convert numeric target into categorical bins if classification is needed',
                        'Choose a different categorical column as target'
                    ]
                elif n_unique == 1:
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" has only 1 unique value. Cannot perform classification.'
                    alternative_suggestions = ['Choose a column with 2 or more classes']
                elif n_unique > 200:
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" has too many classes ({n_unique}). Classification with >200 classes is impractical.'
                    alternative_suggestions = ['Group rare classes together', 'Filter to top 20-50 classes', 'Use different target column']
                elif n_unique > 50:
                    data_suitability = 'Fair'
                    suitability_reasoning = f'Target column "{target}" has many classes ({n_unique}). This may be challenging.'
                elif n_samples < 50:
                    data_suitability = 'Fair'
                    suitability_reasoning = f'Very small dataset ({n_samples} samples). Classification may be unreliable.'
                else:
                    data_suitability = 'Good'
                    suitability_reasoning = f'Target column "{target}" appears suitable for classification ({n_unique} classes, {n_samples} samples).'
        
        else:  # regression
            # Rule-based regression target detection with suitability check
            numeric_cols = df.select_dtypes(include=['number']).columns
            target = None
            data_suitability = 'Unknown'
            suitability_reasoning = 'Rule-based fallback - limited analysis'
            alternative_suggestions = []
            
            # Look for columns with 'value', 'price', 'medv' patterns
            for col in numeric_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['medv', 'value', 'price', 'target', 'amount', 'quantity', 'revenue', 'salary']):
                    target = col
                    break
            
            if not target:
                target = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[-1]
            
            # Assess suitability for regression
            if target and target in df.columns:
                n_unique = df[target].nunique()
                n_samples = len(df)
                
                # Check if target is categorical (BAD for regression)
                if not pd.api.types.is_numeric_dtype(df[target]):
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" is not numeric. Regression requires continuous numeric targets.'
                    alternative_suggestions = [
                        'Use ML Classification instead for categorical targets',
                        'Choose a numeric column as target'
                    ]
                elif n_unique < 10 and pd.api.types.is_integer_dtype(df[target]):
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" has very few unique values ({n_unique}). This appears categorical, not continuous.'
                    alternative_suggestions = ['Use ML Classification for categorical targets', 'Choose a continuous numeric column']
                elif n_unique == 1:
                    data_suitability = 'Poor'
                    suitability_reasoning = f'Target column "{target}" has only 1 unique value. Cannot perform regression.'
                    alternative_suggestions = ['Choose a column with varying values']
                elif n_samples < 50:
                    data_suitability = 'Fair'
                    suitability_reasoning = f'Very small dataset ({n_samples} samples). Regression may be unreliable.'
                else:
                    data_suitability = 'Good'
                    suitability_reasoning = f'Target column "{target}" appears suitable for regression (continuous numeric with {n_unique} unique values).'
        
        # Return format for ML tasks
        return {
            'target_column': target,
            'target_reasoning': 'Rule-based detection (AI unavailable). Column selected based on data type and naming patterns.',
            'reasoning': 'Using rule-based detection (AI unavailable). This column was selected based on data type and position.',
            'confidence': 'Medium',
            'data_suitability': data_suitability,
            'suitability_reasoning': suitability_reasoning,
            'alternative_suggestions': alternative_suggestions,
            'features_to_use': [col for col in df.columns if col != target],
            'features_to_exclude': [],
            'recommended_cv_folds': 5,
            'recommended_test_size': 20,
            'warnings': ['AI-powered detection unavailable - using rule-based fallback'],
            'data_quality': 'Unknown',
            'performance_risk': 'Medium',
            'performance_warnings': [],
            'optimization_suggestions': []
        }
    
    @staticmethod
    def display_ai_recommendation(recommendations: Dict, expanded: bool = False):
        """Display AI recommendations in Streamlit UI."""
        
        # Confidence badge
        confidence_colors = {
            'High': '🟢',
            'Medium': '🟡',
            'Low': '🔴'
        }
        confidence_badge = confidence_colors.get(recommendations.get('confidence', 'Medium'), '🟡')
        
        with st.expander(f"🤖 AI Smart Detection {confidence_badge} {recommendations.get('confidence', 'Medium')} Confidence", expanded=expanded):
            st.write("**🎯 Recommended Target Column:**")
            st.success(f"**{recommendations['target_column']}**")
            
            st.write("**💡 AI Reasoning:**")
            st.info(recommendations['reasoning'])
            
            # Data quality indicator
            quality = recommendations.get('data_quality', 'Unknown')
            quality_emoji = {'Excellent': '🌟', 'Good': '✅', 'Fair': '⚠️', 'Poor': '❌'}.get(quality, '❓')
            st.write(f"**📊 Data Quality:** {quality_emoji} {quality}")
            
            # Warnings
            if recommendations.get('warnings'):
                st.write("**⚠️ Warnings:**")
                for warning in recommendations['warnings']:
                    st.warning(warning)
            
            # Configuration recommendations
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recommended CV Folds", recommendations.get('recommended_cv_folds', 5))
            with col2:
                st.metric("Recommended Test Size", f"{recommendations.get('recommended_test_size', 20)}%")
            
            # Class imbalance and SMOTE recommendations
            if recommendations.get('class_imbalance_detected'):
                st.write("**⚖️ Class Imbalance Analysis:**")
                severity = recommendations.get('imbalance_severity', 'Unknown')
                severity_emoji = {'None': '✅', 'Mild': '🟡', 'Moderate': '🟠', 'Severe': '🔴'}.get(severity, '❓')
                st.write(f"{severity_emoji} **Severity:** {severity}")
                
                if recommendations.get('recommend_smote'):
                    st.success(f"🤖 **AI Recommends SMOTE:** {recommendations.get('smote_reasoning', 'Class imbalance detected')}")
                else:
                    st.info(f"ℹ️ **SMOTE Not Needed:** {recommendations.get('smote_reasoning', 'Classes are relatively balanced')}")
            
            # Performance Risk Assessment
            if recommendations.get('performance_risk'):
                risk_level = recommendations.get('performance_risk', 'Unknown')
                risk_emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(risk_level, '❓')
                st.write(f"**⚡ Performance Risk:** {risk_emoji} {risk_level}")
                
                # Performance warnings
                if recommendations.get('performance_warnings'):
                    st.write("**⚠️ Performance Warnings:**")
                    for warning in recommendations['performance_warnings']:
                        st.warning(f"⚡ {warning}")
                
                # Optimization suggestions
                if recommendations.get('optimization_suggestions'):
                    with st.expander("🚀 Performance Optimization Suggestions"):
                        for suggestion in recommendations['optimization_suggestions']:
                            st.write(f"• {suggestion}")
            
            # Features to exclude with detailed reasons
            if recommendations.get('features_to_exclude'):
                with st.expander("🚫 Features to Consider Excluding"):
                    for feature_info in recommendations['features_to_exclude']:
                        if isinstance(feature_info, dict):
                            st.write(f"• **{feature_info['column']}**: {feature_info['reason']}")
                        else:
                            # Backward compatibility for old format
                            st.write(f"• {feature_info}")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ai_detection(df_hash: str, df: pd.DataFrame, task_type: str) -> Dict:
    """Cached version of AI detection to avoid repeated API calls."""
    return AISmartDetection.analyze_dataset_for_ml(df, task_type)


def get_ai_recommendation(df: pd.DataFrame, task_type: str = 'classification') -> Dict:
    """
    Get AI-powered smart detection recommendation with caching.
    
    Args:
        df: Input DataFrame
        task_type: 'classification', 'regression', 'anomaly_detection', or 'data_cleaning'
        
    Returns:
        Dictionary with AI recommendations
    """
    # Create hash of dataframe for caching
    df_hash = str(hash(str(df.columns.tolist()) + str(len(df))))
    
    return cached_ai_detection(df_hash, df, task_type)
