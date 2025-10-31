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
            print(f"Checking for OpenAI API key...")
            api_key = os.getenv('OPENAI_API_KEY')
            print(f"Environment variable OPENAI_API_KEY: {'Found' if api_key else 'Not found'}")
            
            # Also check for alternative environment variable names
            alt_keys = ['OPENAI_KEY', 'OPENAI_TOKEN', 'GPT_API_KEY']
            for alt_key in alt_keys:
                alt_value = os.getenv(alt_key)
                if alt_value:
                    print(f"Alternative key {alt_key}: Found")
                    if not api_key:
                        api_key = alt_value
                        print(f"Using {alt_key} as API key")
            
            # Also check Streamlit secrets if available
            if not api_key:
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                        api_key = st.secrets['OPENAI_API_KEY']
                        print(f"Found API key in Streamlit secrets")
                    elif hasattr(st, 'secrets') and 'openai_api_key' in st.secrets:
                        api_key = st.secrets['openai_api_key']
                        print(f"Found API key in Streamlit secrets (lowercase)")
                except:
                    print(f"Could not access Streamlit secrets")
            
            if not api_key:
                print(f"No OpenAI API key found in environment variables or Streamlit secrets - using fallback detection for {task_type}")
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
                
                # Add diverse sample values for better content analysis
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # Get diverse samples: first few, middle, and last few
                    sample_indices = []
                    if len(non_null_values) >= 5:
                        sample_indices = [0, 1, len(non_null_values)//2, -2, -1]
                    else:
                        sample_indices = list(range(len(non_null_values)))
                    
                    sample_values = [non_null_values.iloc[i] for i in sample_indices[:5]]
                    # Convert to JSON-serializable format and remove duplicates
                    info['sample_values'] = list(dict.fromkeys([str(v) for v in sample_values]))
                else:
                    info['sample_values'] = []
                
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
1. PERFORMANCE RISK: Assess based on dataset size and complexity
   - Low: <1K rows, <20 columns
   - Medium: 1K-10K rows, 20-50 columns  
   - High: >10K rows, >50 columns
2. ALGORITHM SELECTION:
   - Isolation Forest: Best for high-dimensional data, large datasets, general purpose
   - Local Outlier Factor: Good for local anomalies, medium datasets with clusters
   - One-Class SVM: Best for well-separated normal data, smaller datasets
3. CONTAMINATION: Expected proportion of anomalies
   - Small datasets (<500): 0.05 (5%)
   - Large datasets (>10K): 0.02 (2%) for precision
   - High performance risk: 0.1 (10%) for speed
4. EXCLUDE: ID columns, timestamps, categorical with >50 categories, constant columns
5. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations (1GB RAM, CPU timeout)
6. MEMORY OPTIMIZATION: Recommend excluding high-cardinality categorical columns
7. Be specific about performance warnings and optimization suggestions

Provide ONLY the JSON response, no additional text."""
            elif task_type == 'market_basket_analysis':
                prompt = f"""You are an expert retail analytics specialist analyzing a dataset for Market Basket Analysis.

Dataset Overview:
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Task Type: MARKET BASKET ANALYSIS

Column Details:
{json.dumps(column_info, indent=2)}

Please analyze this dataset and provide Market Basket Analysis recommendations in the following JSON format:
{{
    "performance_risk": "Low/Medium/High",
    "performance_warnings": ["List of performance concerns for Streamlit Cloud"],
    "optimization_suggestions": ["List of specific suggestions to improve MBA performance"],
    "data_suitability": "Excellent/Good/Fair/Poor",
    "suitability_reasoning": "Why this dataset is/isn't suitable for Market Basket Analysis",
    "recommended_transaction_column": "column_name",
    "recommended_item_column": "column_name",
    "column_reasoning": "Why these columns are best for transaction ID and item identification",
    "recommended_min_support": 0.02,
    "recommended_min_confidence": 0.4,
    "recommended_min_lift": 1.5,
    "support_reasoning": "Why this support threshold is appropriate for this dataset",
    "confidence_reasoning": "Why this confidence threshold is appropriate",
    "lift_reasoning": "Why this lift threshold is appropriate",
    "transaction_structure_issues": ["List of potential issues with transaction structure"],
    "preprocessing_recommendations": ["List of data preprocessing steps specific to MBA"]
}}

Guidelines for Market Basket Analysis:
1. PERFORMANCE RISK: Assess based on transaction volume and item diversity
   - Low: <5K transactions, <500 unique items
   - Medium: 5K-50K transactions, 500-2K unique items
   - High: >50K transactions, >2K unique items (memory intensive for Apriori)
2. DATA SUITABILITY: Evaluate transaction data structure using MBA industry standards
   - Excellent: Clear transaction grouping, descriptive item names, diverse baskets (2-20 items)
   - Good: Minor formatting issues, reasonable transaction sizes (avg 3-15 items)
   - Fair: Some structural problems, very small (<2 items) or very large (>50 items) baskets
   - Poor: Missing transaction structure, single-item transactions, no interpretable items
3. COLUMN SELECTION: Analyze actual data content, not just column names
   - Transaction Column: Must contain identifiers that group items into baskets/transactions
     * Analyze: Does the column have repeated values that logically group items?
     * Content: Should be IDs, order numbers, session IDs, customer visits, etc.
   - Item Column: Must contain meaningful item identifiers for association analysis
     * Analyze: Does the column contain descriptive, human-readable item information?
     * Content: Product names, descriptions, categories - NOT numeric codes or IDs
     * Test: Can you understand what the item is from the column value alone?
     * Avoid: Pure numeric codes, SKUs, IDs that don't describe the actual product
4. THRESHOLD RECOMMENDATIONS:
   - Support: Based on transaction count and item frequency distribution
     * Small datasets (<1K trans): 0.01-0.05 (1-5%)
     * Medium datasets (1K-10K): 0.005-0.02 (0.5-2%)
     * Large datasets (>10K): 0.001-0.01 (0.1-1%)
   - Confidence: Based on business requirements
     * Conservative: 0.6-0.8 (60-80%)
     * Balanced: 0.3-0.6 (30-60%)
     * Exploratory: 0.1-0.3 (10-30%)
   - Lift: Always start with 1.0+ for meaningful associations
     * Weak associations: 1.0-1.5
     * Moderate associations: 1.5-3.0
     * Strong associations: 3.0+
4. TRANSACTION STRUCTURE: Identify potential issues
   - Missing transaction IDs
   - Single-item transactions (no associations possible)
   - Extremely large baskets (computational complexity)
   - Item naming inconsistencies
5. PERFORMANCE CONSTRAINTS: Consider Streamlit Cloud limitations
   - Apriori algorithm is memory-intensive
   - Large item catalogs create exponential complexity
   - Recommend sampling for very large datasets
6. PREPROCESSING: MBA-specific data preparation
   - Item name standardization
   - Transaction grouping validation
   - Basket size analysis
   - Item frequency distribution

Focus ONLY on Market Basket Analysis requirements. Do not consider ML classification, regression, or other analysis types.

ANALYSIS METHODOLOGY:
1. Examine actual data samples from each column, not just column names
2. For Transaction Column: Look for values that repeat and logically group items (order IDs, session IDs, etc.)
3. For Item Column: Look for values that are human-readable and describe actual products
4. Apply MBA industry standards: Items must be interpretable for business insights

IMPORTANT: You MUST provide ALL required fields in the JSON response. Do not leave any field empty or null.
- data_suitability must be one of: Excellent, Good, Fair, Poor
- suitability_reasoning must provide specific reasoning based on actual data content analysis
- recommended_transaction_column and recommended_item_column must be actual column names from the dataset
- column_reasoning must explain why these columns were chosen based on content analysis, not names
- Analyze sample values from each column to determine suitability for MBA
- Prioritize columns with meaningful, descriptive content over codes or IDs

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

            # Call OpenAI API
            print(f"Making AI API call for {task_type}...")
            print(f"API key present: {bool(api_key)}")
            print(f"API key length: {len(api_key) if api_key else 0}")
            
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing ML configuration recommendations. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            print(f"AI API call successful for {task_type}")
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            print(f"Raw AI response for {task_type}: {ai_response[:200]}...")
            
            # Remove markdown code blocks if present
            if ai_response.startswith('```'):
                ai_response = ai_response.split('```')[1]
                if ai_response.startswith('json'):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            print(f"Cleaned AI response for {task_type}: {ai_response[:200]}...")
            recommendations = json.loads(ai_response)
            print(f"Successfully parsed AI recommendations for {task_type}")
            
            # Validate target column exists (skip for anomaly detection and data cleaning)
            if task_type not in ['anomaly_detection', 'data_cleaning'] and recommendations.get('target_column') not in df.columns:
                return AISmartDetection._fallback_detection(df, task_type)
            
            return recommendations
            
        except Exception as e:
            # Fallback to rule-based detection on any error
            print(f"AI detection error for {task_type}: {str(e)}")
            print(f"Falling back to rule-based detection...")
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
            
            return {
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
            # Rule-based MBA recommendations
            n_samples = len(df)
            n_features = len(df.columns)
            
            # Performance risk assessment
            if n_samples > 50000 or n_features > 2000:
                performance_risk = 'High'
                performance_warnings = [
                    f'Large dataset ({n_samples:,} transactions, {n_features} potential items) may be memory intensive',
                    'Consider using higher support thresholds to reduce memory usage'
                ]
            elif n_samples > 5000 or n_features > 500:
                performance_risk = 'Medium'
                performance_warnings = ['Medium-sized dataset - monitor memory usage during analysis']
            else:
                performance_risk = 'Low'
                performance_warnings = []
            
            # Smart column detection for MBA
            transaction_col = None
            item_col = None
            
            # Look for transaction ID patterns
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['invoice', 'order', 'transaction', 'basket', 'receipt']):
                    # Check if it has repeated values (good for grouping)
                    if df[col].nunique() < len(df) * 0.8:  # Less than 80% unique (has repeats)
                        transaction_col = col
                        break
            
            # Look for item description patterns (avoid codes)
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['description', 'product', 'item', 'name']):
                    # Check if it contains descriptive text (not just codes)
                    sample_values = df[col].dropna().head(5).astype(str)
                    if any(len(str(val)) > 5 and not str(val).isdigit() for val in sample_values):
                        item_col = col
                        break
            
            # Fallbacks if no good matches found
            if not transaction_col:
                transaction_col = df.columns[0]  # First column as fallback
            if not item_col:
                # Avoid numeric columns and codes
                for col in df.columns:
                    if col != transaction_col and not pd.api.types.is_numeric_dtype(df[col]):
                        item_col = col
                        break
                if not item_col:
                    item_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            return {
                'performance_risk': performance_risk,
                'performance_warnings': performance_warnings,
                'optimization_suggestions': [
                    'Use higher support thresholds for large datasets',
                    'Consider filtering out very rare or very common items'
                ],
                'data_suitability': 'Good',
                'suitability_reasoning': 'Rule-based analysis suggests this data can be used for Market Basket Analysis',
                'recommended_transaction_column': transaction_col,
                'recommended_item_column': item_col,
                'column_reasoning': f'Selected {transaction_col} for transactions and {item_col} for items based on column names and content patterns',
                'recommended_min_support': 0.01,
                'recommended_min_confidence': 0.3,
                'recommended_min_lift': 1.2,
                'support_reasoning': 'Conservative support threshold for rule-based detection',
                'confidence_reasoning': 'Moderate confidence threshold for exploratory analysis',
                'lift_reasoning': 'Standard lift threshold for meaningful associations',
                'transaction_structure_issues': [],
                'preprocessing_recommendations': ['Verify transaction grouping is correct', 'Check for item name consistency']
            }
        elif task_type == 'classification':
            # Simple classification target detection
            target = None
            # Look for common target patterns
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['target', 'label', 'class', 'category', 'species']):
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 50:
                        target = col
                        break
            
            # Fallback: find categorical column with 2-50 unique values
            if not target:
                for col in df.columns:
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 50:
                        target = col
                        break
            
            # Ultimate fallback: last column
            if not target:
                target = df.columns[-1]
        else:
            # Simple regression detection
            numeric_cols = df.select_dtypes(include=['number']).columns
            # Look for columns with 'value', 'price', 'medv' patterns
            target = None
            for col in numeric_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['medv', 'value', 'price', 'target']):
                    target = col
                    break
            
            if not target:
                target = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[-1]
        
        # Return format for ML tasks
        return {
            'target_column': target,
            'reasoning': 'Using rule-based detection (AI unavailable). This column was selected based on data type and position.',
            'confidence': 'Medium',
            'features_to_use': [col for col in df.columns if col != target],
            'features_to_exclude': [],
            'recommended_cv_folds': 5,
            'recommended_test_size': 20,
            'warnings': ['AI-powered detection unavailable - using rule-based fallback'],
            'data_quality': 'Unknown'
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
