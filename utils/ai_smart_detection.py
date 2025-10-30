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
            task_type: 'classification', 'regression', or 'anomaly_detection'
            
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
            if task_type == 'anomaly_detection':
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
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if ai_response.startswith('```'):
                ai_response = ai_response.split('```')[1]
                if ai_response.startswith('json'):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()
            
            recommendations = json.loads(ai_response)
            
            # Validate target column exists (skip for anomaly detection)
            if task_type != 'anomaly_detection' and recommendations.get('target_column') not in df.columns:
                return AISmartDetection._fallback_detection(df, task_type)
            
            return recommendations
            
        except Exception as e:
            # Fallback to rule-based detection on any error
            print(f"AI detection error: {str(e)}")
            return AISmartDetection._fallback_detection(df, task_type)
    
    @staticmethod
    def _fallback_detection(df: pd.DataFrame, task_type: str) -> Dict:
        """Fallback to rule-based detection if AI fails or API key missing."""
        if task_type == 'anomaly_detection':
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
        task_type: 'classification', 'regression', or 'anomaly_detection'
        
    Returns:
        Dictionary with AI recommendations
    """
    # Create hash of dataframe for caching
    df_hash = str(hash(str(df.columns.tolist()) + str(len(df))))
    
    return cached_ai_detection(df_hash, df, task_type)
