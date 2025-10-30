"""
Class Balancing Utilities for Imbalanced Classification Datasets

This module provides SMOTE, undersampling, and combined techniques to handle
imbalanced class distributions in classification tasks.

Author: DataInsights Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import streamlit as st


class ClassBalancer:
    """
    Handles class balancing for imbalanced classification datasets.
    
    Provides SMOTE (Synthetic Minority Over-sampling Technique), random undersampling,
    and combined methods to improve model performance on minority classes.
    """
    
    def __init__(self):
        """Initialize the ClassBalancer."""
        pass
    
    @staticmethod
    def analyze_imbalance(df: pd.DataFrame, target_col: str) -> Dict:
        """
        Analyze class distribution and calculate imbalance metrics.
        
        Args:
            df: DataFrame containing the data
            target_col: Name of the target column
            
        Returns:
            Dictionary with imbalance analysis:
                - class_counts: Series with counts per class
                - n_classes: Number of classes
                - imbalance_ratio: Max/min class ratio
                - minority_class: Name of minority class
                - majority_class: Name of majority class
                - recommendation: Recommended balancing method
        """
        class_counts = df[target_col].value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        
        imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
        
        # Determine recommendation
        if imbalance_ratio > 50:
            recommendation = "SMOTE + Tomek Links"
            severity = "Severe"
        elif imbalance_ratio > 10:
            recommendation = "SMOTE"
            severity = "Moderate"
        elif imbalance_ratio > 3:
            recommendation = "Random Undersampling"
            severity = "Mild"
        else:
            recommendation = "No balancing needed"
            severity = "Balanced"
        
        return {
            'class_counts': class_counts,
            'n_classes': len(class_counts),
            'imbalance_ratio': imbalance_ratio,
            'minority_class': class_counts.idxmin(),
            'majority_class': class_counts.idxmax(),
            'min_samples': min_class_size,
            'max_samples': max_class_size,
            'recommendation': recommendation,
            'severity': severity
        }
    
    @staticmethod
    def get_smart_preset(imbalance_ratio: float) -> Dict:
        """
        Get smart preset parameters based on imbalance severity.
        
        Args:
            imbalance_ratio: Ratio of majority to minority class
            
        Returns:
            Dictionary with recommended parameters
        """
        if imbalance_ratio > 50:
            return {
                'method': 'SMOTE + Tomek Links',
                'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'description': 'Severe imbalance - create synthetic samples and remove noisy borderline samples'
            }
        elif imbalance_ratio > 10:
            return {
                'method': 'SMOTE',
                'k_neighbors': 5,
                'sampling_strategy': 0.5,
                'description': 'Moderate imbalance - create synthetic minority samples'
            }
        elif imbalance_ratio > 3:
            return {
                'method': 'Random Undersampling',
                'k_neighbors': 5,
                'sampling_strategy': 0.7,
                'description': 'Mild imbalance - remove majority class samples'
            }
        else:
            return {
                'method': 'None',
                'k_neighbors': 5,
                'sampling_strategy': 1.0,
                'description': 'Balanced dataset - no resampling needed'
            }
    
    @staticmethod
    def apply_balancing(
        df: pd.DataFrame,
        target_col: str,
        method: str,
        sampling_strategy: float = 0.5,
        k_neighbors: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Apply class balancing to the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            method: Balancing method ('SMOTE', 'Random Undersampling', 'SMOTE + Tomek Links')
            sampling_strategy: Target ratio for minority class (0.3 to 1.0)
            k_neighbors: Number of neighbors for SMOTE
            random_state: Random seed for reproducibility
            
        Returns:
            Balanced DataFrame
            
        Raises:
            ValueError: If method is invalid or insufficient samples for SMOTE
        """
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTETomek
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine if binary or multi-class
        n_classes = y.nunique()
        
        # For multi-class (>2 classes), use 'auto' instead of float
        if n_classes > 2:
            sampling_strategy_param = 'auto'  # Balance all minority classes to majority
        else:
            sampling_strategy_param = sampling_strategy  # Float works for binary
        
        # Handle non-numeric features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            # Simple encoding for categorical variables
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        else:
            X_encoded = X.copy()
        
        # Apply balancing method
        try:
            if method == 'SMOTE':
                sampler = SMOTE(
                    sampling_strategy=sampling_strategy_param,
                    k_neighbors=min(k_neighbors, y.value_counts().min() - 1),
                    random_state=random_state
                )
            elif method == 'Random Undersampling':
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy_param,
                    random_state=random_state
                )
            elif method == 'SMOTE + Tomek Links':
                sampler = SMOTETomek(
                    sampling_strategy=sampling_strategy_param,
                    random_state=random_state,
                    smote=SMOTE(
                        k_neighbors=min(k_neighbors, y.value_counts().min() - 1),
                        random_state=random_state
                    )
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Resample
            X_resampled, y_resampled = sampler.fit_resample(X_encoded, y)
            
            # Create balanced DataFrame
            df_balanced = pd.DataFrame(X_resampled, columns=X_encoded.columns)
            df_balanced[target_col] = y_resampled
            
            return df_balanced
            
        except ValueError as e:
            if "k_neighbors" in str(e):
                raise ValueError(
                    f"Not enough samples in minority class for SMOTE with k_neighbors={k_neighbors}. "
                    f"Try reducing k_neighbors or use Random Undersampling instead."
                )
            else:
                raise e
    
    @staticmethod
    def display_class_distribution(
        df: pd.DataFrame,
        target_col: str,
        title: str = "Class Distribution"
    ):
        """
        Display class distribution as a bar chart and table.
        
        Args:
            df: DataFrame
            target_col: Target column name
            title: Chart title
        """
        import plotly.express as px
        
        class_counts = df[target_col].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Percentage'] = (class_counts['Count'] / len(df) * 100).round(2)
        
        # Bar chart
        fig = px.bar(
            class_counts,
            x='Class',
            y='Count',
            title=title,
            text='Count',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(class_counts, use_container_width=True, hide_index=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Number of Classes", len(class_counts))
        with col3:
            imbalance = class_counts['Count'].max() / class_counts['Count'].min()
            st.metric("Imbalance Ratio", f"{imbalance:.1f}:1")


@st.cache_data(ttl=1800)
def cached_class_balancing(
    df_hash: str,
    target_col: str,
    method: str,
    sampling_strategy: float,
    k_neighbors: int
) -> pd.DataFrame:
    """
    Cached version of class balancing for performance.
    
    Args:
        df_hash: MD5 hash of the dataframe
        target_col: Target column name
        method: Balancing method
        sampling_strategy: Sampling strategy
        k_neighbors: Number of neighbors for SMOTE
        
    Returns:
        Balanced DataFrame
    """
    # Note: This function signature is for caching purposes
    # Actual balancing happens in ClassBalancer.apply_balancing()
    pass
