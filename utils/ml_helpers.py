"""
Machine Learning Helper Functions for Optimization
Provides utilities for smart CV fold selection and results caching
"""

import pandas as pd
import streamlit as st
import hashlib
import json
from typing import Tuple, Dict, Any


def get_recommended_cv_folds(n_samples: int, n_classes: int = None) -> Tuple[int, str]:
    """
    Determine optimal CV folds based on dataset characteristics.
    
    Args:
        n_samples: Number of samples in dataset
        n_classes: Number of classes (for classification, None for regression)
        
    Returns:
        tuple: (recommended_folds, explanation)
    """
    if n_samples < 100:
        folds = 3
        reason = "Small dataset - using minimum 3-fold CV"
    elif n_samples < 500:
        folds = 3
        reason = "Small dataset - 3-fold CV is optimal"
    elif n_samples < 2000:
        folds = 5
        reason = "Medium dataset - 5-fold CV for reliability"
    elif n_samples < 5000:
        folds = 5
        reason = "Large dataset - 5-fold CV balances speed/accuracy"
    else:
        folds = 3
        reason = "Very large dataset - 3-fold CV to reduce compute time"
    
    # For multi-class with many classes, use fewer folds
    if n_classes and n_classes > 20:
        folds = min(folds, 3)
        reason += " (reduced for multi-class problem)"
    
    return folds, reason


def create_data_hash(df: pd.DataFrame, target_col: str, params: Dict[str, Any]) -> str:
    """
    Create unique hash for caching based on data and parameters.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        params: Dictionary of training parameters
        
    Returns:
        str: MD5 hash of data + parameters
    """
    hash_input = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'target': target_col,
        'params': params
    }
    hash_str = json.dumps(hash_input, sort_keys=True, default=str)
    return hashlib.md5(hash_str.encode()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_classification_training(
    data_hash: str,
    df_dict: Dict,
    target_col: str,
    test_size: float,
    cv_folds: int,
    models_to_train: list,
    max_samples: int = 10000
) -> Tuple[list, Any, Dict]:
    """
    Train classification models with caching (1 hour TTL).
    
    Args:
        data_hash: Unique hash for cache key
        df_dict: DataFrame as dictionary (for caching)
        target_col: Target column name
        test_size: Test set size (0-1)
        cv_folds: Number of CV folds
        models_to_train: List of model names to train
        max_samples: Maximum samples for training
        
    Returns:
        tuple: (results list, trainer object, prep_info dict)
    """
    from utils.ml_training import MLTrainer
    
    # Reconstruct DataFrame from dict
    df = pd.DataFrame(df_dict)
    
    # Initialize trainer
    trainer = MLTrainer(df, target_col, max_samples=max_samples)
    prep_info = trainer.prepare_data(test_size=test_size)
    
    # Get all models
    all_models = trainer.get_all_models()
    
    # Train models
    results = []
    for model_name in models_to_train:
        model = all_models.get(model_name)
        if model:
            result = trainer.train_single_model(model_name, model, cv_folds=cv_folds)
            results.append(result)
    
    return results, trainer, prep_info


@st.cache_data(ttl=3600, show_spinner=False)
def cached_regression_training(
    data_hash: str,
    df_dict: Dict,
    target_col: str,
    test_size: float,
    cv_folds: int,
    models_to_train: list,
    max_samples: int = 10000
) -> Tuple[list, Any]:
    """
    Train regression models with caching (1 hour TTL).
    
    Args:
        data_hash: Unique hash for cache key
        df_dict: DataFrame as dictionary (for caching)
        target_col: Target column name
        test_size: Test set size (0-1)
        cv_folds: Number of CV folds
        models_to_train: List of model names to train
        max_samples: Maximum samples for training
        
    Returns:
        tuple: (results list, regressor object)
    """
    from utils.ml_regression import MLRegressor
    
    # Reconstruct DataFrame from dict
    df = pd.DataFrame(df_dict)
    
    # Initialize regressor
    regressor = MLRegressor(df, target_col, max_samples=max_samples)
    regressor.prepare_data(test_size=test_size)
    
    # Train models
    results = regressor.train_all_models(
        selected_models=models_to_train,
        cv_folds=cv_folds
    )
    
    return results, regressor
