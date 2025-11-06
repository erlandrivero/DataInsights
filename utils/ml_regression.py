"""
Comprehensive Machine Learning Regression Module
Implements 15 regression algorithms with full evaluation pipeline
Optimized with LazyModuleLoader for reduced memory footprint
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
import warnings
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# Import LazyModuleLoader for dynamic model loading
from utils.lazy_loader import LazyModuleLoader

# Check external library availability (without importing)
try:
    import importlib.util
    XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
except:
    XGBOOST_AVAILABLE = False

try:
    LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
except:
    LIGHTGBM_AVAILABLE = False

try:
    CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
except:
    CATBOOST_AVAILABLE = False


class MLRegressor:
    """
    Comprehensive ML regression pipeline with 15+ algorithms.
    Optimized for performance and memory efficiency.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str, max_samples: int = 10000):
        """
        Initialize ML Regressor.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (continuous numerical)
            max_samples: Maximum samples for training (performance optimization)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.max_samples = max_samples
        
        # Stratified sampling for regression (based on target quantiles)
        if len(self.df) > max_samples:
            try:
                # Create bins for stratified sampling
                self.df['_temp_bins'] = pd.qcut(self.df[target_column], q=10, labels=False, duplicates='drop')
                self.df = self.df.groupby('_temp_bins', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(max_samples * len(x) / len(self.df)))), random_state=42)
                ).reset_index(drop=True)
                self.df = self.df.drop('_temp_bins', axis=1)
                self.sampled = True
            except:
                # Fallback to random sampling
                self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
                self.sampled = True
        else:
            self.sampled = False
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Prepare data for training with proper encoding and scaling.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with preparation details
        """
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column]).copy()
        self.y = self.df[self.target_column].copy()
        
        # Handle datetime columns - convert to numeric features or drop
        datetime_cols = []
        for col in self.X.columns:
            if pd.api.types.is_datetime64_any_dtype(self.X[col]):
                datetime_cols.append(col)
                try:
                    # Try to extract useful features from datetime
                    self.X[f'{col}_year'] = pd.to_datetime(self.X[col]).dt.year
                    self.X[f'{col}_month'] = pd.to_datetime(self.X[col]).dt.month
                    self.X[f'{col}_day'] = pd.to_datetime(self.X[col]).dt.day
                    self.X[f'{col}_dayofweek'] = pd.to_datetime(self.X[col]).dt.dayofweek
                    # Drop original datetime column
                    self.X = self.X.drop(columns=[col])
                    warnings.warn(f"Converted datetime column '{col}' to numeric features (year, month, day, dayofweek)")
                except:
                    # If extraction fails, just drop the column
                    self.X = self.X.drop(columns=[col])
                    warnings.warn(f"Dropped datetime column '{col}' - could not convert to numeric features")
        
        # Store feature names
        self.feature_names = list(self.X.columns)
        
        # Encode categorical features
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'n_features': len(self.feature_names),
            'sampled': self.sampled
        }
    
    def _lazy_load_model(self, model_name: str) -> Any:
        """
        Lazy load a model using LazyModuleLoader.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Instantiated model object
        """
        model_configs = {
            # Linear Models
            'Linear Regression': ('sklearn.linear_model', 'LinearRegression', {}),
            'Ridge': ('sklearn.linear_model', 'Ridge', {'alpha': 1.0}),
            'Lasso': ('sklearn.linear_model', 'Lasso', {'alpha': 1.0}),
            'ElasticNet': ('sklearn.linear_model', 'ElasticNet', {'alpha': 1.0}),
            'Bayesian Ridge': ('sklearn.linear_model', 'BayesianRidge', {}),
            
            # Support Vector Machines
            'SVR': ('sklearn.svm', 'SVR', {'kernel': 'rbf'}),
            
            # Tree-Based Models
            'Decision Tree': ('sklearn.tree', 'DecisionTreeRegressor', {'max_depth': 10, 'random_state': 42}),
            'Random Forest': ('sklearn.ensemble', 'RandomForestRegressor', {'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1, 'random_state': 42}),
            'Extra Trees': ('sklearn.ensemble', 'ExtraTreesRegressor', {'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1, 'random_state': 42}),
            
            # Boosting Models
            'AdaBoost': ('sklearn.ensemble', 'AdaBoostRegressor', {'n_estimators': 50, 'random_state': 42}),
            'Gradient Boosting': ('sklearn.ensemble', 'GradientBoostingRegressor', {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
            'Hist Gradient Boosting': ('sklearn.ensemble', 'HistGradientBoostingRegressor', {'max_iter': 100, 'random_state': 42}),
            
            # Ensemble Models
            'Bagging': ('sklearn.ensemble', 'BaggingRegressor', {'n_estimators': 10, 'n_jobs': -1, 'random_state': 42}),
            
            # External Models
            'XGBoost': ('xgboost', 'XGBRegressor', {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbosity': 0}),
            'LightGBM': ('lightgbm', 'LGBMRegressor', {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbose': -1}),
            'CatBoost': ('catboost', 'CatBoostRegressor', {'iterations': 100, 'depth': 5, 'random_state': 42, 'verbose': 0}),
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        module_name, class_name, params = model_configs[model_name]
        
        # Lazy load the module
        module = LazyModuleLoader.load_module(module_name)
        if module is None:
            raise ImportError(f"Could not load module: {module_name}")
        
        # Get the model class and instantiate
        model_class = getattr(module, class_name)
        model = model_class(**params)
        
        return model
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Get list of all available regression model names.
        Models are NOT instantiated here - they will be lazy loaded when needed.
        
        Returns:
            Dict with model names as keys and None as values (placeholder)
        """
        models = {
            # Linear Models
            'Linear Regression': None,
            'Ridge': None,
            'Lasso': None,
            'ElasticNet': None,
            'Bayesian Ridge': None,
            
            # Support Vector Machines
            'SVR': None,
            
            # Tree-Based Models
            'Decision Tree': None,
            'Random Forest': None,
            'Extra Trees': None,
            
            # Boosting Models
            'AdaBoost': None,
            'Gradient Boosting': None,
            'Hist Gradient Boosting': None,
            
            # Ensemble Models
            'Bagging': None,
        }
        
        # Add external models if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = None
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = None
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = None
        
        return models
    
    def train_model(self, model_name: str, model: Any, cv_folds: int = 3) -> Dict[str, Any]:
        """
        Train a single model and return evaluation metrics.
        
        Args:
            model_name: Name of the model
            model: Model instance
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model results
        """
        start_time = time.time()
        
        try:
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # MAPE (handle zero values)
            try:
                mape = mean_absolute_percentage_error(self.y_test, y_pred)
            except:
                mape = None
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=min(cv_folds, len(self.X_train)),
                    scoring='r2',
                    n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = None
                cv_std = None
            
            training_time = time.time() - start_time
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(self.feature_names, np.abs(model.coef_)))
            
            return {
                'model_name': model_name,
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'feature_importance': feature_importance,
                'success': True
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def train_all_models(self, selected_models: Optional[List[str]] = None, 
                        cv_folds: int = 3,
                        progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Train all models and return results.
        
        Args:
            selected_models: List of model names to train (None = all)
            cv_folds: Number of cross-validation folds
            progress_callback: Callback function for progress updates
            
        Returns:
            List of result dictionaries
        """
        all_models = self.get_all_models()
        
        if selected_models:
            models_to_train = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            models_to_train = all_models
        
        results = []
        total_models = len(models_to_train)
        
        for idx, (model_name, model) in enumerate(models_to_train.items(), 1):
            result = self.train_model(model_name, model, cv_folds)
            
            if result['success']:
                results.append(result)
                
                if progress_callback:
                    progress_callback(idx, total_models, model_name, result)
        
        return results
    
    def train_models_sequentially(
        self,
        model_names: List[str],
        cv_folds: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models one at a time using LazyModuleLoader to prevent memory overload.
        
        This method trains models sequentially with lazy loading and garbage collection
        between each model to minimize memory usage. Only essential results are stored.
        Models are loaded dynamically, trained, and immediately unloaded.
        
        Args:
            model_names: List of model names to train
            cv_folds: Number of cross-validation folds
            progress_callback: Optional callback function(current, total, model_name)
        
        Returns:
            Dict of model results (without full model objects)
        """
        all_models = self.get_all_models()
        results = {}
        total = len(model_names)
        
        for i, model_name in enumerate(model_names):
            module_name = None
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total, model_name)
                
                # Check if model exists
                if model_name not in all_models:
                    results[model_name] = {'error': f'Unknown model: {model_name}'}
                    continue
                
                # Lazy load model (load module dynamically)
                model = self._lazy_load_model(model_name)
                
                # Determine module name for cleanup
                model_configs = {
                    'Linear Regression': 'sklearn.linear_model',
                    'Ridge': 'sklearn.linear_model',
                    'Lasso': 'sklearn.linear_model',
                    'ElasticNet': 'sklearn.linear_model',
                    'Bayesian Ridge': 'sklearn.linear_model',
                    'SVR': 'sklearn.svm',
                    'Decision Tree': 'sklearn.tree',
                    'Random Forest': 'sklearn.ensemble',
                    'Extra Trees': 'sklearn.ensemble',
                    'AdaBoost': 'sklearn.ensemble',
                    'Gradient Boosting': 'sklearn.ensemble',
                    'Hist Gradient Boosting': 'sklearn.ensemble',
                    'Bagging': 'sklearn.ensemble',
                    'XGBoost': 'xgboost',
                    'LightGBM': 'lightgbm',
                    'CatBoost': 'catboost',
                }
                module_name = model_configs.get(model_name)
                
                # Train single model
                model_result = self.train_model(model_name, model, cv_folds)
                
                # Store only essential results (not the full model object)
                if model_result.get('success'):
                    results[model_name] = {
                        'r2': model_result.get('r2'),
                        'rmse': model_result.get('rmse'),
                        'mae': model_result.get('mae'),
                        'mape': model_result.get('mape'),
                        'cv_mean': model_result.get('cv_mean'),
                        'cv_std': model_result.get('cv_std'),
                        'train_time': model_result.get('training_time', 0),
                        'feature_importance': model_result.get('feature_importance')
                    }
                else:
                    results[model_name] = {'error': model_result.get('error', 'Training failed')}
                
                # Clean up model object to free memory
                if 'model' in model_result:
                    del model_result['model']
                del model
                
                # Unload the module to free memory
                if module_name:
                    LazyModuleLoader.unload_module(module_name)
                
                # Force garbage collection after each model
                gc.collect()
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
                # Unload module even on error
                if module_name:
                    LazyModuleLoader.unload_module(module_name)
                gc.collect()
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get detailed information about a model."""
        model_info = {
            'Linear Regression': {
                'description': 'Ordinary Least Squares regression with no regularization',
                'strengths': '• Simple and interpretable\n• Fast training\n• Good for linear relationships',
                'weaknesses': '• Sensitive to outliers\n• Prone to overfitting with many features\n• Assumes linear relationship',
                'use_cases': 'Price prediction, demand forecasting, simple trend analysis'
            },
            'Ridge': {
                'description': 'Linear regression with L2 regularization',
                'strengths': '• Reduces overfitting\n• Handles multicollinearity\n• More stable than OLS',
                'weaknesses': '• All features retained (not sparse)\n• Requires tuning alpha\n• Linear assumptions',
                'use_cases': 'High-dimensional data, correlated features, regularized predictions'
            },
            'Lasso': {
                'description': 'Linear regression with L1 regularization (feature selection)',
                'strengths': '• Automatic feature selection\n• Sparse solutions\n• Interpretable',
                'weaknesses': '• Can be unstable with correlated features\n• Requires tuning\n• Linear only',
                'use_cases': 'Feature selection, sparse models, interpretable predictions'
            },
            'Random Forest': {
                'description': 'Ensemble of decision trees with bagging',
                'strengths': '• Handles non-linear relationships\n• Robust to outliers\n• Feature importance\n• Minimal tuning needed',
                'weaknesses': '• Can be slow\n• Less interpretable\n• Memory intensive',
                'use_cases': 'Complex relationships, robust predictions, feature ranking'
            },
            'XGBoost': {
                'description': 'Extreme Gradient Boosting - optimized gradient boosting',
                'strengths': '• State-of-the-art performance\n• Handles missing data\n• Regularization built-in\n• Fast training',
                'weaknesses': '• Many hyperparameters\n• Can overfit\n• Requires careful tuning',
                'use_cases': 'Kaggle competitions, production systems, best accuracy'
            },
            # Add more model info as needed
        }
        
        return model_info.get(model_name, {
            'description': f'{model_name} regression model',
            'strengths': 'Effective regression algorithm',
            'weaknesses': 'May require tuning',
            'use_cases': 'General regression tasks'
        })
    
    def get_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics from results."""
        if not results:
            return {}
        
        best_model = max(results, key=lambda x: x.get('r2', -float('inf')))
        
        return {
            'total_models': len(results),
            'best_model': best_model['model_name'],
            'best_r2': best_model['r2'],
            'total_time': sum(r['training_time'] for r in results)
        }
