"""Comprehensive Machine Learning Classification Training Module.

This module implements a complete ML classification pipeline with 15 algorithms including
linear models, tree-based methods, boosting ensembles, and meta-learners. Features automatic
data preprocessing, stratified sampling, cross-validation, and comprehensive evaluation metrics.

Typical usage example:
    trainer = MLTrainer(df, target_column='Churn', max_samples=10000)
    prep_info = trainer.prepare_data(test_size=0.2)
    all_models = trainer.get_all_models()
    results = trainer.train_all_models(selected_models=['Random Forest', 'XGBoost'])
    best_model_info = trainer.get_best_model_details(results)

Classes:
    MLTrainer: Main class for ML classification training and evaluation.

Supported Algorithms:
    Linear: Ridge, SGD, Perceptron (3)
    Trees: Decision Tree, Random Forest, Extra Trees (3)
    Boosting: AdaBoost, Gradient Boosting, Hist GB, XGBoost, LightGBM, CatBoost (6)
    Ensembles: Bagging, Voting, Stacking (3)

Features:
    - Automatic categorical encoding with LabelEncoder
    - Datetime feature extraction (year, month, day, dayofweek)
    - Stratified train-test split
    - StandardScaler normalization
    - Cross-validation with F1 scoring
    - Multi-class compatible metrics
    - Feature importance extraction
    - Intelligent sampling for large datasets (stratified)

Notes:
    - XGBoost, LightGBM, CatBoost are optional dependencies
    - Handles both binary and multi-class classification
    - Optimized for datasets up to 10,000 samples (configurable)
    - All models use random_state=42 for reproducibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Linear Models
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Boosting Models
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

# Ensemble Models
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier

# External libraries (with fallbacks)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class MLTrainer:
    """
    Comprehensive ML training pipeline with 15 classification algorithms.
    Optimized for performance and memory efficiency.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str, max_samples: int = 10000) -> None:
        """Initialize ML classification trainer with data and configuration.
        
        Creates a new MLTrainer instance with automatic stratified sampling for large
        datasets to optimize training performance while maintaining class distribution.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing features and target column.
            target_column (str): Name of the column to predict (classification target).
            max_samples (int): Maximum number of samples to use for training. Datasets
                larger than this will be stratified sampled to maintain class proportions.
                Default is 10,000.
        
        Examples:
            >>> df = pd.read_csv('customer_churn.csv')
            >>> trainer = MLTrainer(df, target_column='Churn', max_samples=5000)
            >>> print(f"Sampled: {trainer.sampled}, Rows: {len(trainer.df)}")
        
        Notes:
            - Makes a copy of input DataFrame
            - Uses stratified sampling to preserve class distribution
            - Falls back to random sampling if stratification fails
            - Sets sampled=True if sampling occurred
            - Call prepare_data() next to preprocess features
        """
        self.df = df.copy()
        self.target_column = target_column
        self.max_samples = max_samples
        
        # Sample if dataset is large (STRATIFIED to maintain class distribution)
        if len(self.df) > max_samples:
            # Use stratified sampling to maintain class proportions
            try:
                self.df = self.df.groupby(target_column, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(max_samples * len(x) / len(self.df)))), random_state=42)
                ).reset_index(drop=True)
                self.sampled = True
            except:
                # Fallback to random sampling if stratified fails
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
        self.target_encoder = None
        self.feature_names = []
        self.class_names = []
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Prepare data for ML training with encoding, scaling, and splitting.
        
        This method performs comprehensive data preprocessing including datetime feature
        extraction, categorical encoding, target encoding, train-test split (stratified),
        and feature scaling.
        
        Args:
            test_size (float): Proportion of dataset to use for testing (0.0-1.0).
                Default is 0.2 (20% test, 80% train).
            random_state (int): Random seed for reproducible splits. Default is 42.
            
        Returns:
            Dict[str, Any]: Preparation summary containing:
                - n_samples (int): Total number of samples
                - n_features (int): Number of features after preprocessing
                - n_classes (int): Number of unique target classes
                - train_size (int): Number of training samples
                - test_size (int): Number of test samples
                - sampled (bool): Whether dataset was sampled
                - class_names (List[str]): Names of target classes
        
        Examples:
            >>> trainer = MLTrainer(df, 'Churn')
            >>> info = trainer.prepare_data(test_size=0.25)
            >>> print(f"Features: {info['n_features']}, Classes: {info['n_classes']}")
            >>> print(f"Train: {info['train_size']}, Test: {info['test_size']}")
        
        Notes:
            - Datetime columns converted to numeric features (year, month, day, dayofweek)
            - Categorical features encoded with LabelEncoder
            - Target variable encoded if categorical
            - StandardScaler applied to features
            - Stratified split maintains class distribution
            - Sets instance attributes: X_train, X_test, y_train, y_test
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
        
        # Encode target
        if self.y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y.astype(str))
            self.class_names = list(self.target_encoder.classes_)
        else:
            self.class_names = [str(c) for c in sorted(self.y.unique())]
        
        # Train-test split (stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return {
            'n_samples': len(self.df),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'sampled': self.sampled,
            'class_names': self.class_names
        }
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get dictionary of all 15 available classification models.
        
        Returns a dictionary mapping model names to initialized model instances.
        Models are configured with default hyperparameters optimized for general use.
        
        Returns:
            Dict[str, Any]: Dictionary with model names as keys and sklearn-compatible
                model instances as values. Total of 12-15 models depending on optional
                library availability (XGBoost, LightGBM, CatBoost).
        
        Examples:
            >>> trainer = MLTrainer(df, 'Target')
            >>> models = trainer.get_all_models()
            >>> print(f"Available models: {list(models.keys())}")
            >>> print(f"Total: {len(models)} models")
        
        Notes:
            - All models use random_state=42 for reproducibility
            - XGBoost, LightGBM, CatBoost only included if installed
            - Models grouped by type: Linear (3), Trees (3), Boosting (6), Ensembles (3)
            - All models are freshly initialized (not fitted)
            - Use train_single_model() or train_all_models() to train
        """
        models = {}
        
        # Linear Models (3)
        models['Ridge Classifier'] = RidgeClassifier(random_state=42)
        models['SGD Classifier'] = SGDClassifier(random_state=42, max_iter=500, n_jobs=1)
        models['Perceptron'] = Perceptron(random_state=42, max_iter=500, n_jobs=1)
        
        # Tree-Based Models (3)
        models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
        models['Random Forest'] = RandomForestClassifier(random_state=42, n_jobs=-1)
        models['Extra Trees'] = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        
        # Boosting Models (6)
        models['AdaBoost'] = AdaBoostClassifier(random_state=42)
        models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
        models['Histogram Gradient Boosting'] = HistGradientBoostingClassifier(random_state=42, max_iter=500)
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
        
        # Ensemble Models (3)
        models['Bagging'] = BaggingClassifier(
            estimator=RidgeClassifier(random_state=42),
            random_state=42,
            n_jobs=-1
        )
        
        models['Voting'] = VotingClassifier(
            estimators=[
                ('ridge', RidgeClassifier(random_state=42)),
                ('sgd', SGDClassifier(random_state=42, max_iter=500)),
                ('perceptron', Perceptron(random_state=42, max_iter=500))
            ],
            voting='hard',
            n_jobs=-1
        )
        
        models['Stacking'] = StackingClassifier(
            estimators=[
                ('extra', ExtraTreesClassifier(n_estimators=20, max_depth=10, random_state=42)),
                ('sgd', SGDClassifier(random_state=42, max_iter=500))
            ],
            final_estimator=RidgeClassifier(random_state=42),
            n_jobs=-1
        )
        
        return models
    
    def train_single_model(
        self,
        model_name: str,
        model: Any,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """Train and comprehensively evaluate a single classification model.
        
        This method trains a model on the training set, generates predictions on the test
        set, computes evaluation metrics (accuracy, precision, recall, F1, ROC-AUC), and
        performs cross-validation for robustness assessment.
        
        Args:
            model_name (str): Human-readable name of the model (e.g., 'Random Forest').
            model (Any): Initialized sklearn-compatible model instance to train.
            cv_folds (int): Number of cross-validation folds for F1 scoring. Default is 3.
                Must be >= 2. Use higher values (5-10) for more reliable estimates.
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results containing:
                - model_name (str): Name of the model
                - model (Any): Fitted model instance
                - accuracy (float): Test set accuracy (0.0-1.0)
                - precision (float): Precision score (binary/weighted)
                - recall (float): Recall score (binary/weighted)
                - f1 (float): F1 score (binary/weighted)
                - roc_auc (float|None): ROC-AUC score (None if unavailable)
                - cv_scores (List[float]): Cross-validation F1 scores
                - cv_mean (float|None): Mean CV score
                - cv_std (float|None): Standard deviation of CV scores
                - training_time (float): Training time in seconds
                - success (bool): True if training succeeded
                - error (str|None): Error message if training failed
        
        Examples:
            >>> trainer = MLTrainer(df, 'Churn')
            >>> trainer.prepare_data()
            >>> models = trainer.get_all_models()
            >>> result = trainer.train_single_model('Random Forest', models['Random Forest'])
            >>> print(f"Accuracy: {result['accuracy']:.3f}")
            >>> print(f"F1: {result['f1']:.3f}, CV Mean: {result['cv_mean']:.3f}")
        
        Notes:
            - Uses weighted averaging for multi-class metrics
            - ROC-AUC computed for both binary and multi-class
            - Cross-validation uses StratifiedKFold
            - Returns error dict with success=False if training fails
            - Training time includes fitting only (not CV)
        """
        try:
            # Track training time
            start_time = time.time()
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Multi-class compatible metrics
            n_classes = len(np.unique(self.y))
            average_method = 'binary' if n_classes == 2 else 'weighted'
            
            precision = precision_score(self.y_test, y_pred, average=average_method, zero_division=0)
            recall = recall_score(self.y_test, y_pred, average=average_method, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average=average_method, zero_division=0)
            
            # ROC-AUC (handle binary and multi-class)
            try:
                if n_classes == 2:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test)[:, 1]
                        roc_auc = roc_auc_score(self.y_test, y_proba)
                    elif hasattr(model, 'decision_function'):
                        y_score = model.decision_function(self.X_test)
                        roc_auc = roc_auc_score(self.y_test, y_score)
                    else:
                        roc_auc = None
                else:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test)
                        roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr', average='weighted')
                    else:
                        roc_auc = None
            except:
                roc_auc = None
            
            # Cross-validation
            try:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_scores = []
                cv_mean = None
                cv_std = None
            
            return {
                'model_name': model_name,
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_scores': cv_scores.tolist() if len(cv_scores) > 0 else [],
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'model': None,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'roc_auc': None,
                'cv_scores': [],
                'cv_mean': None,
                'cv_std': None,
                'training_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def train_all_models(
        self,
        selected_models: Optional[List[str]] = None,
        cv_folds: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Train all or selected models.
        
        Args:
            selected_models: List of model names to train (None = all)
            cv_folds: Number of cross-validation folds
            progress_callback: Function to call after each model (model_name, results)
            
        Returns:
            List of results sorted by F1 score (descending)
        """
        all_models = self.get_all_models()
        
        if selected_models:
            models_to_train = {k: v for k, v in all_models.items() if k in selected_models}
        else:
            models_to_train = all_models
        
        results = []
        
        for i, (model_name, model) in enumerate(models_to_train.items(), 1):
            result = self.train_single_model(model_name, model, cv_folds)
            results.append(result)
            
            if progress_callback:
                progress_callback(i, len(models_to_train), model_name, result)
        
        # Sort by F1 score (descending)
        results.sort(key=lambda x: x['f1'], reverse=True)
        
        return results
    
    def train_models_sequentially(
        self,
        model_names: List[str],
        cv_folds: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models one at a time to prevent memory overload.
        
        This method trains models sequentially with garbage collection between
        each model to minimize memory usage. Only essential results are stored.
        
        Args:
            model_names: List of model names to train
            cv_folds: Number of cross-validation folds
            progress_callback: Optional callback function(current, total, model_name)
        
        Returns:
            Dict of model results (without full model objects)
        """
        import gc
        
        all_models = self.get_all_models()
        results = {}
        total = len(model_names)
        
        for i, model_name in enumerate(model_names):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total, model_name)
                
                # Get model
                if model_name not in all_models:
                    results[model_name] = {'error': f'Unknown model: {model_name}'}
                    continue
                
                model = all_models[model_name]
                
                # Train single model
                model_result = self.train_single_model(model_name, model, cv_folds)
                
                # Store only essential results (not the full model object)
                results[model_name] = {
                    'accuracy': model_result.get('accuracy'),
                    'precision': model_result.get('precision'),
                    'recall': model_result.get('recall'),
                    'f1': model_result.get('f1'),
                    'cv_scores': model_result.get('cv_scores'),
                    'train_time': model_result.get('train_time'),
                    'feature_importance': model_result.get('feature_importance')
                }
                
                # Clean up model object to free memory
                if 'model' in model_result:
                    del model_result['model']
                del model
                
                # Force garbage collection after each model
                gc.collect()
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
                gc.collect()
        
        return results
    
    def get_best_model_details(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed information about the best model.
        
        Args:
            results: List of training results
            
        Returns:
            Dictionary with best model details
        """
        if not results or len(results) == 0:
            return None
        
        best_result = results[0]
        best_model = best_result['model']
        
        if best_model is None:
            return None
        
        # Confusion matrix
        y_pred = best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = {
                'features': self.feature_names,
                'importances': importances.tolist()
            }
        elif hasattr(best_model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = np.abs(best_model.coef_)
            if len(coef.shape) > 1:
                coef = coef.mean(axis=0)
            feature_importance = {
                'features': self.feature_names,
                'importances': coef.tolist()
            }
        
        return {
            'model_name': best_result['model_name'],
            'metrics': {
                'accuracy': best_result['accuracy'],
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1': best_result['f1'],
                'roc_auc': best_result['roc_auc'],
                'cv_mean': best_result['cv_mean'],
                'cv_std': best_result['cv_std'],
                'training_time': best_result['training_time']
            },
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance,
            'model_object': best_model,
            'class_names': self.class_names
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model description, strengths, weaknesses, use cases
        """
        model_info = {
            'Ridge Classifier': {
                'description': 'Linear classifier with L2 regularization',
                'strengths': 'Fast, works well with high-dimensional data, prevents overfitting',
                'weaknesses': 'Assumes linear relationships, may underfit complex patterns',
                'use_cases': 'Text classification, high-dimensional sparse data, when speed is critical'
            },
            'SGD Classifier': {
                'description': 'Stochastic Gradient Descent classifier for online learning',
                'strengths': 'Very fast, memory efficient, works with streaming data',
                'weaknesses': 'Sensitive to feature scaling, requires tuning',
                'use_cases': 'Large datasets, real-time learning, limited memory'
            },
            'Perceptron': {
                'description': 'Simple linear classifier based on neural network principles',
                'strengths': 'Extremely fast, simple to understand',
                'weaknesses': 'Only works for linearly separable data',
                'use_cases': 'Simple binary classification, baseline model'
            },
            'Decision Tree': {
                'description': 'Tree-based model that splits data based on features',
                'strengths': 'Interpretable, handles non-linear relationships, no scaling needed',
                'weaknesses': 'Prone to overfitting, unstable',
                'use_cases': 'When interpretability is crucial, mixed data types'
            },
            'Random Forest': {
                'description': 'Ensemble of decision trees with bagging',
                'strengths': 'High accuracy, reduces overfitting, provides feature importance',
                'weaknesses': 'Slower training, less interpretable than single tree',
                'use_cases': 'General-purpose classifier, feature selection, high accuracy needed'
            },
            'Extra Trees': {
                'description': 'Extremely randomized trees ensemble',
                'strengths': 'Faster than Random Forest, good generalization',
                'weaknesses': 'Less accurate than Random Forest sometimes',
                'use_cases': 'When speed matters, large datasets'
            },
            'AdaBoost': {
                'description': 'Adaptive boosting that combines weak learners',
                'strengths': 'High accuracy, works well with weak classifiers',
                'weaknesses': 'Sensitive to noisy data and outliers',
                'use_cases': 'Binary classification, when you have clean data'
            },
            'Gradient Boosting': {
                'description': 'Sequential tree building with gradient descent',
                'strengths': 'Very high accuracy, handles complex patterns',
                'weaknesses': 'Slow training, prone to overfitting without tuning',
                'use_cases': 'Kaggle competitions, when accuracy is paramount'
            },
            'Histogram Gradient Boosting': {
                'description': 'Optimized gradient boosting with binning',
                'strengths': 'Faster than GB, handles missing values, memory efficient',
                'weaknesses': 'May lose precision with binning',
                'use_cases': 'Large datasets, when GB is too slow'
            },
            'XGBoost': {
                'description': 'Extreme Gradient Boosting (industry standard)',
                'strengths': 'Best-in-class accuracy, highly optimized, feature importance',
                'weaknesses': 'Many hyperparameters, can overfit',
                'use_cases': 'Production systems, competitions, maximum accuracy'
            },
            'LightGBM': {
                'description': 'Light Gradient Boosting Machine',
                'strengths': 'Very fast training, low memory, handles large data',
                'weaknesses': 'Can overfit small datasets',
                'use_cases': 'Large datasets (>10k rows), when speed is critical'
            },
            'CatBoost': {
                'description': 'Categorical Boosting',
                'strengths': 'Handles categorical features natively, robust, less tuning',
                'weaknesses': 'Slower than LightGBM',
                'use_cases': 'Datasets with many categorical features'
            },
            'Bagging': {
                'description': 'Bootstrap aggregating ensemble',
                'strengths': 'Reduces variance, parallelizable',
                'weaknesses': 'May not improve bias',
                'use_cases': 'When base model is high-variance'
            },
            'Voting': {
                'description': 'Combines predictions from multiple models',
                'strengths': 'Often more robust than single model',
                'weaknesses': 'Slower, all models must succeed',
                'use_cases': 'When you want to combine different model types'
            },
            'Stacking': {
                'description': 'Meta-learning ensemble',
                'strengths': 'Can achieve highest accuracy, learns optimal combination',
                'weaknesses': 'Complex, slow, risk of overfitting',
                'use_cases': 'Competitions, when maximum accuracy is needed'
            }
        }
        
        return model_info.get(model_name, {
            'description': 'No information available',
            'strengths': 'N/A',
            'weaknesses': 'N/A',
            'use_cases': 'N/A'
        })
