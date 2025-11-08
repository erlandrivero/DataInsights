"""Anomaly Detection utilities using multiple machine learning algorithms.

This module provides comprehensive anomaly detection capabilities using
Isolation Forest, Local Outlier Factor, and One-Class SVM algorithms.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import numpy as np
import gc
import importlib.util
from typing import List, Dict, Any, Tuple, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from utils.lazy_loader import LazyModuleLoader

# Check sklearn availability
SKLEARN_AVAILABLE = importlib.util.find_spec('sklearn') is not None


class AnomalyDetector:
    """Handles anomaly detection using multiple machine learning algorithms.
    
    This class provides comprehensive anomaly detection capabilities using three
    different algorithms: Isolation Forest (ensemble-based), Local Outlier Factor
    (density-based), and One-Class SVM (boundary-based).
    
    Attributes:
        df (pd.DataFrame): Original input dataframe
        features (Optional[pd.DataFrame]): Selected feature columns
        scaled_features (Optional[pd.DataFrame]): Standardized feature matrix
        scaler (StandardScaler): Scikit-learn scaler for normalization
        model: Fitted anomaly detection model (IF, LOF, or SVM)
        anomaly_results (Optional[pd.DataFrame]): Detection results with scores
    
    Example:
        >>> # Basic anomaly detection workflow
        >>> detector = AnomalyDetector(df)
        >>> 
        >>> # Set features
        >>> detector.set_features(['age', 'income', 'transactions'])
        >>> 
        >>> # Run Isolation Forest
        >>> results = detector.run_isolation_forest(contamination=0.1)
        >>> 
        >>> # Get anomalies
        >>> anomalies = results[results['is_anomaly']]
        >>> print(f"Found {len(anomalies)} anomalies")
        >>> 
        >>> # Visualize
        >>> fig = detector.create_2d_scatter(use_pca=True)
        >>> st.plotly_chart(fig)
    
    Note:
        - All features must be numeric
        - Features are automatically scaled using StandardScaler
        - Missing values are imputed with column means
        - Contamination parameter controls expected outlier percentage
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize AnomalyDetector with a dataframe.
        
        Creates a copy of the input dataframe and initializes all components
        for anomaly detection.
        
        Args:
            df: Input dataframe with data to analyze
        
        Example:
            >>> df = pd.read_csv('transactions.csv')
            >>> detector = AnomalyDetector(df)
        
        Note:
            - Creates a copy to avoid modifying original data
            - StandardScaler loaded lazily when needed
            - All attributes set to None until features are selected
        """
        self.df: pd.DataFrame = df.copy()
        self.features: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[pd.DataFrame] = None
        self.scaler = None  # Lazy loaded
        self.model = None  # Lazy loaded
        self.anomaly_results: Optional[pd.DataFrame] = None
    
    @staticmethod
    def _lazy_load_model(model_name: str):
        """Lazy load sklearn models to reduce memory footprint.
        
        Args:
            model_name: Name of the model to load
                       ('StandardScaler', 'IsolationForest', 'LocalOutlierFactor', 
                        'OneClassSVM', 'PCA')
        
        Returns:
            Model class or None if sklearn not available
        """
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            if model_name == 'StandardScaler':
                from sklearn.preprocessing import StandardScaler
                return StandardScaler
            elif model_name == 'IsolationForest':
                from sklearn.ensemble import IsolationForest
                return IsolationForest
            elif model_name == 'LocalOutlierFactor':
                from sklearn.neighbors import LocalOutlierFactor
                return LocalOutlierFactor
            elif model_name == 'OneClassSVM':
                from sklearn.svm import OneClassSVM
                return OneClassSVM
            elif model_name == 'PCA':
                from sklearn.decomposition import PCA
                return PCA
            else:
                return None
        except ImportError:
            return None
    
    def set_features(self, feature_cols: List[str]) -> pd.DataFrame:
        """Set and scale feature columns for anomaly detection.
        
        Validates columns exist and are numeric, handles missing values,
        and applies standard scaling for algorithm compatibility.
        
        Args:
            feature_cols: List of column names to use as features
        
        Returns:
            DataFrame containing selected and cleaned features
        
        Raises:
            ValueError: If column doesn't exist or isn't numeric
        
        Example:
            >>> detector = AnomalyDetector(df)
            >>> features = detector.set_features(['price', 'quantity', 'total'])
            >>> print(f"Using {len(features.columns)} features")
        
        Note:
            - All columns must be numeric (int or float)
            - Missing values filled with column means
            - Features scaled to mean=0, std=1 using StandardScaler
            - Scaled features stored in self.scaled_features
        """
        # Validate columns
        for col in feature_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        # Select features
        self.features = self.df[feature_cols].copy()
        
        # Handle missing values (impute with mean)
        if self.features.isnull().any().any():
            self.features = self.features.fillna(self.features.mean())
        
        # Lazy load StandardScaler
        StandardScaler = self._lazy_load_model('StandardScaler')
        if StandardScaler is None:
            raise ImportError("sklearn not available")
        
        try:
            # Scale features for algorithm compatibility
            scaler = StandardScaler()
            self.scaled_features = pd.DataFrame(
                scaler.fit_transform(self.features),
                columns=self.features.columns,
                index=self.features.index
            )
            self.scaler = scaler
            
            return self.features
        finally:
            # Cleanup
            gc.collect()
    
    def run_isolation_forest(self, contamination: float = 0.1) -> pd.DataFrame:
        """Run Isolation Forest algorithm for anomaly detection.
        
        Isolation Forest detects anomalies by isolating outliers through
        random partitioning. Anomalies require fewer splits to isolate.
        
        Args:
            contamination: Expected proportion of outliers in dataset (0.01 to 0.5)
                          - 0.01 = expect 1% outliers
                          - 0.1 = expect 10% outliers (default)
                          - 0.5 = maximum allowed
        
        Returns:
            DataFrame with columns:
                - Original dataframe columns
                - anomaly_score (float): Isolation score (lower = more anomalous)
                - is_anomaly (bool): True if classified as anomaly
                - anomaly_type (str): 'Anomaly' or 'Normal'
        
        Example:
            >>> detector.set_features(['amount', 'frequency'])
            >>> results = detector.run_isolation_forest(contamination=0.05)
            >>> 
            >>> # Get top 10 anomalies
            >>> top_anomalies = results[results['is_anomaly']].nsmallest(
            >>>     10, 'anomaly_score'
            >>> )
            >>> st.dataframe(top_anomalies)
        
        Note:
            - Fast algorithm, works well with high-dimensional data
            - Uses 100 trees for stable results
            - Score threshold automatically determined by contamination
            - Best for global outliers (different from overall pattern)
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Lazy load IsolationForest
        IsolationForest = self._lazy_load_model('IsolationForest')
        if IsolationForest is None:
            raise ImportError("sklearn not available")
        
        try:
            # Initialize and fit Isolation Forest model
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict anomalies (-1 for outliers, 1 for inliers)
            predictions = model.fit_predict(self.scaled_features)
            
            # Get anomaly scores (lower values = more anomalous)
            scores = model.score_samples(self.scaled_features)
            
            # Create results dataframe with original data + predictions
            results_df = self.df.copy()
            results_df['anomaly_score'] = scores
            results_df['is_anomaly'] = predictions == -1
            results_df['anomaly_type'] = results_df['is_anomaly'].map(
                {True: 'Anomaly', False: 'Normal'}
            )
            
            self.model = model
            self.anomaly_results = results_df
            return results_df
        finally:
            # Cleanup
            gc.collect()
    
    def run_local_outlier_factor(
        self, 
        contamination: float = 0.1, 
        n_neighbors: int = 20
    ) -> pd.DataFrame:
        """Run Local Outlier Factor (LOF) algorithm for anomaly detection.
        
        LOF detects anomalies by measuring local density deviation. Points in
        sparse regions compared to neighbors are flagged as outliers.
        
        Args:
            contamination: Expected proportion of outliers (0.01 to 0.5)
            n_neighbors: Number of neighbors to consider for density calculation
                        - Low values (5-10): Detect local anomalies
                        - High values (20-50): Detect global anomalies
        
        Returns:
            DataFrame with columns:
                - Original dataframe columns
                - anomaly_score (float): Negative outlier factor (lower = more anomalous)
                - is_anomaly (bool): True if classified as anomaly
                - anomaly_type (str): 'Anomaly' or 'Normal'
        
        Example:
            >>> # Detect local anomalies with small neighborhoods
            >>> results = detector.run_local_outlier_factor(
            >>>     contamination=0.1,
            >>>     n_neighbors=10
            >>> )
            >>> 
            >>> print(f"Found {sum(results['is_anomaly'])} local outliers")
        
        Note:
            - Better for local outliers (different from nearby points)
            - More sensitive to n_neighbors parameter
            - Slower than Isolation Forest for large datasets
            - Works well for clustered data
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Lazy load LocalOutlierFactor
        LocalOutlierFactor = self._lazy_load_model('LocalOutlierFactor')
        if LocalOutlierFactor is None:
            raise ImportError("sklearn not available")
        
        try:
            # Initialize and fit LOF model
            model = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=n_neighbors,
                novelty=False
            )
            
            # Predict anomalies (-1 for outliers, 1 for inliers)
            predictions = model.fit_predict(self.scaled_features)
            
            # Get negative outlier factor (lower = more anomalous)
            scores = model.negative_outlier_factor_
            
            # Create results dataframe
            results_df = self.df.copy()
            results_df['anomaly_score'] = scores
            results_df['is_anomaly'] = predictions == -1
            results_df['anomaly_type'] = results_df['is_anomaly'].map(
                {True: 'Anomaly', False: 'Normal'}
            )
            
            self.model = model
            self.anomaly_results = results_df
            return results_df
        finally:
            # Cleanup
            gc.collect()
    
    def run_one_class_svm(
        self, 
        nu: float = 0.1, 
        kernel: str = 'rbf'
    ) -> pd.DataFrame:
        """Run One-Class SVM algorithm for anomaly detection.
        
        One-Class SVM learns a decision boundary around normal data. Points
        outside this boundary are classified as anomalies.
        
        Args:
            nu: Upper bound on fraction of outliers and lower bound on fraction
                of support vectors (0.01 to 0.5)
                - Similar to contamination in other algorithms
            kernel: Kernel type for decision boundary
                   - 'rbf': Radial basis function (default, flexible)
                   - 'linear': Linear decision boundary
                   - 'poly': Polynomial boundary
                   - 'sigmoid': Sigmoid boundary
        
        Returns:
            DataFrame with columns:
                - Original dataframe columns
                - anomaly_score (float): Decision function value (lower = more anomalous)
                - is_anomaly (bool): True if outside decision boundary
                - anomaly_type (str): 'Anomaly' or 'Normal'
        
        Example:
            >>> # Use RBF kernel for non-linear boundaries
            >>> results = detector.run_one_class_svm(nu=0.05, kernel='rbf')
            >>> 
            >>> # Or linear kernel for simple boundaries
            >>> results = detector.run_one_class_svm(nu=0.05, kernel='linear')
        
        Note:
            - Slower than Isolation Forest
            - Effective for well-separated normal data
            - RBF kernel works well for most cases
            - nu parameter controls boundary tightness
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Lazy load OneClassSVM
        OneClassSVM = self._lazy_load_model('OneClassSVM')
        if OneClassSVM is None:
            raise ImportError("sklearn not available")
        
        try:
            # Initialize and fit One-Class SVM
            model = OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma='auto'
            )
            
            # Fit model and predict
            model.fit(self.scaled_features)
            predictions = model.predict(self.scaled_features)
            
            # Get decision function values (lower = more anomalous)
            scores = model.decision_function(self.scaled_features)
            
            # Create results dataframe
            results_df = self.df.copy()
            results_df['anomaly_score'] = scores
            results_df['is_anomaly'] = predictions == -1
            results_df['anomaly_type'] = results_df['is_anomaly'].map(
                {True: 'Anomaly', False: 'Normal'}
            )
            
            self.model = model
            self.anomaly_results = results_df
            return results_df
        finally:
            # Cleanup
            gc.collect()
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Calculate feature importance for Isolation Forest algorithm.
        
        Analyzes decision trees to determine which features were most
        important for detecting anomalies.
        
        Returns:
            DataFrame with columns:
                - Feature (str): Feature name
                - Importance (float): Normalized importance score (0 to 1)
            Sorted by importance (descending), or None if not using Isolation Forest
        
        Example:
            >>> detector.run_isolation_forest(contamination=0.1)
            >>> importance = detector.get_feature_importance()
            >>> 
            >>> if importance is not None:
            >>>     st.bar_chart(importance.set_index('Feature'))
        
        Note:
            - Only works with Isolation Forest algorithm
            - Based on frequency of features used in tree splits
            - Higher values = more important for anomaly detection
            - Returns None if another algorithm was used last
        """
        # Lazy load IsolationForest for type checking
        IsolationForest = self._lazy_load_model('IsolationForest')
        if IsolationForest is None or self.model is None:
            return None
        
        if not isinstance(self.model, IsolationForest):
            return None
        
        try:
            # Calculate feature importance from tree splits
            feature_importances = np.zeros(len(self.features.columns))
            
            for tree in self.model.estimators_:
                # Count feature usage in splits
                feature_importances += np.bincount(
                    tree.tree_.feature[tree.tree_.feature >= 0],
                    minlength=len(self.features.columns)
                )
            
            # Normalize to sum to 1
            feature_importances = feature_importances / feature_importances.sum()
            
            # Create sorted dataframe
            importance_df = pd.DataFrame({
                'Feature': self.features.columns,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        finally:
            # Cleanup
            gc.collect()
    
    def get_anomaly_profiles(self, top_n: int = 5) -> pd.DataFrame:
        """Get statistical profiles of top N anomalies compared to normal data.
        
        Analyzes how anomalies differ from normal data across all features,
        measuring deviation in standard deviation units.
        
        Args:
            top_n: Number of top anomalies to profile
        
        Returns:
            DataFrame with one row per anomaly containing:
                - Index: Original dataframe index
                - Anomaly_Score: Detection score
                - {feature}_value: Actual value for each feature
                - {feature}_normal_mean: Mean of normal data
                - {feature}_deviation: Z-score (std devs from normal mean)
        
        Raises:
            ValueError: If anomaly detection hasn't been run
        
        Example:
            >>> profiles = detector.get_anomaly_profiles(top_n=10)
            >>> 
            >>> # Find which feature deviates most
            >>> deviation_cols = [c for c in profiles.columns if '_deviation' in c]
            >>> max_deviation = profiles[deviation_cols].abs().max(axis=1)
            >>> print(f"Max deviation: {max_deviation.max():.2f} standard deviations")
        
        Note:
            - Helps understand WHY points are anomalies
            - Deviation of 2-3 = moderately unusual
            - Deviation >5 = very unusual
            - Compares only to non-anomalous (normal) data
        """
        if self.anomaly_results is None:
            raise ValueError("Anomaly detection must be run first")
        
        # Split data into normal and anomalous
        normal_data = self.features[~self.anomaly_results['is_anomaly']]
        anomalies = self.features[self.anomaly_results['is_anomaly']]
        
        # Get top N anomalies by score (lowest scores)
        top_anomalies = self.anomaly_results[
            self.anomaly_results['is_anomaly']
        ].nsmallest(
            min(top_n, sum(self.anomaly_results['is_anomaly'])),
            'anomaly_score'
        )
        
        # Calculate statistics for each anomaly
        profiles = []
        for idx in top_anomalies.index:
            profile = {
                'Index': idx,
                'Anomaly_Score': self.anomaly_results.loc[idx, 'anomaly_score']
            }
            
            # Compare each feature to normal distribution
            for col in self.features.columns:
                anomaly_value = self.features.loc[idx, col]
                normal_mean = normal_data[col].mean()
                normal_std = normal_data[col].std()
                
                profile[f'{col}_value'] = anomaly_value
                profile[f'{col}_normal_mean'] = normal_mean
                profile[f'{col}_deviation'] = (
                    (anomaly_value - normal_mean) / normal_std 
                    if normal_std > 0 else 0
                )
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def create_2d_scatter(
        self, 
        use_pca: bool = False, 
        show_only_anomalies: bool = True
    ) -> go.Figure:
        """Create interactive 2D scatter plot visualizing detected anomalies.
        
        Visualizes anomalies in 2D space using either first two features or
        PCA for dimensionality reduction.
        
        Args:
            use_pca: If True, use PCA to reduce all features to 2D
                    If False and >2 features, PCA is used automatically
            show_only_anomalies: If True, show only anomaly points for clarity
                                If False, show all points (slower with large data)
        
        Returns:
            Plotly Figure object with interactive scatter plot
        
        Example:
            >>> # Show only anomalies with PCA
            >>> fig = detector.create_2d_scatter(use_pca=True, show_only_anomalies=True)
            >>> st.plotly_chart(fig, use_container_width=True)
            >>> 
            >>> # Show all points without PCA (for 2D data)
            >>> fig = detector.create_2d_scatter(use_pca=False, show_only_anomalies=False)
        
        Raises:
            ValueError: If anomaly detection hasn't been run
        
        Note:
            - Red points = anomalies, Blue points = normal
            - Hover shows anomaly score
            - PCA shows explained variance percentage
            - show_only_anomalies=True recommended for >1000 points
        """
        if self.anomaly_results is None:
            raise ValueError("Anomaly detection must be run first")
        
        # Filter to anomalies only if requested (better performance)
        if show_only_anomalies:
            sample_results = self.anomaly_results[
                self.anomaly_results['anomaly_type'] == 'Anomaly'
            ]
            sample_indices = sample_results.index
            sample_features = self.scaled_features.loc[sample_indices]
        else:
            sample_results = self.anomaly_results
            sample_features = self.scaled_features
        
        # Determine coordinates (PCA or first 2 features)
        if use_pca or len(self.features.columns) > 2:
            # Lazy load PCA
            PCA = self._lazy_load_model('PCA')
            if PCA is None:
                raise ImportError("sklearn not available")
            
            try:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=2)
                coords = pca.fit_transform(sample_features)
                x_label = f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
                y_label = f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            finally:
                gc.collect()
        else:
            # Use first two features directly
            coords = sample_features.iloc[:, :2].values
            x_label = self.features.columns[0]
            y_label = self.features.columns[1]
        
        # Create scatter plot
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=sample_results['anomaly_type'],
            color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
            title=f'🔴 Detected Anomalies ({len(sample_results):,} anomalies shown)',
            labels={'x': x_label, 'y': y_label},
            hover_data={'Anomaly Score': sample_results['anomaly_score']}
        )
        
        # Customize appearance
        fig.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='darkred')),
            hovertemplate='<b>Anomaly</b><br>' + 
                         x_label + ': %{x:.2f}<br>' + 
                         y_label + ': %{y:.2f}<br>' +
                         'Score: %{customdata[0]:.3f}<extra></extra>'
        )
        fig.update_layout(
            height=600, 
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def get_summary_stats(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics of anomaly detection results.
        
        Calculates counts and percentages for anomalies and normal points.
        
        Returns:
            Dictionary containing:
                - total_records (int): Total number of data points
                - num_anomalies (int): Count of detected anomalies
                - pct_anomalies (float): Percentage of anomalies
                - num_normal (int): Count of normal points
                - pct_normal (float): Percentage of normal points
        
        Raises:
            ValueError: If anomaly detection hasn't been run
        
        Example:
            >>> stats = detector.get_summary_stats()
            >>> 
            >>> col1, col2 = st.columns(2)
            >>> col1.metric("Total Records", f"{stats['total_records']:,}")
            >>> col2.metric("Anomalies", f"{stats['num_anomalies']:,} ({stats['pct_anomalies']:.1f}%)")
        
        Note:
            - Percentages sum to 100%
            - Useful for dashboard displays
            - Works with all algorithms
        """
        if self.anomaly_results is None:
            raise ValueError("Anomaly detection must be run first")
        
        total_records = len(self.anomaly_results)
        num_anomalies = sum(self.anomaly_results['is_anomaly'])
        pct_anomalies = (num_anomalies / total_records) * 100
        
        return {
            'total_records': total_records,
            'num_anomalies': num_anomalies,
            'pct_anomalies': pct_anomalies,
            'num_normal': total_records - num_anomalies,
            'pct_normal': 100 - pct_anomalies
        }
