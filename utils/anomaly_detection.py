"""
Anomaly Detection utilities using multiple algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go

class AnomalyDetector:
    """Handles anomaly detection using multiple algorithms."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AnomalyDetector with a dataframe.
        
        Args:
            df: Input dataframe
        """
        self.df = df.copy()
        self.features = None
        self.scaled_features = None
        self.scaler = StandardScaler()
        self.model = None
        self.anomaly_results = None
    
    def set_features(self, feature_cols: List[str]) -> pd.DataFrame:
        """
        Set and scale the feature columns for anomaly detection.
        
        Args:
            feature_cols: List of column names to use as features
            
        Returns:
            DataFrame with selected features
        """
        # Validate columns
        for col in feature_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        # Select features
        self.features = self.df[feature_cols].copy()
        
        # Handle missing values
        if self.features.isnull().any().any():
            self.features = self.features.fillna(self.features.mean())
        
        # Scale features
        self.scaled_features = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        return self.features
    
    def run_isolation_forest(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Run Isolation Forest algorithm for anomaly detection.
        
        Args:
            contamination: Expected proportion of outliers (0.01 to 0.5)
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Initialize and fit model
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for outliers, 1 for inliers)
        predictions = self.model.fit_predict(self.scaled_features)
        
        # Get anomaly scores (lower is more anomalous)
        scores = self.model.score_samples(self.scaled_features)
        
        # Create results dataframe
        results_df = self.df.copy()
        results_df['anomaly_score'] = scores
        results_df['is_anomaly'] = predictions == -1
        results_df['anomaly_type'] = results_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'})
        
        self.anomaly_results = results_df
        return results_df
    
    def run_local_outlier_factor(self, contamination: float = 0.1, n_neighbors: int = 20) -> pd.DataFrame:
        """
        Run Local Outlier Factor (LOF) algorithm for anomaly detection.
        
        Args:
            contamination: Expected proportion of outliers (0.01 to 0.5)
            n_neighbors: Number of neighbors to use
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Initialize and fit model
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=False
        )
        
        # Predict anomalies (-1 for outliers, 1 for inliers)
        predictions = self.model.fit_predict(self.scaled_features)
        
        # Get negative outlier factor (lower is more anomalous)
        scores = self.model.negative_outlier_factor_
        
        # Create results dataframe
        results_df = self.df.copy()
        results_df['anomaly_score'] = scores
        results_df['is_anomaly'] = predictions == -1
        results_df['anomaly_type'] = results_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'})
        
        self.anomaly_results = results_df
        return results_df
    
    def run_one_class_svm(self, nu: float = 0.1, kernel: str = 'rbf') -> pd.DataFrame:
        """
        Run One-Class SVM algorithm for anomaly detection.
        
        Args:
            nu: Upper bound on fraction of outliers (0.01 to 0.5)
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if self.scaled_features is None:
            raise ValueError("Features must be set first using set_features()")
        
        # Initialize and fit model
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma='auto'
        )
        
        # Fit and predict
        self.model.fit(self.scaled_features)
        predictions = self.model.predict(self.scaled_features)
        
        # Get decision function values (lower is more anomalous)
        scores = self.model.decision_function(self.scaled_features)
        
        # Create results dataframe
        results_df = self.df.copy()
        results_df['anomaly_score'] = scores
        results_df['is_anomaly'] = predictions == -1
        results_df['anomaly_type'] = results_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'})
        
        self.anomaly_results = results_df
        return results_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance for Isolation Forest.
        Only works if Isolation Forest was the last algorithm run.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not isinstance(self.model, IsolationForest):
            return None
        
        # Calculate feature importance based on split counts
        feature_importances = np.zeros(len(self.features.columns))
        
        for tree in self.model.estimators_:
            # Get feature indices used in splits
            feature_importances += np.bincount(
                tree.tree_.feature[tree.tree_.feature >= 0],
                minlength=len(self.features.columns)
            )
        
        # Normalize
        feature_importances = feature_importances / feature_importances.sum()
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_anomaly_profiles(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get profiles of top N anomalies compared to normal data.
        
        Args:
            top_n: Number of top anomalies to profile
            
        Returns:
            DataFrame with anomaly profiles
        """
        if self.anomaly_results is None:
            raise ValueError("Anomaly detection must be run first")
        
        # Get normal and anomaly data
        normal_data = self.features[~self.anomaly_results['is_anomaly']]
        anomalies = self.features[self.anomaly_results['is_anomaly']]
        
        # Get top N anomalies by score
        top_anomalies = self.anomaly_results[self.anomaly_results['is_anomaly']].nsmallest(
            min(top_n, sum(self.anomaly_results['is_anomaly'])),
            'anomaly_score'
        )
        
        # Calculate statistics
        profiles = []
        for idx in top_anomalies.index:
            profile = {
                'Index': idx,
                'Anomaly_Score': self.anomaly_results.loc[idx, 'anomaly_score']
            }
            
            for col in self.features.columns:
                anomaly_value = self.features.loc[idx, col]
                normal_mean = normal_data[col].mean()
                normal_std = normal_data[col].std()
                
                profile[f'{col}_value'] = anomaly_value
                profile[f'{col}_normal_mean'] = normal_mean
                profile[f'{col}_deviation'] = (anomaly_value - normal_mean) / normal_std if normal_std > 0 else 0
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def create_2d_scatter(self, use_pca: bool = False, show_only_anomalies: bool = True) -> go.Figure:
        """
        Create 2D scatter plot of anomalies.
        
        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            show_only_anomalies: If True, only show anomaly points (faster, clearer)
            
        Returns:
            Plotly figure
        """
        if self.anomaly_results is None:
            raise ValueError("Anomaly detection must be run first")
        
        # Filter to show only anomalies for better performance and clarity
        if show_only_anomalies:
            sample_results = self.anomaly_results[self.anomaly_results['anomaly_type'] == 'Anomaly']
            sample_indices = sample_results.index
            sample_features = self.scaled_features.loc[sample_indices]
        else:
            sample_results = self.anomaly_results
            sample_features = self.scaled_features
        
        if use_pca or len(self.features.columns) > 2:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            coords = pca.fit_transform(sample_features)
            x_label = f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
            y_label = f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
        else:
            # Use first two features
            coords = sample_features.iloc[:, :2].values
            x_label = self.features.columns[0]
            y_label = self.features.columns[1]
        
        # Create scatter plot with hover (fewer points = better performance)
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=sample_results['anomaly_type'],
            color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
            title=f'🔴 Detected Anomalies ({len(sample_results):,} anomalies shown)',
            labels={'x': x_label, 'y': y_label},
            hover_data={'Anomaly Score': sample_results['anomaly_score']}
        )
        
        # Enable hover with clean styling (safe with fewer points)
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
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of anomaly detection results.
        
        Returns:
            Dictionary with summary stats
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
