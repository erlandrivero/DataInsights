"""
Churn Prediction Module

Provides comprehensive customer churn prediction with automated feature engineering,
model training, and actionable retention strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


class ChurnPredictor:
    """Predicts customer churn with automated feature engineering and model training."""
    
    def __init__(self):
        """Initialize the ChurnPredictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None
        self.results = {}
        
    def engineer_features(self, data: pd.DataFrame,
                         customer_id_col: str,
                         date_col: str,
                         value_col: Optional[str] = None,
                         churn_col: Optional[str] = None) -> pd.DataFrame:
        """Engineer features for churn prediction from transactional data.
        
        Creates RFM-like features, engagement metrics, and behavioral patterns.
        
        Args:
            data: DataFrame with customer transaction data.
            customer_id_col: Column with customer IDs.
            date_col: Column with transaction dates.
            value_col: Column with transaction values (optional).
            churn_col: Column indicating churn status (if available).
        
        Returns:
            DataFrame with engineered features per customer.
        
        Examples:
            >>> features = predictor.engineer_features(df, 'customer_id', 'date', 'amount')
        """
        # Ensure date column is datetime with error handling
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Unable to parse '{date_col}' as datetime. Please select a valid date column. Error: {str(e)}")
        
        # Check for successful parsing
        if data[date_col].isna().all():
            raise ValueError(f"Column '{date_col}' could not be parsed as dates. Please ensure you've selected a column containing date/datetime values.")
        
        # Drop rows with invalid dates
        valid_rows = data[date_col].notna()
        if valid_rows.sum() < len(data):
            data = data[valid_rows].copy()
        
        # Reference date (most recent transaction)
        max_date = data[date_col].max()
        
        # Group by customer
        customer_features = []
        
        for customer, group in data.groupby(customer_id_col):
            features = {'customer_id': customer}
            
            # Recency (days since last transaction)
            features['recency_days'] = (max_date - group[date_col].max()).days
            
            # Frequency (number of transactions)
            features['frequency'] = len(group)
            
            # Monetary (if value column provided)
            if value_col and value_col in data.columns:
                features['monetary_total'] = group[value_col].sum()
                features['monetary_avg'] = group[value_col].mean()
                features['monetary_std'] = group[value_col].std() if len(group) > 1 else 0
                features['monetary_max'] = group[value_col].max()
                features['monetary_min'] = group[value_col].min()
            
            # Time-based features
            date_range = (group[date_col].max() - group[date_col].min()).days
            features['customer_lifetime_days'] = max(date_range, 1)
            features['avg_days_between_transactions'] = date_range / max(len(group) - 1, 1)
            
            # Engagement trend (comparing recent vs. historical activity)
            mid_point = group[date_col].min() + timedelta(days=date_range/2)
            recent_count = len(group[group[date_col] >= mid_point])
            historical_count = len(group[group[date_col] < mid_point])
            features['engagement_trend'] = recent_count / max(historical_count, 1)
            
            # Activity in last 30/60/90 days
            features['transactions_last_30d'] = len(group[group[date_col] >= max_date - timedelta(days=30)])
            features['transactions_last_60d'] = len(group[group[date_col] >= max_date - timedelta(days=60)])
            features['transactions_last_90d'] = len(group[group[date_col] >= max_date - timedelta(days=90)])
            
            # Declining activity indicator
            features['is_declining'] = 1 if features['engagement_trend'] < 0.5 else 0
            
            # High risk indicators
            features['is_dormant'] = 1 if features['recency_days'] > 90 else 0
            features['is_low_frequency'] = 1 if features['frequency'] < 3 else 0
            
            # Add churn label if provided
            if churn_col and churn_col in data.columns:
                # Take the churn status from any row (should be same for customer)
                features['churned'] = group[churn_col].iloc[0]
            
            customer_features.append(features)
        
        return pd.DataFrame(customer_features)
    
    def train_model(self, features: pd.DataFrame,
                   target_col: str = 'churned',
                   model_type: str = 'random_forest',
                   test_size: float = 0.3,
                   random_state: int = 42) -> Dict[str, Any]:
        """Train churn prediction model.
        
        Args:
            features: DataFrame with engineered features.
            target_col: Column with churn labels (0/1).
            model_type: Model to use ('random_forest', 'gradient_boosting', 'logistic').
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
        
        Returns:
            Dictionary with training results and metrics.
        
        Examples:
            >>> results = predictor.train_model(features, 'churned', 'random_forest')
        """
        # Separate features and target
        X = features.drop(['customer_id', target_col], axis=1)
        y = features[target_col]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select and train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        results = {
            'model_type': model_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'churn_rate_train': y_train.mean(),
            'churn_rate_test': y_test.mean()
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.results = results
        return results
    
    def predict_churn_risk(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict churn risk for customers.
        
        Args:
            features: DataFrame with customer features.
        
        Returns:
            DataFrame with customer IDs and churn risk scores.
        
        Examples:
            >>> predictions = predictor.predict_churn_risk(new_customers)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract customer IDs
        customer_ids = features['customer_id'].values
        
        # Prepare features
        X = features.drop(['customer_id'], axis=1)
        if 'churned' in X.columns:
            X = X.drop('churned', axis=1)
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        churn_probability = self.model.predict_proba(X_scaled)[:, 1]
        churn_prediction = self.model.predict(X_scaled)
        
        # Risk categorization
        risk_category = pd.cut(
            churn_probability,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Results
        predictions = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'risk_category': risk_category
        })
        
        return predictions.sort_values('churn_probability', ascending=False)
    
    def get_retention_strategies(self, predictions: pd.DataFrame,
                                features: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate personalized retention strategies based on churn risk.
        
        Args:
            predictions: DataFrame with churn predictions.
            features: DataFrame with customer features.
        
        Returns:
            Dictionary mapping risk categories to customer recommendations.
        
        Examples:
            >>> strategies = predictor.get_retention_strategies(predictions, features)
        """
        # Merge predictions with features
        customer_data = predictions.merge(
            features,
            on='customer_id',
            how='left'
        )
        
        strategies = {
            'High Risk': [],
            'Medium Risk': [],
            'Low Risk': []
        }
        
        for risk in ['High Risk', 'Medium Risk', 'Low Risk']:
            risk_customers = customer_data[customer_data['risk_category'] == risk]
            
            if len(risk_customers) == 0:
                continue
            
            # Analyze patterns in this risk group
            avg_recency = risk_customers['recency_days'].mean()
            avg_frequency = risk_customers['frequency'].mean()
            dormant_pct = risk_customers['is_dormant'].mean() * 100
            declining_pct = risk_customers['is_declining'].mean() * 100
            
            strategy = {
                'risk_level': risk,
                'customer_count': len(risk_customers),
                'avg_churn_probability': risk_customers['churn_probability'].mean(),
                'characteristics': {
                    'avg_recency_days': avg_recency,
                    'avg_frequency': avg_frequency,
                    'pct_dormant': dormant_pct,
                    'pct_declining': declining_pct
                },
                'recommended_actions': []
            }
            
            # Generate recommendations based on patterns
            if risk == 'High Risk':
                if avg_recency > 60:
                    strategy['recommended_actions'].append({
                        'action': 'Win-Back Campaign',
                        'reason': f'{dormant_pct:.0f}% are dormant (no activity >90 days)',
                        'tactics': ['Special discount offer', 'Personalized email', 'Phone outreach']
                    })
                if declining_pct > 50:
                    strategy['recommended_actions'].append({
                        'action': 'Re-Engagement Program',
                        'reason': f'{declining_pct:.0f}% show declining engagement',
                        'tactics': ['Product recommendations', 'Loyalty rewards', 'Feature education']
                    })
                strategy['recommended_actions'].append({
                    'action': 'Priority Retention',
                    'reason': 'Highest churn risk group',
                    'tactics': ['Dedicated account manager', 'VIP treatment', 'Exit survey']
                })
            
            elif risk == 'Medium Risk':
                if avg_frequency < 5:
                    strategy['recommended_actions'].append({
                        'action': 'Engagement Boost',
                        'reason': 'Low transaction frequency',
                        'tactics': ['Cross-sell campaign', 'Usage tips', 'Community engagement']
                    })
                strategy['recommended_actions'].append({
                    'action': 'Preventive Monitoring',
                    'reason': 'At-risk but salvageable',
                    'tactics': ['Monthly check-ins', 'Satisfaction surveys', 'Early warning alerts']
                })
            
            else:  # Low Risk
                strategy['recommended_actions'].append({
                    'action': 'Maintain Satisfaction',
                    'reason': 'Low churn risk, keep engaged',
                    'tactics': ['Regular newsletters', 'Exclusive previews', 'Referral program']
                })
            
            strategies[risk] = [strategy]
        
        return strategies
    
    @staticmethod
    def create_feature_importance_plot(feature_importance: pd.DataFrame,
                                      top_n: int = 10) -> go.Figure:
        """Create feature importance visualization.
        
        Args:
            feature_importance: DataFrame with features and importance scores.
            top_n: Number of top features to display.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = ChurnPredictor.create_feature_importance_plot(importance)
        """
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='indianred'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Churn Drivers',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_risk_distribution_plot(predictions: pd.DataFrame) -> go.Figure:
        """Create churn risk distribution visualization.
        
        Args:
            predictions: DataFrame with churn predictions.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = ChurnPredictor.create_risk_distribution_plot(predictions)
        """
        risk_counts = predictions['risk_category'].value_counts()
        
        colors = {
            'High Risk': '#d62728',
            'Medium Risk': '#ff7f0e',
            'Low Risk': '#2ca02c'
        }
        
        fig = go.Figure(go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[colors.get(cat, 'gray') for cat in risk_counts.index],
            text=risk_counts.values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Customer Distribution by Churn Risk',
            xaxis_title='Risk Category',
            yaxis_title='Number of Customers',
            height=400
        )
        
        return fig
