"""
RFM Analysis and Customer Segmentation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class RFMAnalyzer:
    """Handles RFM Analysis and Customer Segmentation."""
    
    def __init__(self):
        self.rfm_data = None
        self.scaled_data = None
        self.clusters = None
        self.scaler = StandardScaler()
    
    @staticmethod
    def calculate_rfm(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str, 
                      reference_date: datetime = None) -> pd.DataFrame:
        """
        Calculate RFM metrics from transactional data.
        
        Args:
            df: DataFrame with transaction data
            customer_col: Customer ID column name
            date_col: Transaction date column name
            amount_col: Transaction amount column name
            reference_date: Reference date for recency (default: max date + 1 day)
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set reference date
        if reference_date is None:
            reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_col).agg({
            date_col: [
                ('Recency', lambda x: (reference_date - x.max()).days),  # Recency
                ('Frequency', 'count')  # Frequency (count of transactions)
            ],
            amount_col: ('Monetary', 'sum')  # Monetary
        }).reset_index()
        
        # Flatten column names
        rfm.columns = [customer_col, 'Recency', 'Frequency', 'Monetary']
        
        return rfm
    
    @staticmethod
    def score_rfm(rfm: pd.DataFrame, method: str = 'quartile') -> pd.DataFrame:
        """
        Score RFM metrics using quartiles or quantiles.
        
        Args:
            rfm: DataFrame with RFM metrics
            method: 'quartile' (1-4) or 'quintile' (1-5)
            
        Returns:
            DataFrame with RFM scores
        """
        rfm_scored = rfm.copy()
        
        # Determine number of bins
        n_bins = 4 if method == 'quartile' else 5
        
        # Score Recency (lower is better, so reverse labels)
        rfm_scored['R_Score'] = pd.qcut(
            rfm['Recency'], 
            q=n_bins, 
            labels=range(n_bins, 0, -1),
            duplicates='drop'
        ).astype(int)
        
        # Score Frequency (higher is better)
        rfm_scored['F_Score'] = pd.qcut(
            rfm['Frequency'], 
            q=n_bins, 
            labels=range(1, n_bins + 1),
            duplicates='drop'
        ).astype(int)
        
        # Score Monetary (higher is better)
        rfm_scored['M_Score'] = pd.qcut(
            rfm['Monetary'], 
            q=n_bins, 
            labels=range(1, n_bins + 1),
            duplicates='drop'
        ).astype(int)
        
        # Calculate RFM Score (concatenated)
        rfm_scored['RFM_Score'] = (
            rfm_scored['R_Score'].astype(str) + 
            rfm_scored['F_Score'].astype(str) + 
            rfm_scored['M_Score'].astype(str)
        )
        
        # Calculate RFM Total (summed)
        rfm_scored['RFM_Total'] = (
            rfm_scored['R_Score'] + 
            rfm_scored['F_Score'] + 
            rfm_scored['M_Score']
        )
        
        return rfm_scored
    
    def perform_kmeans_clustering(self, rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform K-Means clustering on RFM data.
        
        Args:
            rfm: DataFrame with RFM metrics
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        # Store original data
        self.rfm_data = rfm.copy()
        
        # Select RFM columns for clustering
        rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Scale features
        self.scaled_data = self.scaler.fit_transform(rfm_features)
        
        # Perform K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.scaled_data)
        
        # Add cluster assignments
        rfm_clustered = rfm.copy()
        rfm_clustered['Cluster'] = self.clusters
        
        return rfm_clustered
    
    def calculate_elbow_curve(self, rfm: pd.DataFrame, max_clusters: int = 10) -> Tuple[List[int], List[float]]:
        """
        Calculate inertia for elbow method.
        
        Args:
            rfm: DataFrame with RFM metrics
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Tuple of (cluster_range, inertias)
        """
        rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
        scaled_data = self.scaler.fit_transform(rfm_features)
        
        cluster_range = range(2, max_clusters + 1)
        inertias = []
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        return list(cluster_range), inertias
    
    @staticmethod
    def segment_customers(rfm_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers based on RFM scores.
        
        Args:
            rfm_scored: DataFrame with RFM scores
            
        Returns:
            DataFrame with customer segments
        """
        rfm_segmented = rfm_scored.copy()
        
        def assign_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # Champions
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            # Loyal Customers
            elif r >= 3 and f >= 4:
                return 'Loyal Customers'
            # Potential Loyalists
            elif r >= 4 and f >= 2 and f <= 3:
                return 'Potential Loyalists'
            # New Customers
            elif r >= 4 and f == 1:
                return 'New Customers'
            # Promising
            elif r >= 3 and f == 1:
                return 'Promising'
            # Need Attention
            elif r >= 2 and r <= 3 and f >= 2 and f <= 3:
                return 'Need Attention'
            # About to Sleep
            elif r >= 2 and r <= 3 and f <= 2:
                return 'About to Sleep'
            # At Risk
            elif r <= 2 and f >= 3:
                return 'At Risk'
            # Cannot Lose Them
            elif r <= 1 and f >= 4:
                return 'Cannot Lose Them'
            # Hibernating
            elif r <= 2 and f <= 2:
                return 'Hibernating'
            # Lost
            else:
                return 'Lost'
        
        rfm_segmented['Segment'] = rfm_segmented.apply(assign_segment, axis=1)
        
        return rfm_segmented
    
    @staticmethod
    def get_segment_profiles(rfm_segmented: pd.DataFrame, customer_col: str = None) -> pd.DataFrame:
        """
        Get aggregate statistics for each segment.
        
        Args:
            rfm_segmented: DataFrame with customer segments
            customer_col: Customer ID column name
            
        Returns:
            DataFrame with segment profiles
        """
        if customer_col:
            group_col = 'Segment'
            agg_dict = {
                customer_col: 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean'
            }
        else:
            group_col = 'Segment'
            agg_dict = {
                'Recency': ['count', 'mean'],
                'Frequency': 'mean',
                'Monetary': 'mean',
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean'
            }
        
        profiles = rfm_segmented.groupby(group_col).agg(agg_dict).reset_index()
        
        if customer_col:
            profiles.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 
                              'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']
        else:
            profiles.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 
                              'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']
        
        # Sort by customer count
        profiles = profiles.sort_values('Customer_Count', ascending=False)
        
        return profiles
    
    @staticmethod
    def create_rfm_scatter_3d(rfm: pd.DataFrame, color_col: str = 'Cluster') -> go.Figure:
        """Create 3D scatter plot of RFM metrics."""
        fig = px.scatter_3d(
            rfm,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color=color_col,
            title='3D RFM Customer Segmentation',
            labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency (transactions)', 
                   'Monetary': 'Monetary (value)'},
            height=600
        )
        
        fig.update_traces(marker=dict(size=5))
        
        return fig
    
    @staticmethod
    def create_elbow_plot(cluster_range: List[int], inertias: List[float]) -> go.Figure:
        """Create elbow curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cluster_range,
            y=inertias,
            mode='lines+markers',
            marker=dict(size=10, color='blue'),
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Elbow Method - Optimal Number of Clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Inertia (Within-Cluster Sum of Squares)',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_segment_distribution(rfm_segmented: pd.DataFrame) -> go.Figure:
        """Create bar chart of segment distribution."""
        segment_counts = rfm_segmented['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig = px.bar(
            segment_counts,
            x='Segment',
            y='Count',
            title='Customer Segment Distribution',
            labels={'Count': 'Number of Customers', 'Segment': 'Customer Segment'},
            color='Count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    @staticmethod
    def generate_segment_insights(rfm_segmented: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate business insights for each segment."""
        insights = {
            'Champions': [
                '🏆 Your best customers - high value, frequent purchases, recent activity',
                '💡 Strategy: Reward with exclusive offers, VIP programs, early access',
                '📈 Action: Increase engagement with premium products, loyalty rewards',
                '⚠️ Risk: Low - but monitor for satisfaction'
            ],
            'Loyal Customers': [
                '💎 Reliable revenue generators with consistent purchase patterns',
                '💡 Strategy: Upsell premium products, create loyalty programs',
                '📈 Action: Personalized recommendations, special discounts',
                '⚠️ Risk: Low - maintain relationship'
            ],
            'Potential Loyalists': [
                '🌱 Recent customers with growing engagement',
                '💡 Strategy: Build relationship through personalized communication',
                '📈 Action: Offer incentives for repeat purchases, membership programs',
                '⚠️ Risk: Medium - nurture carefully'
            ],
            'New Customers': [
                '✨ Recently acquired - need onboarding and engagement',
                '💡 Strategy: Welcome campaigns, product education, support',
                '📈 Action: Quick wins with easy purchases, build trust',
                '⚠️ Risk: High - critical first impression period'
            ],
            'Promising': [
                '🎯 Potential for growth with right engagement',
                '💡 Strategy: Targeted offers based on purchase history',
                '📈 Action: Introduce complementary products, limited-time offers',
                '⚠️ Risk: Medium - opportunity to convert'
            ],
            'Need Attention': [
                '⚡ Declining engagement - requires intervention',
                '💡 Strategy: Re-engagement campaigns, feedback surveys',
                '📈 Action: Special offers, remind of value, address issues',
                '⚠️ Risk: High - at risk of churning'
            ],
            'About to Sleep': [
                '😴 Showing signs of disengagement',
                '💡 Strategy: Win-back campaigns, limited-time offers',
                '📈 Action: Aggressive discounts, new product highlights',
                '⚠️ Risk: Very High - last chance to retain'
            ],
            'At Risk': [
                '🚨 Previously valuable but now inactive',
                '💡 Strategy: Urgent win-back with significant incentives',
                '📈 Action: Survey for issues, personalized recovery offers',
                '⚠️ Risk: Critical - immediate action needed'
            ],
            'Cannot Lose Them': [
                '⛔ High-value customers who have gone quiet',
                '💡 Strategy: Priority re-engagement with executive involvement',
                '📈 Action: Direct outreach, VIP recovery offers, problem solving',
                '⚠️ Risk: Critical - highest priority to recover'
            ],
            'Hibernating': [
                '💤 Long inactive with low past engagement',
                '💡 Strategy: Low-cost reactivation attempts',
                '📈 Action: Automated win-back emails, surveys',
                '⚠️ Risk: High - likely churned'
            ],
            'Lost': [
                '❌ Churned customers - minimal recovery potential',
                '💡 Strategy: Learn from loss, minimal investment',
                '📈 Action: Exit surveys, passive win-back',
                '⚠️ Risk: Very High - focus on learning'
            ]
        }
        
        return insights
