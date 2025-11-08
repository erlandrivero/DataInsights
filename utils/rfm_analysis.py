"""RFM Analysis and Customer Segmentation Utilities.

This module provides comprehensive tools for performing RFM (Recency, Frequency, Monetary)
analysis and customer segmentation. It implements industry-standard RFM scoring, K-Means
clustering, and rule-based customer segmentation with detailed business insights.

Typical usage example:
    analyzer = RFMAnalyzer()
    rfm_data = analyzer.calculate_rfm(df, 'CustomerID', 'InvoiceDate', 'TotalAmount')
    rfm_scored = analyzer.score_rfm(rfm_data, method='quartile')
    rfm_segmented = analyzer.segment_customers(rfm_scored)
    profiles = analyzer.get_segment_profiles(rfm_segmented, 'CustomerID')

Attributes:
    rfm_data (Optional[pd.DataFrame]): Original RFM metrics data.
    scaled_data (Optional[np.ndarray]): Scaled features for clustering.
    clusters (Optional[np.ndarray]): Cluster assignments from K-Means.
    scaler (StandardScaler): Scaler for feature normalization.

"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Lazy loading for sklearn
from utils.lazy_loader import LazyModuleLoader

class RFMAnalyzer:
    """Handles RFM Analysis and Customer Segmentation.
    
    This class provides a complete toolkit for RFM (Recency, Frequency, Monetary) analysis,
    including RFM calculation, scoring methods, K-Means clustering, rule-based segmentation,
    and comprehensive visualizations.
    
    Attributes:
        rfm_data (Optional[pd.DataFrame]): Original RFM metrics data.
        scaled_data (Optional[np.ndarray]): Scaled features for clustering.
        clusters (Optional[np.ndarray]): Cluster assignments from K-Means.
        scaler (StandardScaler): Scaler for feature normalization.
    
    Examples:
        >>> analyzer = RFMAnalyzer()
        >>> rfm = analyzer.calculate_rfm(transactions_df, 'CustomerID', 'Date', 'Amount')
        >>> scored = analyzer.score_rfm(rfm, method='quartile')
        >>> segmented = analyzer.segment_customers(scored)
        >>> print(segmented[['CustomerID', 'Segment']].head())
    
    Notes:
        - Recency: Days since last purchase (lower is better)
        - Frequency: Number of transactions (higher is better)
        - Monetary: Total spending (higher is better)
        - Supports both quartile (1-4) and quintile (1-5) scoring
    """
    
    def __init__(self):
        self.rfm_data = None
        self.scaled_data = None
        self.clusters = None
        # Lazy load StandardScaler when needed
        self.scaler = None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def calculate_rfm(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str, 
                      reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) metrics from transactional data.
        
        This method aggregates transaction-level data into customer-level RFM metrics.
        Recency represents days since last purchase, Frequency counts transactions,
        and Monetary sums transaction amounts.
        
        Args:
            df (pd.DataFrame): DataFrame containing transaction data with customer IDs,
                dates, and amounts.
            customer_col (str): Name of the column containing customer identifiers.
            date_col (str): Name of the column containing transaction dates.
            amount_col (str): Name of the column containing transaction amounts.
            reference_date (Optional[datetime]): Reference date for calculating recency.
                If None, uses the maximum date in the dataset plus 1 day.
                Default is None.
            
        Returns:
            pd.DataFrame: DataFrame with one row per customer containing:
                - customer_col: Customer identifier
                - Recency: Days since last purchase
                - Frequency: Number of transactions
                - Monetary: Total transaction amount
        
        Examples:
            >>> transactions = pd.DataFrame({
            ...     'CustomerID': [1, 1, 2, 2, 3],
            ...     'Date': ['2024-01-01', '2024-01-15', '2024-01-10', '2024-01-20', '2024-01-05'],
            ...     'Amount': [100, 150, 200, 300, 50]
            ... })
            >>> rfm = RFMAnalyzer.calculate_rfm(
            ...     transactions, 'CustomerID', 'Date', 'Amount'
            ... )
            >>> print(rfm.columns)
            Index(['CustomerID', 'Recency', 'Frequency', 'Monetary'], dtype='object')
        
        Notes:
            - Date column must be convertible to datetime format
            - Recency is calculated in days (integer)
            - Frequency is a count of transactions (not unique days)
            - Monetary is the sum of all transaction amounts
            - Missing values in amount column will affect monetary calculations
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set reference date
        if reference_date is None:
            reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM metrics using separate aggregations
        recency = df.groupby(customer_col)[date_col].apply(
            lambda x: (reference_date - x.max()).days
        ).reset_index()
        recency.columns = [customer_col, 'Recency']
        
        frequency = df.groupby(customer_col).size().reset_index()
        frequency.columns = [customer_col, 'Frequency']
        
        monetary = df.groupby(customer_col)[amount_col].sum().reset_index()
        monetary.columns = [customer_col, 'Monetary']
        
        # Merge all metrics
        rfm = recency.merge(frequency, on=customer_col).merge(monetary, on=customer_col)
        
        return rfm
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def score_rfm(rfm: pd.DataFrame, method: str = 'quartile') -> pd.DataFrame:
        """Score RFM metrics using quartile or quintile binning.
        
        This method assigns numerical scores (1-4 for quartile, 1-5 for quintile) to each
        RFM metric. Recency scores are reversed (lower days = higher score) since recent
        customers are more valuable. Frequency and Monetary scores are direct (higher = better).
        
        Args:
            rfm (pd.DataFrame): DataFrame containing Recency, Frequency, and Monetary columns.
            method (str): Scoring method - either 'quartile' for 1-4 scores or 'quintile'
                for 1-5 scores. Default is 'quartile'.
            
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - R_Score: Recency score (4/5 = most recent)
                - F_Score: Frequency score (4/5 = highest frequency)
                - M_Score: Monetary score (4/5 = highest value)
                - RFM_Score: Concatenated string score (e.g., '444')
                - RFM_Total: Sum of individual scores (e.g., 12)
        
        Examples:
            >>> rfm_data = pd.DataFrame({
            ...     'CustomerID': [1, 2, 3, 4],
            ...     'Recency': [5, 30, 60, 90],
            ...     'Frequency': [10, 5, 2, 1],
            ...     'Monetary': [1000, 500, 200, 100]
            ... })
            >>> scored = RFMAnalyzer.score_rfm(rfm_data, method='quartile')
            >>> print(scored[['R_Score', 'F_Score', 'M_Score', 'RFM_Score']].head())
        
        Notes:
            - Uses pd.qcut for equal-frequency binning
            - Handles duplicate bin edges with duplicates='drop'
            - RFM_Score is string concatenation for pattern matching
            - RFM_Total is numeric sum for ranking
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
        """Perform K-Means clustering on RFM metrics for customer segmentation.
        
        This method applies K-Means clustering to scaled RFM features to identify
        natural customer segments. Features are standardized using StandardScaler
        before clustering to ensure equal weighting.
        
        Args:
            rfm (pd.DataFrame): DataFrame containing Recency, Frequency, and Monetary columns.
            n_clusters (int): Number of customer segments to create. Default is 4.
                Typical values: 3-6 clusters.
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'Cluster' column containing
                cluster assignments (0 to n_clusters-1).
        
        Examples:
            >>> analyzer = RFMAnalyzer()
            >>> rfm_clustered = analyzer.perform_kmeans_clustering(rfm_data, n_clusters=5)
            >>> print(rfm_clustered['Cluster'].value_counts())
        
        Notes:
            - Uses StandardScaler for feature normalization
            - K-Means parameters: random_state=42, n_init=10
            - Stores scaled_data and clusters as instance attributes
            - For optimal k, use calculate_elbow_curve() first
        """
        # Store original data
        self.rfm_data = rfm.copy()
        
        # Select RFM columns
        # Lazy load sklearn modules
        preprocessing = LazyModuleLoader.load_module('sklearn.preprocessing')
        cluster = LazyModuleLoader.load_module('sklearn.cluster')
        
        StandardScaler = getattr(preprocessing, 'StandardScaler')
        KMeans = getattr(cluster, 'KMeans')
        
        # Scale features
        rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].values
        self.scaler = StandardScaler()
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
        # Lazy load KMeans
        cluster = LazyModuleLoader.load_module('sklearn.cluster')
        KMeans = getattr(cluster, 'KMeans')
        
        inertias = []
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        return list(cluster_range), inertias
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def segment_customers(rfm_scored: pd.DataFrame) -> pd.DataFrame:
        """Assign customers to business segments using rule-based RFM scoring.
        
        This method implements industry-standard RFM segmentation logic to classify
        customers into 11 distinct business segments (Champions, Loyal, At Risk, etc.)
        based on their R_Score, F_Score, and M_Score values.
        
        Args:
            rfm_scored (pd.DataFrame): DataFrame containing R_Score, F_Score, and M_Score
                columns (typically from score_rfm() method).
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'Segment' column containing
                one of 11 segment labels:
                - Champions: Best customers (high RFM)
                - Loyal Customers: Reliable high-value customers
                - Potential Loyalists: Recent, growing engagement
                - New Customers: Recently acquired
                - Promising: Potential for growth
                - Need Attention: Declining engagement
                - About to Sleep: Showing disengagement
                - At Risk: Previously valuable, now inactive
                - Cannot Lose Them: High-value customers at risk
                - Hibernating: Long inactive, low engagement
                - Lost: Churned customers
        
        Examples:
            >>> scored_rfm = RFMAnalyzer.score_rfm(rfm_data)
            >>> segmented = RFMAnalyzer.segment_customers(scored_rfm)
            >>> print(segmented['Segment'].value_counts())
            Champions             1250
            Loyal Customers        980
            At Risk                456
            ...
        
        Notes:
            - Assumes quartile scoring (1-4 scale)
            - Segments are mutually exclusive
            - Logic prioritizes high-value at-risk customers
            - Use get_segment_profiles() for segment analysis
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
    @st.cache_data(ttl=1800)
    def get_segment_profiles(rfm_segmented: pd.DataFrame, customer_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate aggregate statistics and profiles for each customer segment.
        
        This method computes mean RFM values and customer counts for each segment,
        providing a high-level overview of segment characteristics.
        
        Args:
            rfm_segmented (pd.DataFrame): DataFrame with 'Segment' column and RFM metrics.
            customer_col (Optional[str]): Name of customer ID column for counting unique
                customers. If None, counts rows instead. Default is None.
            
        Returns:
            pd.DataFrame: Segment profile summary with columns:
                - Segment: Segment name
                - Customer_Count: Number of customers in segment
                - Avg_Recency: Mean recency (days)
                - Avg_Frequency: Mean frequency (transactions)
                - Avg_Monetary: Mean monetary value
                - Avg_R_Score: Mean R score
                - Avg_F_Score: Mean F score  
                - Avg_M_Score: Mean M score
                Sorted by Customer_Count descending.
        
        Examples:
            >>> profiles = RFMAnalyzer.get_segment_profiles(segmented_df, 'CustomerID')
            >>> print(profiles[['Segment', 'Customer_Count', 'Avg_Monetary']].head())
        
        Notes:
            - All averages are calculated using mean()
            - Results sorted by customer count (largest segments first)
            - Useful for executive summaries and segment comparison
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
            # When no customer column, use Segment as the grouping column
            # and count the number of rows in each segment
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
            # Flatten MultiIndex columns when customer_col is None
            profiles.columns = profiles.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
            # Rename columns to match expected format
            profiles.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency', 
                              'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']
        
        # Sort by customer count
        profiles = profiles.sort_values('Customer_Count', ascending=False)
        
        return profiles
    
    @staticmethod
    def create_rfm_scatter_3d(rfm: pd.DataFrame, color_col: str = 'Cluster') -> go.Figure:
        """Create interactive 3D scatter plot visualizing RFM customer segments.
        
        This visualization displays customers in 3D space with Recency, Frequency,
        and Monetary as axes, colored by segment or cluster assignment.
        
        Args:
            rfm (pd.DataFrame): DataFrame with Recency, Frequency, Monetary columns.
            color_col (str): Column name to use for color coding points. Typically
                'Cluster' or 'Segment'. Default is 'Cluster'.
            
        Returns:
            go.Figure: Plotly 3D scatter plot figure object, ready to display.
        
        Examples:
            >>> fig = RFMAnalyzer.create_rfm_scatter_3d(rfm_clustered, color_col='Cluster')
            >>> fig.show()
        
        Notes:
            - Height fixed at 600px
            - Marker size: 5
            - Fully interactive (rotate, zoom, pan)
            - Axes labeled with units
        """
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
        """Create bar chart showing customer distribution across segments.
        
        This visualization displays the number of customers in each segment,
        helping identify the largest and smallest customer groups.
        
        Args:
            rfm_segmented (pd.DataFrame): DataFrame with 'Segment' column.
            
        Returns:
            go.Figure: Plotly bar chart figure showing segment distribution.
        
        Examples:
            >>> fig = RFMAnalyzer.create_segment_distribution(segmented_data)
            >>> fig.show()
        
        Notes:
            - Bars colored with Blues color scale
            - X-axis labels rotated -45° for readability
            - Height: 500px
            - Interactive tooltips show exact counts
        """
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
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def calculate_clv(transactions_df: pd.DataFrame, customer_col: str, date_col: str, 
                      amount_col: str, time_period_months: int = 12) -> pd.DataFrame:
        """Calculate Customer Lifetime Value (CLV) metrics per customer.
        
        This method calculates both historic and predictive CLV metrics using
        transaction history. Historic CLV represents total actual value, while
        predictive CLV estimates future value based on purchasing patterns.
        
        Args:
            transactions_df (pd.DataFrame): Transaction data with customer IDs, dates, and amounts.
            customer_col (str): Name of the column containing customer identifiers.
            date_col (str): Name of the column containing transaction dates.
            amount_col (str): Name of the column containing transaction amounts.
            time_period_months (int): Time period in months for calculating average metrics.
                Default is 12 months (annual basis).
        
        Returns:
            pd.DataFrame: DataFrame with one row per customer containing:
                - customer_col: Customer identifier
                - Historic_CLV: Total lifetime spending (sum of all transactions)
                - Avg_Order_Value: Average transaction amount
                - Purchase_Frequency: Average purchases per month
                - Customer_Lifespan_Months: Months between first and last purchase
                - Predicted_CLV: Estimated future value (12-month projection)
        
        Examples:
            >>> clv_data = RFMAnalyzer.calculate_clv(
            ...     transactions_df, 'CustomerID', 'InvoiceDate', 'Amount', time_period_months=12
            ... )
            >>> print(clv_data[['CustomerID', 'Historic_CLV', 'Predicted_CLV']].head())
        
        Notes:
            - Historic CLV = Total spending to date
            - Predicted CLV = (Avg Order Value × Monthly Frequency × 12 months)
            - Customers with single purchase get conservative estimates
            - Assumes continuation of current purchasing patterns
        """
        df = transactions_df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Group by customer and calculate CLV metrics
        clv_metrics = df.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            date_col: ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        clv_metrics.columns = [customer_col, 'Historic_CLV', 'Avg_Order_Value', 
                               'Transaction_Count', 'First_Purchase', 'Last_Purchase']
        
        # Calculate customer lifespan in months
        clv_metrics['Customer_Lifespan_Days'] = (
            clv_metrics['Last_Purchase'] - clv_metrics['First_Purchase']
        ).dt.days
        clv_metrics['Customer_Lifespan_Months'] = clv_metrics['Customer_Lifespan_Days'] / 30.44
        
        # For single-purchase customers, set lifespan to 1 month minimum
        clv_metrics['Customer_Lifespan_Months'] = clv_metrics['Customer_Lifespan_Months'].apply(
            lambda x: max(x, 1)
        )
        
        # Calculate purchase frequency (purchases per month)
        clv_metrics['Purchase_Frequency'] = (
            clv_metrics['Transaction_Count'] / clv_metrics['Customer_Lifespan_Months']
        )
        
        # Predicted CLV = Average Order Value × Monthly Frequency × Projection Period (12 months)
        clv_metrics['Predicted_CLV'] = (
            clv_metrics['Avg_Order_Value'] * 
            clv_metrics['Purchase_Frequency'] * 
            time_period_months
        )
        
        # Select and order final columns
        clv_final = clv_metrics[[
            customer_col, 'Historic_CLV', 'Avg_Order_Value', 'Purchase_Frequency',
            'Customer_Lifespan_Months', 'Transaction_Count', 'Predicted_CLV'
        ]].copy()
        
        return clv_final
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def merge_rfm_with_clv(rfm_segmented: pd.DataFrame, clv_data: pd.DataFrame, 
                           customer_col: str) -> pd.DataFrame:
        """Merge RFM segmentation data with CLV metrics.
        
        This method combines RFM analysis results with Customer Lifetime Value metrics,
        providing a comprehensive view of customer value and behavior.
        
        Args:
            rfm_segmented (pd.DataFrame): RFM data with segments and scores.
            clv_data (pd.DataFrame): CLV metrics calculated from transaction history.
            customer_col (str): Name of the column containing customer identifiers.
        
        Returns:
            pd.DataFrame: Combined DataFrame with both RFM and CLV metrics per customer.
        
        Examples:
            >>> rfm_clv = RFMAnalyzer.merge_rfm_with_clv(
            ...     rfm_segmented, clv_data, 'CustomerID'
            ... )
            >>> print(rfm_clv[['CustomerID', 'Segment', 'Historic_CLV']].head())
        
        Notes:
            - Uses left join to preserve all RFM customers
            - Missing CLV values filled with 0 (new/no-purchase customers)
            - Enables segment-level CLV analysis
        """
        rfm_with_clv = rfm_segmented.merge(clv_data, on=customer_col, how='left')
        
        # Fill NaN values with 0 for customers without CLV data
        clv_columns = ['Historic_CLV', 'Avg_Order_Value', 'Purchase_Frequency', 
                       'Customer_Lifespan_Months', 'Transaction_Count', 'Predicted_CLV']
        
        for col in clv_columns:
            if col in rfm_with_clv.columns:
                rfm_with_clv[col] = rfm_with_clv[col].fillna(0)
        
        return rfm_with_clv
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_segment_profiles_with_clv(rfm_clv: pd.DataFrame, 
                                      customer_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate segment profiles including CLV metrics.
        
        This method extends standard segment profiling to include Customer Lifetime Value
        metrics, providing insights into the financial value of each customer segment.
        
        Args:
            rfm_clv (pd.DataFrame): Combined RFM and CLV data from merge_rfm_with_clv().
            customer_col (Optional[str]): Name of the column containing customer identifiers.
                If None, uses row count for customer count. Default is None.
        
        Returns:
            pd.DataFrame: Segment profiles with RFM averages and CLV metrics:
                - Segment: Segment name
                - Customer_Count: Number of customers
                - Avg_Recency, Avg_Frequency, Avg_Monetary: RFM averages
                - Avg_R_Score, Avg_F_Score, Avg_M_Score: Score averages
                - Avg_Historic_CLV: Average lifetime value to date
                - Avg_Predicted_CLV: Average predicted future value
                - Total_Historic_CLV: Total segment historic value
                - Total_Predicted_CLV: Total segment predicted value
        
        Examples:
            >>> profiles = RFMAnalyzer.get_segment_profiles_with_clv(rfm_clv, 'CustomerID')
            >>> print(profiles[['Segment', 'Avg_Historic_CLV', 'Total_Historic_CLV']].head())
        
        Notes:
            - Sorted by total historic CLV (most valuable segments first)
            - Helps prioritize marketing spend by segment value
            - Champions typically show highest CLV metrics
        """
        if customer_col:
            agg_dict = {
                customer_col: 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean',
                'Historic_CLV': ['mean', 'sum'],
                'Predicted_CLV': ['mean', 'sum']
            }
        else:
            agg_dict = {
                'Recency': ['count', 'mean'],
                'Frequency': 'mean',
                'Monetary': 'mean',
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean',
                'Historic_CLV': ['mean', 'sum'],
                'Predicted_CLV': ['mean', 'sum']
            }
        
        profiles = rfm_clv.groupby('Segment').agg(agg_dict).reset_index()
        
        # Flatten MultiIndex columns
        if customer_col:
            profiles.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency',
                              'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score',
                              'Avg_Historic_CLV', 'Total_Historic_CLV',
                              'Avg_Predicted_CLV', 'Total_Predicted_CLV']
        else:
            profiles.columns = profiles.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
            profiles.columns = ['Segment', 'Customer_Count', 'Avg_Recency', 'Avg_Frequency',
                              'Avg_Monetary', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score',
                              'Avg_Historic_CLV', 'Total_Historic_CLV',
                              'Avg_Predicted_CLV', 'Total_Predicted_CLV']
        
        # Sort by total historic CLV (most valuable segments first)
        profiles = profiles.sort_values('Total_Historic_CLV', ascending=False)
        
        return profiles
