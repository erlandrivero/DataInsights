"""Recommendation System Utilities.

This module provides tools for building recommendation systems including
collaborative filtering (user-based and item-based) and content-based filtering.

Typical usage example:
    engine = RecommendationEngine()
    engine.fit(ratings_df, user_col='user_id', item_col='item_id', rating_col='rating')
    recommendations = engine.recommend_items(user_id=123, n_recommendations=10)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


class RecommendationEngine:
    """Handles Recommendation System algorithms.
    
    This class provides collaborative filtering (user-based and item-based) and
    content-based recommendation algorithms with evaluation metrics.
    
    Attributes:
        user_item_matrix (Optional[pd.DataFrame]): User-item rating matrix.
        user_similarity (Optional[np.ndarray]): User similarity matrix.
        item_similarity (Optional[np.ndarray]): Item similarity matrix.
    
    Examples:
        >>> engine = RecommendationEngine()
        >>> engine.fit(df, 'user_id', 'product_id', 'rating')
        >>> recs = engine.recommend_items(user_id=123, n_recommendations=5)
    """
    
    def __init__(self):
        """Initialize the Recommendation Engine."""
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.users = None
        self.items = None
    
    @st.cache_data(ttl=1800)
    def fit(_self, df: pd.DataFrame, user_col: str, item_col: str, 
            rating_col: str, similarity_metric: str = 'cosine') -> None:
        """Fit the recommendation engine on user-item ratings.
        
        Args:
            df: DataFrame with user-item ratings.
            user_col: Column name for user identifiers.
            item_col: Column name for item identifiers.
            rating_col: Column name for ratings.
            similarity_metric: Similarity metric - 'cosine' or 'pearson'.
        
        Examples:
            >>> engine.fit(df, 'user_id', 'movie_id', 'rating')
        """
        # Create user-item matrix
        _self.user_item_matrix = df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=0
        )
        
        _self.users = _self.user_item_matrix.index.tolist()
        _self.items = _self.user_item_matrix.columns.tolist()
        
        # Calculate similarity matrices
        if similarity_metric == 'cosine':
            # User similarity (rows)
            _self.user_similarity = cosine_similarity(_self.user_item_matrix)
            # Item similarity (columns)
            _self.item_similarity = cosine_similarity(_self.user_item_matrix.T)
        else:  # Pearson correlation
            _self.user_similarity = np.corrcoef(_self.user_item_matrix)
            _self.item_similarity = np.corrcoef(_self.user_item_matrix.T)
        
        # Replace NaN with 0
        _self.user_similarity = np.nan_to_num(_self.user_similarity)
        _self.item_similarity = np.nan_to_num(_self.item_similarity)
    
    def recommend_items_user_based(self, user_id: Any, n_recommendations: int = 10,
                                    n_similar_users: int = 5) -> pd.DataFrame:
        """Generate recommendations using user-based collaborative filtering.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to return.
            n_similar_users: Number of similar users to consider.
        
        Returns:
            DataFrame with recommended items and scores.
        
        Examples:
            >>> recs = engine.recommend_items_user_based(user_id=123, n_recommendations=5)
        """
        if user_id not in self.users:
            return pd.DataFrame()
        
        user_idx = self.users.index(user_id)
        
        # Get similar users (excluding self)
        user_similarities = self.user_similarity[user_idx].copy()
        user_similarities[user_idx] = 0  # Exclude self
        similar_user_indices = np.argsort(user_similarities)[::-1][:n_similar_users]
        
        # Get items user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        predictions = {}
        for item in unrated_items:
            # Weighted average of similar users' ratings
            similar_users_ratings = self.user_item_matrix.iloc[similar_user_indices][item]
            similar_users_sims = user_similarities[similar_user_indices]
            
            # Only consider users who rated this item
            mask = similar_users_ratings > 0
            if mask.sum() == 0:
                continue
            
            weighted_sum = np.sum(similar_users_ratings[mask] * similar_users_sims[mask])
            sim_sum = np.sum(similar_users_sims[mask])
            
            if sim_sum > 0:
                predictions[item] = weighted_sum / sim_sum
        
        # Sort and return top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return pd.DataFrame(sorted_predictions[:n_recommendations], 
                           columns=['item_id', 'predicted_rating'])
    
    def recommend_items_item_based(self, user_id: Any, n_recommendations: int = 10,
                                    n_similar_items: int = 5) -> pd.DataFrame:
        """Generate recommendations using item-based collaborative filtering.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to return.
            n_similar_items: Number of similar items to consider per rated item.
        
        Returns:
            DataFrame with recommended items and scores.
        
        Examples:
            >>> recs = engine.recommend_items_item_based(user_id=123, n_recommendations=5)
        """
        if user_id not in self.users:
            return pd.DataFrame()
        
        # Get items user has rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        # Get items user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        predictions = {}
        for item in unrated_items:
            item_idx = self.items.index(item)
            
            # Get similar items from user's rated items
            weighted_sum = 0
            sim_sum = 0
            
            for rated_item in rated_items.index:
                rated_item_idx = self.items.index(rated_item)
                similarity = self.item_similarity[item_idx][rated_item_idx]
                
                if similarity > 0:
                    weighted_sum += similarity * rated_items[rated_item]
                    sim_sum += similarity
            
            if sim_sum > 0:
                predictions[item] = weighted_sum / sim_sum
        
        # Sort and return top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return pd.DataFrame(sorted_predictions[:n_recommendations], 
                           columns=['item_id', 'predicted_rating'])
    
    def get_similar_users(self, user_id: Any, n_users: int = 10) -> pd.DataFrame:
        """Find most similar users.
        
        Args:
            user_id: User identifier.
            n_users: Number of similar users to return.
        
        Returns:
            DataFrame with similar users and similarity scores.
        
        Examples:
            >>> similar = engine.get_similar_users(user_id=123, n_users=5)
        """
        if user_id not in self.users:
            return pd.DataFrame()
        
        user_idx = self.users.index(user_id)
        similarities = self.user_similarity[user_idx].copy()
        similarities[user_idx] = -1  # Exclude self
        
        similar_indices = np.argsort(similarities)[::-1][:n_users]
        similar_users = [self.users[i] for i in similar_indices]
        similar_scores = similarities[similar_indices]
        
        return pd.DataFrame({
            'user_id': similar_users,
            'similarity_score': similar_scores
        })
    
    def get_similar_items(self, item_id: Any, n_items: int = 10) -> pd.DataFrame:
        """Find most similar items.
        
        Args:
            item_id: Item identifier.
            n_items: Number of similar items to return.
        
        Returns:
            DataFrame with similar items and similarity scores.
        
        Examples:
            >>> similar = engine.get_similar_items(item_id=456, n_items=5)
        """
        if item_id not in self.items:
            return pd.DataFrame()
        
        item_idx = self.items.index(item_id)
        similarities = self.item_similarity[item_idx].copy()
        similarities[item_idx] = -1  # Exclude self
        
        similar_indices = np.argsort(similarities)[::-1][:n_items]
        similar_items = [self.items[i] for i in similar_indices]
        similar_scores = similarities[similar_indices]
        
        return pd.DataFrame({
            'item_id': similar_items,
            'similarity_score': similar_scores
        })
    
    def get_popular_items(self, n_items: int = 10) -> pd.DataFrame:
        """Get most popular items based on rating count and average rating.
        
        Used for cold start - recommending to new users with no history.
        
        Args:
            n_items: Number of popular items to return.
        
        Returns:
            DataFrame with popular items, their avg ratings, and rating counts.
        """
        if self.user_item_matrix is None:
            return pd.DataFrame()
        
        # Calculate popularity score: count * avg_rating
        item_ratings = self.user_item_matrix.replace(0, np.nan)
        rating_counts = item_ratings.count(axis=0)
        avg_ratings = item_ratings.mean(axis=0)
        
        # Popularity score: weighted by both count and rating
        popularity_scores = rating_counts * avg_ratings
        
        # Sort and get top N
        top_items = popularity_scores.nlargest(n_items)
        
        return pd.DataFrame({
            'item_id': top_items.index.tolist(),
            'avg_rating': avg_ratings.loc[top_items.index].values,
            'rating_count': rating_counts.loc[top_items.index].values,
            'popularity_score': top_items.values
        })
    
    def get_global_average(self) -> float:
        """Get global average rating across all users and items.
        
        Returns:
            Global average rating.
        """
        if self.user_item_matrix is None:
            return 0.0
        
        # Replace 0s with NaN to get true average
        ratings = self.user_item_matrix.replace(0, np.nan)
        return ratings.mean().mean()
    
    def recommend_with_cold_start(self, user_id: Any, n_recommendations: int = 10,
                                   method: str = 'user_based') -> Tuple[pd.DataFrame, str]:
        """Recommend items with cold start handling.
        
        Detects cold start scenarios and uses appropriate fallback strategy:
        - New user (no ratings): Return popular items
        - Existing user: Use collaborative filtering
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations.
            method: 'user_based' or 'item_based'.
        
        Returns:
            Tuple of (recommendations DataFrame, strategy used).
        """
        # Check if user exists
        if user_id not in self.users:
            # Cold start: New user with no history
            popular_items = self.get_popular_items(n_recommendations)
            return popular_items[['item_id', 'avg_rating']].rename(
                columns={'avg_rating': 'predicted_rating'}
            ), 'popularity_fallback'
        
        # Check if user has any ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        if user_ratings.sum() == 0:
            # Cold start: User exists but has no ratings
            popular_items = self.get_popular_items(n_recommendations)
            return popular_items[['item_id', 'avg_rating']].rename(
                columns={'avg_rating': 'predicted_rating'}
            ), 'popularity_fallback'
        
        # Normal case: Use collaborative filtering
        if method == 'user_based':
            recs = self.recommend_items_user_based(user_id, n_recommendations)
        else:
            recs = self.recommend_items_item_based(user_id, n_recommendations)
        
        # If collaborative filtering returns no results, fallback to popular items
        if recs.empty:
            popular_items = self.get_popular_items(n_recommendations)
            return popular_items[['item_id', 'avg_rating']].rename(
                columns={'avg_rating': 'predicted_rating'}
            ), 'popularity_fallback'
        
        return recs, 'collaborative_filtering'
    
    def evaluate_recommendations(self, test_df: pd.DataFrame, user_col: str, 
                                 item_col: str, rating_col: str,
                                 method: str = 'user_based', k: int = 10) -> Dict[str, float]:
        """Evaluate recommendation quality.
        
        Args:
            test_df: Test set DataFrame.
            user_col: Column name for user identifiers.
            item_col: Column name for item identifiers.
            rating_col: Column name for ratings.
            method: Method - 'user_based' or 'item_based'.
            k: Number of recommendations to generate.
        
        Returns:
            Dictionary with evaluation metrics.
        
        Examples:
            >>> metrics = engine.evaluate_recommendations(test_df, 'user_id', 'item_id', 'rating')
        """
        hits = 0
        total_predictions = 0
        precision_sum = 0
        recall_sum = 0
        users_evaluated = 0
        
        for user_id in test_df[user_col].unique():
            if user_id not in self.users:
                continue
            
            # Get actual items user liked in test set
            user_test = test_df[test_df[user_col] == user_id]
            actual_items = set(user_test[user_test[rating_col] >= 4][item_col])
            
            if len(actual_items) == 0:
                continue
            
            # Get recommendations
            if method == 'user_based':
                recs = self.recommend_items_user_based(user_id, n_recommendations=k)
            else:
                recs = self.recommend_items_item_based(user_id, n_recommendations=k)
            
            if len(recs) == 0:
                continue
            
            recommended_items = set(recs['item_id'])
            
            # Calculate metrics
            hits_for_user = len(actual_items & recommended_items)
            hits += hits_for_user
            total_predictions += len(recommended_items)
            
            precision = hits_for_user / len(recommended_items) if len(recommended_items) > 0 else 0
            recall = hits_for_user / len(actual_items) if len(actual_items) > 0 else 0
            
            precision_sum += precision
            recall_sum += recall
            users_evaluated += 1
        
        if users_evaluated == 0:
            return {'precision@k': 0, 'recall@k': 0, 'f1@k': 0, 'hit_rate': 0}
        
        avg_precision = precision_sum / users_evaluated
        avg_recall = recall_sum / users_evaluated
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        hit_rate = hits / total_predictions if total_predictions > 0 else 0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1,
            'hit_rate': hit_rate,
            'users_evaluated': users_evaluated
        }
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: np.ndarray, 
                                  labels: List[str], 
                                  title: str = 'Similarity Matrix') -> go.Figure:
        """Create similarity matrix heatmap.
        
        Args:
            similarity_matrix: Similarity matrix.
            labels: Labels for rows/columns.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = RecommendationEngine.create_similarity_heatmap(user_sim, users)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        # Limit to top 50 for readability
        if len(labels) > 50:
            similarity_matrix = similarity_matrix[:50, :50]
            labels = labels[:50]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Similarity")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='',
            yaxis_title='',
            height=600
        )
        
        return fig
