"""Geospatial Analysis Utilities.

This module provides tools for geospatial analysis including location-based
insights, distance calculations, clustering, and interactive maps.

Typical usage example:
    analyzer = GeospatialAnalyzer()
    analyzer.fit(df, lat_col='latitude', lon_col='longitude')
    clusters = analyzer.perform_clustering(n_clusters=5)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


class GeospatialAnalyzer:
    """Handles Geospatial Analysis and Location Intelligence.
    
    This class provides tools for analyzing geographic data, calculating distances,
    performing spatial clustering, and creating interactive maps.
    
    Attributes:
        data (Optional[pd.DataFrame]): Geographic data with coordinates.
        clusters (Optional[np.ndarray]): Cluster assignments.
    
    Examples:
        >>> analyzer = GeospatialAnalyzer()
        >>> analyzer.fit(df, 'latitude', 'longitude')
        >>> clusters = analyzer.perform_clustering(n_clusters=5)
    """
    
    def __init__(self):
        """Initialize the Geospatial Analyzer."""
        self.data = None
        self.clusters = None
        self.lat_col = None
        self.lon_col = None
    
    def fit(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> None:
        """Fit the analyzer on geographic data.
        
        Args:
            df: DataFrame with geographic coordinates.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
        
        Examples:
            >>> analyzer.fit(df, 'latitude', 'longitude')
        """
        self.data = df.copy()
        self.lat_col = lat_col
        self.lon_col = lon_col
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers.
        
        Args:
            lat1: Latitude of point 1.
            lon1: Longitude of point 1.
            lat2: Latitude of point 2.
            lon2: Longitude of point 2.
        
        Returns:
            Distance in kilometers.
        
        Examples:
            >>> distance = GeospatialAnalyzer.haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
            >>> print(f"Distance: {distance:.2f} km")
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """Calculate pairwise distance matrix for all points.
        
        Returns:
            Distance matrix in kilometers.
        
        Examples:
            >>> distances = analyzer.calculate_distance_matrix()
        """
        n = len(self.data)
        distance_matrix = np.zeros((n, n))
        
        lats = self.data[self.lat_col].values
        lons = self.data[self.lon_col].values
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.haversine_distance(lats[i], lons[i], lats[j], lons[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    @st.cache_data(ttl=1800)
    def perform_clustering(_self, n_clusters: int = 5, method: str = 'kmeans') -> np.ndarray:
        """Perform spatial clustering.
        
        Args:
            n_clusters: Number of clusters (for K-Means).
            method: Clustering method - 'kmeans' or 'dbscan'.
        
        Returns:
            Array of cluster assignments.
        
        Examples:
            >>> clusters = analyzer.perform_clustering(n_clusters=5, method='kmeans')
        """
        X = _self.data[[_self.lat_col, _self.lon_col]].values
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            _self.clusters = clusterer.fit_predict(X)
        else:  # DBSCAN
            # Convert lat/lon to radians for DBSCAN
            X_rad = np.radians(X)
            # eps is in radians, roughly 0.1 km = 0.0001 radians
            clusterer = DBSCAN(eps=0.01, min_samples=5, metric='haversine')
            _self.clusters = clusterer.fit_predict(X_rad)
        
        return _self.clusters
    
    def find_nearest_points(self, target_lat: float, target_lon: float, 
                           n_points: int = 10) -> pd.DataFrame:
        """Find nearest points to a target location.
        
        Args:
            target_lat: Target latitude.
            target_lon: Target longitude.
            n_points: Number of nearest points to return.
        
        Returns:
            DataFrame with nearest points and distances.
        
        Examples:
            >>> nearest = analyzer.find_nearest_points(40.7128, -74.0060, n_points=5)
        """
        distances = []
        
        for idx, row in self.data.iterrows():
            dist = self.haversine_distance(target_lat, target_lon,
                                          row[self.lat_col], row[self.lon_col])
            distances.append((idx, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, _ in distances[:n_points]]
        nearest_distances = [dist for _, dist in distances[:n_points]]
        
        result = self.data.loc[nearest_indices].copy()
        result['distance_km'] = nearest_distances
        
        return result
    
    def calculate_density_grid(self, grid_size: int = 20) -> pd.DataFrame:
        """Calculate point density on a grid.
        
        Args:
            grid_size: Number of grid cells per dimension.
        
        Returns:
            DataFrame with grid cell counts.
        
        Examples:
            >>> density = analyzer.calculate_density_grid(grid_size=20)
        """
        lat_bins = np.linspace(self.data[self.lat_col].min(), 
                              self.data[self.lat_col].max(), grid_size)
        lon_bins = np.linspace(self.data[self.lon_col].min(), 
                              self.data[self.lon_col].max(), grid_size)
        
        # Assign each point to a grid cell
        lat_indices = np.digitize(self.data[self.lat_col], lat_bins)
        lon_indices = np.digitize(self.data[self.lon_col], lon_bins)
        
        # Count points per cell
        density_data = []
        for i in range(len(lat_bins)):
            for j in range(len(lon_bins)):
                count = ((lat_indices == i) & (lon_indices == j)).sum()
                if count > 0:
                    density_data.append({
                        'lat': lat_bins[i] if i < len(lat_bins) else lat_bins[-1],
                        'lon': lon_bins[j] if j < len(lon_bins) else lon_bins[-1],
                        'count': count
                    })
        
        return pd.DataFrame(density_data)
    
    @staticmethod
    def create_scatter_map(df: pd.DataFrame, lat_col: str, lon_col: str,
                          color_col: Optional[str] = None,
                          size_col: Optional[str] = None,
                          hover_data: Optional[List[str]] = None,
                          title: str = 'Geographic Distribution') -> go.Figure:
        """Create scatter plot map.
        
        Args:
            df: DataFrame with geographic data.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            color_col: Optional column for color coding.
            size_col: Optional column for size coding.
            hover_data: Optional columns to show on hover.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = GeospatialAnalyzer.create_scatter_map(df, 'lat', 'lon', color_col='cluster')
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            size=size_col,
            hover_data=hover_data,
            title=title,
            zoom=3,
            height=600
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        
        return fig
    
    @staticmethod
    def create_density_heatmap(df: pd.DataFrame, lat_col: str, lon_col: str,
                               title: str = 'Density Heatmap') -> go.Figure:
        """Create density heatmap.
        
        Args:
            df: DataFrame with geographic data.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = GeospatialAnalyzer.create_density_heatmap(df, 'lat', 'lon')
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            radius=10,
            title=title,
            zoom=3,
            height=600
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        
        return fig
    
    @staticmethod
    def create_choropleth_map(df: pd.DataFrame, location_col: str, 
                             value_col: str, title: str = 'Choropleth Map') -> go.Figure:
        """Create choropleth map (for country/state level data).
        
        Args:
            df: DataFrame with location and value data.
            location_col: Column with location codes (e.g., country codes).
            value_col: Column with values to visualize.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = GeospatialAnalyzer.create_choropleth_map(df, 'country_code', 'sales')
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        fig = px.choropleth(
            df,
            locations=location_col,
            color=value_col,
            title=title,
            color_continuous_scale='Viridis',
            height=600
        )
        
        return fig
