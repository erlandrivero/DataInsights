"""Network Analysis Utilities.

This module provides tools for social network analysis including graph metrics,
community detection, centrality measures, and network visualizations.

Typical usage example:
    analyzer = NetworkAnalyzer()
    analyzer.build_graph(edges_df, 'source', 'target')
    communities = analyzer.detect_communities()
    centrality = analyzer.calculate_centrality('betweenness')
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
import plotly.graph_objects as go


class NetworkAnalyzer:
    """Handles Network Analysis and Graph Theory.
    
    This class provides tools for analyzing networks including centrality measures,
    community detection, path finding, and network visualizations.
    
    Attributes:
        graph (Optional[nx.Graph]): NetworkX graph object.
        communities (Optional[List]): Detected communities.
    
    Examples:
        >>> analyzer = NetworkAnalyzer()
        >>> analyzer.build_graph(edges_df, 'from', 'to', weight_col='strength')
        >>> centrality = analyzer.calculate_centrality('degree')
    """
    
    def __init__(self):
        """Initialize the Network Analyzer."""
        self.graph = None
        self.communities = None
        self.node_positions = None
    
    def build_graph(self, edges_df: pd.DataFrame, source_col: str, 
                    target_col: str, weight_col: Optional[str] = None,
                    directed: bool = False) -> None:
        """Build network graph from edge list.
        
        Args:
            edges_df: DataFrame with edge list.
            source_col: Column name for source nodes.
            target_col: Column name for target nodes.
            weight_col: Optional column name for edge weights.
            directed: Whether to create directed graph.
        
        Examples:
            >>> analyzer.build_graph(df, 'user1', 'user2', weight_col='interactions')
        """
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        
        # Add edges
        if weight_col and weight_col in edges_df.columns:
            edges = [(row[source_col], row[target_col], row[weight_col]) 
                    for _, row in edges_df.iterrows()]
            self.graph.add_weighted_edges_from(edges)
        else:
            edges = [(row[source_col], row[target_col]) 
                    for _, row in edges_df.iterrows()]
            self.graph.add_edges_from(edges)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get basic network statistics.
        
        Returns:
            Dictionary with network metrics.
        
        Examples:
            >>> stats = analyzer.get_network_stats()
            >>> print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
        """
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph) if not self.graph.is_directed() else nx.is_weakly_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph) if not self.graph.is_directed() else nx.number_weakly_connected_components(self.graph)
        }
        
        # Average clustering coefficient
        if not self.graph.is_directed():
            stats['avg_clustering'] = nx.average_clustering(self.graph)
        
        # Average degree
        degrees = dict(self.graph.degree())
        stats['avg_degree'] = np.mean(list(degrees.values()))
        stats['max_degree'] = max(degrees.values())
        
        return stats
    
    @st.cache_data(ttl=1800)
    def calculate_centrality(_self, measure: str = 'degree') -> pd.DataFrame:
        """Calculate node centrality measures.
        
        Args:
            measure: Centrality measure - 'degree', 'betweenness', 'closeness', 'eigenvector', or 'pagerank'.
        
        Returns:
            DataFrame with node centrality scores.
        
        Examples:
            >>> centrality = analyzer.calculate_centrality('betweenness')
        """
        if _self.graph is None:
            raise ValueError("Must build graph first")
        
        if measure == 'degree':
            centrality_dict = nx.degree_centrality(_self.graph)
        elif measure == 'betweenness':
            centrality_dict = nx.betweenness_centrality(_self.graph)
        elif measure == 'closeness':
            centrality_dict = nx.closeness_centrality(_self.graph)
        elif measure == 'eigenvector':
            try:
                centrality_dict = nx.eigenvector_centrality(_self.graph, max_iter=1000)
            except:
                centrality_dict = nx.eigenvector_centrality_numpy(_self.graph)
        elif measure == 'pagerank':
            centrality_dict = nx.pagerank(_self.graph)
        else:
            raise ValueError(f"Unknown measure: {measure}")
        
        # Convert to DataFrame
        df = pd.DataFrame(list(centrality_dict.items()), 
                         columns=['node', 'centrality'])
        df = df.sort_values('centrality', ascending=False)
        
        return df
    
    @st.cache_data(ttl=1800)
    def detect_communities(_self, method: str = 'louvain') -> List[set]:
        """Detect communities in the network.
        
        Args:
            method: Community detection method - 'louvain', 'greedy', or 'label_propagation'.
        
        Returns:
            List of sets, each containing nodes in a community.
        
        Examples:
            >>> communities = analyzer.detect_communities(method='louvain')
            >>> print(f"Found {len(communities)} communities")
        """
        if _self.graph is None:
            raise ValueError("Must build graph first")
        
        # Convert to undirected if needed
        G = _self.graph.to_undirected() if _self.graph.is_directed() else _self.graph
        
        if method == 'louvain':
            # Requires python-louvain package
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                # Convert to list of sets
                communities_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities_dict:
                        communities_dict[comm_id] = set()
                    communities_dict[comm_id].add(node)
                _self.communities = list(communities_dict.values())
            except ImportError:
                # Fallback to greedy
                _self.communities = list(nx.community.greedy_modularity_communities(G))
        elif method == 'greedy':
            _self.communities = list(nx.community.greedy_modularity_communities(G))
        elif method == 'label_propagation':
            _self.communities = list(nx.community.label_propagation_communities(G))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return _self.communities
    
    def get_shortest_path(self, source: Any, target: Any) -> List:
        """Find shortest path between two nodes.
        
        Args:
            source: Source node.
            target: Target node.
        
        Returns:
            List of nodes in the shortest path.
        
        Examples:
            >>> path = analyzer.get_shortest_path('Alice', 'Bob')
        """
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []
    
    def find_influencers(self, top_n: int = 10) -> pd.DataFrame:
        """Find most influential nodes based on multiple centrality measures.
        
        Args:
            top_n: Number of top influencers to return.
        
        Returns:
            DataFrame with influencer rankings.
        
        Examples:
            >>> influencers = analyzer.find_influencers(top_n=10)
        """
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        # Calculate multiple centrality measures
        degree_cent = nx.degree_centrality(self.graph)
        betweenness_cent = nx.betweenness_centrality(self.graph)
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.graph, max_iter=1000)
        except:
            eigenvector_cent = {node: 0 for node in self.graph.nodes()}
        
        # Combine scores
        combined_scores = {}
        for node in self.graph.nodes():
            combined_scores[node] = (
                degree_cent.get(node, 0) + 
                betweenness_cent.get(node, 0) + 
                eigenvector_cent.get(node, 0)
            ) / 3
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'node': node,
                'influence_score': combined_scores[node],
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent[node],
                'eigenvector_centrality': eigenvector_cent[node]
            }
            for node in self.graph.nodes()
        ])
        
        df = df.sort_values('influence_score', ascending=False).head(top_n)
        
        return df
    
    @staticmethod
    def create_network_visualization(graph: nx.Graph, 
                                     node_color_attr: Optional[str] = None,
                                     node_size_attr: Optional[str] = None,
                                     title: str = 'Network Graph') -> go.Figure:
        """Create interactive network visualization.
        
        Args:
            graph: NetworkX graph.
            node_color_attr: Node attribute for coloring.
            node_size_attr: Node attribute for sizing.
            title: Plot title.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = NetworkAnalyzer.create_network_visualization(analyzer.graph)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        # Calculate layout
        pos = nx.spring_layout(graph, k=0.5, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            degree = graph.degree(node)
            node_text.append(f'Node: {node}<br>Degree: {degree}')
        
        # Node colors
        if node_color_attr and all(node_color_attr in graph.nodes[n] for n in graph.nodes()):
            node_colors = [graph.nodes[n][node_color_attr] for n in graph.nodes()]
        else:
            node_colors = [graph.degree(n) for n in graph.nodes()]
        
        # Node sizes
        if node_size_attr and all(node_size_attr in graph.nodes[n] for n in graph.nodes()):
            node_sizes = [graph.nodes[n][node_size_attr] * 10 for n in graph.nodes()]
        else:
            node_sizes = [max(10, graph.degree(n) * 3) for n in graph.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree',
                    xanchor='left'
                ),
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=title, font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600))
        
        return fig
    
    @staticmethod
    def create_degree_distribution(graph: nx.Graph) -> go.Figure:
        """Create degree distribution histogram.
        
        Args:
            graph: NetworkX graph.
        
        Returns:
            Plotly figure.
        
        Examples:
            >>> fig = NetworkAnalyzer.create_degree_distribution(analyzer.graph)
            >>> st.plotly_chart(fig, use_container_width=True)
        """
        degrees = [graph.degree(n) for n in graph.nodes()]
        
        fig = go.Figure(data=[go.Histogram(x=degrees, nbinsx=30)])
        
        fig.update_layout(
            title='Node Degree Distribution',
            xaxis_title='Degree',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
