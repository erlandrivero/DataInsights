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
    
    def simulate_influence_spread(self, seed_nodes: List[Any], 
                                  propagation_prob: float = 0.1,
                                  model: str = 'independent_cascade',
                                  max_iterations: int = 100) -> Dict[str, Any]:
        """Simulate influence propagation through network.
        
        Simulates how information/influence spreads from seed nodes through the network.
        
        Args:
            seed_nodes: Initial set of influenced nodes.
            propagation_prob: Probability of influence spreading along an edge.
            model: Propagation model ('independent_cascade' or 'linear_threshold').
            max_iterations: Maximum simulation iterations.
        
        Returns:
            Dictionary with spread results.
        
        Examples:
            >>> top_nodes = ['user_1', 'user_5', 'user_10']
            >>> spread = analyzer.simulate_influence_spread(top_nodes, propagation_prob=0.15)
        """
        if self.graph is None:
            return {}
        
        if model == 'independent_cascade':
            return self._independent_cascade(seed_nodes, propagation_prob, max_iterations)
        elif model == 'linear_threshold':
            return self._linear_threshold(seed_nodes, propagation_prob, max_iterations)
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _independent_cascade(self, seed_nodes: List[Any], prob: float, 
                           max_iter: int) -> Dict[str, Any]:
        """Independent Cascade propagation model.
        
        Each activated node gets one chance to activate each inactive neighbor
        with probability prob.
        """
        # Track activated nodes and when they were activated
        activated = set(seed_nodes)
        newly_activated = set(seed_nodes)
        activation_time = {node: 0 for node in seed_nodes}
        
        iteration = 0
        spread_over_time = [len(activated)]
        
        while newly_activated and iteration < max_iter:
            iteration += 1
            next_activated = set()
            
            for node in newly_activated:
                # Try to activate neighbors
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in activated:
                        # Activation probability
                        if np.random.random() < prob:
                            next_activated.add(neighbor)
                            activation_time[neighbor] = iteration
            
            newly_activated = next_activated
            activated.update(newly_activated)
            spread_over_time.append(len(activated))
        
        return {
            'activated_nodes': list(activated),
            'num_activated': len(activated),
            'activation_time': activation_time,
            'spread_over_time': spread_over_time,
            'final_reach_pct': len(activated) / self.graph.number_of_nodes() * 100,
            'iterations': iteration
        }
    
    def _linear_threshold(self, seed_nodes: List[Any], threshold: float,
                         max_iter: int) -> Dict[str, Any]:
        """Linear Threshold propagation model.
        
        A node becomes active when the sum of influences from active neighbors
        exceeds its threshold.
        """
        # Assign random thresholds to nodes
        thresholds = {node: np.random.random() for node in self.graph.nodes()}
        
        # Calculate influence weights (1/degree)
        influence_weights = {}
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            influence_weights[node] = 1.0 / degree if degree > 0 else 0
        
        activated = set(seed_nodes)
        activation_time = {node: 0 for node in seed_nodes}
        spread_over_time = [len(activated)]
        
        iteration = 0
        changed = True
        
        while changed and iteration < max_iter:
            iteration += 1
            changed = False
            newly_activated = set()
            
            for node in self.graph.nodes():
                if node not in activated:
                    # Calculate influence from active neighbors
                    active_neighbors = [n for n in self.graph.neighbors(node) if n in activated]
                    total_influence = sum(influence_weights[n] for n in active_neighbors)
                    
                    # Activate if threshold exceeded
                    if total_influence >= thresholds[node]:
                        newly_activated.add(node)
                        activation_time[node] = iteration
                        changed = True
            
            activated.update(newly_activated)
            spread_over_time.append(len(activated))
        
        return {
            'activated_nodes': list(activated),
            'num_activated': len(activated),
            'activation_time': activation_time,
            'spread_over_time': spread_over_time,
            'final_reach_pct': len(activated) / self.graph.number_of_nodes() * 100,
            'iterations': iteration
        }
    
    def find_optimal_seed_nodes(self, num_seeds: int = 5, 
                               propagation_prob: float = 0.1,
                               model: str = 'independent_cascade',
                               simulations: int = 10) -> pd.DataFrame:
        """Find optimal seed nodes for maximizing influence spread.
        
        Uses greedy algorithm: iteratively selects node that maximizes
        marginal gain in influence spread.
        
        Args:
            num_seeds: Number of seed nodes to find.
            propagation_prob: Propagation probability.
            model: Propagation model to use.
            simulations: Number of simulations per candidate.
        
        Returns:
            DataFrame with seed nodes and their expected influence.
        
        Examples:
            >>> seeds = analyzer.find_optimal_seed_nodes(num_seeds=5, propagation_prob=0.15)
        """
        if self.graph is None:
            return pd.DataFrame()
        
        seed_set = []
        candidates = list(self.graph.nodes())
        seed_results = []
        
        for i in range(num_seeds):
            best_node = None
            best_marginal_gain = -1
            
            # Try each candidate
            for candidate in candidates:
                if candidate in seed_set:
                    continue
                
                # Test this candidate with current seed set
                test_seeds = seed_set + [candidate]
                
                # Run multiple simulations and average
                total_spread = 0
                for _ in range(simulations):
                    result = self.simulate_influence_spread(
                        test_seeds, 
                        propagation_prob=propagation_prob,
                        model=model
                    )
                    total_spread += result['num_activated']
                
                avg_spread = total_spread / simulations
                marginal_gain = avg_spread - sum([r['avg_spread'] for r in seed_results])
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = candidate
            
            if best_node:
                # Run final simulations for selected seed
                total_spread = 0
                for _ in range(simulations * 2):  # More simulations for final
                    result = self.simulate_influence_spread(
                        seed_set + [best_node],
                        propagation_prob=propagation_prob,
                        model=model
                    )
                    total_spread += result['num_activated']
                
                avg_spread = total_spread / (simulations * 2)
                
                seed_set.append(best_node)
                seed_results.append({
                    'rank': i + 1,
                    'node': best_node,
                    'avg_spread': avg_spread,
                    'marginal_gain': best_marginal_gain,
                    'reach_pct': avg_spread / self.graph.number_of_nodes() * 100
                })
        
        return pd.DataFrame(seed_results)
    
    def predict_links(self, method: str = 'common_neighbors', 
                     top_k: int = 20) -> pd.DataFrame:
        """Predict potential future links in the network.
        
        Uses various link prediction algorithms to identify node pairs
        likely to form connections in the future.
        
        Args:
            method: Link prediction method to use.
                   Options: 'common_neighbors', 'adamic_adar', 'jaccard',
                           'preferential_attachment', 'resource_allocation'
            top_k: Number of top predictions to return.
        
        Returns:
            DataFrame with predicted links and scores.
        
        Examples:
            >>> predictions = analyzer.predict_links('adamic_adar', top_k=10)
            >>> print(f"Top prediction: {predictions.iloc[0]}")
        """
        if self.graph is None:
            return pd.DataFrame()
        
        # Convert to undirected graph for link prediction
        # Most link prediction algorithms are designed for undirected graphs
        G = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        
        # Get non-edges (node pairs without connections)
        non_edges = list(nx.non_edges(G))
        
        # Sample if too many (for performance)
        if len(non_edges) > 10000:
            import random
            non_edges = random.sample(non_edges, 10000)
        
        predictions = []
        
        # Apply selected method (use undirected graph G)
        if method == 'common_neighbors':
            scores = nx.common_neighbor_centrality(G, ebunch=non_edges)
        elif method == 'adamic_adar':
            scores = nx.adamic_adar_index(G, ebunch=non_edges)
        elif method == 'jaccard':
            scores = nx.jaccard_coefficient(G, ebunch=non_edges)
        elif method == 'preferential_attachment':
            scores = nx.preferential_attachment(G, ebunch=non_edges)
        elif method == 'resource_allocation':
            scores = nx.resource_allocation_index(G, ebunch=non_edges)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Collect predictions
        for u, v, score in scores:
            predictions.append({
                'node_1': u,
                'node_2': v,
                'score': score,
                'method': method
            })
        
        # Convert to DataFrame and sort by score
        df = pd.DataFrame(predictions)
        if not df.empty:
            df = df.sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def evaluate_link_prediction(self, test_edges: List[Tuple[Any, Any]],
                                 method: str = 'common_neighbors',
                                 top_k: int = 100) -> Dict[str, float]:
        """Evaluate link prediction accuracy.
        
        Measures how well the prediction method can identify actual future links
        from a test set.
        
        Args:
            test_edges: List of actual edges to predict (held-out test set).
            method: Link prediction method to use.
            top_k: Number of top predictions to consider.
        
        Returns:
            Dictionary with evaluation metrics (precision, recall, F1).
        
        Examples:
            >>> test_edges = [(1, 5), (2, 7), (3, 9)]
            >>> metrics = analyzer.evaluate_link_prediction(test_edges, 'adamic_adar')
        """
        if self.graph is None:
            return {}
        
        # Get predictions
        predictions = self.predict_links(method=method, top_k=top_k)
        
        if predictions.empty:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Convert predictions to set of tuples
        predicted_edges = set(zip(predictions['node_1'], predictions['node_2']))
        
        # Also add reverse (for undirected graphs)
        predicted_edges_symmetric = predicted_edges | {(v, u) for u, v in predicted_edges}
        
        # Convert test edges to set
        test_edges_set = set(test_edges)
        test_edges_symmetric = test_edges_set | {(v, u) for u, v in test_edges_set}
        
        # Calculate metrics
        true_positives = len(predicted_edges_symmetric & test_edges_symmetric)
        
        precision = true_positives / len(predicted_edges) if len(predicted_edges) > 0 else 0
        recall = true_positives / len(test_edges_set) if len(test_edges_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'predicted_count': len(predicted_edges),
            'test_count': len(test_edges_set)
        }
    
    def analyze_link_formation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in existing link formation.
        
        Examines structural properties of existing edges to understand
        link formation patterns in the network.
        
        Returns:
            Dictionary with pattern analysis results.
        
        Examples:
            >>> patterns = analyzer.analyze_link_formation_patterns()
            >>> print(f"Clustering effect: {patterns['clustering_coefficient']:.3f}")
        """
        if self.graph is None:
            return {}
        
        # Calculate various network properties
        clustering_coef = nx.average_clustering(self.graph)
        
        # Degree assortativity (homophily)
        try:
            assortativity = nx.degree_assortativity_coefficient(self.graph)
        except:
            assortativity = 0.0
        
        # Transitivity (global clustering)
        transitivity = nx.transitivity(self.graph)
        
        # Average path length (if connected)
        try:
            if nx.is_connected(self.graph):
                avg_path_length = nx.average_shortest_path_length(self.graph)
            else:
                # Use largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = None
        
        # Degree distribution stats
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = np.mean(degrees)
        degree_variance = np.var(degrees)
        
        return {
            'clustering_coefficient': clustering_coef,
            'degree_assortativity': assortativity,
            'transitivity': transitivity,
            'avg_path_length': avg_path_length,
            'avg_degree': avg_degree,
            'degree_variance': degree_variance,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph)
        }
    
    @staticmethod
    def compare_prediction_methods(graph: nx.Graph, 
                                   test_edges: List[Tuple[Any, Any]],
                                   methods: List[str] = None,
                                   top_k: int = 100) -> pd.DataFrame:
        """Compare performance of different link prediction methods.
        
        Args:
            graph: NetworkX graph.
            test_edges: Test set of edges to predict.
            methods: List of methods to compare (None = all).
            top_k: Number of top predictions per method.
        
        Returns:
            DataFrame with comparison results.
        
        Examples:
            >>> comparison = NetworkAnalyzer.compare_prediction_methods(G, test_edges)
            >>> print(comparison.sort_values('f1', ascending=False))
        """
        if methods is None:
            methods = ['common_neighbors', 'adamic_adar', 'jaccard', 
                      'preferential_attachment', 'resource_allocation']
        
        analyzer = NetworkAnalyzer()
        analyzer.graph = graph
        
        results = []
        for method in methods:
            try:
                metrics = analyzer.evaluate_link_prediction(test_edges, method=method, top_k=top_k)
                results.append({
                    'method': method,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'true_positives': metrics['true_positives']
                })
            except Exception as e:
                results.append({
                    'method': method,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'true_positives': 0,
                    'error': str(e)
                })
        
        return pd.DataFrame(results).sort_values('f1', ascending=False)
