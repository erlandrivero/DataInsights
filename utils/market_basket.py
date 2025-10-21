"""
Market Basket Analysis utilities using Apriori algorithm.
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List, Tuple, Dict, Any
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

class MarketBasketAnalyzer:
    """Handles Market Basket Analysis operations."""
    
    def __init__(self):
        self.transactions = None
        self.df_encoded = None
        self.frequent_itemsets = None
        self.rules = None
    
    @staticmethod
    def load_groceries_data() -> List[List[str]]:
        """Load the groceries dataset from URL."""
        import requests
        
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            transactions = []
            for line in response.text.splitlines():
                transactions.append(line.split(','))
            
            return transactions
        except Exception as e:
            raise Exception(f"Error loading groceries data: {str(e)}")
    
    @staticmethod
    def parse_uploaded_transactions(df: pd.DataFrame, transaction_col: str, item_col: str) -> List[List[str]]:
        """
        Parse uploaded CSV into transaction format.
        
        Args:
            df: DataFrame with transaction data
            transaction_col: Column name for transaction ID
            item_col: Column name for items
            
        Returns:
            List of transactions (each transaction is a list of items)
        """
        # Clean the data before processing
        df_clean = df.copy()
        
        # Remove rows where item is NaN/None
        df_clean = df_clean.dropna(subset=[item_col])
        
        # Convert all items to strings and strip whitespace
        df_clean[item_col] = df_clean[item_col].astype(str).str.strip()
        
        # Remove empty strings
        df_clean = df_clean[df_clean[item_col] != '']
        df_clean = df_clean[df_clean[item_col] != 'nan']
        df_clean = df_clean[df_clean[item_col] != 'None']
        
        # Group by transaction and create lists
        transactions = df_clean.groupby(transaction_col)[item_col].apply(list).tolist()
        
        # Remove any empty transactions
        transactions = [t for t in transactions if len(t) > 0]
        
        return transactions
    
    def encode_transactions(self, transactions: List[List[str]]) -> pd.DataFrame:
        """
        One-hot encode transactions for Apriori algorithm.
        
        Args:
            transactions: List of transactions
            
        Returns:
            One-hot encoded DataFrame
        """
        self.transactions = transactions
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        return self.df_encoded
    
    def find_frequent_itemsets(self, min_support: float = 0.01) -> pd.DataFrame:
        """
        Find frequent itemsets using Apriori algorithm.
        
        Args:
            min_support: Minimum support threshold
            
        Returns:
            DataFrame of frequent itemsets
        """
        if self.df_encoded is None:
            raise ValueError("Transactions must be encoded first")
        
        self.frequent_itemsets = apriori(
            self.df_encoded, 
            min_support=min_support, 
            use_colnames=True
        )
        
        # Add itemset length
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        return self.frequent_itemsets
    
    def generate_association_rules(
        self, 
        metric: str = 'lift',
        min_threshold: float = 1.0,
        min_confidence: float = 0.2,
        min_support: float = 0.01
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            metric: Metric to use for filtering ('lift', 'confidence', 'support')
            min_threshold: Minimum threshold for the metric
            min_confidence: Minimum confidence
            min_support: Minimum support
            
        Returns:
            DataFrame of association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError("Frequent itemsets must be generated first")
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric=metric, 
            min_threshold=min_threshold
        )
        
        # Filter by confidence and support
        self.rules = self.rules[
            (self.rules['confidence'] >= min_confidence) & 
            (self.rules['support'] >= min_support)
        ]
        
        # Sort by lift, confidence, support
        self.rules = self.rules.sort_values(
            ['lift', 'confidence', 'support'], 
            ascending=False
        )
        
        return self.rules
    
    @staticmethod
    def format_itemset(itemset) -> str:
        """Format frozenset as readable string."""
        try:
            # Convert all items to strings and sort
            items = [str(item) for item in itemset]
            return '{' + ', '.join(sorted(items)) + '}'
        except Exception as e:
            # Fallback if sorting fails
            return '{' + ', '.join(str(item) for item in itemset) + '}'
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the rules."""
        if self.rules is None or len(self.rules) == 0:
            return {
                'total_rules': 0,
                'avg_support': 0,
                'avg_confidence': 0,
                'avg_lift': 0
            }
        
        return {
            'total_rules': len(self.rules),
            'avg_support': self.rules['support'].mean(),
            'avg_confidence': self.rules['confidence'].mean(),
            'avg_lift': self.rules['lift'].mean(),
            'max_lift': self.rules['lift'].max(),
            'max_confidence': self.rules['confidence'].max()
        }
    
    def get_top_items(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N most frequent items."""
        if self.df_encoded is None:
            return pd.DataFrame()
        
        item_freq = self.df_encoded.sum().sort_values(ascending=False).head(top_n)
        
        return pd.DataFrame({
            'Item': item_freq.index,
            'Frequency': item_freq.values,
            'Support': item_freq.values / len(self.df_encoded)
        })
    
    def create_scatter_plot(self) -> go.Figure:
        """Create scatter plot of Support vs Confidence (size = Lift)."""
        if self.rules is None or len(self.rules) == 0:
            return go.Figure()
        
        # Calculate bubble sizes based on lift
        sizes = 50 * (self.rules['lift'] - self.rules['lift'].min() + 0.1)
        
        fig = go.Figure(data=go.Scatter(
            x=self.rules['support'],
            y=self.rules['confidence'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=self.rules['lift'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Lift"),
                line=dict(width=1, color='white')
            ),
            text=[
                f"Rule: {self.format_itemset(row['antecedents'])} → {self.format_itemset(row['consequents'])}<br>"
                f"Support: {row['support']:.4f}<br>"
                f"Confidence: {row['confidence']:.4f}<br>"
                f"Lift: {row['lift']:.4f}"
                for _, row in self.rules.iterrows()
            ],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Association Rules: Support vs Confidence (size = Lift)',
            xaxis_title='Support',
            yaxis_title='Confidence',
            hovermode='closest',
            height=500
        )
        
        return fig
    
    def create_network_graph(self, top_n: int = 15) -> go.Figure:
        """Create network graph of top association rules."""
        if self.rules is None or len(self.rules) == 0:
            return go.Figure()
        
        # Get top N rules by lift
        top_rules = self.rules.nlargest(top_n, 'lift')
        
        # Create directed graph
        G = nx.DiGraph()
        
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            for ant in antecedents:
                for cons in consequents:
                    G.add_edge(ant, cons, weight=rule['lift'])
        
        # Generate layout
        pos = nx.spring_layout(G, seed=42, k=0.7)
        
        # Create edge traces
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title=f'Network Graph of Top {top_n} Association Rules',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def generate_insights(self, top_n: int = 5) -> List[str]:
        """Generate business insights from top rules."""
        if self.rules is None or len(self.rules) == 0:
            return ["No rules found. Try adjusting the thresholds."]
        
        insights = []
        top_rules = self.rules.nlargest(top_n, 'lift')
        
        for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
            ant = self.format_itemset(rule['antecedents'])
            cons = self.format_itemset(rule['consequents'])
            
            insight = (
                f"**{i}. {ant} → {cons}**\n"
                f"   - Customers who buy {ant} are **{rule['lift']:.2f}x** more likely to buy {cons}\n"
                f"   - This happens in **{rule['support']*100:.2f}%** of all transactions\n"
                f"   - When {ant} is purchased, {cons} is also purchased **{rule['confidence']*100:.1f}%** of the time\n"
                f"   - **Recommendation:** Place these items together or create a bundle promotion"
            )
            insights.append(insight)
        
        return insights
