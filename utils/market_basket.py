"""Market Basket Analysis utilities using Apriori algorithm for DataInsights.

This module provides comprehensive market basket analysis capabilities including
frequent itemset mining, association rule generation, and business insights.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List, Tuple, Dict, Any, Optional, Union
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px


class MarketBasketAnalyzer:
    """Handles Market Basket Analysis using the Apriori algorithm.
    
    This class provides complete functionality for analyzing transaction data
    to discover frequent itemsets and generate association rules for business
    insights like product recommendations and store layout optimization.
    
    Attributes:
        transactions (List[List[str]]): Original transaction data
        df_encoded (pd.DataFrame): One-hot encoded transaction matrix
        frequent_itemsets (pd.DataFrame): Discovered frequent itemsets
        rules (pd.DataFrame): Generated association rules
    
    Example:
        >>> # Basic workflow
        >>> analyzer = MarketBasketAnalyzer()
        >>> 
        >>> # Load data
        >>> transactions = analyzer.load_groceries_data()
        >>> 
        >>> # Encode and analyze
        >>> analyzer.encode_transactions(transactions)
        >>> itemsets = analyzer.find_frequent_itemsets(min_support=0.01)
        >>> rules = analyzer.generate_association_rules(
        >>>     metric='lift',
        >>>     min_threshold=1.0
        >>> )
        >>> 
        >>> # Get insights
        >>> insights = analyzer.generate_insights(top_n=5)
    
    Note:
        - Uses the Apriori algorithm for efficient itemset mining
        - Supports both sample datasets and user-uploaded data
        - All metrics (support, confidence, lift) are calculated automatically
    """
    
    def __init__(self):
        """Initialize MarketBasketAnalyzer with empty state.
        
        All attributes are set to None initially and populated through
        the analysis workflow methods.
        """
        self.transactions: Optional[List[List[str]]] = None
        self.df_encoded: Optional[pd.DataFrame] = None
        self.frequent_itemsets: Optional[pd.DataFrame] = None
        self.rules: Optional[pd.DataFrame] = None
    
    @staticmethod
    def load_groceries_data() -> List[List[str]]:
        """Load the standard groceries dataset from remote repository.
        
        Downloads the popular groceries dataset commonly used for market
        basket analysis examples and tutorials.
        
        Returns:
            List of transactions where each transaction is a list of item names
        
        Raises:
            Exception: If download fails or data cannot be parsed
        
        Example:
            >>> analyzer = MarketBasketAnalyzer()
            >>> transactions = analyzer.load_groceries_data()
            >>> print(f"Loaded {len(transactions)} transactions")
            Loaded 9835 transactions
        
        Note:
            - Data source: Machine Learning with R datasets repository
            - No preprocessing required - data is ready to use
            - Contains ~9,800 grocery store transactions
        """
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
    def parse_uploaded_transactions(
        df: pd.DataFrame, 
        transaction_col: str, 
        item_col: str
    ) -> List[List[str]]:
        """Parse uploaded CSV data into transaction format for analysis.
        
        Converts a DataFrame with transaction IDs and item columns into
        the list-of-lists format required by the Apriori algorithm. Includes
        comprehensive data cleaning to handle missing values and invalid items.
        
        Args:
            df: DataFrame containing transaction data
            transaction_col: Name of column containing transaction/basket IDs
            item_col: Name of column containing item names
        
        Returns:
            List of transactions where each transaction is a list of item strings
        
        Example:
            >>> # Sample DataFrame format
            >>> df = pd.DataFrame({
            >>>     'TransactionID': [1, 1, 2, 2, 3],
            >>>     'Item': ['Milk', 'Bread', 'Milk', 'Eggs', 'Bread']
            >>> })
            >>> 
            >>> analyzer = MarketBasketAnalyzer()
            >>> transactions = analyzer.parse_uploaded_transactions(
            >>>     df, 'TransactionID', 'Item'
            >>> )
            >>> print(transactions)
            [['Milk', 'Bread'], ['Milk', 'Eggs'], ['Bread']]
        
        Note:
            - Automatically removes NaN values and empty strings
            - Converts all items to strings
            - Strips whitespace from item names
            - Filters out common null representations ('nan', 'None', etc.)
            - Removes empty transactions
        """
        # Clean the data before processing
        df_clean = df.copy()
        
        # Remove rows where item is NaN/None
        df_clean = df_clean.dropna(subset=[item_col])
        
        # Convert all items to strings and strip whitespace
        df_clean.loc[:, item_col] = df_clean[item_col].astype(str).str.strip()
        
        # Remove empty strings and 'nan' strings
        df_clean = df_clean[df_clean[item_col].str.len() > 0]
        df_clean = df_clean[~df_clean[item_col].isin(['nan', 'None', 'NaN', 'NONE'])]
        
        # Group by transaction and create lists
        transactions = df_clean.groupby(transaction_col)[item_col].apply(
            lambda x: [str(i) for i in x]
        ).tolist()
        
        # Remove any empty transactions
        transactions = [t for t in transactions if len(t) > 0]
        
        return transactions
    
    def encode_transactions(self, transactions: List[List[str]]) -> pd.DataFrame:
        """One-hot encode transactions for Apriori algorithm processing.
        
        Transforms transaction data from list format to binary matrix format
        (one-hot encoding) where each row is a transaction and each column
        is an item. Cell values are True/False indicating item presence.
        
        Args:
            transactions: List of transactions (each transaction is a list of items)
        
        Returns:
            One-hot encoded DataFrame with True/False values
            - Rows: Transactions
            - Columns: Unique items
            - Values: Boolean indicating item presence
        
        Example:
            >>> transactions = [['Milk', 'Bread'], ['Milk'], ['Bread', 'Eggs']]
            >>> analyzer = MarketBasketAnalyzer()
            >>> encoded = analyzer.encode_transactions(transactions)
            >>> print(encoded)
               Bread   Eggs   Milk
            0   True  False   True
            1  False  False   True
            2   True   True  False
        
        Note:
            - Uses mlxtend's TransactionEncoder for efficient encoding
            - Automatically handles mixed types by converting to strings
            - Filters out null values and empty strings
            - Stores result in self.df_encoded for later use
        """
        self.transactions = transactions
        
        # Extra safety: ensure all items in all transactions are strings
        # This prevents mlxtend TransactionEncoder from encountering mixed types
        cleaned_transactions = []
        for transaction in transactions:
            # Convert each item to string, filter out 'nan' and empty strings
            cleaned_items = [
                str(item) for item in transaction 
                if str(item) not in ['nan', 'None', '']
            ]
            if cleaned_items:  # Only add non-empty transactions
                cleaned_transactions.append(cleaned_items)
        
        te = TransactionEncoder()
        te_ary = te.fit(cleaned_transactions).transform(cleaned_transactions)
        self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        return self.df_encoded
    
    def find_frequent_itemsets(self, min_support: float = 0.01, max_len: Optional[int] = None) -> pd.DataFrame:
        """Find frequent itemsets using the Apriori algorithm.
        
        Discovers all itemsets (combinations of items) that appear together
        in transactions with frequency above the minimum support threshold.
        
        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
                        - 0.01 = itemset must appear in at least 1% of transactions
                        - 0.1 = itemset must appear in at least 10% of transactions
                        - Lower values find more (but potentially less meaningful) itemsets
            max_len: Maximum length of itemsets to find (None = no limit)
                    - 2 = only find single items and pairs
                    - 3 = find up to 3-item combinations
                    - Limits memory usage for large datasets (Streamlit Cloud protection)
        
        Returns:
            DataFrame with columns:
                - support: Frequency of the itemset (0.0 to 1.0)
                - itemsets: Frozenset of items in the itemset
                - length: Number of items in the itemset
        
        Raises:
            ValueError: If transactions haven't been encoded yet
        
        Example:
            >>> analyzer.encode_transactions(transactions)
            >>> itemsets = analyzer.find_frequent_itemsets(min_support=0.02, max_len=3)
            >>> 
            >>> # View most frequent itemsets
            >>> print(itemsets.sort_values('support', ascending=False).head())
            >>> 
            >>> # Filter by itemset size
            >>> pairs = itemsets[itemsets['length'] == 2]
            >>> print(f"Found {len(pairs)} frequent pairs")
        
        Note:
            - Uses Apriori algorithm for efficient mining
            - Result is stored in self.frequent_itemsets
            - Required before generating association rules
            - Higher min_support = faster computation, fewer results
            - max_len limits memory usage (critical for Streamlit Cloud)
        """
        if self.df_encoded is None:
            raise ValueError("Transactions must be encoded first using encode_transactions()")
        
        self.frequent_itemsets = apriori(
            self.df_encoded, 
            min_support=min_support, 
            use_colnames=True,
            max_len=max_len
        )
        
        # Add itemset length for easy filtering
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        return self.frequent_itemsets
    
    def generate_association_rules(
        self, 
        metric: str = 'lift',
        min_threshold: float = 1.0,
        min_confidence: float = 0.2,
        min_support: float = 0.01
    ) -> pd.DataFrame:
        """Generate association rules from frequent itemsets.
        
        Creates "if-then" rules showing relationships between items, useful for
        product recommendations, bundling strategies, and store layout optimization.
        
        Args:
            metric: Metric to use for initial filtering
                   - 'lift': Strength of association (default)
                   - 'confidence': Conditional probability
                   - 'support': Frequency of rule
            min_threshold: Minimum value for the chosen metric
                          - For lift: 1.0 = no association, >1.0 = positive association
                          - For confidence: 0.0 to 1.0 (probability)
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
                           - 0.5 = consequent bought 50% of time when antecedent bought
            min_support: Minimum support threshold (0.0 to 1.0)
                        - 0.01 = rule must occur in at least 1% of transactions
        
        Returns:
            DataFrame with columns:
                - antecedents: Items on left side of rule (IF items)
                - consequents: Items on right side of rule (THEN items)
                - support: Frequency of complete rule
                - confidence: P(consequents | antecedents)
                - lift: Strength of association
                - And other metrics (leverage, conviction, etc.)
            Sorted by lift (descending), then confidence, then support
        
        Raises:
            ValueError: If frequent itemsets haven't been generated yet
        
        Example:
            >>> # Generate rules
            >>> rules = analyzer.generate_association_rules(
            >>>     metric='lift',
            >>>     min_threshold=2.0,  # 2x more likely than random
            >>>     min_confidence=0.3  # 30% probability
            >>> )
            >>> 
            >>> # View top rules
            >>> for _, rule in rules.head().iterrows():
            >>>     print(f"{rule['antecedents']} → {rule['consequents']}")
            >>>     print(f"  Lift: {rule['lift']:.2f}")
        
        Note:
            - Lift > 1.0 = positive association (items bought together)
            - Lift = 1.0 = no association (independent)
            - Lift < 1.0 = negative association (items rarely bought together)
            - Result stored in self.rules
        """
        if self.frequent_itemsets is None:
            raise ValueError(
                "Frequent itemsets must be generated first using find_frequent_itemsets()"
            )
        
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
    def format_itemset(itemset: Union[frozenset, set]) -> str:
        """Format frozenset or set as human-readable string.
        
        Converts frozenset to sorted, comma-separated string for display.
        
        Args:
            itemset: Frozenset or set of items
        
        Returns:
            Formatted string like "{Item1, Item2, Item3}"
        
        Example:
            >>> itemset = frozenset(['Milk', 'Bread', 'Eggs'])
            >>> formatted = MarketBasketAnalyzer.format_itemset(itemset)
            >>> print(formatted)
            {Bread, Eggs, Milk}
        
        Note:
            - Items are sorted alphabetically for consistency
            - Falls back to unsorted if sorting fails
        """
        try:
            # Convert all items to strings and sort
            items = [str(item) for item in itemset]
            return '{' + ', '.join(sorted(items)) + '}'
        except Exception as e:
            # Fallback if sorting fails
            return '{' + ', '.join(str(item) for item in itemset) + '}'
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated association rules.
        
        Calculates aggregate metrics for all discovered rules including
        averages and maximums for support, confidence, and lift.
        
        Returns:
            Dictionary containing:
                - total_rules (int): Number of rules found
                - avg_support (float): Average support across all rules
                - avg_confidence (float): Average confidence across all rules
                - avg_lift (float): Average lift across all rules
                - max_lift (float): Highest lift value (if rules exist)
                - max_confidence (float): Highest confidence value (if rules exist)
        
        Example:
            >>> analyzer.generate_association_rules()
            >>> summary = analyzer.get_rules_summary()
            >>> 
            >>> st.metric("Total Rules", summary['total_rules'])
            >>> st.metric("Average Lift", f"{summary['avg_lift']:.2f}")
        
        Note:
            - Returns zeros if no rules found
            - Useful for dashboard displays and reporting
        """
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
        """Get top N most frequently purchased items.
        
        Calculates item frequencies and support values from encoded transactions.
        
        Args:
            top_n: Number of top items to return (default: 10)
        
        Returns:
            DataFrame with columns:
                - Item: Item name
                - Frequency: Count of transactions containing item
                - Support: Proportion of transactions containing item (0.0 to 1.0)
            Sorted by frequency (descending)
        
        Example:
            >>> top_items = analyzer.get_top_items(top_n=5)
            >>> st.dataframe(top_items)
        
        Note:
            - Returns empty DataFrame if transactions not encoded yet
            - Support = Frequency / Total Transactions
        """
        if self.df_encoded is None:
            return pd.DataFrame()
        
        item_freq = self.df_encoded.sum().sort_values(ascending=False).head(top_n)
        
        return pd.DataFrame({
            'Item': item_freq.index,
            'Frequency': item_freq.values,
            'Support': item_freq.values / len(self.df_encoded)
        })
    
    def create_scatter_plot(self) -> go.Figure:
        """Create interactive scatter plot of association rules.
        
        Visualizes rules with support on x-axis, confidence on y-axis,
        and lift represented by bubble size and color.
        
        Returns:
            Plotly Figure object for display with st.plotly_chart()
        
        Example:
            >>> fig = analyzer.create_scatter_plot()
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            - Bubble size proportional to lift
            - Color intensity represents lift value
            - Hover shows complete rule details
            - Returns empty figure if no rules exist
        """
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
        """Create network graph visualization of top association rules.
        
        Shows items as nodes and rules as directed edges, useful for
        understanding complex item relationships.
        
        Args:
            top_n: Number of top rules to display (default: 15)
        
        Returns:
            Plotly Figure object showing network graph
        
        Example:
            >>> fig = analyzer.create_network_graph(top_n=20)
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            - Nodes: Items
            - Edges: Rules (directed from antecedent to consequent)
            - Edge weight represents lift value
            - Uses spring layout algorithm for positioning
            - Returns empty figure if no rules exist
        """
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
        """Generate actionable business insights from top association rules.
        
        Creates human-readable recommendations based on strongest rules,
        including specific business actions like bundling and placement.
        
        Args:
            top_n: Number of top rules to generate insights for (default: 5)
        
        Returns:
            List of formatted insight strings with markdown formatting
        
        Example:
            >>> insights = analyzer.generate_insights(top_n=3)
            >>> for insight in insights:
            >>>     st.markdown(insight)
        
        Note:
            - Insights sorted by lift (strongest associations first)
            - Each insight includes lift, support, confidence interpretations
            - Provides specific business recommendations
            - Returns helpful message if no rules found
        """
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
