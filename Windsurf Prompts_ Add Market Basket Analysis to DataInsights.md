# Windsurf Prompts: Add Market Basket Analysis to DataInsights

## Overview

These prompts will extend your existing **DataInsights** app with Market Basket Analysis functionality for Module 1 assignment (+2 extra credit).

**What You'll Add:**
- New "Market Basket Analysis" page
- Apriori algorithm implementation
- Association rules mining
- Interactive threshold controls
- Network graph visualization
- Business insights for retail patterns

**Total Prompts:** 6  
**Estimated Time:** 3-4 hours  
**Result:** Complete MBA functionality in your existing app

---

# PROMPT 1: Install Dependencies and Create MBA Utility Module

## Context
Add Market Basket Analysis capabilities to the existing DataInsights app by installing required libraries and creating a new utility module for MBA operations.

## Instructions

### Step 1: Update `requirements.txt`

Add these new dependencies to your existing `requirements.txt`:

```txt
# Existing dependencies remain...

# Market Basket Analysis
mlxtend==0.23.0
networkx==3.2.1
```

### Step 2: Create `utils/market_basket.py`

Create a new file `utils/market_basket.py`:

```python
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
        transactions = df.groupby(transaction_col)[item_col].apply(list).tolist()
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
        return '{' + ', '.join(sorted(list(itemset))) + '}'
    
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
```

## Testing

1. Install new dependencies:
```bash
pip install mlxtend==0.23.0 networkx==3.2.1
```

2. Test the module in Python:
```python
from utils.market_basket import MarketBasketAnalyzer

# Test loading data
mba = MarketBasketAnalyzer()
transactions = mba.load_groceries_data()
print(f"Loaded {len(transactions)} transactions")

# Test encoding
df_encoded = mba.encode_transactions(transactions)
print(f"Encoded shape: {df_encoded.shape}")

# Test Apriori
itemsets = mba.find_frequent_itemsets(min_support=0.01)
print(f"Found {len(itemsets)} frequent itemsets")

# Test rules
rules = mba.generate_association_rules(min_confidence=0.2)
print(f"Generated {len(rules)} association rules")
```

## Expected Output

- ✅ New dependencies installed
- ✅ `utils/market_basket.py` created
- ✅ `MarketBasketAnalyzer` class works
- ✅ Can load groceries data
- ✅ Can encode transactions
- ✅ Can find frequent itemsets
- ✅ Can generate association rules

## Review Checklist

- [ ] `requirements.txt` updated with mlxtend and networkx
- [ ] `utils/market_basket.py` created
- [ ] All methods in `MarketBasketAnalyzer` implemented
- [ ] Test script runs without errors
- [ ] Groceries data loads successfully
- [ ] Apriori algorithm works

---

# PROMPT 2: Add Market Basket Analysis Page to App

## Context
Add a new "Market Basket Analysis" page to the existing DataInsights Streamlit app navigation.

## Instructions

### Update `app.py` - Add MBA to Navigation

Find the navigation section in `app.py` (around line 65-75) and update it:

```python
def main():
    # Load custom CSS
    load_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio(
            "Select a page:",
            ["Home", "Data Upload", "Analysis", "Insights", "Reports", "Market Basket Analysis"],  # Added MBA
            key="navigation"
        )
        
        # ... rest of sidebar code ...
    
    # Page routing
    if page == "Home":
        show_home()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Analysis":
        show_analysis()
    elif page == "Insights":
        show_insights()
    elif page == "Reports":
        show_reports()
    elif page == "Market Basket Analysis":
        show_market_basket_analysis()  # New page
    
    # ... rest of main() ...
```

### Add MBA Page Function

Add this new function to `app.py` (before the `main()` function):

```python
def show_market_basket_analysis():
    """Market Basket Analysis page."""
    st.header("🧺 Market Basket Analysis")
    
    st.markdown("""
    Discover hidden patterns in transactional data using the **Apriori algorithm**.
    Find which items are frequently purchased together and generate actionable business insights.
    """)
    
    # Import MBA utilities
    from utils.market_basket import MarketBasketAnalyzer
    
    # Initialize analyzer in session state
    if 'mba' not in st.session_state:
        st.session_state.mba = MarketBasketAnalyzer()
    
    mba = st.session_state.mba
    
    # Data source selection
    st.subheader("📤 1. Load Transaction Data")
    
    data_source = st.radio(
        "Choose data source:",
        ["Sample Groceries Dataset", "Upload Custom Data"],
        key="mba_data_source"
    )
    
    transactions = None
    
    if data_source == "Sample Groceries Dataset":
        if st.button("📥 Load Groceries Data", type="primary"):
            with st.spinner("Loading groceries dataset..."):
                try:
                    transactions = mba.load_groceries_data()
                    st.session_state.mba_transactions = transactions
                    st.success(f"✅ Loaded {len(transactions)} transactions!")
                    
                    # Show sample
                    st.write("**Sample transactions:**")
                    for i, trans in enumerate(transactions[:5], 1):
                        st.write(f"{i}. {trans}")
                        
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    else:  # Upload custom data
        st.info("""
        **Upload Format:**
        - CSV file with two columns: `transaction_id` and `item`
        - Each row represents one item in a transaction
        - Example:
          ```
          transaction_id,item
          1,bread
          1,milk
          2,eggs
          2,bread
          ```
        """)
        
        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=['csv'],
            key="mba_upload"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Let user select columns
                col1, col2 = st.columns(2)
                with col1:
                    trans_col = st.selectbox("Transaction ID column:", df.columns, key="trans_col")
                with col2:
                    item_col = st.selectbox("Item column:", df.columns, key="item_col")
                
                if st.button("Process Uploaded Data", type="primary"):
                    transactions = mba.parse_uploaded_transactions(df, trans_col, item_col)
                    st.session_state.mba_transactions = transactions
                    st.success(f"✅ Processed {len(transactions)} transactions!")
                    
                    # Show sample
                    st.write("**Sample transactions:**")
                    for i, trans in enumerate(transactions[:5], 1):
                        st.write(f"{i}. {trans}")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Only show analysis if transactions are loaded
    if 'mba_transactions' not in st.session_state:
        st.info("👆 Load transaction data to begin analysis")
        return
    
    transactions = st.session_state.mba_transactions
    
    # Encode transactions
    if 'mba_encoded' not in st.session_state:
        with st.spinner("Encoding transactions..."):
            df_encoded = mba.encode_transactions(transactions)
            st.session_state.mba_encoded = df_encoded
    
    df_encoded = st.session_state.mba_encoded
    
    # Display dataset info
    st.divider()
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(transactions):,}")
    with col2:
        st.metric("Unique Items", f"{len(df_encoded.columns):,}")
    with col3:
        avg_basket = sum(len(t) for t in transactions) / len(transactions)
        st.metric("Avg Basket Size", f"{avg_basket:.1f}")
    
    # Threshold controls
    st.divider()
    st.subheader("🎛️ 2. Adjust Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Minimum frequency of itemsets (e.g., 0.01 = 1% of transactions)"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.05,
            format="%.2f",
            help="Minimum probability of consequent given antecedent"
        )
    
    with col3:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Minimum lift value (>1 means positive correlation)"
        )
    
    # Run analysis button
    if st.button("🚀 Run Market Basket Analysis", type="primary", use_container_width=True):
        with st.spinner("Mining frequent itemsets and generating rules..."):
            try:
                # Find frequent itemsets
                itemsets = mba.find_frequent_itemsets(min_support=min_support)
                st.session_state.mba_itemsets = itemsets
                
                # Generate rules
                rules = mba.generate_association_rules(
                    metric='lift',
                    min_threshold=min_lift,
                    min_confidence=min_confidence,
                    min_support=min_support
                )
                st.session_state.mba_rules = rules
                
                st.success(f"✅ Found {len(itemsets)} frequent itemsets and {len(rules)} association rules!")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Show results if available
    if 'mba_rules' in st.session_state:
        rules = st.session_state.mba_rules
        
        if len(rules) == 0:
            st.warning("⚠️ No rules found with current thresholds. Try lowering the values.")
        else:
            # Summary metrics
            st.divider()
            st.subheader("📈 Analysis Summary")
            
            summary = mba.get_rules_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rules", f"{summary['total_rules']:,}")
            with col2:
                st.metric("Avg Support", f"{summary['avg_support']:.4f}")
            with col3:
                st.metric("Avg Confidence", f"{summary['avg_confidence']:.3f}")
            with col4:
                st.metric("Avg Lift", f"{summary['avg_lift']:.2f}")
            
            # Placeholder for next prompts
            st.info("📋 Rules table, visualizations, and insights will be added in next steps...")

## Testing

1. Run the app: `streamlit run app.py`
2. Navigate to "Market Basket Analysis" page
3. Load sample groceries data
4. Verify dataset overview displays
5. Adjust thresholds
6. Click "Run Market Basket Analysis"
7. Verify analysis completes successfully

## Expected Output

- ✅ New "Market Basket Analysis" page in navigation
- ✅ Can load sample groceries dataset
- ✅ Can upload custom transaction data
- ✅ Dataset overview displays correctly
- ✅ Threshold sliders work
- ✅ Analysis runs successfully
- ✅ Summary metrics display

## Review Checklist

- [ ] Navigation includes "Market Basket Analysis"
- [ ] MBA page function added
- [ ] Can load groceries data
- [ ] Can upload custom data
- [ ] Dataset overview displays
- [ ] Threshold controls work
- [ ] Analysis runs without errors

---

# PROMPT 3: Add Association Rules Table and Export

## Context
Display the generated association rules in an interactive table with sorting and export capabilities.

## Instructions

### Update `show_market_basket_analysis()` in `app.py`

Replace the placeholder comment `# Placeholder for next prompts` with this code:

```python
            # Association Rules Table
            st.divider()
            st.subheader("📋 3. Association Rules")
            
            # Prepare display dataframe
            display_rules = rules.copy()
            
            # Format itemsets as strings
            display_rules['Antecedents'] = display_rules['antecedents'].apply(
                lambda x: mba.format_itemset(x)
            )
            display_rules['Consequents'] = display_rules['consequents'].apply(
                lambda x: mba.format_itemset(x)
            )
            
            # Select columns to display
            display_cols = [
                'Antecedents', 
                'Consequents', 
                'support', 
                'confidence', 
                'lift',
                'leverage',
                'conviction'
            ]
            
            # Rename for better display
            display_rules_formatted = display_rules[display_cols].copy()
            display_rules_formatted.columns = [
                'Antecedents (If)', 
                'Consequents (Then)', 
                'Support', 
                'Confidence', 
                'Lift',
                'Leverage',
                'Conviction'
            ]
            
            # Round numeric columns
            numeric_cols = ['Support', 'Confidence', 'Lift', 'Leverage', 'Conviction']
            display_rules_formatted[numeric_cols] = display_rules_formatted[numeric_cols].round(4)
            
            # Sorting options
            col1, col2 = st.columns([1, 3])
            with col1:
                sort_by = st.selectbox(
                    "Sort by:",
                    ['Lift', 'Confidence', 'Support', 'Leverage', 'Conviction'],
                    key="sort_by"
                )
            with col2:
                top_n = st.slider(
                    "Show top N rules:",
                    min_value=5,
                    max_value=min(100, len(display_rules_formatted)),
                    value=min(15, len(display_rules_formatted)),
                    step=5,
                    key="top_n_rules"
                )
            
            # Sort and display
            sorted_rules = display_rules_formatted.sort_values(sort_by, ascending=False).head(top_n)
            
            st.dataframe(
                sorted_rules,
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                # Export all rules as CSV
                csv = display_rules_formatted.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Rules (CSV)",
                    data=csv,
                    file_name=f"association_rules_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export top rules as CSV
                csv_top = sorted_rules.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Top {top_n} Rules (CSV)",
                    data=csv_top,
                    file_name=f"top_{top_n}_rules_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Search functionality
            with st.expander("🔍 Search Rules"):
                search_item = st.text_input(
                    "Search for item in rules:",
                    placeholder="e.g., whole milk",
                    key="search_item"
                )
                
                if search_item:
                    # Filter rules containing the search item
                    filtered_rules = display_rules_formatted[
                        display_rules_formatted['Antecedents (If)'].str.contains(search_item, case=False, na=False) |
                        display_rules_formatted['Consequents (Then)'].str.contains(search_item, case=False, na=False)
                    ]
                    
                    if len(filtered_rules) > 0:
                        st.write(f"**Found {len(filtered_rules)} rules containing '{search_item}':**")
                        st.dataframe(
                            filtered_rules.sort_values('Lift', ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(f"No rules found containing '{search_item}'")
```

## Testing

1. Run the app and navigate to MBA page
2. Load data and run analysis
3. Verify rules table displays correctly
4. Test sorting by different metrics
5. Adjust "top N" slider
6. Test download buttons (all rules and top N)
7. Test search functionality with different items
8. Verify CSV exports contain correct data

## Expected Output

- ✅ Rules table displays with formatted itemsets
- ✅ Sorting works for all metrics
- ✅ Top N slider filters correctly
- ✅ Download buttons work
- ✅ CSV exports are valid
- ✅ Search functionality works
- ✅ Filtered results display correctly

## Review Checklist

- [ ] Rules table displays correctly
- [ ] Itemsets are formatted as readable strings
- [ ] Sorting works for all metrics
- [ ] Top N slider functions properly
- [ ] Both download buttons work
- [ ] CSV files contain correct data
- [ ] Search functionality works
- [ ] Filtered results are accurate

---

# PROMPT 4: Add Interactive Visualizations

## Context
Add interactive visualizations to help users explore association rules visually.

## Instructions

### Update `show_market_basket_analysis()` in `app.py`

Add this code after the search functionality section:

```python
            # Visualizations
            st.divider()
            st.subheader("📈 4. Visualizations")
            
            # Tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "📊 Scatter Plot", 
                "🕸️ Network Graph", 
                "📊 Top Items"
            ])
            
            with viz_tab1:
                st.markdown("""
                **Support vs Confidence Scatter Plot**
                - Each point represents an association rule
                - Size of bubble indicates Lift value
                - Color intensity shows Lift strength
                - Hover for rule details
                """)
                
                fig_scatter = mba.create_scatter_plot()
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Interpretation guide
                with st.expander("💡 How to interpret this chart"):
                    st.markdown("""
                    - **Top-right corner:** High support AND high confidence (strong, frequent rules)
                    - **Large bubbles:** High lift (strong association)
                    - **Small bubbles:** Low lift (weak association)
                    - **Bottom-left:** Low support AND low confidence (weak, rare rules)
                    
                    **Best rules:** Look for large bubbles in the top-right area!
                    """)
            
            with viz_tab2:
                st.markdown("""
                **Network Graph of Item Associations**
                - Shows relationships between items
                - Arrows point from antecedent → consequent
                - Based on top rules by lift
                """)
                
                # Number of rules to show
                network_top_n = st.slider(
                    "Number of top rules to visualize:",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=5,
                    key="network_top_n"
                )
                
                fig_network = mba.create_network_graph(top_n=network_top_n)
                st.plotly_chart(fig_network, use_container_width=True)
                
                with st.expander("💡 How to interpret this graph"):
                    st.markdown("""
                    - **Nodes:** Individual items
                    - **Arrows:** Association rules (A → B means "if A, then B")
                    - **Clusters:** Items that frequently appear together
                    - **Central nodes:** Items involved in many rules
                    
                    **Business insight:** Items connected by arrows should be:
                    - Placed near each other in store
                    - Bundled in promotions
                    - Cross-promoted in marketing
                    """)
            
            with viz_tab3:
                st.markdown("""
                **Most Frequent Items**
                - Shows items that appear most often in transactions
                - Helps identify popular products
                """)
                
                top_items = mba.get_top_items(top_n=15)
                
                if not top_items.empty:
                    import plotly.express as px
                    
                    fig_items = px.bar(
                        top_items,
                        x='Frequency',
                        y='Item',
                        orientation='h',
                        title='Top 15 Most Frequent Items',
                        labels={'Frequency': 'Number of Transactions', 'Item': 'Item'},
                        color='Support',
                        color_continuous_scale='Blues'
                    )
                    
                    fig_items.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=500
                    )
                    
                    st.plotly_chart(fig_items, use_container_width=True)
                    
                    # Show table
                    st.dataframe(
                        top_items.style.format({
                            'Frequency': '{:,.0f}',
                            'Support': '{:.4f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No item frequency data available")
```

## Testing

1. Run the app and navigate to MBA page
2. Load data and run analysis
3. Navigate to each visualization tab:
   - **Scatter Plot:** Verify bubbles display, hover works, colors show lift
   - **Network Graph:** Adjust slider, verify graph updates, check node labels
   - **Top Items:** Verify bar chart displays, table shows correct data
4. Test interpretation expanders
5. Verify all charts are interactive (zoom, pan, hover)

## Expected Output

- ✅ Three visualization tabs display
- ✅ Scatter plot shows rules with bubble sizes
- ✅ Hover tooltips work on scatter plot
- ✅ Network graph displays item relationships
- ✅ Network graph slider updates visualization
- ✅ Top items bar chart displays
- ✅ Top items table shows frequency and support
- ✅ All charts are interactive

## Review Checklist

- [ ] Three visualization tabs created
- [ ] Scatter plot displays correctly
- [ ] Bubble sizes reflect lift values
- [ ] Hover tooltips show rule details
- [ ] Network graph displays
- [ ] Network slider works
- [ ] Top items chart displays
- [ ] Top items table is formatted correctly
- [ ] All charts are interactive (zoom, pan, hover)

---

# PROMPT 5: Add Business Insights and Recommendations

## Context
Generate AI-powered business insights and actionable recommendations based on the association rules.

## Instructions

### Update `show_market_basket_analysis()` in `app.py`

Add this code after the visualizations section:

```python
            # Business Insights
            st.divider()
            st.subheader("💡 5. Business Insights & Recommendations")
            
            st.markdown("""
            Based on the association rules discovered, here are actionable business recommendations:
            """)
            
            # Generate insights
            insights = mba.generate_insights(top_n=5)
            
            for insight in insights:
                st.markdown(insight)
                st.markdown("---")
            
            # Additional analysis
            with st.expander("📊 Advanced Insights"):
                st.markdown("### Rule Strength Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Lift distribution
                    import plotly.express as px
                    
                    fig_lift = px.histogram(
                        rules,
                        x='lift',
                        nbins=30,
                        title='Distribution of Lift Values',
                        labels={'lift': 'Lift', 'count': 'Number of Rules'}
                    )
                    fig_lift.add_vline(
                        x=rules['lift'].mean(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {rules['lift'].mean():.2f}"
                    )
                    st.plotly_chart(fig_lift, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    fig_conf = px.histogram(
                        rules,
                        x='confidence',
                        nbins=30,
                        title='Distribution of Confidence Values',
                        labels={'confidence': 'Confidence', 'count': 'Number of Rules'}
                    )
                    fig_conf.add_vline(
                        x=rules['confidence'].mean(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {rules['confidence'].mean():.2f}"
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Top antecedents and consequents
                st.markdown("### Most Common Items in Rules")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Antecedents (If):**")
                    # Count antecedents
                    ant_items = []
                    for itemset in rules['antecedents']:
                        ant_items.extend(list(itemset))
                    
                    ant_counts = pd.Series(ant_items).value_counts().head(10)
                    st.dataframe(
                        pd.DataFrame({
                            'Item': ant_counts.index,
                            'Appears in Rules': ant_counts.values
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("**Top Consequents (Then):**")
                    # Count consequents
                    cons_items = []
                    for itemset in rules['consequents']:
                        cons_items.extend(list(itemset))
                    
                    cons_counts = pd.Series(cons_items).value_counts().head(10)
                    st.dataframe(
                        pd.DataFrame({
                            'Item': cons_counts.index,
                            'Appears in Rules': cons_counts.values
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Business strategies
            with st.expander("🎯 Strategic Recommendations"):
                st.markdown("""
                ### How to Use These Insights
                
                #### 1. **Store Layout Optimization**
                - Place frequently associated items near each other
                - Create "discovery zones" for high-lift pairs
                - Use end-cap displays for complementary products
                
                #### 2. **Promotional Strategies**
                - **Bundle Deals:** Combine items with high confidence
                - **Cross-Promotions:** "Customers who bought X also bought Y"
                - **Discount Strategies:** Discount antecedent to drive consequent sales
                
                #### 3. **Inventory Management**
                - Stock associated items proportionally
                - Predict demand for consequents based on antecedent sales
                - Avoid stockouts of frequently paired items
                
                #### 4. **Marketing & Personalization**
                - **Email Campaigns:** Recommend consequents to antecedent buyers
                - **Website Recommendations:** "You might also like..."
                - **Targeted Ads:** Show consequent ads to antecedent purchasers
                
                #### 5. **Product Development**
                - Create new products combining popular associations
                - Develop private-label bundles
                - Design combo packages
                
                ### Metrics to Track
                - **Basket Size:** Average items per transaction
                - **Attachment Rate:** % of antecedent buyers who also buy consequent
                - **Bundle Performance:** Sales lift from bundled promotions
                - **Cross-Sell Success:** Conversion rate of recommendations
                """)
            
            # Export full report
            st.divider()
            
            if st.button("📄 Generate Full Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    # Create report content
                    report = f"""
# Market Basket Analysis Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- **Total Transactions:** {len(transactions):,}
- **Unique Items:** {len(df_encoded.columns):,}
- **Average Basket Size:** {sum(len(t) for t in transactions) / len(transactions):.2f}

## Analysis Parameters
- **Minimum Support:** {min_support}
- **Minimum Confidence:** {min_confidence}
- **Minimum Lift:** {min_lift}

## Results
- **Frequent Itemsets Found:** {len(st.session_state.mba_itemsets):,}
- **Association Rules Generated:** {len(rules):,}
- **Average Support:** {summary['avg_support']:.4f}
- **Average Confidence:** {summary['avg_confidence']:.4f}
- **Average Lift:** {summary['avg_lift']:.2f}

## Top 10 Association Rules

{sorted_rules.head(10).to_markdown(index=False)}

## Business Insights

{chr(10).join(insights)}

## Recommendations

Based on this analysis, we recommend:

1. **Product Placement:** Position highly associated items in proximity
2. **Promotional Bundles:** Create bundles from high-confidence rules
3. **Cross-Selling:** Implement recommendation systems based on these rules
4. **Inventory Planning:** Stock associated items proportionally
5. **Marketing Campaigns:** Target customers with personalized recommendations

---
*Report generated by DataInsight AI - Market Basket Analysis Module*
"""
                    
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=report,
                        file_name=f"mba_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated! Click download button above.")
```

## Testing

1. Run the app and complete MBA analysis
2. Verify business insights display
3. Check that top 5 rules have insights
4. Test "Advanced Insights" expander:
   - Lift distribution histogram
   - Confidence distribution histogram
   - Top antecedents table
   - Top consequents table
5. Test "Strategic Recommendations" expander
6. Click "Generate Full Report" button
7. Verify report downloads correctly
8. Check report content is complete

## Expected Output

- ✅ Business insights display for top 5 rules
- ✅ Insights are actionable and specific
- ✅ Advanced insights expander works
- ✅ Distribution histograms display
- ✅ Top items tables show correctly
- ✅ Strategic recommendations display
- ✅ Full report generates
- ✅ Report download works
- ✅ Report content is comprehensive

## Review Checklist

- [ ] Business insights display correctly
- [ ] Insights are specific and actionable
- [ ] Advanced insights section works
- [ ] Lift distribution shows
- [ ] Confidence distribution shows
- [ ] Top antecedents/consequents display
- [ ] Strategic recommendations section works
- [ ] Full report button works
- [ ] Report downloads successfully
- [ ] Report content is complete and formatted

---

# PROMPT 6: Final Polish and Testing

## Context
Add final touches, error handling, and comprehensive testing to ensure the MBA module is production-ready.

## Instructions

### Add Help Section

Add this at the beginning of `show_market_basket_analysis()` function:

```python
def show_market_basket_analysis():
    """Market Basket Analysis page."""
    st.header("🧺 Market Basket Analysis")
    
    # Help section
    with st.expander("ℹ️ What is Market Basket Analysis?"):
        st.markdown("""
        **Market Basket Analysis (MBA)** is a data mining technique that discovers relationships 
        between items in transactional data.
        
        ### Key Concepts:
        
        - **Support:** How frequently an itemset appears in transactions
          - Formula: `support(A) = transactions containing A / total transactions`
          - Example: If milk appears in 500 of 1000 transactions, support = 0.5
        
        - **Confidence:** Probability of buying B given A was purchased
          - Formula: `confidence(A→B) = support(A,B) / support(A)`
          - Example: If 80% of milk buyers also buy bread, confidence = 0.8
        
        - **Lift:** How much more likely B is purchased when A is purchased
          - Formula: `lift(A→B) = support(A,B) / (support(A) × support(B))`
          - Lift > 1: Positive correlation (items bought together)
          - Lift = 1: No correlation (independent)
          - Lift < 1: Negative correlation (items not bought together)
        
        ### The Apriori Algorithm:
        
        1. **Find frequent itemsets:** Items/combinations that appear often
        2. **Generate rules:** Create "if-then" associations
        3. **Filter by metrics:** Keep only strong, meaningful rules
        
        ### Business Applications:
        
        - 🛒 **Retail:** Product placement, bundling, promotions
        - 🎬 **Entertainment:** Movie/music recommendations
        - 🏥 **Healthcare:** Symptom co-occurrence, treatment patterns
        - 📚 **Education:** Course recommendations
        - 🍔 **Food Service:** Menu combinations, upselling
        """)
    
    st.markdown("""
    Discover hidden patterns in transactional data using the **Apriori algorithm**.
    Find which items are frequently purchased together and generate actionable business insights.
    """)
    
    # ... rest of existing code ...
```

### Add Error Handling and Validation

Update the "Run Market Basket Analysis" button section with better error handling:

```python
    # Run analysis button
    if st.button("🚀 Run Market Basket Analysis", type="primary", use_container_width=True):
        with st.spinner("Mining frequent itemsets and generating rules..."):
            try:
                # Validate thresholds
                if min_support <= 0 or min_support > 1:
                    st.error("❌ Minimum support must be between 0 and 1")
                    return
                
                if min_confidence <= 0 or min_confidence > 1:
                    st.error("❌ Minimum confidence must be between 0 and 1")
                    return
                
                if min_lift < 0:
                    st.error("❌ Minimum lift must be positive")
                    return
                
                # Find frequent itemsets
                itemsets = mba.find_frequent_itemsets(min_support=min_support)
                
                if len(itemsets) == 0:
                    st.warning(f"⚠️ No frequent itemsets found with support >= {min_support}. Try lowering the minimum support.")
                    return
                
                st.session_state.mba_itemsets = itemsets
                
                # Generate rules
                rules = mba.generate_association_rules(
                    metric='lift',
                    min_threshold=min_lift,
                    min_confidence=min_confidence,
                    min_support=min_support
                )
                
                if len(rules) == 0:
                    st.warning(f"""
                    ⚠️ No association rules found with current thresholds:
                    - Support >= {min_support}
                    - Confidence >= {min_confidence}
                    - Lift >= {min_lift}
                    
                    **Try:**
                    - Lowering minimum support
                    - Lowering minimum confidence
                    - Lowering minimum lift
                    """)
                    return
                
                st.session_state.mba_rules = rules
                
                st.success(f"✅ Found {len(itemsets)} frequent itemsets and {len(rules)} association rules!")
                
            except ValueError as e:
                st.error(f"❌ Validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try adjusting the thresholds or checking your data format.")
```

### Add README Section for MBA

Create a new file `MBA_GUIDE.md` in your repository:

```markdown
# Market Basket Analysis Module - User Guide

## Overview

The Market Basket Analysis (MBA) module in DataInsight AI allows you to discover hidden patterns in transactional data using the Apriori algorithm.

## Getting Started

### 1. Load Data

**Option A: Sample Dataset**
- Click "Load Groceries Data" to use the sample grocery transactions dataset
- Contains 9,835 transactions with 169 unique items

**Option B: Upload Custom Data**
- Prepare CSV file with columns: `transaction_id`, `item`
- Each row = one item in a transaction
- Example:
  ```csv
  transaction_id,item
  1,bread
  1,milk
  2,eggs
  2,bread
  ```

### 2. Adjust Thresholds

- **Minimum Support** (0.001-0.1): How often itemsets appear
  - Lower = more rules, but less frequent
  - Higher = fewer rules, but more common
  - Recommended: Start with 0.01 (1%)

- **Minimum Confidence** (0.1-1.0): Strength of association
  - Lower = more rules, weaker associations
  - Higher = fewer rules, stronger associations
  - Recommended: Start with 0.2 (20%)

- **Minimum Lift** (1.0-5.0): Correlation strength
  - 1.0 = no correlation
  - >1.0 = positive correlation (items bought together)
  - Recommended: Start with 1.0

### 3. Run Analysis

- Click "Run Market Basket Analysis"
- Wait for processing (usually 5-30 seconds)
- View results in tables and visualizations

### 4. Explore Results

**Association Rules Table:**
- Sort by Lift, Confidence, or Support
- Download as CSV
- Search for specific items

**Visualizations:**
- **Scatter Plot:** See rule strength visually
- **Network Graph:** Understand item relationships
- **Top Items:** Identify most frequent products

**Business Insights:**
- Read AI-generated recommendations
- Review strategic suggestions
- Download full report

## Understanding the Metrics

### Support
**What it means:** How often an itemset appears

**Example:** Support({milk}) = 0.25 means milk appears in 25% of transactions

**Business use:** Identify popular products

### Confidence
**What it means:** Probability of buying B given A was purchased

**Example:** Confidence({milk} → {bread}) = 0.6 means 60% of milk buyers also buy bread

**Business use:** Predict customer behavior

### Lift
**What it means:** How much more likely items are bought together vs. independently

**Example:** Lift({milk} → {bread}) = 2.0 means buying milk makes you 2x more likely to buy bread

**Business use:** Find strong associations

## Business Applications

### Retail
- **Product Placement:** Put associated items near each other
- **Bundling:** Create combo deals from high-confidence rules
- **Promotions:** "Buy X, get Y at discount"

### E-commerce
- **Recommendations:** "Customers who bought X also bought Y"
- **Upselling:** Suggest complementary products
- **Personalization:** Tailor homepage to purchase history

### Inventory
- **Stock Planning:** Order associated items proportionally
- **Demand Forecasting:** Predict consequent sales from antecedent sales
- **Warehouse Layout:** Store related items together

## Tips for Best Results

1. **Start Conservative:** Begin with default thresholds, then adjust
2. **Focus on Lift:** Rules with lift > 2 are usually most actionable
3. **Consider Context:** Not all high-lift rules make business sense
4. **Test Strategies:** Implement recommendations and measure results
5. **Iterate:** Rerun analysis as customer behavior changes

## Troubleshooting

**No rules found:**
- Lower minimum support (try 0.005)
- Lower minimum confidence (try 0.15)
- Check data format is correct

**Too many rules:**
- Raise minimum support (try 0.02)
- Raise minimum confidence (try 0.3)
- Raise minimum lift (try 1.5)

**Analysis is slow:**
- Reduce dataset size
- Increase minimum support
- Use fewer transactions

## Example Workflow

1. Load groceries dataset
2. Set thresholds: support=0.01, confidence=0.2, lift=1.0
3. Run analysis
4. Sort rules by Lift (descending)
5. Review top 10 rules
6. Check network graph for clusters
7. Read business insights
8. Download report
9. Implement recommendations
10. Track results

## Support

For questions or issues:
- Check this guide
- Review the "What is MBA?" section in the app
- Consult the main README.md

---

**Made with ❤️ for Data Mining Module 1 Assignment**
```

### Update Main README

Add this section to your main `README.md`:

```markdown
## Market Basket Analysis Module

DataInsight AI now includes a comprehensive **Market Basket Analysis** module for discovering patterns in transactional data.

### Features:
- 🧺 Apriori algorithm implementation
- 📊 Interactive threshold controls
- 📈 Multiple visualizations (scatter, network, bar charts)
- 💡 AI-generated business insights
- 📥 Export rules and reports
- 🔍 Search and filter functionality

### Quick Start:
1. Navigate to "Market Basket Analysis" page
2. Load sample groceries data or upload your own
3. Adjust thresholds (support, confidence, lift)
4. Click "Run Market Basket Analysis"
5. Explore results and download reports

See [MBA_GUIDE.md](MBA_GUIDE.md) for detailed instructions.
```

## Final Testing Checklist

Test the complete MBA module:

### Data Loading
- [ ] Sample groceries data loads successfully
- [ ] Custom CSV upload works
- [ ] Column selection works for custom data
- [ ] Dataset overview displays correctly

### Analysis
- [ ] Threshold sliders work
- [ ] Analysis runs without errors
- [ ] Handles no results gracefully
- [ ] Summary metrics display correctly

### Rules Table
- [ ] Rules display in table
- [ ] Sorting works for all columns
- [ ] Top N slider filters correctly
- [ ] Download buttons work
- [ ] CSV exports are valid
- [ ] Search functionality works

### Visualizations
- [ ] Scatter plot displays
- [ ] Hover tooltips work
- [ ] Network graph displays
- [ ] Network slider updates graph
- [ ] Top items chart displays
- [ ] All charts are interactive

### Insights
- [ ] Business insights generate
- [ ] Advanced insights display
- [ ] Distribution charts show
- [ ] Top items tables work
- [ ] Strategic recommendations display
- [ ] Full report generates
- [ ] Report downloads correctly

### Error Handling
- [ ] Invalid thresholds show errors
- [ ] No data scenario handled
- [ ] No rules scenario handled
- [ ] Upload errors handled gracefully

### Documentation
- [ ] Help section displays
- [ ] MBA_GUIDE.md created
- [ ] README updated
- [ ] All instructions clear

## Expected Output

After completing all prompts:

- ✅ Fully functional Market Basket Analysis module
- ✅ Sample data and custom upload support
- ✅ Interactive threshold controls
- ✅ Comprehensive visualizations
- ✅ Business insights and recommendations
- ✅ Export capabilities
- ✅ Complete documentation
- ✅ Production-ready code

## Deployment

1. Commit all changes to GitHub
2. Push to main branch
3. Streamlit Cloud will auto-deploy
4. Test deployed app thoroughly
5. Submit app URL for Module 1 assignment

## Submission Checklist

For Module 1 Assignment:

- [ ] Jupyter notebook with MBA analysis
- [ ] Deployed app URL
- [ ] README with instructions
- [ ] MBA_GUIDE.md included
- [ ] All features working
- [ ] Screenshots for documentation

---

## 🎉 Congratulations!

You've successfully added Market Basket Analysis to DataInsight AI!

**Total Implementation:**
- 6 detailed prompts
- ~3-4 hours of work
- Production-ready MBA module
- +2 extra credit points

**Your app now has:**
- General data analysis (original)
- AI-powered insights (original)
- **Market Basket Analysis (NEW!)**

**Perfect for:**
- Module 1 Assignment (+2 extra credit)
- Portfolio project
- Real-world business applications

---

*End of Windsurf Prompts - Market Basket Analysis Module*

