import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DataInsight AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def load_custom_css():
    """Load custom CSS from file."""
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file not found, use default styles

# Main app
def main():
    # Load custom CSS
    load_custom_css()
    # Header
    st.markdown('<div class="main-header">🎯 DataInsight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-Powered Business Intelligence Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio(
            "Select a page:",
            ["Home", "Data Upload", "Analysis", "Insights", "Reports", "Market Basket Analysis", "RFM Analysis", "Monte Carlo Simulation"],
            key="navigation"
        )
        
        st.divider()
        
        # Export section
        if st.session_state.data is not None:
            st.header("📥 Quick Export")
            
            from utils.export_helper import ExportHelper
            from datetime import datetime
            export = ExportHelper()
            
            df = st.session_state.data
            profile = st.session_state.get('profile', {})
            issues = st.session_state.get('issues', [])
            
            # Export data
            export_format = st.selectbox(
                "Export data as:",
                ["CSV", "Excel", "JSON"],
                key="export_format"
            )
            
            if st.button("📥 Export Data", use_container_width=True):
                try:
                    format_map = {'CSV': 'csv', 'Excel': 'excel', 'JSON': 'json'}
                    data = export.export_cleaned_data(df, format_map[export_format])
                    
                    file_ext = format_map[export_format]
                    if file_ext == 'excel':
                        file_ext = 'xlsx'
                    
                    st.download_button(
                        label=f"Download {export_format}",
                        data=data,
                        file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                        mime=f"application/{file_ext}",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
            
            # Export data dictionary
            if st.button("📚 Export Data Dictionary", use_container_width=True):
                dictionary = export.create_data_dictionary(df, profile)
                st.download_button(
                    label="Download Dictionary",
                    data=dictionary,
                    file_name=f"data_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # Export analysis summary
            if st.button("📊 Export Analysis", use_container_width=True):
                summary = export.export_analysis_summary(profile, issues)
                st.download_button(
                    label="Download Analysis",
                    data=summary,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        **DataInsight AI** helps you:
        - Upload and analyze data
        - Ask questions in natural language
        - Generate automated insights
        - Create professional reports
        """)
    
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
        show_market_basket_analysis()
    elif page == "RFM Analysis":
        show_rfm_analysis()
    elif page == "Monte Carlo Simulation":
        show_monte_carlo_simulation()

def show_home():
    st.header("Welcome to DataInsight AI! 👋")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📤 Upload Data</h3>
            <p>Upload CSV or Excel files and get instant data profiling<br>&nbsp;</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI Analysis</h3>
            <p>Ask questions in natural language and get intelligent answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Visualizations</h3>
            <p>Interactive charts and dashboards generated automatically<br>&nbsp;</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🚀 Getting Started")
    st.write("""
    1. Navigate to **Data Upload** to upload your dataset
    2. View automatic **Analysis** of your data
    3. Ask questions and get **Insights** from AI
    4. Generate professional **Reports** to share
    """)

def show_data_upload():
    st.header("📤 Data Upload")
    
    # Tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["📁 Local Upload", "🌐 OpenML", "🏆 Kaggle"])
    
    with tab1:
        st.subheader("Upload from Your Computer")
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to get started with analysis"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and analyzing your data..."):
                    from utils.data_processor import DataProcessor
                    
                    # Load data
                    df = DataProcessor.load_data(uploaded_file)
                    st.session_state.data = df
                    
                    # Profile data
                    profile = DataProcessor.profile_data(df)
                    st.session_state.profile = profile
                    
                    # Detect issues
                    issues = DataProcessor.detect_data_quality_issues(df)
                    st.session_state.issues = issues
                
                st.success(f"✅ Successfully loaded {uploaded_file.name}!")
                
                # Display basic info
                st.subheader("📋 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", profile['basic_info']['rows'])
                with col2:
                    st.metric("Columns", profile['basic_info']['columns'])
                with col3:
                    st.metric("Duplicates", profile['basic_info']['duplicates'])
                with col4:
                    st.metric("Memory", profile['basic_info']['memory_usage'])
                
                # Data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column information
                st.subheader("📊 Column Information")
                col_df = pd.DataFrame(profile['column_info'])
                st.dataframe(
                    col_df[['name', 'dtype', 'missing', 'missing_pct', 'unique', 'unique_pct']],
                    use_container_width=True
                )
                
                # Data quality issues
                if issues:
                    st.subheader("⚠️ Data Quality Issues")
                    for issue in issues:
                        severity_color = {
                            'High': '🔴',
                            'Medium': '🟡',
                            'Low': '🟢'
                        }
                        st.warning(
                            f"{severity_color[issue['severity']]} **{issue['type']}** in `{issue['column']}`: {issue['description']}"
                        )
                else:
                    st.success("✅ No significant data quality issues detected!")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            # Show sample data option
            st.info("💡 Don't have data? Try our sample datasets!")
        
        if st.button("Load Sample Sales Data"):
            # Create sample data
            import numpy as np
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'product': np.random.choice(['Product A', 'Product B', 'Product C'], 100),
                'revenue': np.random.uniform(100, 1000, 100),
                'quantity': np.random.randint(1, 20, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
            })
            st.session_state.data = sample_data
            st.rerun()
    
    with tab2:
        st.subheader("Load from OpenML")
        st.markdown("""
        [OpenML](https://www.openml.org/) is a public repository with thousands of machine learning datasets.
        Browse datasets at: [openml.org/search?type=data](https://www.openml.org/search?type=data)
        """)
        
        # Dataset selection method
        openml_method = st.radio(
            "Select dataset by:",
            ["Popular Datasets", "Dataset ID"],
            horizontal=True,
            key="openml_method"
        )
        
        dataset_id = None
        dataset_name = None
        
        if openml_method == "Popular Datasets":
            # Popular OpenML datasets
            openml_datasets = {
                "Iris": 61,
                "Titanic": 40945,
                "Wine Quality": 187,
                "Diabetes": 37,
                "Credit-g": 31,
                "Adult": 1590,
                "Boston Housing": 531,
                "MNIST (Small)": 554
            }
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                dataset_name = st.selectbox(
                    "Choose a dataset:",
                    list(openml_datasets.keys()),
                    key="openml_dataset"
                )
                dataset_id = openml_datasets[dataset_name]
                st.caption(f"Dataset ID: {dataset_id}")
            
            with col2:
                load_button = st.button("📥 Load Dataset", type="primary", use_container_width=True, key="load_openml_popular")
        
        else:  # Custom Dataset ID
            col1, col2 = st.columns([2, 1])
            
            with col1:
                dataset_id = st.number_input(
                    "Enter OpenML Dataset ID:",
                    min_value=1,
                    value=61,
                    step=1,
                    help="Find dataset IDs at openml.org/search",
                    key="openml_custom_id"
                )
                st.caption("💡 Example IDs: 61 (Iris), 40945 (Titanic), 187 (Wine)")
            
            with col2:
                load_button = st.button("📥 Load Dataset", type="primary", use_container_width=True, key="load_openml_custom")
        
        # Load button handler
        if load_button:
            with st.spinner(f"Loading dataset {dataset_id} from OpenML..."):
                try:
                    import openml
                    
                    dataset = openml.datasets.get_dataset(dataset_id)
                    X, y, categorical_indicator, attribute_names = dataset.get_data(
                        dataset_format="dataframe"
                    )
                    
                    # Combine features and target
                    if y is not None:
                        df = X.copy()
                        df['target'] = y
                    else:
                        df = X
                    
                    st.session_state.data = df
                    
                    # Profile data
                    from utils.data_processor import DataProcessor
                    profile = DataProcessor.profile_data(df)
                    st.session_state.profile = profile
                    
                    issues = DataProcessor.detect_data_quality_issues(df)
                    st.session_state.issues = issues
                    
                    success_msg = f"✅ Successfully loaded {dataset.name} (ID: {dataset_id})!"
                    st.success(success_msg)
                    st.info(f"**Description:** {dataset.description[:300]}...")
                    
                    # Show dataset info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Dataset ID", dataset_id)
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading OpenML dataset: {str(e)}")
                    st.info("""
                    💡 **Troubleshooting:**
                    - Verify the dataset ID exists on OpenML
                    - Some datasets may require additional processing
                    - Check your internet connection
                    """)
    
    with tab3:
        st.subheader("Load from Kaggle")
        st.markdown("""
        [Kaggle](https://www.kaggle.com/) hosts thousands of datasets for data science competitions.
        **Requires:** Kaggle API credentials (kaggle.json)
        """)
        
        # Kaggle setup instructions
        with st.expander("🔧 How to set up Kaggle API"):
            st.markdown("""
            1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
            2. Scroll to **API** section
            3. Click **Create New API Token**
            4. Download `kaggle.json` file
            5. For **Streamlit Cloud deployment:**
               - Add to Secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY`
            6. For **local use:**
               - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\\Users\\<username>\\.kaggle\\` (Windows)
            """)
        
        # Kaggle dataset input
        kaggle_dataset = st.text_input(
            "Enter Kaggle dataset (format: username/dataset-name):",
            placeholder="e.g., uciml/iris or heptapod/titanic",
            key="kaggle_dataset"
        )
        
        if st.button("📥 Load Kaggle Dataset", type="primary", disabled=not kaggle_dataset):
            with st.spinner(f"Downloading {kaggle_dataset} from Kaggle..."):
                try:
                    import os
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    
                    # Check for Streamlit secrets first (for cloud deployment)
                    if hasattr(st, 'secrets') and 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
                        os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                        os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
                        st.info("🔑 Using Kaggle credentials from Streamlit Secrets")
                    
                    # Initialize Kaggle API (will use environment variables or kaggle.json)
                    api = KaggleApi()
                    api.authenticate()
                    
                    # Download dataset
                    download_path = "./kaggle_data"
                    os.makedirs(download_path, exist_ok=True)
                    
                    api.dataset_download_files(kaggle_dataset, path=download_path, unzip=True)
                    
                    # Find CSV files
                    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
                    
                    if csv_files:
                        # Load first CSV file
                        csv_file = csv_files[0]
                        df = pd.read_csv(os.path.join(download_path, csv_file))
                        
                        st.session_state.data = df
                        
                        # Profile data
                        from utils.data_processor import DataProcessor
                        profile = DataProcessor.profile_data(df)
                        st.session_state.profile = profile
                        
                        issues = DataProcessor.detect_data_quality_issues(df)
                        st.session_state.issues = issues
                        
                        st.success(f"✅ Successfully loaded {csv_file} from Kaggle!")
                        
                        if len(csv_files) > 1:
                            st.info(f"📁 Found {len(csv_files)} CSV files. Loaded: {csv_file}")
                            st.write("Available files:", csv_files)
                        
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Cleanup
                        import shutil
                        shutil.rmtree(download_path)
                    else:
                        st.warning("⚠️ No CSV files found in the dataset.")
                        
                except Exception as e:
                    st.error(f"Error loading Kaggle dataset: {str(e)}")
                    st.info("""
                    💡 **Troubleshooting:**
                    - Ensure Kaggle API credentials are configured
                    - Check dataset name format (username/dataset-name)
                    - Verify dataset exists and is public
                    """)

def show_analysis():
    st.header("📊 Data Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    profile = st.session_state.get('profile', {})
    issues = st.session_state.get('issues', [])
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistics", "📊 Visualizations", "🤖 AI Insights", "🔧 Cleaning Suggestions"])
    
    with tab1:
        st.subheader("Statistical Summary")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.write("**Categorical Columns:**")
            for col in categorical_cols[:5]:  # Show first 5
                st.write(f"**{col}** - Value Counts:")
                st.dataframe(df[col].value_counts().head(10), use_container_width=True)
    
    with tab2:
        st.subheader("📊 Data Visualizations")
        
        from utils.visualizations import Visualizer
        viz = Visualizer()
        
        # Get visualization suggestions
        if 'viz_suggestions' not in st.session_state:
            st.session_state.viz_suggestions = viz.suggest_visualizations(df)
        
        suggestions = st.session_state.viz_suggestions
        
        st.write("**Suggested Visualizations:**")
        
        # Display suggestions
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"📈 {suggestion['title']}"):
                st.write(suggestion['description'])
                
                try:
                    if suggestion['type'] == 'histogram':
                        fig = viz.create_histogram(df, suggestion['columns'][0], suggestion['title'])
                    elif suggestion['type'] == 'bar':
                        fig = viz.create_bar_chart(df, suggestion['columns'][0], suggestion['title'])
                    elif suggestion['type'] == 'scatter':
                        fig = viz.create_scatter(df, suggestion['columns'][0], suggestion['columns'][1], suggestion['title'])
                    elif suggestion['type'] == 'box':
                        fig = viz.create_box_plot(df, suggestion['columns'][0], suggestion['title'])
                    elif suggestion['type'] == 'correlation':
                        fig = viz.create_correlation_heatmap(df, suggestion['columns'], suggestion['title'])
                    else:
                        continue
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
        
        # Custom visualization builder
        st.divider()
        st.subheader("🎨 Custom Visualization")
        
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
        )
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if viz_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Select column:", numeric_cols)
                if st.button("Create Histogram"):
                    fig = viz.create_histogram(df, col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for histogram")
        
        elif viz_type == "Bar Chart":
            if categorical_cols:
                col = st.selectbox("Select column:", categorical_cols)
                if st.button("Create Bar Chart"):
                    fig = viz.create_bar_chart(df, col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns available for bar chart")
        
        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select X axis:", numeric_cols, key="scatter_x")
                col2 = st.selectbox("Select Y axis:", numeric_cols, key="scatter_y")
                if st.button("Create Scatter Plot"):
                    fig = viz.create_scatter(df, col1, col2)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Box Plot":
            if numeric_cols:
                col = st.selectbox("Select column:", numeric_cols)
                if st.button("Create Box Plot"):
                    fig = viz.create_box_plot(df, col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for box plot")
        
        elif viz_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
                if st.button("Create Heatmap"):
                    if len(selected_cols) >= 2:
                        fig = viz.create_correlation_heatmap(df, selected_cols)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
    
    with tab3:
        st.subheader("🤖 AI-Generated Insights")
        
        if 'ai_insights' not in st.session_state:
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("🤖 AI is analyzing your data..."):
                    try:
                        from utils.ai_helper import AIHelper
                        ai = AIHelper()
                        insights = ai.generate_data_insights(df, profile)
                        st.session_state.ai_insights = insights
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
        else:
            st.markdown(st.session_state.ai_insights)
            if st.button("Regenerate Insights"):
                del st.session_state.ai_insights
                st.rerun()
    
    with tab3:
        st.subheader("🔧 AI-Powered Cleaning Suggestions")
        
        if not issues:
            st.success("✅ No data quality issues detected!")
        else:
            if 'cleaning_suggestions' not in st.session_state:
                if st.button("Get AI Cleaning Suggestions", type="primary"):
                    with st.spinner("🤖 Generating cleaning suggestions..."):
                        try:
                            from utils.ai_helper import AIHelper
                            ai = AIHelper()
                            suggestions = ai.generate_cleaning_suggestions(df, issues)
                            st.session_state.cleaning_suggestions = suggestions
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating suggestions: {str(e)}")
            else:
                suggestions = st.session_state.cleaning_suggestions
                for i, suggestion in enumerate(suggestions):
                    with st.expander(f"💡 Suggestion {i+1}: {suggestion.get('issue', 'N/A')}"):
                        st.write("**What to do:**", suggestion.get('suggestion', 'N/A'))
                        st.write("**Why:**", suggestion.get('reason', 'N/A'))
                        if suggestion.get('code'):
                            st.code(suggestion['code'], language='python')

def show_insights():
    st.header("🤖 AI Insights & Natural Language Querying")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    
    st.write("Ask questions about your data in natural language!")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.write("""
        - What are the top 5 values in [column name]?
        - Show me the average of [numeric column]
        - How many missing values are in each column?
        - What is the correlation between [column1] and [column2]?
        - Find outliers in [numeric column]
        - Group by [column] and show the sum of [numeric column]
        """)
    
    # Question input
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What are the top 5 products by revenue?",
            key="question_input"
        )
    with col2:
        ask_button = st.button("Ask AI", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("🤖 AI is analyzing..."):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                # Get answer
                result = ai.answer_data_question(question, df)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'result': result
                })
                
                # Clear input
                st.session_state.question_input = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                
                result = chat['result']
                
                # Display answer
                st.markdown(f"**Answer:** {result.get('answer', 'No answer provided')}")
                
                # Display code if available
                if result.get('code'):
                    with st.expander("📝 See Python Code"):
                        st.code(result['code'], language='python')
                        
                        # Option to execute code
                        if st.button(f"Execute Code", key=f"exec_{i}"):
                            try:
                                import numpy as np
                                # Create a safe execution environment
                                exec_globals = {'df': df, 'pd': pd, 'np': np}
                                exec(result['code'], exec_globals)
                                
                                # Try to get result
                                if 'result' in exec_globals:
                                    st.write("**Result:**")
                                    st.write(exec_globals['result'])
                                else:
                                    st.success("✅ Code executed successfully!")
                            except Exception as e:
                                st.error(f"Error executing code: {str(e)}")
                
                # Display insights
                if result.get('insights'):
                    st.info(f"💡 **Insights:** {result['insights']}")
                
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

def show_reports():
    st.header("📄 Business Reports")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    profile = st.session_state.get('profile', {})
    issues = st.session_state.get('issues', [])
    
    st.write("Generate professional business reports from your data analysis.")
    
    # Report options
    st.subheader("📋 Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_insights = st.checkbox("Include AI Insights", value=True)
        include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    with col2:
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_code = st.checkbox("Include Code Snippets", value=False)
    
    # Generate report button
    if st.button("🎯 Generate Report", type="primary", use_container_width=True):
        with st.spinner("📝 Generating professional report..."):
            try:
                from utils.report_generator import ReportGenerator
                from utils.ai_helper import AIHelper
                from datetime import datetime
                
                # Get or generate insights
                if include_insights:
                    if 'ai_insights' not in st.session_state:
                        ai = AIHelper()
                        insights = ai.generate_data_insights(df, profile)
                        st.session_state.ai_insights = insights
                    else:
                        insights = st.session_state.ai_insights
                else:
                    insights = "AI insights not included in this report."
                
                # Get or generate suggestions
                if include_recommendations:
                    if 'cleaning_suggestions' not in st.session_state and issues:
                        ai = AIHelper()
                        suggestions = ai.generate_cleaning_suggestions(df, issues)
                        st.session_state.cleaning_suggestions = suggestions
                    else:
                        suggestions = st.session_state.get('cleaning_suggestions', [])
                else:
                    suggestions = []
                
                # Generate report
                report = ReportGenerator.generate_full_report(
                    df, profile, issues, insights, suggestions
                )
                
                st.session_state.generated_report = report
                st.success("✅ Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Display report
    if 'generated_report' in st.session_state:
        st.divider()
        st.subheader("📄 Generated Report")
        
        # Display report
        st.markdown(st.session_state.generated_report)
        
        # Download buttons
        from datetime import datetime
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=st.session_state.generated_report,
                file_name=f"datainsight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Text)",
                data=st.session_state.generated_report,
                file_name=f"datainsight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

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
                # Validate thresholds
                if min_support <= 0 or min_support > 1:
                    st.error("❌ Minimum support must be between 0 and 1")
                    st.stop()
                
                if min_confidence <= 0 or min_confidence > 1:
                    st.error("❌ Minimum confidence must be between 0 and 1")
                    st.stop()
                
                if min_lift < 0:
                    st.error("❌ Minimum lift must be positive")
                    st.stop()
                
                # Find frequent itemsets
                itemsets = mba.find_frequent_itemsets(min_support=min_support)
                
                if len(itemsets) == 0:
                    st.warning(f"⚠️ No frequent itemsets found with support >= {min_support}. Try lowering the minimum support.")
                    st.stop()
                
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
                    st.stop()
                
                st.session_state.mba_rules = rules
                
                st.success(f"✅ Found {len(itemsets)} frequent itemsets and {len(rules)} association rules!")
                
            except ValueError as e:
                st.error(f"❌ Validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try adjusting the thresholds or checking your data format.")
    
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

def show_rfm_analysis():
    """RFM Analysis and Customer Segmentation page."""
    st.header("👥 RFM Analysis & Customer Segmentation")
    
    # Help section
    with st.expander("ℹ️ What is RFM Analysis?"):
        st.markdown("""
        **RFM Analysis** is a customer segmentation technique that categorizes customers based on their purchasing behavior.
        
        ### RFM Metrics:
        
        - **Recency (R):** Days since last purchase
          - Lower is better (recent customers are more engaged)
          - Scored 1-5 (5 = most recent)
        
        - **Frequency (F):** Number of purchases
          - Higher is better (frequent buyers are more loyal)
          - Scored 1-5 (5 = most frequent)
        
        - **Monetary (M):** Total spending amount
          - Higher is better (high spenders are more valuable)
          - Scored 1-5 (5 = highest spending)
        
        ### Customer Segments:
        
        - 🏆 **Champions:** Best customers (R=5, F=5, M=5)
        - 💎 **Loyal Customers:** Consistent purchasers
        - 🌱 **Potential Loyalists:** Growing engagement
        - ✨ **New Customers:** Recently acquired
        - ⚠️ **At Risk:** Previously valuable but declining
        - ❌ **Lost:** Churned customers
        
        ### K-Means Clustering:
        
        - Unsupervised machine learning algorithm
        - Groups similar customers automatically
        - Find optimal clusters using elbow method
        """)
    
    st.markdown("""
    Segment your customers based on purchasing behavior and create targeted marketing strategies.
    """)
    
    # Import RFM utilities
    from utils.rfm_analysis import RFMAnalyzer
    
    # Initialize analyzer in session state
    if 'rfm_analyzer' not in st.session_state:
        st.session_state.rfm_analyzer = RFMAnalyzer()
    
    rfm_analyzer = st.session_state.rfm_analyzer
    
    # Data source selection
    st.subheader("📤 1. Load Transaction Data")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload Custom Data", "Use Sample Data"],
        key="rfm_data_source"
    )
    
    transactions_df = None
    
    if data_source == "Upload Custom Data":
        st.info("""
        **Upload Format:**
        - CSV file with columns: `customer_id`, `transaction_date`, `amount`
        - Each row represents one transaction
        - Example:
          ```
          customer_id,transaction_date,amount
          C001,2024-01-15,150.00
          C001,2024-02-10,200.00
          C002,2024-01-20,75.50
          ```
        """)
        
        uploaded_file = st.file_uploader(
            "Upload transaction CSV",
            type=['csv'],
            key="rfm_upload"
        )
        
        if uploaded_file is not None:
            try:
                transactions_df = pd.read_csv(uploaded_file)
                
                # Let user select columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    customer_col = st.selectbox("Customer ID column:", transactions_df.columns, key="cust_col")
                with col2:
                    date_col = st.selectbox("Transaction Date column:", transactions_df.columns, key="date_col")
                with col3:
                    amount_col = st.selectbox("Amount column:", transactions_df.columns, key="amount_col")
                
                if st.button("Process Data", type="primary"):
                    st.session_state.rfm_transactions = transactions_df
                    st.session_state.rfm_columns = {
                        'customer': customer_col,
                        'date': date_col,
                        'amount': amount_col
                    }
                    st.success(f"✅ Loaded {len(transactions_df)} transactions!")
                    st.dataframe(transactions_df.head(10), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Sample data
        if st.button("📥 Load Sample E-commerce Data", type="primary"):
            # Create realistic sample data
            import numpy as np
            np.random.seed(42)
            
            # Generate 500 transactions for 100 customers
            n_transactions = 500
            n_customers = 100
            
            customer_ids = [f"C{str(i).zfill(3)}" for i in range(1, n_customers + 1)]
            
            # Generate dates over last 365 days
            dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
            
            sample_transactions = []
            for _ in range(n_transactions):
                customer = np.random.choice(customer_ids)
                date = np.random.choice(dates)
                amount = np.random.gamma(shape=2, scale=50)  # Realistic purchase amounts
                
                sample_transactions.append({
                    'customer_id': customer,
                    'transaction_date': date,
                    'amount': round(amount, 2)
                })
            
            transactions_df = pd.DataFrame(sample_transactions)
            
            st.session_state.rfm_transactions = transactions_df
            st.session_state.rfm_columns = {
                'customer': 'customer_id',
                'date': 'transaction_date',
                'amount': 'amount'
            }
            
            st.success(f"✅ Loaded {len(transactions_df)} sample transactions from {n_customers} customers!")
            st.dataframe(transactions_df.head(10), use_container_width=True)
    
    # Only show analysis if transactions are loaded
    if 'rfm_transactions' not in st.session_state:
        st.info("👆 Load transaction data to begin RFM analysis")
        return
    
    transactions_df = st.session_state.rfm_transactions
    cols = st.session_state.rfm_columns
    
    # Display dataset info
    st.divider()
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(transactions_df):,}")
    with col2:
        st.metric("Unique Customers", f"{transactions_df[cols['customer']].nunique():,}")
    with col3:
        total_revenue = transactions_df[cols['amount']].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col4:
        avg_transaction = transactions_df[cols['amount']].mean()
        st.metric("Avg Transaction", f"${avg_transaction:.2f}")
    
    # Calculate RFM button
    st.divider()
    st.subheader("🔢 2. Calculate RFM Metrics")
    
    if st.button("📊 Calculate RFM", type="primary", use_container_width=True):
        with st.spinner("Calculating RFM metrics..."):
            try:
                # Calculate RFM
                rfm_data = rfm_analyzer.calculate_rfm(
                    transactions_df, 
                    cols['customer'], 
                    cols['date'], 
                    cols['amount']
                )
                
                # Score RFM
                rfm_scored = rfm_analyzer.score_rfm(rfm_data, method='quartile')
                
                # Segment customers
                rfm_segmented = rfm_analyzer.segment_customers(rfm_scored)
                
                # Store in session state
                st.session_state.rfm_data = rfm_data
                st.session_state.rfm_scored = rfm_scored
                st.session_state.rfm_segmented = rfm_segmented
                
                st.success(f"✅ RFM calculated for {len(rfm_data)} customers!")
                
            except Exception as e:
                st.error(f"Error calculating RFM: {str(e)}")
    
    # Show results if available
    if 'rfm_segmented' in st.session_state:
        rfm_segmented = st.session_state.rfm_segmented
        rfm_data = st.session_state.rfm_data
        
        # RFM Summary
        st.divider()
        st.subheader("📈 RFM Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Recency (days)", f"{rfm_data['Recency'].mean():.1f}")
        with col2:
            st.metric("Avg Frequency", f"{rfm_data['Frequency'].mean():.1f}")
        with col3:
            st.metric("Avg Monetary", f"${rfm_data['Monetary'].mean():.2f}")
        
        # RFM Data Preview
        with st.expander("👀 View RFM Data"):
            st.dataframe(rfm_segmented.head(20), use_container_width=True)
        
        # K-Means Clustering
        st.divider()
        st.subheader("🎯 3. K-Means Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=8,
                value=4,
                help="Use elbow method to determine optimal number"
            )
            
            if st.button("🔄 Run K-Means Clustering", use_container_width=True):
                with st.spinner("Performing K-Means clustering..."):
                    rfm_clustered = rfm_analyzer.perform_kmeans_clustering(rfm_data, n_clusters)
                    st.session_state.rfm_clustered = rfm_clustered
                    st.success(f"✅ Created {n_clusters} customer clusters!")
        
        with col2:
            if st.button("📉 Show Elbow Curve", use_container_width=True):
                with st.spinner("Calculating elbow curve..."):
                    cluster_range, inertias = rfm_analyzer.calculate_elbow_curve(rfm_data, max_clusters=10)
                    fig_elbow = rfm_analyzer.create_elbow_plot(cluster_range, inertias)
                    st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Segment Analysis
        st.divider()
        st.subheader("👥 4. Customer Segments")
        
        # Segment distribution
        fig_segments = rfm_analyzer.create_segment_distribution(rfm_segmented)
        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Segment profiles
        st.write("**Segment Profiles:**")
        profiles = rfm_analyzer.get_segment_profiles(rfm_segmented, cols['customer'])
        st.dataframe(profiles, use_container_width=True)
        
        # 3D Visualization
        st.divider()
        st.subheader("📊 5. 3D Visualization")
        
        viz_option = st.radio(
            "Color by:",
            ["Segment", "Cluster"],
            horizontal=True,
            key="viz_option"
        )
        
        if viz_option == "Cluster" and 'rfm_clustered' in st.session_state:
            fig_3d = rfm_analyzer.create_rfm_scatter_3d(st.session_state.rfm_clustered, color_col='Cluster')
        else:
            fig_3d = rfm_analyzer.create_rfm_scatter_3d(rfm_segmented, color_col='Segment')
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Business Insights
        st.divider()
        st.subheader("💡 6. Business Insights & Recommendations")
        
        insights_dict = rfm_analyzer.generate_segment_insights(rfm_segmented)
        
        # Show insights for each segment present in data
        segments_present = rfm_segmented['Segment'].unique()
        
        for segment in segments_present:
            if segment in insights_dict:
                with st.expander(f"📋 {segment} ({len(rfm_segmented[rfm_segmented['Segment']==segment])} customers)"):
                    for insight in insights_dict[segment]:
                        st.markdown(insight)
        
        # Export Options
        st.divider()
        st.subheader("📥 7. Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = rfm_segmented.to_csv(index=False)
            st.download_button(
                label="📥 Download RFM Data (CSV)",
                data=csv,
                file_name=f"rfm_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            profiles_csv = profiles.to_csv(index=False)
            st.download_button(
                label="📥 Download Segment Profiles (CSV)",
                data=profiles_csv,
                file_name=f"segment_profiles_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("📄 Generate Full Report", use_container_width=True):
                report = f"""
# RFM Analysis Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- **Total Transactions:** {len(transactions_df):,}
- **Unique Customers:** {transactions_df[cols['customer']].nunique():,}
- **Total Revenue:** ${total_revenue:,.2f}
- **Average Transaction:** ${avg_transaction:.2f}

## RFM Metrics
- **Average Recency:** {rfm_data['Recency'].mean():.1f} days
- **Average Frequency:** {rfm_data['Frequency'].mean():.1f} transactions
- **Average Monetary:** ${rfm_data['Monetary'].mean():.2f}

## Segment Distribution

{profiles.to_markdown(index=False)}

## Recommendations

Based on the RFM analysis:

1. **Champions & Loyal Customers:** Focus on retention with VIP programs
2. **Potential Loyalists:** Nurture with personalized offers
3. **At Risk & Cannot Lose:** Urgent win-back campaigns needed
4. **Hibernating & Lost:** Minimal investment, focus on learning

---
*Report generated by DataInsight AI - RFM Analysis Module*
"""
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"rfm_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

def show_monte_carlo_simulation():
    """Monte Carlo Simulation page for financial forecasting."""
    st.header("📈 Monte Carlo Simulation")
    
    # Help section
    with st.expander("ℹ️ What is Monte Carlo Simulation?"):
        st.markdown("""
        **Monte Carlo Simulation** uses random sampling to model uncertainty and predict future outcomes.
        
        ### How It Works:
        
        1. **Historical Analysis:** Analyze past stock price movements
        2. **Calculate Statistics:** Mean return and volatility (standard deviation)
        3. **Generate Scenarios:** Create thousands of possible future price paths
        4. **Risk Assessment:** Calculate probabilities and confidence intervals
        
        ### Key Metrics:
        
        - **Expected Return:** Average predicted return across all simulations
        - **VaR (Value at Risk):** Maximum expected loss at a given confidence level
        - **CVaR (Conditional VaR):** Average loss when VaR threshold is exceeded
        - **Confidence Intervals:** Range of outcomes at different probability levels
        
        ### Applications:
        
        - 💰 **Portfolio Management:** Asset allocation and risk assessment
        - 📊 **Investment Planning:** Long-term return projections
        - ⚠️ **Risk Analysis:** Worst-case scenario planning
        - 🎯 **Decision Making:** Compare investment opportunities
        """)
    
    st.markdown("""
    Forecast stock prices and assess investment risk using Monte Carlo simulation.
    """)
    
    # Import Monte Carlo utilities
    from utils.monte_carlo import MonteCarloSimulator
    from datetime import datetime, timedelta
    
    # Initialize simulator in session state
    if 'mc_simulator' not in st.session_state:
        st.session_state.mc_simulator = MonteCarloSimulator()
    
    simulator = st.session_state.mc_simulator
    
    # Stock selection
    st.subheader("📊 1. Select Stock & Time Period")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL, TSLA)"
        ).upper()
    
    with col2:
        lookback_days = st.number_input(
            "Historical Data (days)",
            min_value=30,
            max_value=730,
            value=365,
            step=30,
            help="Days of historical data for analysis"
        )
    
    with col3:
        forecast_days = st.number_input(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="Days to simulate into the future"
        )
    
    # Fetch data button
    if st.button("📥 Fetch Stock Data", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {ticker} data..."):
            try:
                start_date = datetime.now() - timedelta(days=lookback_days)
                stock_data = simulator.fetch_stock_data(ticker, start_date)
                
                st.session_state.mc_stock_data = stock_data
                st.session_state.mc_ticker = ticker
                
                # Calculate returns
                returns = simulator.calculate_returns(stock_data['Close'])
                st.session_state.mc_returns = returns
                
                # Calculate statistics
                stats = simulator.get_statistics(returns)
                st.session_state.mc_stats = stats
                
                st.success(f"✅ Loaded {len(stock_data)} days of {ticker} data!")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("💡 Try a different ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    # Only show analysis if data is loaded
    if 'mc_stock_data' not in st.session_state:
        st.info("👆 Fetch stock data to begin Monte Carlo simulation")
        return
    
    stock_data = st.session_state.mc_stock_data
    returns = st.session_state.mc_returns
    stats = st.session_state.mc_stats
    ticker = st.session_state.mc_ticker
    
    # Display historical data summary
    st.divider()
    st.subheader("📊 Historical Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Mean Daily Return", f"{stats['mean']*100:.3f}%")
    with col3:
        st.metric("Volatility (Std Dev)", f"{stats['std']*100:.3f}%")
    with col4:
        total_return = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    # Historical returns distribution
    with st.expander("📈 View Historical Returns Distribution"):
        fig_returns = simulator.create_returns_distribution(returns)
        st.plotly_chart(fig_returns, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Skewness", f"{stats['skewness']:.3f}")
        with col2:
            st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
        with col3:
            st.metric("Median Return", f"{stats['median']*100:.3f}%")
    
    # Simulation parameters
    st.divider()
    st.subheader("🎲 2. Configure Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations = more accurate but slower"
        )
    
    with col2:
        confidence_levels = st.multiselect(
            "Confidence Intervals",
            [5, 10, 25, 50, 75, 90, 95],
            default=[5, 25, 50, 75, 95],
            help="Percentiles to display in simulation plot"
        )
    
    # Run simulation button
    if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {num_simulations} simulations..."):
            try:
                start_price = stock_data['Close'].iloc[-1]
                
                # Run simulation
                simulations = simulator.run_simulation(
                    start_price=start_price,
                    mean_return=stats['mean'],
                    std_return=stats['std'],
                    days=forecast_days,
                    num_simulations=num_simulations
                )
                
                # Calculate confidence intervals
                intervals_dict = simulator.calculate_confidence_intervals(
                    simulations,
                    [level / 100 for level in confidence_levels]
                )
                
                # Calculate risk metrics
                final_prices = simulations[:, -1]
                risk_metrics = simulator.get_risk_metrics(final_prices, start_price)
                
                # Store in session state
                st.session_state.mc_simulations = simulations
                st.session_state.mc_intervals = intervals_dict
                st.session_state.mc_risk_metrics = risk_metrics
                st.session_state.mc_forecast_days = forecast_days
                
                st.success(f"✅ Completed {num_simulations} simulations for {forecast_days} days!")
                
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
    
    # Show results if available
    if 'mc_simulations' in st.session_state:
        simulations = st.session_state.mc_simulations
        intervals = st.session_state.mc_intervals
        risk_metrics = st.session_state.mc_risk_metrics
        forecast_days_actual = st.session_state.mc_forecast_days
        
        # Summary metrics
        st.divider()
        st.subheader("📊 3. Simulation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Price", f"${risk_metrics['expected_price']:.2f}")
        with col2:
            st.metric("Expected Return", f"{risk_metrics['expected_return']:.2f}%")
        with col3:
            st.metric("Probability of Profit", f"{risk_metrics['probability_profit']:.1f}%")
        with col4:
            st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2f}%")
        
        # Simulation plot
        st.subheader("📈 4. Simulation Paths")
        fig_sim = simulator.create_simulation_plot(simulations, intervals, ticker, forecast_days_actual)
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Distribution plot
        st.subheader("📊 5. Final Price Distribution")
        fig_dist = simulator.create_distribution_plot(simulations[:, -1], stock_data['Close'].iloc[-1])
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Risk metrics table
        st.subheader("⚠️ 6. Risk Metrics")
        
        risk_df = pd.DataFrame({
            'Metric': [
                'Expected Return',
                'Standard Deviation',
                'Value at Risk (95%)',
                'Value at Risk (99%)',
                'Conditional VaR (95%)',
                'Probability of Profit',
                'Minimum Price',
                'Maximum Price'
            ],
            'Value': [
                f"{risk_metrics['expected_return']:.2f}%",
                f"{risk_metrics['std_dev']:.2f}%",
                f"{risk_metrics['var_95']:.2f}%",
                f"{risk_metrics['var_99']:.2f}%",
                f"{risk_metrics['cvar_95']:.2f}%",
                f"{risk_metrics['probability_profit']:.1f}%",
                f"${risk_metrics['min_price']:.2f}",
                f"${risk_metrics['max_price']:.2f}"
            ]
        })
        
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Business insights
        st.divider()
        st.subheader("💡 7. Business Insights")
        
        insights = simulator.generate_insights(risk_metrics, ticker, forecast_days_actual)
        
        for insight in insights:
            st.markdown(insight)
        
        # Strategic recommendations
        with st.expander("🎯 Strategic Recommendations"):
            st.markdown("""
            ### Investment Strategies Based on Results:
            
            #### If Probability of Profit > 60%:
            - ✅ Consider **long position** (buying the stock)
            - 📈 Set target prices at 75th-90th percentile levels
            - 🛡️ Use stop-loss at 10th-25th percentile levels
            
            #### If VaR (95%) > -10%:
            - ⚖️ **Moderate risk** - suitable for balanced portfolios
            - 💼 Consider position sizing based on portfolio allocation
            - 📊 Monitor volatility indicators
            
            #### If VaR (95%) > -20%:
            - ⚠️ **High risk** - only for aggressive investors
            - 🎲 Consider options strategies for hedging
            - 📉 Prepare exit strategy in advance
            
            ### Risk Management:
            
            1. **Diversification:** Don't allocate more than 5-10% to single stock
            2. **Stop-Loss:** Set at VaR level comfortable for your risk tolerance
            3. **Rebalance:** Review positions regularly (monthly/quarterly)
            4. **Hedge:** Consider protective puts if downside risk is high
            5. **Dollar-Cost Averaging:** Spread investment over time to reduce risk
            """)
        
        # Export options
        st.divider()
        st.subheader("📥 8. Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export simulation data
            sim_df = pd.DataFrame(simulations).T
            sim_df.insert(0, 'Day', range(len(sim_df)))
            sim_csv = sim_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Paths (CSV)",
                data=sim_csv,
                file_name=f"mc_simulation_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export report
            report = f"""
# Monte Carlo Simulation Report: {ticker}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Simulation Parameters
- **Ticker:** {ticker}
- **Forecast Period:** {forecast_days_actual} days
- **Number of Simulations:** {num_simulations}
- **Starting Price:** ${stock_data['Close'].iloc[-1]:.2f}

## Historical Analysis
- **Mean Daily Return:** {stats['mean']*100:.3f}%
- **Volatility:** {stats['std']*100:.3f}%
- **Skewness:** {stats['skewness']:.3f}
- **Kurtosis:** {stats['kurtosis']:.3f}

## Risk Metrics
{risk_df.to_markdown(index=False)}

## Business Insights
{chr(10).join(insights)}

---
*Report generated by DataInsight AI - Monte Carlo Simulation Module*
"""
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"mc_report_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
