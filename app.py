import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime
import time
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DataInsights",
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
    /* Prevent grey spinner overlay from appearing */
    .stSpinner > div {
        border-color: transparent !important;
    }
    div[data-testid="stSpinner"] {
        position: relative !important;
    }
    /* Remove the grey overlay background */
    .stSpinner::before {
        display: none !important;
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
    
    # Automatic memory cleanup on page load
    if 'last_cleanup' not in st.session_state:
        st.session_state.last_cleanup = time.time()
    
    # Clean up every 5 minutes (300 seconds)
    if time.time() - st.session_state.last_cleanup > 300:
        from utils.process_manager import ProcessManager
        ProcessManager.cleanup_large_session_state_items()
        st.session_state.last_cleanup = time.time()
    
    # Header with fallback - Centered branding
    st.markdown("<h1 style='text-align: center;'>🎯 DataInsights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Your AI-Powered Business Intelligence Assistant</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        
        # Check if any process is running
        from utils.process_manager import NavigationGuard
        guard = NavigationGuard()
        
        # Show warning if process is running
        is_processing = guard.is_any_process_running()
        if is_processing:
            st.warning("⚠️ Process running - please wait before navigating")
        
        page = st.radio(
            "Select a page:",
            ["Home", "Data Upload", "Data Analysis & Cleaning", "Anomaly Detection", "Insights", 
             "Market Basket Analysis", "RFM Analysis", "Time Series Forecasting", "Text Mining & NLP", 
             "ML Classification", "ML Regression", "Monte Carlo Simulation", 
             "A/B Testing", "Cohort Analysis", "Recommendation Systems", 
             "Geospatial Analysis", "Survival Analysis", "Network Analysis", 
             "Churn Prediction",
             "Reports"],
            key="navigation",
            disabled=is_processing
        )
        
        st.divider()
        
        # Memory monitor
        st.subheader("💾 Memory Monitor")
        
        from utils.process_manager import ProcessManager
        
        memory_stats = ProcessManager.get_memory_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory", f"{memory_stats['rss_mb']:.0f}MB")
        with col2:
            st.metric("Usage", f"{memory_stats['percent']:.1f}%")
        
        # Warning if memory is high
        if memory_stats['percent'] > 80:
            st.warning("⚠️ High memory usage!")
            if st.button("🧹 Clean Up Memory", use_container_width=True):
                ProcessManager.cleanup_large_session_state_items()
                st.rerun()
        
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
        **DataInsights** helps you:
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
    elif page == "Data Analysis & Cleaning":
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
    elif page == "ML Classification":
        show_ml_classification()
    elif page == "ML Regression":
        show_ml_regression()
    elif page == "Anomaly Detection":
        show_anomaly_detection()
    elif page == "Time Series Forecasting":
        show_time_series_forecasting()
    elif page == "Text Mining & NLP":
        show_text_mining()
    elif page == "A/B Testing":
        show_ab_testing()
    elif page == "Cohort Analysis":
        show_cohort_analysis()
    elif page == "Recommendation Systems":
        show_recommendation_systems()
    elif page == "Geospatial Analysis":
        show_geospatial_analysis()
    elif page == "Survival Analysis":
        show_survival_analysis()
    elif page == "Network Analysis":
        show_network_analysis()
    elif page == "Churn Prediction":
        show_churn_prediction()

def show_home():
    st.markdown("<h2 style='text-align: center;'>Welcome to DataInsights! 👋</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📤 Upload Data</h3>
            <p>Upload CSV or Excel files and get<br>instant data profiling</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI Analysis</h3>
            <p>Ask questions in natural language<br>and get intelligent answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Visualizations</h3>
            <p>Interactive charts and dashboards<br>generated automatically</p>
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
    st.markdown("<h2 style='text-align: center;'>📤 Data Upload</h2>", unsafe_allow_html=True)
    
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
                with st.status("Loading and analyzing your data...", expanded=True) as status:
                    from utils.data_processor import DataProcessor
                    from utils.data_optimizer import DataOptimizer
                    
                    # Load data
                    status.write("📁 Reading file...")
                    df = DataProcessor.load_data(uploaded_file)
                    
                    # Check data size and optimize
                    status.write("💾 Checking data size...")
                    original_memory = DataOptimizer.get_memory_usage(df)
                    
                    # Optimize DataFrame
                    status.write("⚡ Optimizing data...")
                    df = DataOptimizer.optimize_dataframe(df)
                    optimized_memory = DataOptimizer.get_memory_usage(df)
                    memory_saved = original_memory['total_mb'] - optimized_memory['total_mb']
                    
                    # Check if sampling is needed
                    if DataOptimizer.should_sample_data(df):
                        status.write("⚠️ Large dataset detected - sampling recommended...")
                        st.session_state.data_full = df  # Keep reference to full data
                        
                        # Show sampling option
                        st.warning(f"""
                        ⚠️ **Large Dataset Detected**
                        
                        Your dataset has **{len(df):,} rows**. For optimal performance on Streamlit Cloud,
                        we recommend sampling to prevent memory issues.
                        """)
                        
                        sample_size = st.slider(
                            "Sample size for analysis:",
                            min_value=10000,
                            max_value=min(len(df), 200000),
                            value=min(50000, len(df)),
                            step=10000,
                            help="Larger samples provide better accuracy but use more memory"
                        )
                        
                        if sample_size < len(df):
                            df_sampled = DataOptimizer.sample_data(df, sample_size=sample_size)
                            st.info(f"📊 Using {len(df_sampled):,} sampled rows (from {len(df):,} total)")
                            st.session_state.data = df_sampled
                        else:
                            st.session_state.data = df
                    else:
                        st.session_state.data = df
                    
                    # Show optimization results
                    if memory_saved > 0.1:
                        st.success(f"✅ Optimized! Saved {memory_saved:.1f}MB ({memory_saved/original_memory['total_mb']*100:.1f}%)")
                    
                    # Profile data
                    status.write("📊 Profiling data...")
                    profile = DataProcessor.profile_data(st.session_state.data)
                    st.session_state.profile = profile
                    
                    # Detect issues (use sampled data if applicable)
                    issues = DataProcessor.detect_data_quality_issues(st.session_state.data)
                    st.session_state.issues = issues
                    
                    # AI Data Profiling (not ML-specific analysis)
                    status.write("🤖 AI analyzing data quality and structure...")
                    from utils.ai_helper import AIHelper
                    ai = AIHelper()
                    
                    # Generate data insights (general profiling, not ML-specific)
                    data_insights = ai.generate_data_insights(st.session_state.data, profile)
                    st.session_state.ai_data_insights = data_insights
                    
                    # Store data hash to track if data changes
                    st.session_state.data_profile_hash = hash(str(st.session_state.data.columns.tolist()) + str(len(st.session_state.data)))
                
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
                
                # AI Data Insights Section
                if 'ai_data_insights' in st.session_state:
                    st.subheader("🤖 AI Data Insights")
                    with st.expander("📊 View AI Analysis", expanded=False):
                        st.markdown(st.session_state.ai_data_insights)
                    
                    st.info("💡 **This data profile will be available to all analytics modules. ML-specific recommendations will be generated when you use ML Classification or Regression.**")
                
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
            with st.status(f"🔄 Loading dataset {dataset_id} from OpenML...", expanded=True) as status:
                try:
                    st.write("⏳ This may take 30-60 seconds for large datasets...")
                    import openml
                    
                    dataset = openml.datasets.get_dataset(dataset_id)
                    st.write("📥 Downloading dataset...")
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
                    st.write("📊 Profiling data...")
                    from utils.data_processor import DataProcessor
                    profile = DataProcessor.profile_data(df)
                    st.session_state.profile = profile
                    
                    issues = DataProcessor.detect_data_quality_issues(df)
                    st.session_state.issues = issues
                    
                    status.update(label=f"✅ Successfully loaded {dataset.name}!", state="complete", expanded=False)
                    success_msg = f"✅ Successfully loaded {dataset.name} (ID: {dataset_id})!"
                    st.success(success_msg)
                    
                    # Show full description and metadata in expander
                    with st.expander("📋 Dataset Information & Citation", expanded=True):
                        st.markdown(dataset.description)
                    
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
        
        # Add helpful hint if input is empty
        if not kaggle_dataset:
            st.caption("💡 Tip: Type the dataset name above and press **Enter** to enable the button")
        
        if st.button("📥 Load Kaggle Dataset", type="primary", disabled=not kaggle_dataset):
            with st.status(f"Downloading {kaggle_dataset} from Kaggle...", expanded=True) as status:
                try:
                    st.write("🔑 Authenticating with Kaggle API...")
                    import os
                    from kaggle.api.kaggle_api_extended import KaggleApi
                    
                    # Check for Streamlit secrets first (for cloud deployment)
                    if hasattr(st, 'secrets') and 'KAGGLE_USERNAME' in st.secrets and 'KAGGLE_KEY' in st.secrets:
                        os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
                        os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']
                        st.info("🔑 Using Kaggle credentials from Streamlit Secrets")
                        st.write(f"Debug - Username: {st.secrets['KAGGLE_USERNAME']}")
                    else:
                        st.warning("⚠️ Kaggle credentials not found in Streamlit Secrets. Trying local kaggle.json...")
                    
                    # Initialize Kaggle API (will use environment variables or kaggle.json)
                    api = KaggleApi()
                    api.authenticate()
                    st.success("✅ Kaggle API authenticated successfully!")
                    
                    # Verify dataset exists before downloading
                    try:
                        dataset_info = api.dataset_list(search=kaggle_dataset)
                        st.info(f"📊 Found dataset. Downloading...")
                    except Exception as e:
                        st.error(f"❌ Cannot access dataset: {str(e)}")
                        st.info("""
                        **Possible reasons:**
                        - Dataset is private or requires competition acceptance
                        - Dataset name is incorrect (format: username/dataset-name)
                        - You need to accept terms on Kaggle website first
                        
                        **Try:** Visit https://www.kaggle.com/datasets/{0} and click "Download" to accept any terms.
                        """.format(kaggle_dataset))
                        raise
                    
                    # Download dataset
                    download_path = "./kaggle_data"
                    os.makedirs(download_path, exist_ok=True)
                    
                    api.dataset_download_files(kaggle_dataset, path=download_path, unzip=True)
                    
                    # Find CSV files
                    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
                    
                    if csv_files:
                        # Load first CSV file
                        csv_file = csv_files[0]
                        csv_path = os.path.join(download_path, csv_file)
                        
                        # Try different encodings for international datasets
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                        df = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(csv_path, encoding=encoding)
                                st.info(f"✅ Loaded with {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if df is None:
                            st.error("❌ Could not read CSV with any standard encoding")
                            raise Exception("Encoding error - unable to decode CSV file")
                        
                        st.session_state.data = df
                        
                        # Profile data
                        st.write("📊 Profiling data...")
                        from utils.data_processor import DataProcessor
                        profile = DataProcessor.profile_data(df)
                        st.session_state.profile = profile
                        
                        issues = DataProcessor.detect_data_quality_issues(df)
                        st.session_state.issues = issues
                        
                        status.update(label=f"✅ Successfully loaded from Kaggle!", state="complete", expanded=False)
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
                    error_msg = str(e).lower()
                    st.error(f"Error loading Kaggle dataset: {str(e)}")
                    
                    if "403" in error_msg or "forbidden" in error_msg:
                        st.error(f"""
                        🔒 **403 Forbidden - Dataset Requires Access**
                        
                        **Fix:** Visit https://www.kaggle.com/datasets/{kaggle_dataset}
                        Click "Download" to accept terms, then try again.
                        
                        **Or try:** `uciml/iris` or `heeraldedhia/groceries-dataset`
                        """)
                    elif "401" in error_msg or "unauthorized" in error_msg:
                        st.error("""
                        🔑 **401 Unauthorized - Invalid Credentials**
                        
                        Generate new API token at kaggle.com/settings
                        """)

def show_analysis():
    st.markdown("<h2 style='text-align: center;'>📊 Data Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    
    # Generate profile if not exists
    if 'profile' not in st.session_state or not st.session_state.profile:
        from utils.data_processor import DataProcessor
        st.session_state.profile = DataProcessor.profile_data(df)
        st.session_state.issues = DataProcessor.detect_data_quality_issues(df)
    
    profile = st.session_state.profile
    issues = st.session_state.get('issues', [])
    
    # Data Quality Overview
    from utils.data_cleaning import DataCleaner
    
    st.subheader("🎯 Data Quality Overview")
    
    # Calculate quality metrics
    cleaner = DataCleaner(df)
    quality_score = cleaner.calculate_quality_score()
    
    # Quality metrics cards with smart indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Quality Score", f"{quality_score:.1f}/100", 
                 delta="Good" if quality_score >= 80 else "Needs Improvement",
                 delta_color="normal" if quality_score >= 80 else "inverse")
    with col2:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        # Best practice: < 5% is good, > 20% needs attention
        if missing_pct < 5:
            delta_text = "Excellent"
            delta_color = "normal"
        elif missing_pct < 20:
            delta_text = "Acceptable"
            delta_color = "off"
        else:
            delta_text = "High"
            delta_color = "inverse"
        st.metric("Missing Values", f"{missing_pct:.1f}%", delta=delta_text, delta_color=delta_color)
    with col3:
        duplicates = df.duplicated().sum()
        dup_pct = (duplicates / len(df)) * 100 if len(df) > 0 else 0
        # Best practice: < 1% is good, > 5% needs attention
        if dup_pct < 1:
            delta_text = "Excellent"
            delta_color = "normal"
        elif dup_pct < 5:
            delta_text = "Acceptable"
            delta_color = "off"
        else:
            delta_text = "High"
            delta_color = "inverse"
        st.metric("Duplicates", f"{duplicates:,}", delta=delta_text, delta_color=delta_color)
    with col4:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        total_cols = len(df.columns)
        numeric_pct = (numeric_cols / total_cols * 100) if total_cols > 0 else 0
        # Best practice: Having numeric data is good for analysis
        if numeric_cols > 0:
            delta_text = f"{numeric_pct:.0f}% of data"
            delta_color = "normal"
        else:
            delta_text = "None found"
            delta_color = "inverse"
        st.metric("Numeric Columns", numeric_cols, delta=delta_text, delta_color=delta_color)
    with col5:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        cat_pct = (categorical_cols / total_cols * 100) if total_cols > 0 else 0
        # Best practice: Having text data is good for analysis
        if categorical_cols > 0:
            delta_text = f"{cat_pct:.0f}% of data"
            delta_color = "normal"
        else:
            delta_text = "None found"
            delta_color = "off"
        st.metric("Text Columns", categorical_cols, delta=delta_text, delta_color=delta_color)
    
    st.divider()
    
    # AI-Powered Cleaning Analysis (Above tabs for better visibility)
    st.subheader("🤖 AI Cleaning Recommendations")
    
    if 'cleaning_ai_recommendations' not in st.session_state:
        if st.button("🔍 Generate AI Cleaning Analysis", type="primary", use_container_width=True):
            with st.status("🤖 AI analyzing dataset for optimal cleaning strategy...", expanded=True) as status:
                try:
                    from utils.ai_smart_detection import get_ai_recommendation
                    
                    # Get performance-aware recommendations for cleaning
                    status.write("Analyzing data structure and performance constraints...")
                    ai_recommendations = get_ai_recommendation(df, task_type='classification')
                    st.session_state.cleaning_ai_recommendations = ai_recommendations
                    
                    status.update(label="✅ AI analysis complete!", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error")
                    st.error(f"Error generating AI recommendations: {str(e)}")
    else:
        ai_recs = st.session_state.cleaning_ai_recommendations
        
        # Performance Risk Assessment
        performance_risk = ai_recs.get('performance_risk', 'Low')
        risk_emoji = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(performance_risk, '❓')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"**⚡ Performance Risk:** {risk_emoji} {performance_risk} - Dataset suitability for Streamlit Cloud")
        with col2:
            if st.button("🔄 Regenerate Analysis", use_container_width=True):
                del st.session_state.cleaning_ai_recommendations
                st.rerun()
        
        # Performance Warnings
        if performance_risk in ['Medium', 'High']:
            perf_warnings = ai_recs.get('performance_warnings', [])
            if perf_warnings:
                st.warning("⚠️ **Performance Warnings:**")
                for warning in perf_warnings:
                    st.write(f"• {warning}")
        
        # Optimization Suggestions
        optimization_suggestions = ai_recs.get('optimization_suggestions', [])
        if optimization_suggestions:
            with st.expander("🚀 AI Optimization Suggestions", expanded=True):
                for suggestion in optimization_suggestions:
                    st.write(f"• {suggestion}")
        
        # Columns to Consider Excluding
        features_to_exclude = ai_recs.get('features_to_exclude', [])
        if features_to_exclude:
            with st.expander("🚫 Columns AI Recommends Excluding Before Analysis", expanded=False):
                for feature_info in features_to_exclude:
                    if isinstance(feature_info, dict):
                        st.write(f"• **{feature_info['column']}**: {feature_info['reason']}")
                    else:
                        st.write(f"• {feature_info}")
    
    st.divider()
    
    # Tabs organized by best practice workflow
    tab1, tab2, tab3, tab4 = st.tabs(["🧹 Quick Clean", "📈 Statistics", "📊 Visualizations", "🤖 AI Insights"])
    
    with tab1:
        st.subheader("🧹 Quick Data Cleaning")
        st.write("Apply automatic data cleaning with one click, or customize the cleaning options.")
        
        # AI-Powered Cleaning Presets
        def get_ai_cleaning_presets(df, profile, ai_recommendations=None):
            """Generate intelligent cleaning presets based on data profile and AI analysis."""
            presets = {}
            
            # Basic Cleaning Presets
            presets['normalize_cols'] = True  # Always recommended
            presets['convert_numeric'] = len([col for col in profile['column_info'] if col['dtype'] == 'object' and col['unique'] < len(df) * 0.8]) > 0
            presets['trim_strings'] = len([col for col in profile['column_info'] if col['dtype'] == 'object']) > 0
            presets['parse_dates'] = len([col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated'])]) > 0
            
            # Data Quality Presets
            presets['remove_dups'] = profile['basic_info']['duplicates'] > 0
            presets['remove_constant'] = len([col for col in profile['column_info'] if col['unique'] == 1]) > 0
            presets['remove_empty_rows'] = len(df.dropna(how='all')) < len(df)
            # Fix: Calculate missing percentage as number, not string
            presets['drop_high_missing'] = len([col for col in profile['column_info'] if (col['missing'] / len(df)) * 100 > 80]) > 0
            
            # Advanced Presets
            missing_pct = sum([col['missing'] for col in profile['column_info']]) / (len(df) * len(df.columns)) * 100
            presets['fill_missing'] = missing_pct > 5 and missing_pct < 50  # Don't fill if too much missing
            
            # Smart missing strategy based on data types
            numeric_cols = len([col for col in profile['column_info'] if col['dtype'] in ['int64', 'float64']])
            categorical_cols = len([col for col in profile['column_info'] if col['dtype'] == 'object'])
            if numeric_cols > categorical_cols:
                presets['missing_strategy'] = 'median'
            elif categorical_cols > numeric_cols:
                presets['missing_strategy'] = 'mode'
            else:
                presets['missing_strategy'] = 'median'
            
            # Outlier detection for numeric data
            presets['remove_outliers'] = numeric_cols > 0 and len(df) > 100  # Only for larger datasets
            presets['outlier_method'] = 'IQR'  # More robust default
            
            # Negative values detection
            negative_cols = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if (df[col] < 0).any():
                    negative_cols.append(col)
            presets['fix_negatives'] = len(negative_cols) > 0 and any(keyword in ' '.join(negative_cols).lower() for keyword in ['amount', 'price', 'cost', 'quantity', 'qty', 'count'])
            presets['negative_method'] = 'abs'  # Most common fix
            
            # Categorical standardization
            presets['standardize_categorical'] = categorical_cols > 0
            
            # AI-specific adjustments
            if ai_recommendations:
                performance_risk = ai_recommendations.get('performance_risk', 'Low')
                if performance_risk == 'High':
                    # More aggressive cleaning for high-risk datasets
                    presets['drop_high_missing'] = True
                    presets['remove_outliers'] = True
                    presets['fill_missing'] = False  # Don't fill, just drop
                elif performance_risk == 'Medium':
                    presets['remove_outliers'] = len(df) < 50000  # Only for smaller datasets
            
            return presets
        
        # Get AI presets
        ai_recs = st.session_state.get('cleaning_ai_recommendations', {})
        cleaning_presets = get_ai_cleaning_presets(df, profile, ai_recs)
        
        # Show AI preset summary if available
        if 'cleaning_ai_recommendations' in st.session_state:
            st.info("🤖 **AI has analyzed your data and preset the cleaning options below based on your data profile.**")
            
            # Show detailed AI reasoning
            with st.expander("🧠 View AI Reasoning for Presets", expanded=False):
                st.markdown("**📊 Data Profile Analysis:**")
                
                # Basic stats
                total_missing = sum([col['missing'] for col in profile['column_info']])
                missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
                numeric_cols = len([col for col in profile['column_info'] if col['dtype'] in ['int64', 'float64']])
                categorical_cols = len([col for col in profile['column_info'] if col['dtype'] == 'object'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                with col2:
                    st.metric("Numeric Columns", numeric_cols)
                with col3:
                    st.metric("Text Columns", categorical_cols)
                
                st.markdown("**🤖 AI Reasoning:**")
                
                # Explain key decisions
                if cleaning_presets.get('remove_dups'):
                    st.write(f"✅ **Remove duplicates**: Found {profile['basic_info']['duplicates']} duplicate rows")
                else:
                    st.write("❌ **Remove duplicates**: No duplicates detected")
                
                if cleaning_presets.get('drop_high_missing'):
                    high_missing_cols = [col['name'] for col in profile['column_info'] if (col['missing'] / len(df)) * 100 > 80]
                    st.write(f"✅ **Drop high missing columns**: Found {len(high_missing_cols)} columns with >80% missing")
                else:
                    st.write("❌ **Drop high missing columns**: No columns with excessive missing values")
                
                if cleaning_presets.get('fill_missing'):
                    st.write(f"✅ **Fill missing values**: {missing_pct:.1f}% missing data is manageable")
                    st.write(f"📊 **Strategy**: {cleaning_presets.get('missing_strategy')} (based on {numeric_cols} numeric vs {categorical_cols} categorical columns)")
                else:
                    if missing_pct > 50:
                        st.write(f"❌ **Fill missing values**: {missing_pct:.1f}% missing is too high - recommend dropping instead")
                    else:
                        st.write(f"❌ **Fill missing values**: {missing_pct:.1f}% missing is minimal")
                
                if cleaning_presets.get('remove_outliers'):
                    st.write(f"✅ **Remove outliers**: Dataset has {numeric_cols} numeric columns and {len(df):,} rows - outlier detection recommended")
                else:
                    if numeric_cols == 0:
                        st.write("❌ **Remove outliers**: No numeric columns for outlier detection")
                    else:
                        st.write(f"❌ **Remove outliers**: Dataset too small ({len(df):,} rows) for reliable outlier detection")
                
                if cleaning_presets.get('fix_negatives'):
                    negative_cols = []
                    for col in df.select_dtypes(include=[np.number]).columns:
                        if (df[col] < 0).any():
                            negative_cols.append(col)
                    st.write(f"✅ **Fix negative values**: Found negative values in quantity/amount-like columns: {', '.join(negative_cols[:3])}")
                else:
                    st.write("❌ **Fix negative values**: No problematic negative values detected")
                
                # Performance considerations
                if ai_recs.get('performance_risk') == 'High':
                    st.warning("⚡ **Performance Mode**: Aggressive cleaning enabled due to high performance risk")
                elif ai_recs.get('performance_risk') == 'Medium':
                    st.info("⚡ **Balanced Mode**: Moderate cleaning for medium performance risk")
        
        # Cleaning options
        with st.form("cleaning_form"):
            st.write("**Select Cleaning Steps:**")
            
            # Create tabs for organizing options
            basic_tab, advanced_tab = st.tabs(["✅ Basic Cleaning", "⚡ Advanced Options"])
            
            with basic_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Structure & Format:**")
                    normalize_cols = st.checkbox("Normalize column names", 
                                                value=cleaning_presets.get('normalize_cols', True), 
                                                help="Convert to lowercase with underscores")
                    convert_numeric = st.checkbox("Convert to numeric", 
                                                 value=cleaning_presets.get('convert_numeric', True),
                                                 help="Convert compatible columns to numbers")
                    trim_strings = st.checkbox("Trim whitespace from text", 
                                               value=cleaning_presets.get('trim_strings', True),
                                               help="Remove leading/trailing spaces")
                    parse_dates = st.checkbox("Auto-parse date columns", 
                                             value=cleaning_presets.get('parse_dates', True),
                                             help="Automatically detect and parse dates")
                
                with col2:
                    st.markdown("**🧹 Data Quality:**")
                    remove_dups = st.checkbox("Remove duplicate rows", 
                                             value=cleaning_presets.get('remove_dups', True))
                    remove_constant = st.checkbox("Remove constant columns", 
                                                 value=cleaning_presets.get('remove_constant', True),
                                                 help="Remove columns with all same values")
                    remove_empty_rows = st.checkbox("Remove empty rows", 
                                                   value=cleaning_presets.get('remove_empty_rows', True),
                                                   help="Remove rows with all missing values")
                    drop_high_missing = st.checkbox("Drop columns with >80% missing", 
                                                   value=cleaning_presets.get('drop_high_missing', False))
            
            with advanced_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Missing Values:**")
                    fill_missing = st.checkbox("Fill missing values", 
                                              value=cleaning_presets.get('fill_missing', True))
                    
                    # Get the index of the preset strategy
                    strategies = ["median", "mean", "mode"]
                    preset_strategy = cleaning_presets.get('missing_strategy', 'median')
                    strategy_index = strategies.index(preset_strategy) if preset_strategy in strategies else 0
                    
                    missing_strategy = st.selectbox("Fill strategy:", 
                                                   strategies,
                                                   index=strategy_index,
                                                   help="Strategy for filling missing values (only used if checkbox is checked)")
                    
                    st.markdown("**🔢 Outliers:**")
                    remove_outliers = st.checkbox("Remove statistical outliers", 
                                                 value=cleaning_presets.get('remove_outliers', False),
                                                 help="Remove data points using IQR method")
                    
                    # Get the index of the preset outlier method
                    outlier_methods = ["IQR", "zscore"]
                    preset_outlier = cleaning_presets.get('outlier_method', 'IQR')
                    outlier_index = outlier_methods.index(preset_outlier) if preset_outlier in outlier_methods else 0
                    
                    outlier_method = st.selectbox("Outlier method:", 
                                                 outlier_methods,
                                                 index=outlier_index,
                                                 help="IQR = 1.5*IQR rule, zscore = 3 std devs (only used if checkbox is checked)")
                
                with col2:
                    st.markdown("**⚠️ Negative Values:**")
                    fix_negatives = st.checkbox("Fix negative quantities/amounts", 
                                               value=cleaning_presets.get('fix_negatives', False),
                                               help="Auto-detect and fix negative values in qty/amount columns")
                    
                    # Get the index of the preset negative method
                    negative_methods = ["abs", "zero", "drop"]
                    preset_negative = cleaning_presets.get('negative_method', 'abs')
                    negative_index = negative_methods.index(preset_negative) if preset_negative in negative_methods else 0
                    
                    negative_method = st.selectbox("Fix method:", 
                                                  negative_methods,
                                                  index=negative_index,
                                                  help="abs=absolute value, zero=replace with 0, drop=remove rows (only used if checkbox is checked)")
                    
                    st.markdown("**📂 Categorical:**")
                    standardize_categorical = st.checkbox("Standardize categorical values", 
                                                        value=cleaning_presets.get('standardize_categorical', False),
                                                        help="Lowercase and trim categorical values")
            
            submitted = st.form_submit_button("🚀 Clean Data Now", type="primary", use_container_width=True)
        
        if submitted:
            from utils.process_manager import ProcessManager
            
            # Create process manager
            pm = ProcessManager("Data_Cleaning")
            
            # Show warning about not navigating
            st.warning("""
            ⚠️ **Important:** Do not navigate away from this page during cleaning.
            Navigation is now locked to prevent data loss.
            """)
            
            # Lock navigation
            pm.lock()
            
            try:
                with st.status("🧹 Cleaning data...", expanded=True) as status:
                    from utils.data_cleaning import DataCleaner
                    
                    # Progress tracking
                    st.divider()
                    st.subheader("⚙️ Cleaning Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing data cleaner...")
                    progress_bar.progress(0.2)
                    
                    cleaner = DataCleaner(df)
                    
                    status_text.text("Running cleaning pipeline...")
                    progress_bar.progress(0.5)
                    
                    result = cleaner.clean_pipeline(
                        normalize_cols=normalize_cols,
                        convert_numeric=convert_numeric,
                        remove_dups=remove_dups,
                        fill_missing=fill_missing,
                        missing_strategy=missing_strategy,
                        drop_high_missing_cols=drop_high_missing,
                        col_threshold=0.8,
                        # New parameters
                        trim_strings=trim_strings,
                        remove_outliers_flag=remove_outliers,
                        outlier_method=outlier_method,
                        remove_constant=remove_constant,
                        parse_dates_flag=parse_dates,
                        remove_empty_rows_flag=remove_empty_rows,
                        fix_negatives=fix_negatives,
                        negative_method=negative_method,
                        standardize_categorical_flag=standardize_categorical
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("Storing cleaned data...")
                    
                    # Store cleaned data and results
                    st.session_state.data = result['cleaned_df']
                    st.session_state.original_data = result['original_df']
                    st.session_state.cleaning_stats = result['stats']
                    st.session_state.cleaning_quality_score = result['quality_score']
                    st.session_state.cleaning_quality_score_before = result.get('quality_score_before', 0)
                    
                    # Clear cached analysis
                    for key in ['profile', 'issues', 'viz_suggestions', 'ai_insights', 'cleaning_suggestions']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Save checkpoint
                    pm.save_checkpoint({
                        'completed': True,
                        'original_rows': result['stats']['original_shape'][0],
                        'cleaned_rows': result['stats']['cleaned_shape'][0],
                        'quality_score': result['quality_score'],
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Cleaning complete!")
                    
                    st.success("✅ Data cleaned successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error during cleaning: {str(e)}")
                pm.save_checkpoint({'error': str(e)})
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Always unlock navigation
                pm.unlock()
                st.info("✅ Navigation unlocked - you can now navigate to other pages.")
                # Small delay before rerun
                import time
                time.sleep(1)
                st.rerun()
        
        # Show cleaning results if available
        if 'cleaning_stats' in st.session_state:
            stats = st.session_state.cleaning_stats
            
            st.divider()
            st.subheader("📊 Cleaning Results")
            
            # Before/After comparison
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container():
                    st.markdown("**Before Cleaning:**")
                    st.markdown(f"- **Rows:** {stats['original_shape'][0]:,}")
                    st.markdown(f"- **Columns:** {stats['original_shape'][1]}")
                    st.markdown(f"- **Missing values:** {stats['original_missing']:,}")
                    st.markdown(f"- **Duplicates:** {stats.get('duplicates_removed', 0):,}")
            
            with col2:
                with st.container():
                    st.markdown("**After Cleaning:**")
                    st.markdown(f"- **Rows:** {stats['cleaned_shape'][0]:,}")
                    st.markdown(f"- **Columns:** {stats['cleaned_shape'][1]}")
                    st.markdown(f"- **Missing values:** {stats['cleaned_missing']:,}")
                    st.markdown(f"- **Rows removed:** {stats.get('rows_removed', 0):,}")
            
            # Quality score comparison
            quality_score_after = st.session_state.get('cleaning_quality_score', 0)
            quality_score_before = st.session_state.get('cleaning_quality_score_before', 0)
            
            # Calculate improvement
            improvement = quality_score_after - quality_score_before
            
            # Display before and after scores
            st.markdown("### Data Quality Score")
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.metric(
                    "Previous Quality Score", 
                    f"{quality_score_before:.1f}/100",
                    delta="Before cleaning",
                    delta_color="off"
                )
            
            with col_after:
                st.metric(
                    "Current Quality Score", 
                    f"{quality_score_after:.1f}/100",
                    delta=f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}",
                    delta_color="normal" if improvement >= 0 else "inverse"
                )
    
    with tab2:
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
    
    with tab3:
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
                col = st.selectbox("Select column:", numeric_cols, key="histogram_col")
                if st.button("Create Histogram", key="create_histogram_btn"):
                    fig = viz.create_histogram(df, col)
                    st.plotly_chart(fig, use_container_width=True, key="histogram_chart")
            else:
                st.warning("No numeric columns available for histogram")
        
        elif viz_type == "Bar Chart":
            if categorical_cols:
                col = st.selectbox("Select column:", categorical_cols, key="bar_col")
                if st.button("Create Bar Chart", key="create_bar_btn"):
                    fig = viz.create_bar_chart(df, col)
                    st.plotly_chart(fig, use_container_width=True, key="bar_chart")
            else:
                st.warning("No categorical columns available for bar chart")
        
        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select X axis:", numeric_cols, key="scatter_x")
                col2 = st.selectbox("Select Y axis:", numeric_cols, key="scatter_y")
                if st.button("Create Scatter Plot", key="create_scatter_btn"):
                    fig = viz.create_scatter(df, col1, col2)
                    st.plotly_chart(fig, use_container_width=True, key="scatter_chart")
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        elif viz_type == "Box Plot":
            if numeric_cols:
                col = st.selectbox("Select column:", numeric_cols, key="box_col")
                if st.button("Create Box Plot", key="create_box_btn"):
                    fig = viz.create_box_plot(df, col)
                    st.plotly_chart(fig, use_container_width=True, key="box_chart")
            else:
                st.warning("No numeric columns available for box plot")
        
        elif viz_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="heatmap_cols")
                if st.button("Create Heatmap", key="create_heatmap_btn"):
                    if len(selected_cols) >= 2:
                        fig = viz.create_correlation_heatmap(df, selected_cols)
                        st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")
                    else:
                        st.warning("Please select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
    
    with tab4:
        st.subheader("🤖 AI-Generated Insights")
        
        if 'ai_insights' not in st.session_state:
            if st.button("Generate AI Insights", type="primary"):
                with st.status("🤖 AI is analyzing your data...", expanded=True) as status:
                    try:
                        from utils.ai_helper import AIHelper
                        ai = AIHelper()
                        st.write("Analyzing data patterns...")
                        insights = ai.generate_data_insights(df, profile)
                        st.session_state.ai_insights = insights
                        status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status.update(label="❌ Analysis failed", state="error", expanded=True)
                        st.error(f"Error generating insights: {str(e)}")
        else:
            st.markdown(st.session_state.ai_insights)
            if st.button("Regenerate Insights"):
                del st.session_state.ai_insights
                st.rerun()


def show_insights():
    st.markdown("<h2 style='text-align: center;'>🤖 AI Insights & Natural Language Querying</h2>", unsafe_allow_html=True)
    
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
        with st.status("🤖 AI is analyzing...", expanded=True) as status:
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                st.write("Analyzing your question...")
                # Get answer
                result = ai.answer_data_question(question, df)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'result': result
                })
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                # Rerun to show new result
                st.rerun()
                
            except Exception as e:
                status.update(label="❌ Analysis failed", state="error", expanded=True)
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
    # Hero header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;'>
        <h1 style='margin: 0; color: white;'>📊 Business Intelligence Reports</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;'>
            Generate comprehensive analytics reports with AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if ANY analysis has been completed
    any_analysis_complete = any([
        'mba_rules' in st.session_state,
        'rfm_segmented' in st.session_state,
        'mc_simulations' in st.session_state,
        'ml_results' in st.session_state,
        'mlr_results' in st.session_state,
        'anomaly_results' in st.session_state,
        'arima_results' in st.session_state or 'prophet_results' in st.session_state,
        'sentiment_results' in st.session_state or 'topics' in st.session_state,
        'ab_test_results' in st.session_state,
        'cohort_retention' in st.session_state,
        'rec_similarity' in st.session_state,
        'geo_results' in st.session_state,
        'surv_results' in st.session_state,
        'net_results' in st.session_state
    ])
    
    if st.session_state.data is None and not any_analysis_complete:
        st.error("⚠️ **No Data or Analyses Completed**")
        st.info("💡 Please upload data in the **Data Upload** section or run some analyses to generate reports")
        return
    
    df = st.session_state.data if st.session_state.data is not None else None
    
    # Check which modules have been run
    st.subheader("📊 Analytics Dashboard")
    
    # Check which modules have been run - using correct session state keys
    modules_status = {
        'Market Basket Analysis': 'mba_rules' in st.session_state,
        'RFM Analysis': 'rfm_segmented' in st.session_state,
        'Monte Carlo Simulation': 'mc_simulations' in st.session_state,
        'ML Classification': 'ml_results' in st.session_state,
        'ML Regression': 'mlr_results' in st.session_state,
        'Anomaly Detection': 'anomaly_results' in st.session_state,
        'Time Series Forecasting': ('arima_results' in st.session_state or 'prophet_results' in st.session_state),
        'Text Mining & NLP': ('sentiment_results' in st.session_state or 'topics' in st.session_state),
        'A/B Testing': 'ab_test_results' in st.session_state,
        'Cohort Analysis': 'cohort_retention' in st.session_state,
        'Recommendation Systems': 'rec_similarity' in st.session_state,
        'Geospatial Analysis': 'geo_results' in st.session_state,
        'Survival Analysis': 'surv_results' in st.session_state,
        'Network Analysis': 'net_results' in st.session_state,
    }
    
    completed = sum(modules_status.values())
    total = len(modules_status)
    completion_pct = (completed/total)*100 if total > 0 else 0
    
    # Key metrics with enhanced styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📊 Completed Analyses", 
            f"{completed}/{total}",
            delta=f"{completion_pct:.0f}% complete",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "📥 Available Reports", 
            completed,
            delta="Ready" if completed > 0 else "Pending",
            delta_color="normal" if completed > 0 else "off"
        )
    
    with col3:
        if df is not None:
            st.metric(
                "📈 Data Rows",
                f"{len(df):,}",
                delta=f"{len(df.columns)} columns"
            )
        else:
            st.metric(
                "📈 Analyses",
                "Module-based",
                delta="No dataset loaded"
            )
    
    # Progress bar
    st.progress(completion_pct / 100, text=f"Overall Completion: {completion_pct:.0f}%")
    
    # Key Insights Cards
    if completed > 0 or 'cleaning_stats' in st.session_state:
        st.divider()
        st.subheader("🎯 Key Insights at a Glance")
        
        # Count total insights available (including data cleaning)
        total_insights = completed + (1 if 'cleaning_stats' in st.session_state else 0)
        insight_cols = st.columns(min(total_insights, 4))
        col_idx = 0
        
        # Data Cleaning Insights (show first if available)
        if 'cleaning_stats' in st.session_state:
            with insight_cols[col_idx % 4]:
                stats = st.session_state.cleaning_stats
                quality = st.session_state.get('cleaning_quality_score', 0)
                rows_removed = stats.get('rows_removed', 0)
                
                st.metric(
                    "🧹 Data Cleaned",
                    f"Quality: {quality:.0f}/100",
                    delta=f"{rows_removed:,} rows removed" if rows_removed > 0 else "No rows removed"
                )
            col_idx += 1
        
        # ML Classification Insights
        if modules_status['ML Classification']:
            with insight_cols[col_idx % 4]:
                ml_results = st.session_state.get('ml_results', [])
                if ml_results:
                    best = max(ml_results, key=lambda x: x.get('accuracy', 0))
                    st.metric(
                        "🤖 Best ML Model",
                        best['model_name'],
                        delta=f"{best['accuracy']:.1%} accuracy"
                    )
            col_idx += 1
        
        # ML Regression Insights
        if modules_status['ML Regression']:
            with insight_cols[col_idx % 4]:
                mlr_results = st.session_state.get('mlr_results', [])
                if mlr_results:
                    best = max(mlr_results, key=lambda x: x.get('r2', 0))
                    st.metric(
                        "📈 Best Regressor",
                        best['model_name'],
                        delta=f"R²: {best['r2']:.3f}"
                    )
            col_idx += 1
        
        # Market Basket Insights
        if modules_status['Market Basket Analysis']:
            with insight_cols[col_idx % 4]:
                mba_rules = st.session_state.get('mba_rules', pd.DataFrame())
                if not mba_rules.empty:
                    st.metric(
                        "🧺 Association Rules",
                        len(mba_rules),
                        delta=f"Max confidence: {mba_rules['confidence'].max():.1%}"
                    )
            col_idx += 1
        
        # RFM Insights
        if modules_status['RFM Analysis']:
            with insight_cols[col_idx % 4]:
                rfm_data = st.session_state.get('rfm_segmented', pd.DataFrame())
                if not rfm_data.empty:
                    segments = rfm_data['Segment'].nunique()
                    st.metric(
                        "👥 Customer Segments",
                        segments,
                        delta=f"{len(rfm_data)} customers"
                    )
            col_idx += 1
        
        # Anomaly Detection Insights
        if modules_status['Anomaly Detection'] and col_idx < 4:
            with insight_cols[col_idx % 4]:
                anomaly_results = st.session_state.get('anomaly_results', None)
                if anomaly_results is not None and isinstance(anomaly_results, pd.DataFrame) and len(anomaly_results) > 0:
                    anomalies = sum(anomaly_results['is_anomaly'])
                    anomaly_total = len(anomaly_results)  # Use specific name to avoid collision
                    st.metric(
                        "🔬 Anomalies Found",
                        anomalies,
                        delta=f"{(anomalies/anomaly_total)*100:.1f}% of data"
                    )
            col_idx += 1
        
        # Monte Carlo Insights
        if modules_status['Monte Carlo Simulation'] and col_idx < 4:
            with insight_cols[col_idx % 4]:
                mc_metrics = st.session_state.get('mc_risk_metrics', {})
                if mc_metrics:
                    st.metric(
                        "📈 Expected Return",
                        f"{mc_metrics.get('expected_return', 0):.2f}%",
                        delta=f"VaR: {mc_metrics.get('var_95', 0):.2f}%"
                    )
            col_idx += 1
    
    st.divider()
    
    # Module status cards with improved styling
    st.write("**📋 Module Status Overview:**")
    
    # Create 2-column layout for module cards
    cols = st.columns(2)
    
    # Add Data Cleaning status first if available
    idx = 0
    if 'cleaning_stats' in st.session_state:
        with cols[0]:
            stats = st.session_state.cleaning_stats
            quality = st.session_state.get('cleaning_quality_score', 0)
            st.markdown(f"""
            <div style='background-color: #d4edda; padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid #28a745; margin-bottom: 0.5rem;'>
                <strong>✅ Data Cleaning</strong><br/>
                <small style='color: #155724;'>Quality Score: {quality:.0f}/100 | {stats.get('rows_removed', 0):,} rows removed</small>
            </div>
            """, unsafe_allow_html=True)
        idx = 1
    
    modules_items = list(modules_status.items())
    
    for module_idx, (module, status) in enumerate(modules_items):
        col_idx = (idx + module_idx) % 2
        with cols[col_idx]:
            if status:
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #28a745; margin-bottom: 0.5rem;'>
                    <strong>✅ {module}</strong><br/>
                    <small style='color: #155724;'>Completed - Report available</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #6c757d; margin-bottom: 0.5rem;'>
                    <strong>⚪ {module}</strong><br/>
                    <small style='color: #6c757d;'>Not run yet</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Individual Module Reports
    st.divider()
    st.subheader("📥 Download Individual Reports")
    
    if completed == 0:
        st.info("👆 No analyses completed yet. Run some analyses to generate reports!")
    else:
        st.write("Download reports from completed analyses:")
        
        # Market Basket Analysis
        if modules_status['Market Basket Analysis']:
            with st.expander("🧺 Market Basket Analysis Report"):
                st.write("**Association rules and product recommendations**")
                mba_rules = st.session_state.get('mba_rules', pd.DataFrame())
                if not mba_rules.empty:
                    report = f"""# Market Basket Analysis Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Rules Generated: {len(mba_rules)}
- Top Rule Support: {mba_rules['support'].max():.4f}
- Top Rule Confidence: {mba_rules['confidence'].max():.4f}

## Top 10 Association Rules
{mba_rules.head(10).to_markdown(index=False)}

## Insights
The analysis reveals key product associations that can be used for:
- Cross-selling recommendations
- Product placement optimization
- Bundle offerings
- Marketing campaigns
"""
                    st.download_button("📥 Download Report", report, f"mba_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_mba")
        
        # RFM Analysis
        if modules_status['RFM Analysis']:
            with st.expander("👥 RFM Analysis Report"):
                st.write("**Customer segmentation and insights**")
                rfm_data = st.session_state.get('rfm_segmented', pd.DataFrame())
                if not rfm_data.empty:
                    segments = rfm_data['Segment'].value_counts()
                    report = f"""# RFM Analysis Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Customers: {len(rfm_data)}
- Customer Segments: {rfm_data['Segment'].nunique()}

## Segment Distribution
{segments.to_markdown()}

## Average RFM Scores by Segment
{rfm_data.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean().to_markdown()}

## Business Recommendations
Use these segments for targeted marketing campaigns and customer retention strategies.
"""
                    st.download_button("📥 Download Report", report, f"rfm_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_rfm")
        
        # Monte Carlo
        if modules_status['Monte Carlo Simulation']:
            with st.expander("📈 Monte Carlo Simulation Report"):
                st.write("**Financial forecasting and risk analysis**")
                mc_metrics = st.session_state.get('mc_risk_metrics', {})
                if mc_metrics:
                    report = f"""# Monte Carlo Simulation Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Expected Return: {mc_metrics.get('expected_return', 0):.2f}%
- Value at Risk (VaR): {mc_metrics.get('var_95', 0):.2f}%
- Conditional VaR: {mc_metrics.get('cvar_95', 0):.2f}%
- Sharpe Ratio: {mc_metrics.get('sharpe_ratio', 0):.4f}

## Risk Assessment
The simulation provides probabilistic forecasts for financial planning and risk management.
"""
                    
                    # Add AI insights if available
                    if 'mc_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Investment Analysis

{st.session_state.mc_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Monte Carlo Simulation Module*
"""
                    st.download_button("📥 Download Report", report, f"montecarlo_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_mc")
        
        # ML Classification
        if modules_status['ML Classification']:
            with st.expander("🤖 ML Classification Report"):
                st.write("**Model performance and best model details**")
                ml_results = st.session_state.get('ml_results', [])
                if ml_results:
                    best = max(ml_results, key=lambda x: x.get('accuracy', 0))
                    report = f"""# ML Classification Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Model: {best['model_name']}
- Accuracy: {best['accuracy']:.4f}
- Precision: {best.get('precision', 0):.4f}
- Recall: {best.get('recall', 0):.4f}
- F1-Score: {best.get('f1', 0):.4f}

## All Models Comparison
{pd.DataFrame(ml_results)[['model_name', 'accuracy', 'precision', 'recall', 'f1']].to_markdown(index=False)}

## Recommendation
Use {best['model_name']} for production deployment based on highest accuracy.
"""
                    st.download_button("📥 Download Report", report, f"classification_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_mlc")
        
        # ML Regression
        if modules_status['ML Regression']:
            with st.expander("📈 ML Regression Report"):
                st.write("**Regression model results and predictions**")
                mlr_results = st.session_state.get('mlr_results', [])
                if mlr_results:
                    best = max(mlr_results, key=lambda x: x.get('r2', 0))
                    report = f"""# ML Regression Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Model: {best['model_name']}
- R² Score: {best['r2']:.4f}
- RMSE: {best.get('rmse', 0):.4f}
- MAE: {best.get('mae', 0):.4f}

## All Models Comparison
{pd.DataFrame(mlr_results)[['model_name', 'r2', 'rmse', 'mae']].to_markdown(index=False)}

## Recommendation
Use {best['model_name']} for prediction based on highest R² score.
"""
                    st.download_button("📥 Download Report", report, f"regression_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_mlr")
        
        # Anomaly Detection
        if modules_status['Anomaly Detection']:
            with st.expander("🔬 Anomaly Detection Report"):
                st.write("**Outlier detection and anomaly insights**")
                anomaly_results = st.session_state.get('anomaly_results', None)
                if anomaly_results is not None and isinstance(anomaly_results, pd.DataFrame) and len(anomaly_results) > 0:
                    num_anomalies = sum(anomaly_results['is_anomaly'])
                    total_points = len(anomaly_results)
                    report = f"""# Anomaly Detection Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Algorithm: {st.session_state.get('anomaly_algorithm', 'N/A')}
- Outliers Detected: {num_anomalies}
- Data Points Analyzed: {total_points}
- Percentage: {(num_anomalies/total_points)*100:.2f}%

## Anomaly Distribution
Anomalies represent unusual patterns that may require investigation.

## Recommendations
Review detected anomalies for:
- Data quality issues
- Fraudulent transactions
- System errors
- Unusual customer behavior
"""
                    
                    # Add AI insights if available
                    if 'anomaly_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Anomaly Explanation

{st.session_state.anomaly_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Anomaly Detection Module*
"""
                    st.download_button("📥 Download Report", report, f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_anom")
        
        # Time Series Forecasting
        if modules_status['Time Series Forecasting']:
            with st.expander("📊 Time Series Forecasting Report"):
                st.write("**Forecast results and trend analysis**")
                has_arima = 'arima_results' in st.session_state
                has_prophet = 'prophet_results' in st.session_state
                
                if has_arima or has_prophet:
                    models_used = []
                    model_details = []
                    
                    if has_arima:
                        models_used.append('ARIMA')
                        arima_results = st.session_state.arima_results
                        model_details.append(f"""
### ARIMA Model
- Model Order: {arima_results.get('model_order', 'N/A')}
- AIC: {arima_results.get('aic', 0):.2f}
- BIC: {arima_results.get('bic', 0):.2f}
""")
                    
                    if has_prophet:
                        models_used.append('Prophet')
                        model_details.append("""
### Prophet Model
- Automatic seasonality detection
- Trend and holiday effects included
""")
                    
                    report = f"""# Time Series Forecasting Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Models Used: {', '.join(models_used)}
- Forecast completed successfully

{''.join(model_details)}

## Applications
Use these forecasts for:
- Demand planning
- Resource allocation
- Budget forecasting
- Capacity planning
"""
                    
                    # Add AI insights if available
                    if 'ts_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Forecast Insights

{st.session_state.ts_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Time Series Forecasting Module*
"""
                    st.download_button("📥 Download Report", report, f"timeseries_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_ts")
        
        # Text Mining
        if modules_status['Text Mining & NLP']:
            with st.expander("💬 Text Mining Report"):
                st.write("**Sentiment analysis and text insights**")
                sentiment_df = st.session_state.get('sentiment_results', pd.DataFrame())
                topics = st.session_state.get('topics', {})
                
                if not sentiment_df.empty or topics:
                    sentiment_summary = ""
                    if not sentiment_df.empty:
                        positive = (sentiment_df['sentiment'] == 'positive').sum()
                        negative = (sentiment_df['sentiment'] == 'negative').sum()
                        neutral = (sentiment_df['sentiment'] == 'neutral').sum()
                        sentiment_summary = f"""
## Sentiment Analysis
- Total Texts Analyzed: {len(sentiment_df)}
- Positive: {positive} ({positive/len(sentiment_df)*100:.1f}%)
- Negative: {negative} ({negative/len(sentiment_df)*100:.1f}%)
- Neutral: {neutral} ({neutral/len(sentiment_df)*100:.1f}%)
"""
                    
                    topic_summary = ""
                    if topics:
                        topic_summary = f"""
## Topic Modeling
- Topics Discovered: {len(topics)}
- Key themes identified in text corpus
"""
                    
                    report = f"""# Text Mining & NLP Report
                    
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{sentiment_summary}

{topic_summary}

## Applications
- Customer feedback analysis
- Brand sentiment monitoring
- Content categorization
- Trend identification
"""
                    
                    # Add AI insights if available
                    if 'text_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Text Summary

{st.session_state.text_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Text Mining & NLP Module*
"""
                    st.download_button("📥 Download Report", report, f"textmining_report_{pd.Timestamp.now().strftime('%Y%m%d')}.md", "text/markdown", key="dl_txt")
    
    # Comprehensive Report
    st.divider()
    st.subheader("📋 Comprehensive Summary Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_insights = st.checkbox("Include AI Insights", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    with col2:
        include_module_summaries = st.checkbox("Include All Module Summaries", value=True)
        include_visualizations = st.checkbox("Include Visualization Descriptions", value=False)
    
    # Generate comprehensive report
    if st.button("🎯 Generate Comprehensive Report", type="primary", use_container_width=True):
        from utils.process_manager import ProcessManager
        
        pm = ProcessManager("Report_Generation")
        pm.lock()
        
        # Show warning BEFORE spinner to prevent text cutoff
        st.warning("⚠️ **Important:** Navigation locked during report generation. Please do not navigate away.")
        
        try:
            with st.status("📝 Generating comprehensive report...", expanded=True) as status:
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                from utils.report_generator import ReportGenerator
                from datetime import datetime
                
                status_text.text("Gathering data profile...")
                progress_bar.progress(0.1)
                
                # Get or create profile and issues (only if df exists)
                if df is not None:
                    if 'profile' not in st.session_state or not st.session_state.profile:
                        from utils.data_processor import DataProcessor
                        st.session_state.profile = DataProcessor.profile_data(df)
                        st.session_state.issues = DataProcessor.detect_data_quality_issues(df)
                    
                    profile = st.session_state.profile
                    issues = st.session_state.get('issues', [])
                else:
                    # No dataset loaded - module-only analysis
                    profile = None
                    issues = []
                
                status_text.text("Gathering AI insights...")
                progress_bar.progress(0.2)
                
                # Gather AI insights if requested
                insights_text = ""
                if include_insights:
                    if 'ai_insights' in st.session_state:
                        insights_text = st.session_state.ai_insights
                    else:
                        insights_text = "No AI insights generated yet. Visit the Insights page to generate automated insights."
                else:
                    insights_text = "AI insights not included in this report."
                
                status_text.text("Building module summaries...")
                progress_bar.progress(0.4)
                
                # Build module summaries
                module_insights = []
                if include_module_summaries:
                    module_insights.append("\n## Advanced Analytics Summary\n")
                    
                    if modules_status['ML Classification']:
                        ml_results = st.session_state.get('ml_results', [])
                        if ml_results:
                            best = max(ml_results, key=lambda x: x.get('accuracy', 0))
                            module_insights.append(f"""
### 🤖 ML Classification
- **Best Model:** {best['model_name']}
- **Accuracy:** {best['accuracy']:.4f}
- **Precision:** {best.get('precision', 0):.4f}
- **Recall:** {best.get('recall', 0):.4f}
- **F1-Score:** {best.get('f1', 0):.4f}
- **Models Trained:** {len(ml_results)}
""")
                            if 'ml_ai_insights' in st.session_state:
                                module_insights.append(f"""
**AI Insights:**
{st.session_state.ml_ai_insights}
""")
                    
                    if modules_status['ML Regression']:
                        mlr_results = st.session_state.get('mlr_results', [])
                        if mlr_results:
                            best = max(mlr_results, key=lambda x: x.get('r2', 0))
                            module_insights.append(f"""
### 📈 ML Regression
- **Best Model:** {best['model_name']}
- **R² Score:** {best['r2']:.4f}
- **RMSE:** {best.get('rmse', 0):.4f}
- **MAE:** {best.get('mae', 0):.4f}
- **Models Trained:** {len(mlr_results)}
""")
                            if 'mlr_ai_insights' in st.session_state:
                                module_insights.append(f"""
**AI Insights:**
{st.session_state.mlr_ai_insights}
""")
                    
                    if modules_status['Market Basket Analysis']:
                        mba_rules = st.session_state.get('mba_rules', pd.DataFrame())
                        if not mba_rules.empty:
                            rules_count = len(mba_rules)
                            max_confidence = mba_rules['confidence'].max()
                            max_lift = mba_rules['lift'].max()
                            avg_support = mba_rules['support'].mean()
                            # Get best rule (highest confidence * lift)
                            mba_rules['score'] = mba_rules['confidence'] * mba_rules['lift']
                            best_rule_idx = mba_rules['score'].idxmax()
                            best_rule = mba_rules.loc[best_rule_idx]
                            module_insights.append(f"""
### 🧺 Market Basket Analysis
- **Association Rules Generated:** {rules_count}
- **Average Support:** {avg_support:.4f}
- **Max Confidence:** {max_confidence:.4f} ({max_confidence*100:.1f}%)
- **Max Lift:** {max_lift:.2f}
- **Best Rule Confidence:** {best_rule['confidence']:.4f}
- **Best Rule Lift:** {best_rule['lift']:.2f}
""")
                        else:
                            module_insights.append("""
### 🧺 Market Basket Analysis
- **Status:** Completed (No rules generated)
""")
                        if 'mba_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.mba_ai_insights}
""")
                    
                    if modules_status['RFM Analysis']:
                        rfm_segmented = st.session_state.get('rfm_segmented', pd.DataFrame())
                        segments = rfm_segmented['Segment'].nunique() if 'Segment' in rfm_segmented.columns else 0
                        module_insights.append(f"""
### 👥 RFM Analysis
- **Status:** Completed
- **Customer Segments:** {segments} identified
- **Customers Analyzed:** {len(rfm_segmented)}
- **RFM Scores:** Recency, Frequency, Monetary calculated
- **Segmentation:** K-Means clustering applied
""")
                        if 'rfm_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.rfm_ai_insights}
""")
                    
                    if modules_status['Anomaly Detection']:
                        anomaly_results = st.session_state.get('anomaly_results', None)
                        algorithm = st.session_state.get('anomaly_algorithm', 'N/A')
                        if anomaly_results is not None and isinstance(anomaly_results, pd.DataFrame):
                            anomalies_count = sum(anomaly_results['is_anomaly']) if 'is_anomaly' in anomaly_results.columns else 0
                            total_records = len(anomaly_results)
                            anomaly_pct = (anomalies_count / total_records * 100) if total_records > 0 else 0
                            avg_score = anomaly_results['anomaly_score'].mean() if 'anomaly_score' in anomaly_results.columns else 0
                            module_insights.append(f"""
### 🔬 Anomaly Detection
- **Algorithm:** {algorithm}
- **Outliers Identified:** {anomalies_count:,} ({anomaly_pct:.2f}% of {total_records:,} records)
- **Average Anomaly Score:** {avg_score:.4f}
- **Detection Threshold:** Automatically determined by contamination parameter
""")
                        else:
                            module_insights.append(f"""
### 🔬 Anomaly Detection
- **Status:** Completed
- **Algorithm:** {algorithm}
- **Results:** Available
""")
                        if 'anomaly_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.anomaly_ai_insights}
""")
                    
                    if modules_status['Time Series Forecasting']:
                        has_arima = 'arima_results' in st.session_state
                        has_prophet = 'prophet_results' in st.session_state
                        models_used = []
                        model_details = []
                        
                        if has_arima:
                            models_used.append('ARIMA')
                            arima_results = st.session_state.arima_results
                            model_order = arima_results.get('model_order', 'N/A')
                            aic = arima_results.get('aic', 0)
                            bic = arima_results.get('bic', 0)
                            forecast_df = arima_results.get('forecast', pd.DataFrame())
                            forecast_periods = len(forecast_df)
                            model_details.append(f"ARIMA{model_order} (AIC: {aic:.2f}, BIC: {bic:.2f})")
                        
                        if has_prophet:
                            models_used.append('Prophet')
                            prophet_results = st.session_state.get('prophet_results', {})
                            prophet_forecast = prophet_results.get('forecast', pd.DataFrame())
                            prophet_periods = len(prophet_forecast)
                            model_details.append(f"Prophet ({prophet_periods} periods forecasted)")
                        
                        module_insights.append(f"""
### 📊 Time Series Forecasting
- **Models Used:** {', '.join(models_used) if models_used else 'N/A'}
- **Model Details:** {' | '.join(model_details) if model_details else 'N/A'}
- **Forecast Periods:** {forecast_periods if has_arima else prophet_periods if has_prophet else 0}
- **Trend Analysis:** Completed with automatic seasonality detection
""")
                        if 'ts_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.ts_ai_insights}
""")
                    
                    if modules_status['Text Mining & NLP']:
                        sentiment_df = st.session_state.get('sentiment_results', pd.DataFrame())
                        topics = st.session_state.get('topics', {})
                        texts_analyzed = len(sentiment_df) if not sentiment_df.empty else 0
                        num_topics = len(topics)
                        
                        sentiment_breakdown = ""
                        if not sentiment_df.empty and 'sentiment' in sentiment_df.columns:
                            positive_count = (sentiment_df['sentiment'] == 'positive').sum()
                            negative_count = (sentiment_df['sentiment'] == 'negative').sum()
                            neutral_count = (sentiment_df['sentiment'] == 'neutral').sum()
                            total_sentiment = len(sentiment_df)
                            sentiment_breakdown = f"""
- **Positive:** {positive_count} ({positive_count/total_sentiment*100:.1f}%)
- **Negative:** {negative_count} ({negative_count/total_sentiment*100:.1f}%)
- **Neutral:** {neutral_count} ({neutral_count/total_sentiment*100:.1f}%)"""
                        
                        module_insights.append(f"""
### 📝 Text Mining & NLP
- **Texts Analyzed:** {texts_analyzed:,}
- **Sentiment Analysis:** {'Completed' if not sentiment_df.empty else 'N/A'}{sentiment_breakdown if sentiment_breakdown else ''}
- **Topic Modeling:** {f'{num_topics} topics discovered' if num_topics > 0 else 'N/A'}
""")
                        if 'text_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.text_ai_insights}
""")
                    
                    if modules_status['Monte Carlo Simulation']:
                        mc_simulations = st.session_state.get('mc_simulations', None)
                        mc_risk_metrics = st.session_state.get('mc_risk_metrics', {})
                        simulations = mc_simulations.shape[0] if mc_simulations is not None else 0
                        expected_return = mc_risk_metrics.get('expected_return', 0)
                        module_insights.append(f"""
### 📈 Monte Carlo Simulation
- **Status:** Completed
- **Simulations Run:** {simulations:,}
- **Expected Return:** {expected_return:.2f}%
- **Risk Metrics:** VaR, CVaR calculated
- **Financial Forecasts:** Generated with confidence intervals
""")
                        if 'mc_ai_insights' in st.session_state:
                            module_insights.append(f"""
**AI Insights:**
{st.session_state.mc_ai_insights}
""")
                
                status_text.text("Generating base report...")
                progress_bar.progress(0.6)
                
                # Gather cleaning suggestions (only if recommendations enabled)
                suggestions = st.session_state.get('cleaning_suggestions', []) if include_recommendations else []
                
                # Generate base report
                if df is not None and profile is not None:
                    # Full report with data profiling
                    base_report = ReportGenerator.generate_full_report(
                        df=df,
                        profile=profile,
                        issues=issues,
                        insights=insights_text,
                        suggestions=suggestions
                    )
                    
                    # Remove recommendations section if not requested
                    if not include_recommendations:
                        import re
                        # Remove the entire Recommendations section
                        base_report = re.sub(r'---\n\n## Recommendations.*?(?=---\n|$)', '', base_report, flags=re.DOTALL)
                    
                    # Append module summaries if requested
                    if module_insights:
                        module_section = "\n".join(module_insights)
                        # Insert module summaries after the executive summary
                        # Try multiple possible insertion points
                        if "## Data Profile" in base_report:
                            base_report = base_report.replace(
                                "\n## Data Profile",
                                f"{module_section}\n\n---\n\n## Data Profile"
                            )
                        else:
                            # Fallback: insert after first occurrence of "---" after Executive Summary
                            parts = base_report.split("## Executive Summary", 1)
                            if len(parts) == 2:
                                remaining = parts[1].split("\n---\n", 1)
                                if len(remaining) == 2:
                                    base_report = parts[0] + "## Executive Summary" + remaining[0] + f"\n---\n{module_section}\n\n---\n" + remaining[1]
                    
                    # Add visualization descriptions if requested
                    if include_visualizations:
                        viz_section = "\n\n---\n\n## Visualizations Generated\n\n"
                        viz_section += "The following visualizations were created during analysis:\n\n"
                        
                        viz_count = 0
                        if modules_status['ML Classification']:
                            viz_section += "### ML Classification\n"
                            viz_section += "- Model performance comparison bar chart (accuracy, precision, recall, F1-score)\n"
                            viz_section += "- Confusion matrix heatmap for best model\n"
                            viz_section += "- ROC curves for model comparison\n\n"
                            viz_count += 3
                        
                        if modules_status['ML Regression']:
                            viz_section += "### ML Regression\n"
                            viz_section += "- Model performance comparison (R², RMSE, MAE)\n"
                            viz_section += "- Actual vs Predicted scatter plot\n"
                            viz_section += "- Residual distribution plot\n\n"
                            viz_count += 3
                        
                        if modules_status['Market Basket Analysis']:
                            viz_section += "### Market Basket Analysis\n"
                            viz_section += "- Association rules scatter plot (support vs confidence, sized by lift)\n"
                            viz_section += "- Top product associations network graph\n"
                            viz_section += "- Support-confidence heatmap\n\n"
                            viz_count += 3
                        
                        if modules_status['RFM Analysis']:
                            viz_section += "### RFM Analysis\n"
                            viz_section += "- Customer segmentation distribution bar chart\n"
                            viz_section += "- RFM scores 3D scatter plot\n"
                            viz_section += "- Segment characteristics heatmap\n\n"
                            viz_count += 3
                        
                        if modules_status['Anomaly Detection']:
                            viz_section += "### Anomaly Detection\n"
                            viz_section += "- Anomaly distribution pie chart\n"
                            viz_section += "- Anomaly scores scatter plot (PCA projection)\n"
                            viz_section += "- Feature importance bar chart\n\n"
                            viz_count += 3
                        
                        if modules_status['Time Series Forecasting']:
                            viz_section += "### Time Series Forecasting\n"
                            viz_section += "- Historical data with trend line\n"
                            viz_section += "- Forecast plot with confidence intervals\n"
                            viz_section += "- Seasonal decomposition plots\n\n"
                            viz_count += 3
                        
                        if modules_status['Text Mining & NLP']:
                            viz_section += "### Text Mining & NLP\n"
                            viz_section += "- Sentiment distribution pie chart\n"
                            viz_section += "- Word cloud of frequent terms\n"
                            viz_section += "- Topic modeling visualization\n\n"
                            viz_count += 3
                        
                        if modules_status['Monte Carlo Simulation']:
                            viz_section += "### Monte Carlo Simulation\n"
                            viz_section += "- Return distribution histogram\n"
                            viz_section += "- Cumulative probability curve\n"
                            viz_section += "- Risk metrics visualization\n\n"
                            viz_count += 3
                        
                        if viz_count > 0:
                            viz_section += f"**Total visualizations:** {viz_count} charts generated\n"
                            viz_section += "\n*Note: All visualizations are interactive and can be downloaded individually.*\n"
                            # Insert before Conclusion section
                            base_report = base_report.replace(
                                "---\n\n## Conclusion",
                                f"{viz_section}\n---\n\n## Conclusion"
                            )
                else:
                    # Module-only report (no dataset loaded)
                    module_section = "\n".join(module_insights) if module_insights else "No modules completed yet."
                    recommendations_section = ""
                    if include_recommendations:
                        recommendations_section = """\n---\n\n## Recommendations\n\n- Upload a dataset in the Data Upload section for comprehensive data profiling\n- Continue running analyses to gain deeper insights\n- Use individual module export functions for detailed results\n"""
                    
                    base_report = f"""# DataInsights - Analytics Report

**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

## Executive Summary

This report contains results from completed analytics modules.

{module_section}{recommendations_section}
"""
                
                status_text.text("Adding AI insights summary...")
                progress_bar.progress(0.75)
                
                # Build comprehensive AI insights section if requested
                if include_insights:
                    ai_insights_section = "\n\n---\n\n## 🤖 AI-Powered Insights Summary\n\n"
                    ai_insights_section += "This section consolidates all AI-generated insights from completed analytics modules.\n\n"
                    
                    has_any_insights = False
                    
                    # General insights from Insights page
                    if 'ai_insights' in st.session_state:
                        ai_insights_section += "### 📊 Overall Data Insights\n\n"
                        ai_insights_section += st.session_state.ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # ML Classification insights
                    if 'ml_ai_insights' in st.session_state and modules_status['ML Classification']:
                        ai_insights_section += "### 🤖 Machine Learning Classification Insights\n\n"
                        ai_insights_section += st.session_state.ml_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # ML Regression insights
                    if 'mlr_ai_insights' in st.session_state and modules_status['ML Regression']:
                        ai_insights_section += "### 📈 Machine Learning Regression Insights\n\n"
                        ai_insights_section += st.session_state.mlr_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # Market Basket Analysis insights
                    if 'mba_ai_insights' in st.session_state and modules_status['Market Basket Analysis']:
                        ai_insights_section += "### 🧺 Market Basket Analysis Insights\n\n"
                        ai_insights_section += st.session_state.mba_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # RFM Analysis insights
                    if 'rfm_ai_insights' in st.session_state and modules_status['RFM Analysis']:
                        ai_insights_section += "### 👥 RFM Customer Segmentation Insights\n\n"
                        ai_insights_section += st.session_state.rfm_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # Anomaly Detection insights
                    if 'anomaly_ai_insights' in st.session_state and modules_status['Anomaly Detection']:
                        ai_insights_section += "### 🔬 Anomaly Detection Insights\n\n"
                        ai_insights_section += st.session_state.anomaly_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # Time Series insights
                    if 'ts_ai_insights' in st.session_state and modules_status['Time Series Forecasting']:
                        ai_insights_section += "### 📊 Time Series Forecasting Insights\n\n"
                        ai_insights_section += st.session_state.ts_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # Text Mining insights
                    if 'text_ai_insights' in st.session_state and modules_status['Text Mining & NLP']:
                        ai_insights_section += "### 📝 Text Mining & Sentiment Analysis Insights\n\n"
                        ai_insights_section += st.session_state.text_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    # Monte Carlo insights
                    if 'mc_ai_insights' in st.session_state and modules_status['Monte Carlo Simulation']:
                        ai_insights_section += "### 📈 Monte Carlo Simulation Insights\n\n"
                        ai_insights_section += st.session_state.mc_ai_insights + "\n\n"
                        has_any_insights = True
                    
                    if not has_any_insights:
                        ai_insights_section += "*No AI insights have been generated yet. Visit each analytics module and click 'Generate AI Insights' to get automated analysis.*\n\n"
                    
                    # Insert AI insights section before Conclusion
                    if "## Conclusion" in base_report:
                        base_report = base_report.replace(
                            "---\n\n## Conclusion",
                            f"{ai_insights_section}---\n\n## Conclusion"
                        )
                    else:
                        # If no conclusion, append at the end
                        base_report += ai_insights_section
                
                status_text.text("Enhancing report...")
                progress_bar.progress(0.8)
                
                # Add analysis completion metrics at the top
                if "# DataInsights - Business Intelligence Report" in base_report:
                    enhanced_report = base_report.replace(
                        "# DataInsights - Business Intelligence Report",
                        f"""# DataInsights - Comprehensive Business Intelligence Report

**Analyses Completed:** {completed}/{total} modules ({(completed/total)*100:.0f}% complete)  
**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"""
                    )
                else:
                    # Already has title from module-only report
                    enhanced_report = base_report.replace(
                        "**Report Generated:**",
                        f"**Analyses Completed:** {completed}/{total} modules ({(completed/total)*100:.0f}% complete)\n\n**Report Generated:**"
                    )
                
                st.session_state.comprehensive_report = enhanced_report
                
                # Save checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'modules_included': completed,
                    'timestamp': datetime.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                status_text.text("✅ Report generation complete!")
                
                st.success("✅ Comprehensive report generated successfully!")
                st.info(f"📊 Report includes: Data profiling, quality assessment, {completed} module summaries, and recommendations")
                
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            pm.save_checkpoint({'error': str(e)})
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            pm.unlock()
            st.info("✅ Navigation unlocked")
    
    # Display and download comprehensive report
    if 'comprehensive_report' in st.session_state:
        st.divider()
        st.subheader("📄 Generated Report")
        
        # Display report
        st.markdown(st.session_state.comprehensive_report)
        
        # Download buttons with multiple formats
        from datetime import datetime
        import io
        from utils.advanced_report_exporter import AdvancedReportExporter
        
        st.write("**📥 Export Options:**")
        
        # Collect all visualizations from session state
        charts = []
        chart_errors = []
        
        # MBA charts
        if 'mba_rules' in st.session_state and not st.session_state.mba_rules.empty:
            try:
                from utils.market_basket import MarketBasketAnalyzer
                analyzer = MarketBasketAnalyzer()  # No arguments!
                analyzer.transactions = st.session_state.mba_transactions
                analyzer.rules = st.session_state.mba_rules
                charts.append(("MBA: Rules Scatter Plot", analyzer.create_scatter_plot()))
            except Exception as e:
                chart_errors.append(f"MBA chart failed: {str(e)}")
        
        # RFM charts
        if 'rfm_analyzer' in st.session_state and 'rfm_segmented' in st.session_state:
            try:
                from utils.rfm_analysis import RFMAnalyzer
                rfm_data = st.session_state.rfm_segmented
                charts.append(("RFM: Segment Distribution", RFMAnalyzer.create_segment_distribution(rfm_data)))
            except Exception as e:
                chart_errors.append(f"RFM chart failed: {str(e)}")
        
        # Anomaly Detection charts
        if 'anomaly_detector' in st.session_state:
            try:
                detector = st.session_state.anomaly_detector
                # Only create_2d_scatter exists, not create_distribution_chart
                charts.append(("Anomaly Detection: Scatter Plot", detector.create_2d_scatter(use_pca=True, show_only_anomalies=False)))
            except Exception as e:
                chart_errors.append(f"Anomaly Detection charts failed: {str(e)}")
        
        # Time Series charts
        if 'ts_analyzer' in st.session_state:
            try:
                analyzer = st.session_state.ts_analyzer
                if 'arima_results' in st.session_state:
                    charts.append(("Time Series: ARIMA Forecast", analyzer.create_forecast_plot('arima')))
                if 'prophet_results' in st.session_state:
                    charts.append(("Time Series: Prophet Forecast", analyzer.create_forecast_plot('prophet')))
            except Exception as e:
                chart_errors.append(f"Time Series charts failed: {str(e)}")
        
        # Monte Carlo charts
        if 'mc_simulations' in st.session_state and 'mc_stock_data' in st.session_state:
            try:
                from utils.monte_carlo import MonteCarloSimulator
                simulator = MonteCarloSimulator()  # No arguments!
                simulator.simulations = st.session_state.mc_simulations
                simulator.stock_data = st.session_state.mc_stock_data
                
                # Get initial price and final prices from simulations
                initial_price = st.session_state.mc_stock_data['Close'].iloc[-1]
                final_prices = simulator.simulations[:, -1]
                
                charts.append(("Monte Carlo: Return Distribution", simulator.create_distribution_plot(final_prices, initial_price)))
            except Exception as e:
                chart_errors.append(f"Monte Carlo chart failed: {str(e)}")
        
        # Text Mining charts
        if 'text_analyzer' in st.session_state:
            try:
                analyzer = st.session_state.text_analyzer
                
                # Sentiment plot
                if 'sentiment_results' in st.session_state:
                    sentiment_df = st.session_state.sentiment_results
                    charts.append(("Text Mining: Sentiment Analysis", analyzer.create_sentiment_plot(sentiment_df)))
                
                # Word frequency plot
                if 'word_freq_results' in st.session_state:
                    word_freq_df = st.session_state.word_freq_results
                    charts.append(("Text Mining: Word Frequency", analyzer.create_word_frequency_plot(word_freq_df)))
                    
            except Exception as e:
                chart_errors.append(f"Text Mining charts failed: {str(e)}")
        
        # ML Regression charts - Create feature importance chart
        if 'mlr_results' in st.session_state:
            try:
                results = st.session_state.mlr_results
                best_model = max(results, key=lambda x: x.get('r2', 0))
                
                # Feature Importance Chart
                if best_model.get('feature_importance'):
                    import plotly.express as px
                    feat_imp = best_model['feature_importance']
                    # Sort descending and take top 10, then reverse for chart display
                    feat_imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                    feat_imp_sorted = sorted(feat_imp_sorted, key=lambda x: x[1])
                    
                    fig_imp = px.bar(
                        x=[x[1] for x in feat_imp_sorted],
                        y=[x[0] for x in feat_imp_sorted],
                        orientation='h',
                        title=f'ML Regression: Top 10 Features ({best_model["model_name"]})',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=[x[1] for x in feat_imp_sorted],
                        color_continuous_scale='Blues'
                    )
                    fig_imp.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
                    charts.append(("ML Regression: Feature Importance", fig_imp))
            except Exception as e:
                chart_errors.append(f"ML Regression chart failed: {str(e)}")
        
        # ML Classification charts - Create feature importance chart
        if 'ml_results' in st.session_state:
            try:
                results = st.session_state.ml_results
                best_model = max(results, key=lambda x: x.get('accuracy', 0))
                best_details = best_model.get('details', {})
                
                # Feature Importance Chart
                if best_details.get('feature_importance'):
                    import plotly.express as px
                    feat_imp = best_details['feature_importance']
                    # Get top 10 features
                    importance_df = pd.DataFrame({
                        'Feature': feat_imp['features'][:10],
                        'Importance': feat_imp['importances'][:10]
                    })
                    # Sort for chart display
                    importance_df = importance_df.sort_values('Importance')
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'ML Classification: Top 10 Features ({best_model["model_name"]})',
                        color='Importance',
                        color_continuous_scale='Greens'
                    )
                    fig_imp.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
                    charts.append(("ML Classification: Feature Importance", fig_imp))
            except Exception as e:
                chart_errors.append(f"ML Classification chart failed: {str(e)}")
        
        # Show debug info
        if chart_errors:
            with st.expander("⚠️ Chart Collection Issues (Debug Info)"):
                for error in chart_errors:
                    st.warning(error)
        
        col1, col2, col3 = st.columns(3)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_text = st.session_state.comprehensive_report
        
        with col1:
            st.download_button(
                label="📄 Markdown",
                data=report_text,
                file_name=f"report_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Best for GitHub, documentation"
            )
        
        with col2:
            # Markdown + Images ZIP
            try:
                # Check if kaleido is available
                try:
                    import kaleido
                    has_kaleido = True
                except ImportError:
                    has_kaleido = False
                
                if has_kaleido and charts:
                    zip_data = AdvancedReportExporter.create_markdown_with_images_zip(
                        report_text,
                        charts,
                        filename_prefix="report"
                    )
                    st.download_button(
                        label="📦 MD + Images",
                        data=zip_data,
                        file_name=f"report_{timestamp}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        help="ZIP with markdown and chart images"
                    )
                else:
                    st.download_button(
                        label="📦 MD + Images",
                        data=report_text,
                        file_name=f"report_{timestamp}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        help="Install kaleido for images: pip install kaleido",
                        disabled=not charts
                    )
            except Exception as e:
                st.download_button(
                    label="📦 MD + Images",
                    data=report_text,
                    file_name=f"report_{timestamp}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    help=f"Error: {str(e)[:50]}",
                    disabled=True
                )
        
        with col3:
            # HTML with Interactive Charts
            try:
                if charts:
                    html_content = AdvancedReportExporter.create_html_report(
                        report_text,
                        charts,
                        title="DataInsights - Comprehensive Report"
                    )
                else:
                    # Basic HTML without charts
                    html_content = AdvancedReportExporter.create_html_report(
                        report_text,
                        [],
                        title="DataInsights - Comprehensive Report"
                    )
                
                st.download_button(
                    label="🌐 HTML",
                    data=html_content,
                    file_name=f"report_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Interactive web page with charts"
                )
            except Exception as e:
                # Fallback to basic HTML
                basic_html = f"<html><head><meta charset='UTF-8'><title>Report</title></head><body><pre>{report_text}</pre></body></html>"
                st.download_button(
                    label="🌐 HTML",
                    data=basic_html,
                    file_name=f"report_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Basic HTML (charts failed)"
                )
        
        # Show visualization count with details
        if charts:
            chart_names = [name for name, _ in charts]
            st.success(f"✅ {len(charts)} visualization(s) collected for export")
            with st.expander("📊 Charts Included"):
                for name in chart_names:
                    st.write(f"• {name}")
        else:
            st.info("ℹ️ No visualizations available. Run analyses to generate charts for export.")
            
            # Debug: Show which modules were checked
            with st.expander("🔍 Debug: Module Status"):
                st.write("**Checking for charts in:**")
                st.write(f"• MBA: {'mba_rules' in st.session_state}")
                st.write(f"• RFM: {'rfm_analyzer' in st.session_state}")
                st.write(f"• Anomaly Detection: {'anomaly_detector' in st.session_state}")
                st.write(f"• Time Series: {'ts_analyzer' in st.session_state}")
                st.write(f"• Monte Carlo: {'mc_simulations' in st.session_state}")
                st.write(f"• Text Mining: {'text_analyzer' in st.session_state}")
                st.write(f"• ML Regression: {'mlr_results' in st.session_state}")
                st.write(f"• ML Classification: {'ml_results' in st.session_state}")

def show_market_basket_analysis():
    """Market Basket Analysis page."""
    st.markdown("<h2 style='text-align: center;'>🧺 Market Basket Analysis</h2>", unsafe_allow_html=True)
    
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
    
    # Add clear cache button
    if 'mba_transactions' in st.session_state:
        if st.button("🔄 Clear MBA Cache & Start Fresh", type="secondary"):
            for key in ['mba_transactions', 'mba_encoded', 'mba_frequent_itemsets', 'mba_rules']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Cache cleared! Reload your data.")
            st.rerun()
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample Groceries Dataset", "Upload Custom Data"]
        default_option = "Use Loaded Dataset"
    else:
        data_options = ["Sample Groceries Dataset", "Upload Custom Data"]
        default_option = "Sample Groceries Dataset"
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="mba_data_source"
    )
    
    transactions = None
    
    if data_source == "Use Loaded Dataset":
        st.success("✅ Using dataset from Data Upload section")
        df = st.session_state.data
        
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Get smart column suggestions
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_mba_column_suggestions(df)
        
        # Validate data suitability
        validation = ColumnDetector.validate_mba_suitability(df)
        
        # Store validation result
        st.session_state.mba_data_suitable = validation['suitable']
        
        if not validation['suitable']:
            st.error("❌ **Dataset Not Suitable for Market Basket Analysis**")
            for warning in validation['warnings']:
                st.warning(warning)
            st.info("**💡 Recommendations:**")
            for rec in validation['recommendations']:
                st.write(f"- {rec}")
            st.write("**Consider using:**")
            st.write("- Sample Groceries Dataset (built-in)")
            st.write("- A different dataset with transactional data")
            st.stop()  # STOP here - don't show process button
        elif len(validation['warnings']) > 0:
            with st.expander("⚠️ Data Quality Warnings", expanded=False):
                for warning in validation['warnings']:
                    st.warning(warning)
                if validation['recommendations']:
                    st.info("**Recommendations:**")
                    for rec in validation['recommendations']:
                        st.write(f"- {rec}")
        else:
            st.success(f"✅ **Dataset looks suitable for MBA** (Confidence: {validation['confidence']})")
        
        # Let user select columns for transaction analysis
        st.write("**Select columns for Market Basket Analysis:**")
        st.info("💡 **Smart Detection:** Columns are auto-selected based on your data. You can change them if needed.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Find index of suggested column
            trans_idx = list(df.columns).index(suggestions['transaction_id']) if suggestions['transaction_id'] in df.columns else 0
            trans_col = st.selectbox(
                "Transaction ID column:", 
                df.columns, 
                index=trans_idx,
                key="loaded_trans_col",
                help="Column that groups items into transactions (e.g., Order ID, Invoice ID)"
            )
        with col2:
            # Find index of suggested column
            item_idx = list(df.columns).index(suggestions['item']) if suggestions['item'] in df.columns else 0
            item_col = st.selectbox(
                "Item column:", 
                df.columns,
                index=item_idx, 
                key="loaded_item_col",
                help="Column containing item names or product descriptions"
            )
        
        # Only show button if data is suitable
        data_suitable = st.session_state.get('mba_data_suitable', True)
        
        if not data_suitable:
            st.error("❌ **Cannot process - data incompatible with Market Basket Analysis**")
        elif st.button("🔄 Process Loaded Data", type="primary"):
            with st.status("Processing transactions...", expanded=True) as status:
                try:
                    transactions = mba.parse_uploaded_transactions(df, trans_col, item_col)
                    st.session_state.mba_transactions = transactions
                    st.success(f"✅ Processed {len(transactions)} transactions!")
                    
                    # Show sample
                    st.write("**Sample transactions:**")
                    for i, trans in enumerate(transactions[:5], 1):
                        st.write(f"{i}. {trans}")
                    
                    # Show stats
                    unique_items = set([item for trans in transactions for item in trans])
                    st.info(f"📊 {len(transactions)} transactions, {len(unique_items)} unique items")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    elif data_source == "Sample Groceries Dataset":
        if st.button("📥 Load Groceries Data", type="primary"):
            with st.status("Loading groceries dataset...", expanded=True) as status:
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
        with st.status("Encoding transactions...", expanded=True) as status:
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
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"**Encoded DataFrame shape:** {df_encoded.shape}")
        st.write(f"**Number of columns (items):** {len(df_encoded.columns)}")
        st.write(f"**Sample items:** {list(df_encoded.columns[:10])}")
        
        # Count items in raw transactions
        all_items_set = set()
        for trans in transactions:
            all_items_set.update(trans)
        st.write(f"**Unique items from raw transactions:** {len(all_items_set)}")
    
    # Threshold controls
    st.divider()
    st.subheader("🎛️ 2. Adjust Thresholds")
    
    st.info("💡 **Memory-Friendly Defaults:** Higher support = less memory usage. Recommended for large datasets (>10k transactions).")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.001,
            max_value=0.5,
            value=0.02,
            step=0.005,
            help="Minimum frequency for an itemset (default: 0.02 = 2% - balanced for most datasets)"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum confidence for a rule (default: 0.4 = 40%)"
        )
    
    with col3:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=1.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="Minimum lift - how much more likely items are bought together (default: 1.5 for stronger patterns)"
        )
    
    # Run analysis button
    if st.button("🚀 Run Market Basket Analysis", type="primary", use_container_width=True):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Market_Basket_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Mining frequent itemsets and generating rules...", expanded=True) as status:
                # Validate thresholds
                if min_support <= 0 or min_support > 1:
                    st.error("❌ Minimum support must be between 0 and 1")
                    pm.unlock()
                    st.stop()
                
                if min_confidence <= 0 or min_confidence > 1:
                    st.error("❌ Minimum confidence must be between 0 and 1")
                    pm.unlock()
                    st.stop()
                
                if min_lift < 0:
                    st.error("❌ Minimum lift must be positive")
                    pm.unlock()
                    st.stop()
                
                # Progress tracking
                st.divider()
                st.subheader("⚙️ Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Mining frequent itemsets...")
                progress_bar.progress(0.3)
                
                # Find frequent itemsets
                itemsets = mba.find_frequent_itemsets(min_support=min_support)
                
                if len(itemsets) == 0:
                    st.warning(f"⚠️ No frequent itemsets found with support >= {min_support}. Try lowering the minimum support.")
                    pm.unlock()
                    st.stop()
                
                st.session_state.mba_itemsets = itemsets
                
                progress_bar.progress(0.6)
                status_text.text(f"Found {len(itemsets)} itemsets, generating rules...")
                
                # Generate rules
                rules = mba.generate_association_rules(
                    metric='lift',
                    min_threshold=min_lift,
                    min_confidence=min_confidence,
                    min_support=min_support
                )
                
                progress_bar.progress(0.9)
                status_text.text("Finalizing results...")
                
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
                    pm.unlock()
                    st.stop()
                
                st.session_state.mba_rules = rules
                
                # Save checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'itemsets_count': len(itemsets),
                    'rules_count': len(rules),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                st.success(f"✅ Found {len(itemsets)} frequent itemsets and {len(rules)} association rules!")
                
        except ValueError as e:
            st.error(f"❌ Validation error: {str(e)}")
            pm.save_checkpoint({'error': str(e)})
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("💡 Try adjusting the thresholds or checking your data format.")
            pm.save_checkpoint({'error': str(e)})
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Always unlock navigation
            pm.unlock()
            st.info("✅ Navigation unlocked - you can now navigate to other pages.")
    
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
            try:
                display_rules['Antecedents'] = display_rules['antecedents'].apply(
                    lambda x: mba.format_itemset(x)
                )
                display_rules['Consequents'] = display_rules['consequents'].apply(
                    lambda x: mba.format_itemset(x)
                )
            except Exception as e:
                st.error(f"Error formatting rules: {str(e)}")
                st.warning("This might be due to selecting the wrong column type for 'Item'. Please ensure you selected a text column (like Description), not a numeric column (like Quantity or Price).")
                return
            
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
            
            # AI Insights
            st.divider()
            st.subheader("✨ AI-Powered Insights")
            
            # Display saved insights if they exist
            if 'mba_ai_insights' in st.session_state:
                st.markdown(st.session_state.mba_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
            
            if st.button("🤖 Generate AI Insights", key="mba_ai_insights_btn", use_container_width=True):
                try:
                    from utils.ai_helper import AIHelper
                    ai = AIHelper()
                    
                    with st.status("🤖 Analyzing market basket patterns...", expanded=True) as status:
                        st.write("Preparing analysis data...")
                        # Get data from session state
                        rules_data = st.session_state.get('mba_rules', pd.DataFrame())
                        transactions_data = st.session_state.get('mba_transactions', [])
                        df_encoded_data = st.session_state.get('mba_encoded', pd.DataFrame())
                        itemsets_data = st.session_state.get('mba_itemsets', pd.DataFrame())
                        
                        if rules_data.empty or len(transactions_data) == 0:
                            st.error("No analysis results available. Please run Market Basket Analysis first.")
                            st.stop()
                        
                        # Get top rules for context
                        top_rules = rules_data.nlargest(10, 'lift')
                        rules_text = ""
                        
                        if len(top_rules) > 0:
                            for idx, row in top_rules.iterrows():
                                # Handle frozensets properly
                                try:
                                    ant_items = list(row['antecedents'])
                                    cons_items = list(row['consequents'])
                                    ant = ', '.join(str(item) for item in ant_items)
                                    cons = ', '.join(str(item) for item in cons_items)
                                    rules_text += f"\n- {ant} → {cons} (Support: {row.get('support', 0):.3f}, Confidence: {row.get('confidence', 0):.3f}, Lift: {row.get('lift', 0):.2f})"
                                except Exception as e:
                                    continue
                        else:
                            rules_text = "\nNo rules found with current thresholds."
                        
                        # Calculate summary
                        avg_confidence = rules_data['confidence'].mean() if 'confidence' in rules_data.columns else 0
                        avg_lift = rules_data['lift'].mean() if 'lift' in rules_data.columns else 0
                        
                        # Prepare context
                        context = f"""
                        Market Basket Analysis Results:
                        
                        Dataset: {len(transactions_data):,} transactions, {len(df_encoded_data.columns):,} unique items
                        Average Basket Size: {sum(len(t) for t in transactions_data) / len(transactions_data):.2f} items
                        
                        Results:
                        - Frequent Itemsets: {len(itemsets_data):,}
                        - Association Rules: {len(rules_data):,}
                        - Average Confidence: {avg_confidence:.3f}
                        - Average Lift: {avg_lift:.2f}
                        
                        Top 10 Association Rules:
                        {rules_text}
                        """
                        
                        prompt = f"""
                        As a retail analytics expert, analyze these market basket analysis results and provide:
                        
                        1. **Key Patterns Discovered** (3-4 sentences): What are the most interesting associations and why do they matter?
                        
                        2. **Business Opportunities** (4-5 bullet points): Specific, actionable strategies for:
                           - Product placement and merchandising
                           - Promotional bundles and cross-selling
                           - Pricing strategies
                           - Inventory optimization
                        
                        3. **Customer Behavior Insights** (2-3 sentences): What do these patterns reveal about customer shopping habits?
                        
                        4. **Implementation Priorities** (3-4 bullet points): Which rules should be acted on first and why?
                        
                        5. **Revenue Impact Estimate** (2-3 sentences): How could implementing these insights affect sales?
                        
                        {context}
                        
                        Be specific with product names from the rules. Focus on actionable strategies that drive revenue.
                        """
                        
                        response = ai.client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a retail analytics expert specializing in market basket analysis and customer behavior."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        
                        # Save to session state
                        st.session_state.mba_ai_insights = response.choices[0].message.content
                        status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.mba_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                        
                except Exception as e:
                    st.error(f"Error generating AI insights: {str(e)}")
            
            # Export full report
            st.divider()
            
            if st.button("📄 Generate Full Report", use_container_width=True):
                with st.status("Generating comprehensive report...", expanded=True) as status:
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
"""
                    
                    # Add AI insights if available
                    if 'mba_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Strategic Insights

{st.session_state.mba_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Market Basket Analysis Module*
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
    st.markdown("<h2 style='text-align: center;'>👥 RFM Analysis & Customer Segmentation</h2>", unsafe_allow_html=True)
    
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
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Upload Custom Data", "Use Sample Data"]
    else:
        data_options = ["Upload Custom Data", "Use Sample Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="rfm_data_source"
    )
    
    transactions_df = None
    
    if data_source == "Use Loaded Dataset":
        st.success("✅ Using dataset from Data Upload section")
        df = st.session_state.data
        
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Get smart column suggestions
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_rfm_column_suggestions(df)
        
        # Validate data suitability
        validation = ColumnDetector.validate_rfm_suitability(df)
        
        # Store validation result
        st.session_state.rfm_data_suitable = validation['suitable']
        
        if not validation['suitable']:
            st.error("❌ **Dataset Not Suitable for RFM Analysis**")
            for warning in validation['warnings']:
                st.warning(warning)
            st.info("**💡 Recommendations:**")
            for rec in validation['recommendations']:
                st.write(f"- {rec}")
            st.write("**Consider using:**")
            st.write("- Sample RFM Data (built-in)")
            st.write("- A dataset with customer transactions over time")
            st.stop()  # STOP here - don't show process button
        elif len(validation['warnings']) > 0:
            with st.expander("⚠️ Data Quality Warnings", expanded=False):
                for warning in validation['warnings']:
                    st.warning(warning)
                if validation['recommendations']:
                    st.info("**Recommendations:**")
                    for rec in validation['recommendations']:
                        st.write(f"- {rec}")
        else:
            st.success(f"✅ **Dataset looks suitable for RFM** (Confidence: {validation['confidence']})")
        
        # Let user select columns for RFM analysis
        st.write("**Select columns for RFM Analysis:**")
        st.info("💡 **Smart Detection:** Columns are auto-selected based on your data. You can change them if needed.")
        
        # Show column types to help user
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        with st.expander("💡 Column Type Hints"):
            st.write(f"**Numeric columns** (for Amount): {', '.join(numeric_cols) if numeric_cols else 'None detected'}")
            st.write(f"**Date columns** (for Date): {', '.join(date_cols) if date_cols else 'None detected - will try to parse'}")
            st.write(f"**All columns**: {', '.join(df.columns.tolist())}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Find index of suggested customer column
            cust_idx = list(df.columns).index(suggestions['customer_id']) if suggestions['customer_id'] in df.columns else 0
            customer_col = st.selectbox(
                "Customer ID column:", 
                df.columns,
                index=cust_idx, 
                key="loaded_rfm_cust_col",
                help="Column that identifies unique customers"
            )
        with col2:
            # Find index of suggested date column
            date_idx = list(df.columns).index(suggestions['date']) if suggestions['date'] in df.columns else 0
            date_col = st.selectbox(
                "Transaction Date column:", 
                df.columns,
                index=date_idx, 
                key="loaded_rfm_date_col",
                help="Column containing transaction dates"
            )
        with col3:
            # Suggest numeric columns first for amount
            amount_options = numeric_cols + [col for col in df.columns if col not in numeric_cols]
            # Find index of suggested amount column
            amount_idx = amount_options.index(suggestions['amount']) if suggestions['amount'] in amount_options else 0
            amount_col = st.selectbox(
                "Amount/Revenue column:", 
                amount_options,
                index=amount_idx, 
                key="loaded_rfm_amount_col",
                help="⚠️ Must be NUMERIC column with transaction amounts"
            )
        
        # Only show button if data is suitable
        data_suitable = st.session_state.get('rfm_data_suitable', True)
        
        if not data_suitable:
            st.error("❌ **Cannot process - data incompatible with RFM Analysis**")
        elif st.button("🔄 Process Loaded Data for RFM", type="primary"):
            with st.status("Processing RFM data...", expanded=True) as status:
                try:
                    # Validate column selections
                    if not pd.api.types.is_numeric_dtype(df[amount_col]):
                        st.error(f"""
                        ❌ **Invalid Amount Column**
                        
                        The selected column '{amount_col}' is not numeric!
                        
                        **Amount column must contain:**
                        - Transaction amounts
                        - Revenue values
                        - Numeric data (integers or decimals)
                        
                        **Please select a numeric column** (e.g., price, total, revenue, amount)
                        """)
                        st.stop()
                    
                    # Try to convert date column
                    try:
                        pd.to_datetime(df[date_col])
                    except:
                        st.error(f"""
                        ❌ **Invalid Date Column**
                        
                        The selected column '{date_col}' cannot be parsed as dates!
                        
                        **Date column must contain:**
                        - Date values (YYYY-MM-DD, MM/DD/YYYY, etc.)
                        - Datetime values
                        
                        **Please select a date column**
                        """)
                        st.stop()
                    
                    st.session_state.rfm_transactions = df
                    st.session_state.rfm_columns = {
                        'customer': customer_col,
                        'date': date_col,
                        'amount': amount_col
                    }
                    st.success("✅ Data processed successfully!")
                    st.info(f"📊 {df[customer_col].nunique()} unique customers, {len(df)} transactions")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    elif data_source == "Upload Custom Data":
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
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("RFM_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Calculating RFM metrics...", expanded=True) as status:
                # Progress tracking
                st.divider()
                st.subheader("⚙️ Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Calculating RFM metrics...")
                progress_bar.progress(0.3)
                
                # Calculate RFM
                rfm_data = rfm_analyzer.calculate_rfm(
                    transactions_df, 
                    cols['customer'], 
                    cols['date'], 
                    cols['amount']
                )
                
                progress_bar.progress(0.6)
                status_text.text("Scoring customers...")
                
                # Score RFM
                rfm_scored = rfm_analyzer.score_rfm(rfm_data, method='quartile')
                
                progress_bar.progress(0.8)
                status_text.text("Segmenting customers...")
                
                # Segment customers
                rfm_segmented = rfm_analyzer.segment_customers(rfm_scored)
                
                progress_bar.progress(0.9)
                status_text.text("Storing results...")
                
                # Store in session state
                st.session_state.rfm_data = rfm_data
                st.session_state.rfm_scored = rfm_scored
                st.session_state.rfm_segmented = rfm_segmented
                
                # Save checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'customers_analyzed': len(rfm_data),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                status_text.text("✅ RFM analysis complete!")
                
                st.success(f"✅ RFM calculated for {len(rfm_data)} customers!")
                
        except Exception as e:
            st.error(f"❌ Error calculating RFM: {str(e)}")
            pm.save_checkpoint({'error': str(e)})
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Always unlock navigation
            pm.unlock()
            st.info("✅ Navigation unlocked - you can now navigate to other pages.")
    
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
                with st.status("Performing K-Means clustering...", expanded=True) as status:
                    rfm_clustered = rfm_analyzer.perform_kmeans_clustering(rfm_data, n_clusters)
                    st.session_state.rfm_clustered = rfm_clustered
                    st.success(f"✅ Created {n_clusters} customer clusters!")
        
        with col2:
            if st.button("📉 Show Elbow Curve", use_container_width=True):
                with st.status("Calculating elbow curve...", expanded=True) as status:
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
        
        # Customer Lifetime Value (CLV) Analysis
        st.divider()
        st.subheader("💰 4b. Customer Lifetime Value (CLV)")
        
        with st.expander("ℹ️ About CLV Metrics", expanded=False):
            st.markdown("""
            **Customer Lifetime Value (CLV)** predicts the total value a customer will bring over their relationship with your business.
            
            ### CLV Metrics:
            
            - **Historic CLV:** Total actual spending to date
            - **Predicted CLV:** Estimated future value (12-month projection)
            - **Average Order Value:** Mean transaction amount per customer
            - **Purchase Frequency:** Average purchases per month
            - **Customer Lifespan:** Months between first and last purchase
            
            ### Why CLV Matters:
            
            - 📊 Prioritize high-value customer segments
            - 💵 Optimize marketing spend and ROI
            - 🎯 Tailor retention strategies by segment value
            - 📈 Forecast future revenue by segment
            """)
        
        # Calculate CLV
        if st.button("💰 Calculate Customer Lifetime Value", key="calculate_clv", use_container_width=True):
            with st.status("Calculating CLV metrics...", expanded=True) as status:
                try:
                    # Get transaction data from session state
                    transactions_df = st.session_state.get('rfm_transactions', pd.DataFrame())
                    cols = st.session_state.get('rfm_columns', {})
                    
                    if transactions_df.empty:
                        st.error("❌ No transaction data available. Please process RFM analysis first.")
                        st.stop()
                    
                    st.write("📊 Calculating CLV for each customer...")
                    # Calculate CLV (call on class, not instance - it's a static method)
                    clv_data = RFMAnalyzer.calculate_clv(
                        transactions_df, 
                        cols['customer'], 
                        cols['date'], 
                        cols['amount'],
                        time_period_months=12
                    )
                    
                    st.write("🔗 Merging CLV with RFM segments...")
                    # Merge with RFM segmented data (call on class, not instance)
                    rfm_with_clv = RFMAnalyzer.merge_rfm_with_clv(
                        rfm_segmented, 
                        clv_data, 
                        cols['customer']
                    )
                    
                    # Save to session state
                    st.session_state.rfm_with_clv = rfm_with_clv
                    st.session_state.clv_data = clv_data
                    
                    status.update(label="✅ CLV calculated successfully!", state="complete", expanded=False)
                    st.success(f"✅ CLV calculated for {len(clv_data)} customers!")
                    
                except Exception as e:
                    st.error(f"❌ Error calculating CLV: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display CLV results if available
        if 'rfm_with_clv' in st.session_state and 'clv_data' in st.session_state:
            rfm_with_clv = st.session_state.rfm_with_clv
            clv_data = st.session_state.clv_data
            
            # CLV Summary Metrics
            st.write("**💰 CLV Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_historic_clv = clv_data['Historic_CLV'].sum()
                st.metric("Total Historic CLV", f"${total_historic_clv:,.2f}")
            
            with col2:
                total_predicted_clv = clv_data['Predicted_CLV'].sum()
                st.metric("Total Predicted CLV", f"${total_predicted_clv:,.2f}")
            
            with col3:
                avg_historic_clv = clv_data['Historic_CLV'].mean()
                st.metric("Avg Historic CLV", f"${avg_historic_clv:,.2f}")
            
            with col4:
                avg_predicted_clv = clv_data['Predicted_CLV'].mean()
                st.metric("Avg Predicted CLV", f"${avg_predicted_clv:,.2f}")
            
            # CLV by Segment
            st.write("**📊 CLV by Customer Segment:**")
            cols = st.session_state.get('rfm_columns', {})
            profiles_with_clv = RFMAnalyzer.get_segment_profiles_with_clv(
                rfm_with_clv, 
                cols['customer']
            )
            
            # Format currency columns for better display
            currency_cols = ['Avg_Monetary', 'Avg_Historic_CLV', 'Total_Historic_CLV', 
                           'Avg_Predicted_CLV', 'Total_Predicted_CLV']
            
            profiles_display = profiles_with_clv.copy()
            for col in currency_cols:
                if col in profiles_display.columns:
                    profiles_display[col] = profiles_display[col].apply(lambda x: f"${x:,.2f}")
            
            # Round numeric columns
            numeric_cols = ['Avg_Recency', 'Avg_Frequency', 'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']
            for col in numeric_cols:
                if col in profiles_display.columns:
                    profiles_display[col] = profiles_with_clv[col].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(profiles_display, use_container_width=True)
            
            # Top CLV Customers
            with st.expander("🏆 Top 20 Customers by CLV"):
                top_clv_customers = rfm_with_clv.nlargest(20, 'Historic_CLV')
                
                # Select relevant columns
                display_cols = [cols['customer'], 'Segment', 'Historic_CLV', 'Predicted_CLV', 
                               'Avg_Order_Value', 'Purchase_Frequency', 'Customer_Lifespan_Months']
                
                top_clv_display = top_clv_customers[display_cols].copy()
                
                # Format for display
                top_clv_display['Historic_CLV'] = top_clv_display['Historic_CLV'].apply(lambda x: f"${x:,.2f}")
                top_clv_display['Predicted_CLV'] = top_clv_display['Predicted_CLV'].apply(lambda x: f"${x:,.2f}")
                top_clv_display['Avg_Order_Value'] = top_clv_display['Avg_Order_Value'].apply(lambda x: f"${x:,.2f}")
                top_clv_display['Purchase_Frequency'] = top_clv_display['Purchase_Frequency'].apply(lambda x: f"{x:.2f}")
                top_clv_display['Customer_Lifespan_Months'] = top_clv_display['Customer_Lifespan_Months'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(top_clv_display, use_container_width=True)
                
                st.info("💡 **Insight:** These high-CLV customers should receive premium treatment and personalized retention efforts.")
        
        # 3D Visualization
        st.divider()
        st.subheader("📊 5. 3D Visualization")
        
        viz_option = st.radio(
            "Color by:",
            ["Segment", "Cluster"],
            horizontal=True,
            key="viz_option"
        )
        
        if viz_option == "Cluster":
            if 'rfm_clustered' in st.session_state:
                fig_3d = rfm_analyzer.create_rfm_scatter_3d(st.session_state.rfm_clustered, color_col='Cluster')
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("⚠️ Please run K-Means Clustering first (Section 3 above) to view cluster-based visualization.")
                st.info("💡 Click the '🔄 Run K-Means Clustering' button in Section 3, then return here to view by cluster.")
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
        
        # AI Insights
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'rfm_ai_insights' in st.session_state:
            st.markdown(st.session_state.rfm_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="rfm_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("Analyzing customer segments and generating strategic insights...", expanded=True) as status:
                    # Get data from session state
                    rfm_segmented_data = st.session_state.get('rfm_segmented', pd.DataFrame())
                    rfm_data_state = st.session_state.get('rfm_data', pd.DataFrame())
                    transactions_df_data = st.session_state.get('rfm_transactions', pd.DataFrame())
                    cols_data = st.session_state.get('rfm_columns', {})
                    rfm_analyzer_data = st.session_state.get('rfm_analyzer')
                    
                    if rfm_segmented_data.empty or rfm_data_state.empty or transactions_df_data.empty:
                        st.error("No RFM analysis results available. Please run RFM Analysis first.")
                        st.stop()
                    
                    # Calculate metrics
                    total_revenue_calc = transactions_df_data[cols_data['amount']].sum()
                    avg_transaction_calc = transactions_df_data[cols_data['amount']].mean()
                    
                    # Get profiles
                    profiles_data = rfm_analyzer_data.get_segment_profiles(rfm_segmented_data, cols_data['customer'])
                    
                    # Prepare segment summary
                    segment_summary = ""
                    for idx, row in profiles_data.iterrows():
                        segment_summary += f"\n- **{row['Segment']}**: {row['Customer_Count']} customers, "
                        segment_summary += f"Avg Recency: {row['Avg_Recency']:.1f} days, "
                        segment_summary += f"Avg Frequency: {row['Avg_Frequency']:.1f}, "
                        segment_summary += f"Avg Monetary: ${row['Avg_Monetary']:.2f}"
                    
                    # Prepare context
                    context = f"""
                    RFM Analysis Results:
                    
                    Dataset Overview:
                    - Total Transactions: {len(transactions_df_data):,}
                    - Unique Customers: {transactions_df_data[cols_data['customer']].nunique():,}
                    - Total Revenue: ${total_revenue_calc:,.2f}
                    - Average Transaction: ${avg_transaction_calc:.2f}
                    
                    RFM Metrics:
                    - Average Recency: {rfm_data_state['Recency'].mean():.1f} days
                    - Average Frequency: {rfm_data_state['Frequency'].mean():.1f} transactions
                    - Average Monetary: ${rfm_data_state['Monetary'].mean():.2f}
                    
                    Customer Segments:
                    {segment_summary}
                    """
                    
                    prompt = f"""
                    As a customer relationship management (CRM) expert, analyze these RFM segmentation results and provide:
                    
                    1. **Strategic Overview** (3-4 sentences): What does this customer distribution tell us about the business health and growth stage?
                    
                    2. **Segment-Specific Strategies** (5-6 bullet points): For the TOP segments by customer count, provide:
                       - Specific marketing campaigns to run
                       - Communication frequency and channels
                       - Offer types and discount levels
                       - Retention or acquisition tactics
                    
                    3. **Revenue Optimization** (3-4 bullet points): How to maximize customer lifetime value:
                       - Which segments to prioritize for growth
                       - Cross-sell and upsell opportunities
                       - Budget allocation recommendations
                    
                    4. **Churn Prevention** (3-4 bullet points): Strategies to prevent customer loss in at-risk segments
                    
                    5. **Quick Wins** (3-4 bullet points): Immediate actions to implement this week that will drive results
                    
                    6. **Expected Impact** (2-3 sentences): Realistic business outcomes from implementing these strategies
                    
                    {context}
                    
                    Be specific, actionable, and focus on ROI. Consider typical e-commerce/retail business models.
                    """
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a CRM and customer analytics expert specializing in RFM segmentation and customer lifecycle marketing."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.rfm_ai_insights = response.choices[0].message.content
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.rfm_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
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
"""
                
                # Add AI insights if available
                if 'rfm_ai_insights' in st.session_state:
                    report += f"""

## 🤖 AI-Powered Strategic Insights

{st.session_state.rfm_ai_insights}

"""
                
                report += """
---
*Report generated by DataInsights - RFM Analysis Module*
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
    st.markdown("<h2 style='text-align: center;'>📈 Monte Carlo Simulation</h2>", unsafe_allow_html=True)
    
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
        with st.status(f"Fetching {ticker} data...", expanded=True) as status:
            try:
                st.write("📡 Fetching stock data and company information...")
                start_date = datetime.now() - timedelta(days=lookback_days)
                
                # Fetch both stock data and company name in single API call
                stock_data, company_name = simulator.fetch_stock_data(ticker, start_date)
                
                st.session_state.mc_stock_data = stock_data
                st.session_state.mc_ticker = ticker
                st.session_state.mc_company_name = company_name
                
                # Calculate returns
                st.write("📊 Calculating statistics...")
                returns = simulator.calculate_returns(stock_data['Close'])
                st.session_state.mc_returns = returns
                
                # Calculate statistics
                stats = simulator.get_statistics(returns)
                st.session_state.mc_stats = stats
                
                status.update(label=f"✅ Loaded {len(stock_data)} days of data!", state="complete", expanded=False)
                if company_name:
                    st.success(f"✅ Loaded {len(stock_data)} days of {ticker} data ({company_name})!")
                else:
                    st.success(f"✅ Loaded {len(stock_data)} days of {ticker} data!")
                    st.info("ℹ️ Company name not available from data provider")
                
            except Exception as e:
                status.update(label="❌ Error fetching data", state="error", expanded=True)
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
    
    # Display ticker and company name
    if 'mc_company_name' in st.session_state and st.session_state.mc_company_name:
        st.caption(f"📌 **{ticker}** = {st.session_state.mc_company_name}")
    else:
        st.caption(f"ℹ️ Ticker: {ticker}")
    
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
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Monte_Carlo_Simulation")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during simulation.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status(f"Running {num_simulations} simulations...", expanded=True) as status:
                start_price = stock_data['Close'].iloc[-1]
                
                # Progress tracking
                st.write(f"🎲 Running {num_simulations} simulations for {forecast_days} days...")
                progress_bar = st.progress(0)
                
                progress_bar.progress(0.3)
                
                # Run simulation
                simulations = simulator.run_simulation(
                    start_price=start_price,
                    mean_return=stats['mean'],
                    std_return=stats['std'],
                    days=forecast_days,
                    num_simulations=num_simulations
                )
                
                progress_bar.progress(0.6)
                st.write("📊 Calculating confidence intervals...")
                
                # Calculate confidence intervals
                intervals_dict = simulator.calculate_confidence_intervals(
                    simulations,
                    [level / 100 for level in confidence_levels]
                )
                
                progress_bar.progress(0.8)
                st.write("⚠️ Calculating risk metrics...")
                
                # Calculate risk metrics
                final_prices = simulations[:, -1]
                risk_metrics = simulator.get_risk_metrics(final_prices, start_price)
                
                progress_bar.progress(0.9)
                st.write("💾 Storing results...")
                
                # Store in session state
                st.session_state.mc_simulations = simulations
                st.session_state.mc_intervals = intervals_dict
                st.session_state.mc_risk_metrics = risk_metrics
                st.session_state.mc_forecast_days = forecast_days
                
                # Save checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'num_simulations': num_simulations,
                    'forecast_days': forecast_days,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                
                status.update(label="✅ Simulation complete!", state="complete", expanded=False)
                st.success(f"✅ Completed {num_simulations} simulations for {forecast_days} days!")
                
        except Exception as e:
            st.error(f"❌ Error running simulation: {str(e)}")
            # Save error checkpoint
            pm.save_checkpoint({
                'error': str(e),
                'num_simulations': num_simulations,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Always unlock navigation
            pm.unlock()
            st.info("✅ Navigation unlocked - you can now navigate to other pages.")
    
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
        
        # AI-Powered Analysis
        st.divider()
        st.subheader("✨ AI-Powered Investment Analysis")
        
        # Display saved insights if they exist
        if 'mc_ai_insights' in st.session_state:
            st.markdown(st.session_state.mc_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Analysis", key="mc_ai_analysis_btn", type="primary", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("Analyzing Monte Carlo simulation results with AI...", expanded=True) as status:
                    # Get data from session state
                    mc_ticker_data = st.session_state.get('mc_ticker', 'Unknown')
                    mc_stock_data = st.session_state.get('mc_stock_data', pd.DataFrame())
                    mc_stats_data = st.session_state.get('mc_stats', {})
                    mc_simulations_data = st.session_state.get('mc_simulations', [])
                    mc_risk_metrics_data = st.session_state.get('mc_risk_metrics', {})
                    mc_forecast_days_data = st.session_state.get('mc_forecast_days', 0)
                    
                    if mc_stock_data.empty or not mc_risk_metrics_data or len(mc_simulations_data) == 0:
                        st.error("No Monte Carlo simulation results available. Please run simulation first.")
                        st.stop()
                    
                    num_simulations_calc = len(mc_simulations_data)
                    
                    # Prepare context
                    context = f"""
                    Monte Carlo Simulation Analysis for {mc_ticker_data}:
                    
                    Simulation Parameters:
                    - Forecast Period: {mc_forecast_days_data} days
                    - Number of Simulations: {num_simulations_calc}
                    - Starting Price: ${mc_stock_data['Close'].iloc[-1]:.2f}
                    
                    Historical Statistics:
                    - Mean Daily Return: {mc_stats_data.get('mean', 0)*100:.3f}%
                    - Volatility (Std Dev): {mc_stats_data.get('std', 0)*100:.3f}%
                    
                    Key Risk Metrics:
                    - Expected Return: {mc_risk_metrics_data.get('expected_return', 0):.2f}%
                    - Expected Price: ${mc_risk_metrics_data.get('expected_price', 0):.2f}
                    - Value at Risk (95%): {mc_risk_metrics_data.get('var_95', 0):.2f}%
                    - Conditional VaR (95%): {mc_risk_metrics_data.get('cvar_95', 0):.2f}%
                    - Probability of Profit: {mc_risk_metrics_data.get('probability_profit', 0):.1f}%
                    - Price Range: ${mc_risk_metrics_data.get('min_price', 0):.2f} - ${mc_risk_metrics_data.get('max_price', 0):.2f}
                    """
                    
                    prompt = f"""
                    As a senior financial advisor, analyze these Monte Carlo simulation results and provide:
                    
                    1. **Investment Recommendation** (2-3 sentences): Should an investor consider this stock? Why or why not based on the risk/reward profile?
                    
                    2. **Risk Assessment** (2-3 sentences): How risky is this investment? Interpret the VaR and volatility in practical terms.
                    
                    3. **Optimal Strategy** (3-4 bullet points): What investment strategy would work best? Consider:
                       - Position sizing
                       - Entry/exit points
                       - Risk management tactics
                       - Time horizon considerations
                    
                    4. **Key Concerns** (2-3 bullet points): What should investors watch out for? Red flags or limitations?
                    
                    5. **Market Context** (2-3 sentences): How do these metrics compare to typical market expectations? Is the expected return reasonable given the risk?
                    
                    {context}
                    
                    Be specific, actionable, and investor-focused. Use clear language that a non-expert can understand.
                    """
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior financial advisor providing actionable investment insights based on quantitative analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.mc_ai_insights = response.choices[0].message.content
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.mc_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI analysis: {str(e)}")
                if "openai" in str(e).lower() or "api" in str(e).lower():
                    st.info("💡 Make sure your OpenAI API key is configured in the secrets.")
        
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
"""
            
            # Add AI insights if available
            if 'mc_ai_insights' in st.session_state:
                report += f"""

## 🤖 AI-Powered Investment Analysis

{st.session_state.mc_ai_insights}

"""
            
            report += """
---
*Report generated by DataInsights - Monte Carlo Simulation Module*
"""
            st.download_button(
                label="📥 Download Full Report (Markdown)",
                data=report,
                file_name=f"mc_report_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_ml_classification():
    """Comprehensive ML Classification with 15 models and full evaluation."""
    
    # Import ML helper functions for optimization
    from utils.ml_helpers import get_recommended_cv_folds, create_data_hash, cached_classification_training
    
    st.markdown("<h2 style='text-align: center;'>🤖 Machine Learning - Classification Models</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Machine Learning Classification?"):
        st.markdown("""
        **Classification** is a supervised machine learning technique that predicts categorical outcomes.
        
        ### This Module Features:
        - **15 Classification Algorithms** - From linear models to advanced boosting
        - **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, ROC-AUC, Cross-Validation
        - **Model Comparison** - Interactive visualizations to compare all models
        - **Feature Importance** - Understand what drives predictions
        - **AI Insights** - Get intelligent recommendations from OpenAI
        
        ### Common Business Applications:
        - **Lead Scoring:** Predict which leads are most likely to convert
        - **Churn Prediction:** Identify customers at risk of leaving
        - **Credit Risk:** Assess loan default probability
        - **Fraud Detection:** Flag suspicious transactions
        
        ### How It Works:
        1. **Upload Data** - Your dataset with features and target column
        2. **Select Models** - Choose from 15 algorithms or train all
        3. **Train** - System trains and evaluates all selected models
        4. **Compare** - View comprehensive metrics and visualizations
        5. **Deploy** - Export best model and make predictions
        """)
    
    # Data selection
    st.divider()
    st.subheader("📤 1. Select Data Source")
    
    # Check if data is already uploaded in session
    if st.session_state.data is not None:
        data_source = st.radio(
            "Choose data source:",
            ["Use uploaded data from Data Upload page", "Sample Iris Dataset", "Upload new file for this analysis"],
            help="You can use the data you already uploaded, try a sample dataset, or upload a new file"
        )
        
        if data_source == "Use uploaded data from Data Upload page":
            from utils.ai_smart_detection import get_ai_recommendation
            
            df = st.session_state.data
            st.session_state.ml_data = df
            
            # AI Analysis if not already done (ML-specific analysis)
            if 'ml_ai_recommendations' not in st.session_state or st.session_state.get('ml_ai_data_hash') != hash(str(df.columns.tolist())):
                with st.spinner("🤖 AI analyzing your dataset for ML Classification..."):
                    ai_recommendations = get_ai_recommendation(df, task_type='classification')
                    st.session_state.ml_ai_recommendations = ai_recommendations
                    st.session_state.ml_ai_data_hash = hash(str(df.columns.tolist()))
            
            st.success(f"✅ Using uploaded data: {len(df):,} rows and {len(df.columns)} columns")
            
            # Show AI quick summary
            if 'ml_ai_recommendations' in st.session_state:
                ai_recs = st.session_state.ml_ai_recommendations
                st.info(f"""
                **🤖 AI Analysis Complete:**
                - 🎯 **Recommended Target:** {ai_recs['target_column']}
                - 📊 **Data Quality:** {ai_recs.get('data_quality', 'Good')}
                - 💡 **Confidence:** {ai_recs.get('confidence', 'High')}
                """)
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
                
        elif data_source == "Sample Iris Dataset":
            if st.button("📥 Load Iris Dataset", type="primary"):
                with st.status("Loading Iris dataset...", expanded=True) as status:
                    try:
                        from sklearn.datasets import load_iris
                        st.write("📊 Loading classic Iris dataset...")
                        iris = load_iris()
                        
                        # Create DataFrame
                        df = pd.DataFrame(iris.data, columns=iris.feature_names)
                        df['species'] = iris.target
                        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                        
                        st.session_state.ml_data = df
                        st.success(f"✅ Loaded Iris dataset: {len(df):,} rows and {len(df.columns)} columns")
                        
                        st.info("""
                        **About this dataset:**
                        - 🎯 **Target:** species (3 iris flower types: setosa, versicolor, virginica)
                        - 📊 **Features:** 4 measurements (sepal length/width, petal length/width)
                        - ✅ **Perfect for ML:** 150 samples, perfectly balanced (50-50-50)
                        - 🌸 **Classic:** Most famous ML classification benchmark dataset
                        - ⚡ **Fast:** Trains in seconds, ideal for testing
                        """)
                        
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
                        
        else:  # Upload new file
            uploaded_file = st.file_uploader(
                "Upload CSV file with features and target column",
                type=['csv'],
                key="ml_upload",
                help="Must include predictor features and target column"
            )
            
            if uploaded_file is not None:
                try:
                    from utils.ai_smart_detection import get_ai_recommendation
                    
                    df = pd.read_csv(uploaded_file)
                    st.session_state.ml_data = df
                    
                    # AI Analysis
                    with st.spinner("🤖 AI analyzing your dataset..."):
                        ai_recommendations = get_ai_recommendation(df, task_type='classification')
                        st.session_state.ml_ai_recommendations = ai_recommendations
                        st.session_state.ml_ai_data_hash = hash(str(df.columns.tolist()))
                    
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    
                    # Show AI quick summary
                    st.info(f"""
                    **🤖 AI Analysis Complete:**
                    - 🎯 **Recommended Target:** {ai_recommendations['target_column']}
                    - 📊 **Data Quality:** {ai_recommendations.get('data_quality', 'Good')}
                    - 💡 **Confidence:** {ai_recommendations.get('confidence', 'High')}
                    """)
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            else:
                st.info("👆 Please upload a CSV file to continue")
    else:
        data_source = st.radio(
            "Choose data source:",
            ["Sample Iris Dataset", "Upload new file for this analysis"],
            help="Try the sample dataset or upload your own"
        )
        
        if data_source == "Sample Iris Dataset":
            if st.button("📥 Load Iris Dataset", type="primary"):
                with st.status("Loading Iris dataset...", expanded=True) as status:
                    try:
                        from sklearn.datasets import load_iris
                        from utils.ai_smart_detection import get_ai_recommendation
                        
                        st.write("📊 Loading classic Iris dataset...")
                        iris = load_iris()
                        
                        # Create DataFrame
                        df = pd.DataFrame(iris.data, columns=iris.feature_names)
                        df['species'] = iris.target
                        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                        
                        st.session_state.ml_data = df
                        
                        # AI Analysis
                        st.write("🤖 AI analyzing dataset...")
                        ai_recommendations = get_ai_recommendation(df, task_type='classification')
                        st.session_state.ml_ai_recommendations = ai_recommendations
                        
                        status.update(label="✅ Dataset loaded and analyzed!", state="complete")
                        
                        st.success(f"✅ Loaded Iris dataset: {len(df):,} rows and {len(df.columns)} columns")
                        
                        # Show AI quick summary
                        st.info(f"""
                        **🤖 AI Analysis Complete:**
                        - 🎯 **Recommended Target:** {ai_recommendations['target_column']}
                        - 📊 **Data Quality:** {ai_recommendations.get('data_quality', 'Good')}
                        - 💡 **Confidence:** {ai_recommendations.get('confidence', 'High')}
                        """)
                        
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
        else:  # Upload custom data
            uploaded_file = st.file_uploader(
                "Upload CSV file with features and target column",
                type=['csv'],
                key="ml_upload_no_data",
                help="Must include predictor features and target column"
            )
            
            if uploaded_file is not None:
                try:
                    from utils.ai_smart_detection import get_ai_recommendation
                    
                    df = pd.read_csv(uploaded_file)
                    st.session_state.ml_data = df
                    
                    # AI Analysis
                    with st.spinner("🤖 AI analyzing your dataset..."):
                        ai_recommendations = get_ai_recommendation(df, task_type='classification')
                        st.session_state.ml_ai_recommendations = ai_recommendations
                        st.session_state.ml_ai_data_hash = hash(str(df.columns.tolist()))
                    
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    
                    # Show AI quick summary
                    st.info(f"""
                    **🤖 AI Analysis Complete:**
                    - 🎯 **Recommended Target:** {ai_recommendations['target_column']}
                    - 📊 **Data Quality:** {ai_recommendations.get('data_quality', 'Good')}
                    - 💡 **Confidence:** {ai_recommendations.get('confidence', 'High')}
                    """)
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    # Configuration and training
    if 'ml_data' in st.session_state:
        df = st.session_state.ml_data
        
        # Check if previous quality check failed and show prominent warning
        if 'ml_training_suitable' in st.session_state and not st.session_state.ml_training_suitable:
            st.error("""
            ### 🚫 DATA NOT SUITABLE FOR CLASSIFICATION
            
            Your dataset has quality issues that prevent ML training. Please review the issues below and fix your data.
            """)
            
            quality_issues = st.session_state.get('ml_quality_issues', [])
            recommendations = st.session_state.get('ml_recommendations', [])
            
            if quality_issues:
                st.write("**Issues Found:**")
                for issue in quality_issues:
                    st.write(f"• {issue}")
            
            if recommendations:
                st.info("**💡 How to Fix:**")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            st.divider()
        
        st.divider()
        st.subheader("🎯 2. Configure Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # AI-Powered Smart Detection (from session state)
            from utils.ai_smart_detection import AISmartDetection
            
            # Use stored AI recommendations or run analysis if not available
            if 'ml_ai_recommendations' in st.session_state:
                ai_recommendations = st.session_state.ml_ai_recommendations
            else:
                # Fallback: run analysis now (shouldn't normally happen)
                from utils.ai_smart_detection import get_ai_recommendation
                with st.spinner("🤖 AI analyzing your dataset..."):
                    ai_recommendations = get_ai_recommendation(df, task_type='classification')
                    st.session_state.ml_ai_recommendations = ai_recommendations
            
            # Display AI recommendations
            AISmartDetection.display_ai_recommendation(ai_recommendations, expanded=True)
            
            # Performance Risk Handling
            performance_risk = ai_recommendations.get('performance_risk', 'Low')
            if performance_risk == 'High':
                st.error("🚨 **High Performance Risk Detected!** This dataset may cause timeouts or crashes on Streamlit Cloud.")
                st.warning("⚡ **Recommendation:** Consider reducing dataset size, excluding high-cardinality columns, or using fewer CV folds.")
            elif performance_risk == 'Medium':
                st.warning("⚠️ **Medium Performance Risk:** This dataset may be slow to process. Monitor for timeouts.")
            
            # AI-Recommended Column Dropping with User Approval
            if ai_recommendations.get('features_to_exclude'):
                st.divider()
                st.subheader("🚫 2a. AI-Recommended Column Exclusions")
                
                st.info("🤖 **AI has identified columns that may hurt model performance.** Review and approve exclusions below:")
                
                excluded_columns = []
                for feature_info in ai_recommendations['features_to_exclude']:
                    if isinstance(feature_info, dict):
                        col_name = feature_info['column']
                        reason = feature_info['reason']
                    else:
                        # Backward compatibility
                        col_name = feature_info
                        reason = "AI recommends excluding this column"
                    
                    if col_name in df.columns:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            exclude = st.checkbox(
                                f"Exclude **{col_name}**",
                                value=False,
                                key=f"exclude_{col_name}",
                                help=f"AI Reason: {reason}"
                            )
                        with col2:
                            st.write(f"**Reason:** {reason}")
                        
                        if exclude:
                            excluded_columns.append(col_name)
                
                # Apply exclusions to dataframe
                if excluded_columns:
                    df_filtered = df.drop(columns=excluded_columns)
                    st.success(f"✅ Excluded {len(excluded_columns)} columns: {', '.join(excluded_columns)}")
                    st.info(f"📊 **Dataset updated:** {len(df_filtered)} rows × {len(df_filtered.columns)} columns (was {len(df.columns)})")
                    df = df_filtered  # Use filtered dataframe for the rest of the analysis
                    st.session_state.ml_data = df  # Update session state
            
            # Use AI-recommended target as default
            suggested_target = ai_recommendations['target_column']
            target_index = list(df.columns).index(suggested_target) if suggested_target in df.columns else 0
            
            target_col = st.selectbox(
                "Select Target Column (what to predict)",
                df.columns,
                index=target_index,
                help="Column containing the categories/classes to predict. AI has analyzed your data and recommended the best option above."
            )
            
            # Show class distribution and data quality check
            if target_col:
                class_counts = df[target_col].value_counts()
                min_class_size = class_counts.min()
                max_class_size = class_counts.max()
                n_classes = len(class_counts)
                total_samples = len(df)
                
                # Calculate class imbalance ratio
                imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
                
                # Simulate stratified sampling to see what happens
                max_samples_for_ml = 10000  # Same as MLTrainer
                if total_samples > max_samples_for_ml:
                    # Calculate what min class would become after sampling
                    sampling_ratio = max_samples_for_ml / total_samples
                    min_class_after_sampling = int(min_class_size * sampling_ratio)
                else:
                    min_class_after_sampling = min_class_size
                
                # Real-time data quality assessment
                issues = []
                warnings = []
                recommendations = []
                
                # Check 1: Minimum samples - ONLY BLOCK if class has 0-1 samples
                if min_class_size < 2:
                    bad_classes_count = (class_counts < 2).sum()
                    issues.append(f"❌ {bad_classes_count} class(es) with < 2 samples (CRITICAL - will cause train/test split to fail)")
                    recommendations.append("Remove or combine classes with < 2 samples using Data Cleaning")
                
                # Check 2: Class imbalance - WARN but don't block
                if imbalance_ratio > 100:
                    warnings.append(f"⚠️ Severe class imbalance: {imbalance_ratio:.0f}:1 ratio (largest:{max_class_size}, smallest:{min_class_size})")
                    recommendations.append("Consider: Filter to top 10-15 classes, or balance with SMOTE/undersampling")
                elif imbalance_ratio > 50:
                    warnings.append(f"⚠️ High class imbalance: {imbalance_ratio:.0f}:1 ratio")
                    recommendations.append("Imbalanced data may reduce minority class accuracy")
                
                # Check 3: After-sampling - BLOCK if classes will have <2 samples (CRITICAL)
                if total_samples > max_samples_for_ml and min_class_after_sampling < 2:
                    issues.append(f"❌ After sampling to 10K, smallest class will have ~{min_class_after_sampling} samples (CRITICAL - will cause stratified split to fail)")
                    recommendations.append(f"Filter out rare classes OR create binary/grouped target (e.g., 'Top 10 Countries' vs 'Other')")
                elif total_samples > max_samples_for_ml and min_class_after_sampling < 5:
                    warnings.append(f"⚠️ After sampling to 10K: smallest class will have ~{min_class_after_sampling} samples (may affect model performance)")
                    recommendations.append(f"Consider filtering rare classes for better model performance")
                
                # Check 4: Too many classes - WARN only
                if n_classes > 50:
                    warnings.append(f"⚠️ {n_classes} classes - training will be slower, accuracy may be lower")
                    recommendations.append("Tip: Group similar classes or filter to top 10-20 classes")
                
                # Check 5: Dataset size - WARN only
                if total_samples < 30:
                    warnings.append(f"⚠️ Small dataset ({total_samples} samples) - results may vary. Recommend 100+")
                elif min_class_size >= 2 and min_class_size < 5:
                    warnings.append(f"⚠️ Smallest class has only {min_class_size} samples - may affect accuracy")
                
                # Check 6: Transactional data - WARN only
                if n_classes > 30 and imbalance_ratio > 100:
                    warnings.append("⚠️ Data pattern suggests transactional/high-cardinality data")
                    recommendations.append("Tip: Create derived targets like 'High Value', 'Frequent Buyer', etc.")
                
                # LEVEL 1: Data Source Compatibility - Only block on CRITICAL issues
                data_compatible = len(issues) == 0  # True if no critical issues
                
                st.session_state.ml_data_compatible = data_compatible
                st.session_state.ml_quality_issues = issues
                st.session_state.ml_quality_warnings = warnings
                st.session_state.ml_recommendations = recommendations
                
                # Display quality indicator
                if len(issues) > 0:
                    st.error("**🚨 Data Quality: NOT SUITABLE FOR TRAINING**")
                    for issue in issues:
                        st.write(issue)
                    
                    if len(recommendations) > 0:
                        st.info("**💡 Recommendations:**")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                            
                elif len(warnings) > 0:
                    st.warning("**⚠️ Data Quality: TRAINING POSSIBLE (with warnings)**")
                    for warning in warnings:
                        st.write(warning)
                    
                    if len(recommendations) > 0:
                        with st.expander("💡 Recommendations"):
                            for rec in recommendations:
                                st.write(f"• {rec}")
                else:
                    st.success("**✅ Data Quality: EXCELLENT FOR TRAINING**")
                    st.write(f"✓ {n_classes} classes, **{min_class_size} min samples/class**")
                
                # Always show actual min class size for debugging
                st.caption(f"🔍 Debug: Target='{target_col}', Min Class Size={min_class_size}, Total Classes={n_classes}")
                
                st.write("**Class Distribution:**")
                fig_dist = px.bar(
                    x=class_counts.index.astype(str),
                    y=class_counts.values,
                    labels={'x': target_col, 'y': 'Count'},
                    title=f'Distribution of {target_col}'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Model selection
            train_all = st.checkbox("Train All Models (15 algorithms)", value=True,
                                   help="Train all 15 models or select specific ones")
            
            if not train_all:
                from utils.ml_training import MLTrainer
                temp_trainer = MLTrainer(df, target_col)
                all_model_names = list(temp_trainer.get_all_models().keys())
                
                selected_models = st.multiselect(
                    "Select Models to Train",
                    all_model_names,
                    default=all_model_names[:5],
                    help="Choose which models to train"
                )
            else:
                selected_models = None
            
            # LEVEL 2: Model Availability Checker - Per-model compatibility
            with st.expander("📋 Model Availability Checker", expanded=False):
                st.write("**Per-model compatibility check:**")
                
                # Get data compatibility from Level 1
                data_compatible = st.session_state.get('ml_data_compatible', True)
                quality_issues = st.session_state.get('ml_quality_issues', [])
                
                # Show data-level warning if incompatible
                if not data_compatible:
                    st.warning("⚠️ **Data has critical issues** - Models may fail")
                    for issue in quality_issues:
                        st.write(f"• {issue}")
                    st.divider()
                
                # Get all models
                from utils.ml_training import MLTrainer
                temp_trainer = MLTrainer(df, target_col if target_col else df.columns[0])
                all_models = temp_trainer.get_all_models()
                
                # Check each model individually
                model_status = []
                n_samples = len(df)
                n_features = len(df.columns) - 1
                
                for model_name in all_models.keys():
                    available = True
                    reason = "✅ Ready"
                    
                    # Check 1: Library availability
                    if model_name == "XGBoost":
                        try:
                            import xgboost
                        except ImportError:
                            available = False
                            reason = "❌ XGBoost not installed"
                    elif model_name == "LightGBM":
                        try:
                            import lightgbm
                        except ImportError:
                            available = False
                            reason = "❌ LightGBM not installed"
                    elif model_name == "CatBoost":
                        try:
                            import catboost
                        except ImportError:
                            available = False
                            reason = "❌ CatBoost not installed"
                    
                    # Check 2: Data-level issues (critical ones block all models)
                    if not data_compatible and available:
                        available = False
                        reason = "❌ Data incompatible"
                    
                    # Check 3: Model-specific requirements
                    if available:
                        if n_samples < 10:
                            available = False
                            reason = f"❌ Need ≥10 samples (have {n_samples})"
                        elif n_samples < 30:
                            reason = f"⚠️ Small dataset ({n_samples} samples)"
                        elif n_samples < 20 and model_name in ["Stacking", "Voting"]:
                            reason = f"⚠️ Ensemble needs more samples"
                    
                    model_status.append({
                        'Model': model_name,
                        'Status': '✅ Available' if available else '❌ Unavailable',
                        'Notes': reason if not available else '✅ Ready'
                    })
                
                # Display as table
                status_df = pd.DataFrame(model_status)
                
                # Color code the display
                def color_status(row):
                    if '❌' in row['Status']:
                        return ['background-color: #ffebee'] * len(row)
                    elif '⚠️' in row['Notes']:
                        return ['background-color: #fff8e1'] * len(row)
                    else:
                        return ['background-color: #f1f8e9'] * len(row)
                
                styled_status = status_df.style.apply(color_status, axis=1)
                st.dataframe(styled_status, use_container_width=True, height=400)
                
                # Summary and store available models
                available_models = [m['Model'] for m in model_status if '✅' in m['Status']]
                available_count = len(available_models)
                unavailable_count = len(model_status) - available_count
                
                # Store for Level 3
                st.session_state.ml_available_models = available_models
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Available Models", available_count, delta="Ready to train" if available_count > 0 else "None available")
                with col_b:
                    if unavailable_count > 0:
                        st.metric("Unavailable Models", unavailable_count, delta="Check notes", delta_color="inverse")
                    else:
                        st.metric("Unavailable Models", 0, delta="All ready!")
            
            # Training config
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data reserved for testing"
            )
            
            # Smart CV fold recommendation
            n_samples = len(df)
            n_classes = df[target_col].nunique()
            recommended_folds, cv_reason = get_recommended_cv_folds(n_samples, n_classes)
            
            # Also check if AI has CV fold recommendation
            ai_cv_folds = ai_recommendations.get('recommended_cv_folds', recommended_folds)
            
            # Performance-aware CV fold adjustment
            performance_risk = ai_recommendations.get('performance_risk', 'Low')
            if performance_risk == 'High':
                # Force lower CV folds for high-risk datasets
                final_recommended = min(3, ai_cv_folds, recommended_folds)
                max_cv_folds = 3
                performance_note = " (Limited to 3 folds due to high performance risk)"
            elif performance_risk == 'Medium':
                final_recommended = min(5, ai_cv_folds, recommended_folds)
                max_cv_folds = 5
                performance_note = " (Limited to 5 folds due to medium performance risk)"
            else:
                final_recommended = max(recommended_folds, ai_cv_folds)
                max_cv_folds = 10
                performance_note = ""
            
            st.info(f"💡 **AI Recommended:** {ai_cv_folds}-fold CV | **Rule-Based:** {recommended_folds}-fold CV - {cv_reason}{performance_note}")
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=max_cv_folds,
                value=final_recommended,  # Use performance-aware recommendation
                help=f"AI recommends {ai_cv_folds} folds, rule-based recommends {recommended_folds} folds for your dataset ({n_samples:,} samples, {n_classes} classes). Performance risk: {performance_risk}"
            )
        
        # Class Balancing Section - AI Enhanced
        st.divider()
        st.subheader("⚖️ 2b. Balance Classes (Optional)")
        
        from utils.class_balancing import ClassBalancer
        
        # Analyze imbalance
        imbalance_info = ClassBalancer.analyze_imbalance(df, target_col)
        
        # Check if AI recommends SMOTE
        ai_recommends_smote = ai_recommendations.get('recommend_smote', False)
        ai_detected_imbalance = ai_recommendations.get('class_imbalance_detected', False)
        
        # Show if either rule-based OR AI detects imbalance
        show_balancing = imbalance_info['imbalance_ratio'] > 3 or ai_detected_imbalance
        
        if show_balancing:
            # Show combined recommendations
            preset = ClassBalancer.get_smart_preset(imbalance_info['imbalance_ratio'])
            
            # Enhanced recommendation combining AI and rule-based analysis
            if ai_recommends_smote and imbalance_info['imbalance_ratio'] > 3:
                recommendation_text = f"""
                🤖 **AI + Rule-Based Analysis Agree:** Both recommend class balancing  
                **AI Severity:** {ai_recommendations.get('imbalance_severity', 'Unknown')}  
                **Rule-Based Ratio:** {imbalance_info['imbalance_ratio']:.1f}:1  
                **AI Reasoning:** {ai_recommendations.get('smote_reasoning', 'Class imbalance detected')}
                """
                expanded_default = True  # Auto-expand when both agree
            elif ai_recommends_smote:
                recommendation_text = f"""
                🤖 **AI Recommends SMOTE:** {ai_recommendations.get('smote_reasoning', 'Class imbalance detected')}  
                **AI Severity:** {ai_recommendations.get('imbalance_severity', 'Unknown')}  
                **Rule-Based Ratio:** {imbalance_info['imbalance_ratio']:.1f}:1 (below threshold)
                """
                expanded_default = True  # Auto-expand when AI recommends
            else:
                recommendation_text = f"""
                💡 **Rule-Based Recommendation:** {preset['method']}  
                **Severity:** {imbalance_info['severity']} imbalance ({imbalance_info['imbalance_ratio']:.1f}:1 ratio)  
                **Description:** {preset['description']}
                """
                expanded_default = False
            
            st.info(recommendation_text)
            
            with st.expander("⚖️ Configure Class Balancing", expanded=expanded_default):
                # Auto-enable balancing when AI strongly recommends it
                default_balancing = ai_recommends_smote and ai_recommendations.get('imbalance_severity') in ['Moderate', 'Severe']
                
                apply_balancing = st.checkbox(
                    "Apply class balancing before training",
                    value=default_balancing,
                    help="✨ Auto-enabled because AI detected class imbalance" if default_balancing else "Recommended for imbalanced datasets to improve minority class performance"
                )
                
                if apply_balancing:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prefer SMOTE when AI recommends it, otherwise use preset
                        if ai_recommends_smote:
                            default_method_index = 0  # SMOTE is first in list
                            help_text = "🤖 SMOTE recommended by AI - creates synthetic samples for minority classes"
                        else:
                            default_method_index = ["SMOTE", "Random Undersampling", "SMOTE + Tomek Links"].index(preset['method']) if preset['method'] in ["SMOTE", "Random Undersampling", "SMOTE + Tomek Links"] else 0
                            help_text = "SMOTE creates synthetic samples, Undersampling removes majority samples"
                        
                        balance_method = st.selectbox(
                            "Balancing Method:",
                            ["SMOTE", "Random Undersampling", "SMOTE + Tomek Links"],
                            index=default_method_index,
                            help=help_text
                        )
                    
                    with col2:
                        if balance_method == "Random Undersampling":
                            default_strategy = 0.7
                        else:
                            default_strategy = preset.get('sampling_strategy', 0.5)
                        
                        sampling_strategy = st.slider(
                            "Target Balance Ratio:",
                            min_value=0.3,
                            max_value=1.0,
                            value=float(default_strategy) if isinstance(default_strategy, (int, float)) else 0.5,
                            step=0.1,
                            help="1.0 = fully balanced, 0.5 = minority becomes 50% of majority size"
                        )
                    
                    # K-neighbors for SMOTE
                    if balance_method in ["SMOTE", "SMOTE + Tomek Links"]:
                        k_neighbors = st.slider(
                            "K-Neighbors (SMOTE):",
                            min_value=1,
                            max_value=min(10, imbalance_info['min_samples'] - 1),
                            value=min(5, imbalance_info['min_samples'] - 1),
                            help="Number of nearest neighbors to use for synthetic sample generation"
                        )
                    else:
                        k_neighbors = 5
                    
                    # Preview button
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Distribution:**")
                        current_dist = imbalance_info['class_counts'].head(10)
                        for idx, (cls, count) in enumerate(current_dist.items()):
                            pct = (count / len(df) * 100)
                            st.write(f"{cls}: {count:,} ({pct:.1f}%)")
                    
                    with col2:
                        if st.button("👁️ Preview Balanced Data", use_container_width=True):
                            try:
                                with st.spinner("Applying balancing..."):
                                    balanced_df = ClassBalancer.apply_balancing(
                                        df,
                                        target_col,
                                        balance_method,
                                        sampling_strategy,
                                        k_neighbors
                                    )
                                    
                                    st.success(f"✅ Balanced data: {len(balanced_df):,} rows")
                                    
                                    st.write("**After Balancing:**")
                                    new_dist = balanced_df[target_col].value_counts().head(10)
                                    for cls, count in new_dist.items():
                                        pct = (count / len(balanced_df) * 100)
                                        st.write(f"{cls}: {count:,} ({pct:.1f}%)")
                                    
                                    # Store preview in session state
                                    st.session_state.ml_balanced_preview = balanced_df
                                    st.session_state.ml_balancing_applied = True
                                    
                            except Exception as e:
                                st.error(f"Error during balancing: {str(e)}")
                                st.info("Try reducing k_neighbors or using Random Undersampling")
                    
                    # If balancing configured, store parameters
                    if apply_balancing:
                        st.session_state.ml_balance_config = {
                            'apply': True,
                            'method': balance_method,
                            'sampling_strategy': sampling_strategy,
                            'k_neighbors': k_neighbors
                        }
                    else:
                        st.session_state.ml_balance_config = {'apply': False}
                else:
                    st.session_state.ml_balance_config = {'apply': False}
        else:
            st.success(f"✅ **Balanced Dataset** - Imbalance ratio: {imbalance_info['imbalance_ratio']:.1f}:1. No balancing needed.")
            st.session_state.ml_balance_config = {'apply': False}
        
        st.divider()
        
        # LEVEL 3: Train Button - Only enable if models available
        available_models = st.session_state.get('ml_available_models', [])
        can_train = len(available_models) > 0
        
        if not can_train:
            st.error("❌ **No Models Available** - Check Model Availability Checker for details")
            st.button("🚀 Train Models", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Train Models", type="primary", use_container_width=True):
            from utils.ml_training import MLTrainer
            from utils.class_balancing import ClassBalancer
            
            # Apply class balancing if configured
            balance_config = st.session_state.get('ml_balance_config', {'apply': False})
            if balance_config.get('apply', False):
                try:
                    with st.status("Applying class balancing...", expanded=True) as status:
                        st.write(f"⚖️ Method: {balance_config['method']}")
                        st.write(f"📊 Target Ratio: {balance_config['sampling_strategy']}")
                        
                        df = ClassBalancer.apply_balancing(
                            df,
                            target_col,
                            balance_config['method'],
                            balance_config['sampling_strategy'],
                            balance_config.get('k_neighbors', 5)
                        )
                        
                        st.write(f"✅ Balanced dataset: {len(df):,} rows")
                        status.update(label="✅ Class balancing applied!", state="complete")
                except Exception as e:
                    st.error(f"Error during balancing: {str(e)}")
                    st.stop()
            
            # Comprehensive validation before training
            class_counts = df[target_col].value_counts()
            min_class_size = class_counts.min()
            n_classes = len(class_counts)
            total_samples = len(df)
            
            # Calculate minimum required samples per class for stratified split
            min_required = max(2, int(np.ceil(1 / (test_size/100))))  # At least 1 sample in test set
            
            # Check 1: Minimum 2 samples per class
            if min_class_size < 2:
                validation_passed = False
                
                # Show detailed debugging info
                bad_classes = class_counts[class_counts < 2]
                
                st.error(f"""
                ⚠️ **Validation Failed: Insufficient Samples per Class**
                
                **Target Column:** '{target_col}'  
                **Total Classes:** {n_classes}  
                **Minimum Class Size:** {min_class_size} sample(s)  
                **Classes with < 2 samples:** {len(bad_classes)}
                
                **Requirement:** Each class needs **at least 2 samples** for stratified train-test split.
                """)
                
                st.write("**Full Class Distribution:**")
                st.dataframe(class_counts.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}), 
                           use_container_width=True)
                
                st.write("**❌ Problem Classes (< 2 samples):**")
                st.dataframe(bad_classes.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}), 
                           use_container_width=True)
                
                st.info("""
                **💡 Solutions:**
                1. **Filter Data** - Remove classes with < 2 samples before training
                2. **Collect More Data** - Gather additional samples for rare classes  
                3. **Combine Classes** - Merge similar rare classes into broader categories
                4. **Choose Different Target** - Select a column with better class balance
                """)
                
            # Check 2: Enough samples for stratified split with current test_size
            elif min_class_size < min_required:
                validation_passed = False
                st.error(f"""
                ⚠️ **Validation Failed: Insufficient Samples for Stratified Split**
                
                With test size of **{test_size}%**, each class should have at least **{min_required} samples**.
                The smallest class has only **{min_class_size} samples**.
                
                **Class Distribution:**
                """)
                st.dataframe(class_counts.reset_index().rename(columns={'index': 'Class', target_col: 'Count'}), 
                           use_container_width=True)
                
                st.info(f"""
                **💡 Solutions:**
                1. **Reduce test size** to 10-15% (currently {test_size}%)
                2. **Filter rare classes** with < {min_required} samples
                3. **Combine classes** to reduce number of small classes
                
                **Action Required:** Adjust settings above and click Train again.
                """)
                
            # Check 3: Too many classes
            elif n_classes > 50:
                st.warning(f"""
                ⚠️ **Warning: Large Number of Classes**
                
                Target has **{n_classes} classes**. This is a multi-class classification problem.
                
                **Note:** Some models may take longer to train with many classes.
                """)
                st.info("Continuing with training... Some models may have reduced performance with many classes.")
                
                # Continue with training
                validation_passed = True
            
            # Check 4: Dataset too small
            elif total_samples < 50:
                st.warning(f"""
                ⚠️ **Warning: Small Dataset**
                
                Dataset has only **{total_samples} samples**. Minimum recommended: **50-100 samples**.
                
                **Impact:** 
                - Results may not be reliable
                - Cross-validation may struggle  
                - Model performance metrics could be unstable
                """)
                
                st.info("You can continue, but consider collecting more data for better results.")
                validation_passed = True
            
            else:
                # All validations passed
                validation_passed = True
            
            # Only proceed if validation passed
            if validation_passed:
                from utils.process_manager import ProcessManager, NavigationGuard
                
                # Create process manager
                pm = ProcessManager("ML_Classification")
                
                # Show warning about not navigating
                st.warning("""
                ⚠️ **Training Process Starting**
                
                Training multiple models may take several minutes. Please do not navigate away during this process.
                Navigation has been temporarily disabled to prevent data loss.
                """)
                
                # Lock navigation
                pm.lock()
                
                try:
                    # Initialize trainer
                    with st.status("Initializing ML Trainer...", expanded=True) as status:
                        trainer = MLTrainer(df, target_col, max_samples=10000)
                    
                    # Prepare data
                    with st.status("Preparing data for training...", expanded=True) as status:
                        prep_info = trainer.prepare_data(test_size=test_size/100)
                
                    # Show preparation info
                    st.success(f"✅ Data prepared: {prep_info['train_size']} train, {prep_info['test_size']} test samples across {prep_info['n_classes']} classes")
                    st.info(f"📊 Training will use cross-validation with {cv_folds} folds")
                    
                    # Get models to train - FILTER to only available
                    if selected_models is None:
                        # Train all AVAILABLE models
                        models_to_train = available_models
                    else:
                        # Train only selected AND available
                        models_to_train = [m for m in selected_models if m in available_models]
                    
                    if len(models_to_train) == 0:
                        st.error("❌ No available models selected")
                        pm.unlock()
                        st.stop()
                    
                    st.info(f"📊 Training {len(models_to_train)} available model(s): {', '.join(models_to_train)}")
                    
                    # Check memory before starting
                    memory_stats = ProcessManager.get_memory_stats()
                    st.info(f"💾 Memory usage before training: {memory_stats['rss_mb']:.1f}MB ({memory_stats['percent']:.1f}%)")
                    
                    # Clean up old results
                    ProcessManager.cleanup_large_session_state_items()
                    
                    # Progress tracking
                    st.divider()
                    st.subheader("⚙️ Training Progress")
                    
                    # Create containers for progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Use expander for training progress details
                    with st.expander("📋 Training Details (Click to expand)", expanded=False):
                        results_container = st.container()
                    
                    # Progress callback function
                    def update_progress(current, total, model_name):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Training {model_name}... ({current}/{total})")
                        with results_container:
                            st.write(f"🔄 Training: **{model_name}**")
                    
                    # Train models sequentially with memory management
                    import gc
                    results_dict = trainer.train_models_sequentially(
                        model_names=models_to_train,
                        cv_folds=cv_folds,
                        progress_callback=update_progress
                    )
                    
                    # Convert dict results to list format for compatibility
                    results = []
                    for model_name, model_result in results_dict.items():
                        if 'error' not in model_result:
                            # Add model name to result
                            model_result['model_name'] = model_name
                            model_result['success'] = True
                            # Add dummy model object (not saved for memory)
                            model_result['model'] = None
                            # Use train_time or default to 0
                            if 'training_time' not in model_result:
                                model_result['training_time'] = model_result.get('train_time', 0)
                            results.append(model_result)
                            
                            with results_container:
                                st.write(f"✅ **{model_name}** - Accuracy: {model_result.get('accuracy', 0):.4f}, F1: {model_result.get('f1', 0):.4f}")
                        else:
                            with results_container:
                                st.warning(f"⚠️ **{model_name}** - Error: {model_result['error']}")
                    
                    # Store final results
                    st.session_state.ml_results = results
                    st.session_state.ml_trainer = trainer
                    
                    # Check memory after training
                    memory_stats_after = ProcessManager.get_memory_stats()
                    memory_used = memory_stats_after['rss_mb'] - memory_stats['rss_mb']
                    st.info(f"💾 Memory used during training: {memory_used:.1f}MB")
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear progress
                    progress_bar.progress(1.0)
                    status_text.text("✅ Training complete!")
                    
                    st.success(f"🎉 Successfully trained {len(results)} models!")
                    
                    # Clear checkpoint on successful completion
                    pm.clear_checkpoint()
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    
                    # Save checkpoint on error for potential recovery
                    if results:
                        pm.save_checkpoint({
                            'error': str(e),
                            'partial_results': results,
                            'failed_at': len(results)
                        })
                        st.info("💾 Partial results saved. You can review them before retrying.")
                    
                finally:
                    # Always unlock navigation
                    pm.unlock()
    
    # Display results
    if 'ml_results' in st.session_state and 'ml_trainer' in st.session_state:
        results = st.session_state.ml_results
        trainer = st.session_state.ml_trainer
        
        st.divider()
        st.subheader("📊 3. Model Performance Results")
        
        # Summary metrics
        successful_results = [r for r in results if r['success']]
        # Sort by F1 Score to get the best model
        successful_results = sorted(successful_results, key=lambda x: x['f1'], reverse=True)
        best_model = successful_results[0] if successful_results else None
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Trained", len(results))
        with col2:
            st.metric("Successful", len(successful_results))
        with col3:
            if best_model:
                st.metric("Best F1 Score", f"{best_model['f1']:.4f}")
        with col4:
            # Safely sum training times, handling non-numeric values
            total_time = 0
            for r in results:
                time_val = r.get('training_time', 0)
                try:
                    if time_val is not None:
                        total_time += float(time_val)
                except (ValueError, TypeError):
                    # Skip non-numeric training times
                    continue
            st.metric("Total Time", f"{total_time:.2f}s")
        
        # Results table
        st.write("**Model Comparison Table:**")
        
        # Create results dataframe
        results_data = []
        for r in successful_results:
            results_data.append({
                'Model': r['model_name'],
                'Accuracy': f"{r['accuracy']:.4f}",
                'Precision': f"{r['precision']:.4f}",
                'Recall': f"{r['recall']:.4f}",
                'F1 Score': f"{r['f1']:.4f}",
                'ROC-AUC': f"{r['roc_auc']:.4f}" if r['roc_auc'] else 'N/A',
                'CV Mean': f"{r['cv_mean']:.4f}" if r['cv_mean'] else 'N/A',
                'CV Std': f"{r['cv_std']:.4f}" if r['cv_std'] else 'N/A',
                'Time (s)': f"{r['training_time']:.3f}"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Sort by F1 Score (best to worst)
        results_df['_f1_sort'] = results_df['F1 Score'].astype(float)
        results_df = results_df.sort_values('_f1_sort', ascending=False).drop('_f1_sort', axis=1).reset_index(drop=True)
        
        # Store in session state for export
        st.session_state.ml_results_df = results_df
        
        # Highlight best model
        def highlight_best(row):
            if row['Model'] == best_model['model_name']:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        st.divider()
        st.subheader("📈 4. Model Comparison Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["F1 Score Comparison", "Metrics Radar", "Cross-Validation", "Training Time"])
        
        with tab1:
            st.write("**F1 Score Comparison (All Models)**")
            
            # Sort by F1 for visualization
            f1_data = [(r['model_name'], r['f1']) for r in successful_results]
            f1_data.sort(key=lambda x: x[1], reverse=True)
            
            models = [d[0] for d in f1_data]
            f1_scores = [d[1] for d in f1_data]
            
            # Color code: top 3 green, rest blue
            colors = ['green' if i < 3 else 'blue' for i in range(len(models))]
            
            fig_f1 = go.Figure(data=[
                go.Bar(x=models, y=f1_scores, marker_color=colors)
            ])
            fig_f1.add_hline(y=np.mean(f1_scores), line_dash="dash", 
                            annotation_text=f"Mean: {np.mean(f1_scores):.4f}",
                            line_color="red")
            fig_f1.update_layout(
                title="F1 Score by Model",
                xaxis_title="Model",
                yaxis_title="F1 Score",
                height=500
            )
            fig_f1.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        with tab2:
            st.write("**Metrics Radar Chart (Top 3 Models)**")
            
            # Get top 3 models
            top3 = successful_results[:3]
            
            fig_radar = go.Figure()
            
            for model_result in top3:
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
                values = [
                    model_result['accuracy'],
                    model_result['precision'],
                    model_result['recall'],
                    model_result['f1'],
                    model_result['roc_auc'] if model_result['roc_auc'] else 0
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model_result['model_name']
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Performance Comparison: Top 3 Models",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab3:
            st.write("**Cross-Validation Score Distribution**")
            
            # Box plot of CV scores
            cv_data = []
            for r in successful_results:
                if r['cv_scores'] and len(r['cv_scores']) > 0:
                    for score in r['cv_scores']:
                        cv_data.append({
                            'Model': r['model_name'],
                            'CV Score': score
                        })
            
            if cv_data:
                cv_df = pd.DataFrame(cv_data)
                fig_cv = px.box(cv_df, x='Model', y='CV Score', 
                               title='Cross-Validation Score Distribution')
                fig_cv.update_xaxes(tickangle=-45)
                fig_cv.update_layout(height=500)
                st.plotly_chart(fig_cv, use_container_width=True)
            else:
                st.info("No cross-validation data available")
        
        with tab4:
            st.write("**Training Time Analysis**")
            
            # Sort by time
            time_data = [(r['model_name'], r['training_time']) for r in successful_results]
            time_data.sort(key=lambda x: x[1])
            
            models_time = [d[0] for d in time_data]
            times = [d[1] for d in time_data]
            
            # Color code by speed
            colors_time = []
            for t in times:
                if t < 1:
                    colors_time.append('green')  # Fast
                elif t < 5:
                    colors_time.append('yellow')  # Medium
                else:
                    colors_time.append('red')  # Slow
            
            fig_time = go.Figure(data=[
                go.Bar(y=models_time, x=times, orientation='h', marker_color=colors_time)
            ])
            fig_time.update_layout(
                title="Training Time by Model (seconds)",
                xaxis_title="Time (seconds)",
                yaxis_title="Model",
                height=600
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Best model details
        if best_model:
            st.divider()
            st.subheader(f"🏆 Best Model: {best_model['model_name']}")
            
            best_details = trainer.get_best_model_details(results)
            
            if best_details:
                # Performance summary
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{best_details['metrics']['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{best_details['metrics']['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{best_details['metrics']['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{best_details['metrics']['f1']:.4f}")
                with col5:
                    if best_details['metrics']['roc_auc']:
                        st.metric("ROC-AUC", f"{best_details['metrics']['roc_auc']:.4f}")
                
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                cm = np.array(best_details['confusion_matrix'])
                
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=best_details['class_names'],
                    y=best_details['class_names'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_cm.update_layout(title="Confusion Matrix", height=500)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Feature Importance
                if best_details['feature_importance']:
                    st.write("**Top 10 Most Important Features:**")
                    
                    feat_imp = best_details['feature_importance']
                    importance_df = pd.DataFrame({
                        'Feature': feat_imp['features'],
                        'Importance': feat_imp['importances']
                    })
                    # Sort descending and take top 10, then reverse for proper chart display (highest at top)
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                    importance_df = importance_df.sort_values('Importance', ascending=True)  # Reverse for chart display
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features (Highest to Lowest)'
                    )
                    fig_imp.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Model info
                with st.expander("📖 About This Model"):
                    model_info = trainer.get_model_info(best_model['model_name'])
                    st.markdown(f"""
                    **Description:** {model_info['description']}
                    
                    **Strengths:**
                    {model_info['strengths']}
                    
                    **Weaknesses:**
                    {model_info['weaknesses']}
                    
                    **Best Use Cases:**
                    {model_info['use_cases']}
                    """)
        
        # AI Insights
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'ml_ai_insights' in st.session_state:
            st.markdown(st.session_state.ml_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="ml_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("Analyzing results and generating insights...", expanded=True) as status:
                    # Get data from session state
                    ml_results_data = st.session_state.get('ml_results', [])
                    ml_trainer_data = st.session_state.get('ml_trainer')
                    ml_data_df = st.session_state.get('ml_data', pd.DataFrame())
                    
                    if not ml_results_data or ml_trainer_data is None or ml_data_df.empty:
                        st.error("No ML results available. Please train models first.")
                        st.stop()
                    
                    # Get successful results and best model
                    successful_results_data = [r for r in ml_results_data if r.get('accuracy') is not None]
                    if not successful_results_data:
                        st.error("No successful model results available.")
                        st.stop()
                    
                    best_model_data = max(successful_results_data, key=lambda x: x.get('f1', 0))
                    
                    # Prepare context
                    context = f"""
                    Machine Learning Classification Results:
                    
                    Dataset: {len(ml_data_df)} rows, {len(ml_data_df.columns)} columns
                    Target: {ml_trainer_data.target_column}
                    Classes: {', '.join(ml_trainer_data.class_names)}
                    
                    Models Trained: {len(successful_results_data)}
                    Best Model: {best_model_data['model_name']}
                    Best F1 Score: {best_model_data['f1']:.4f}
                    
                    Top 3 Models:
                    """
                    
                    for i, r in enumerate(successful_results_data[:3], 1):
                        context += f"\n{i}. {r['model_name']}: F1={r['f1']:.4f}, Accuracy={r['accuracy']:.4f}"
                    
                    prompt = f"""
                    As a senior data science consultant, analyze these machine learning results and provide:
                    
                    1. **Performance Analysis** (2-3 sentences): Why did {best_model_data['model_name']} perform best?
                    
                    2. **Model Comparison** (2-3 sentences): Key differences between top 3 models and when to use each.
                    
                    3. **Business Recommendations** (3-4 bullet points): Which model to deploy and why? Consider accuracy, speed, interpretability.
                    
                    4. **Improvement Suggestions** (3-4 bullet points): How to potentially improve performance?
                    
                    5. **Deployment Considerations** (2-3 bullet points): What to watch for in production?
                    
                    {context}
                    
                    Be specific, actionable, and business-focused. Use clear language.
                    """
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior data science consultant providing actionable ML insights."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.ml_ai_insights = response.choices[0].message.content
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.ml_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
        # Export section
        st.divider()
        st.subheader("📥 5. Export & Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export results
            if 'ml_results_df' in st.session_state:
                results_export = st.session_state.ml_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results Table (CSV)",
                    data=results_export,
                    file_name=f"ml_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("Results not available for export")
        
        with col2:
            # Export best model report
            if best_model:
                try:
                    # Generate markdown table from results
                    if 'ml_results_df' in st.session_state:
                        results_table = st.session_state.ml_results_df.to_markdown(index=False)
                    else:
                        results_table = "Results table not available"
                    
                    # Format metrics safely
                    roc_auc_str = f"{best_model['roc_auc']:.4f}" if best_model['roc_auc'] is not None else 'N/A'
                    cv_mean_str = f"{best_model['cv_mean']:.4f}" if best_model['cv_mean'] is not None else 'N/A'
                    
                    report = f"""
# Machine Learning Classification Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Model: {best_model['model_name']}

### Performance Metrics
- **Accuracy:** {best_model['accuracy']:.4f}
- **Precision:** {best_model['precision']:.4f}
- **Recall:** {best_model['recall']:.4f}
- **F1 Score:** {best_model['f1']:.4f}
- **ROC-AUC:** {roc_auc_str}
- **CV Mean:** {cv_mean_str}
- **Training Time:** {best_model['training_time']:.3f}s

## All Models Performance

{results_table}

## Dataset Information
- **Rows:** {len(df)}
- **Features:** {len(trainer.feature_names)}
- **Target:** {target_col}
- **Classes:** {', '.join(trainer.class_names)}
"""
                    
                    # Add AI insights if available
                    if 'ml_ai_insights' in st.session_state:
                        report += f"""

## 🤖 AI-Powered Strategic Insights

{st.session_state.ml_ai_insights}

"""
                    
                    report += """
---
*Report generated by DataInsights - Machine Learning Module*
"""
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    report = "Error generating report"
                st.download_button(
                    label="📄 Download Best Model Report (MD)",
                    data=report,
                    file_name=f"ml_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

def show_ml_regression():
    """Machine Learning Regression page with 15+ algorithms."""
    
    # Import ML helper functions for optimization
    from utils.ml_helpers import get_recommended_cv_folds, create_data_hash, cached_regression_training
    
    st.markdown("<h2 style='text-align: center;'>📈 ML Regression</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Machine Learning Regression?"):
        st.markdown("""
        **Machine Learning Regression** predicts continuous numerical values based on input features.
        
        ### Use Cases:
        - 🏠 **House Price Prediction** - Predict property values
        - 💰 **Sales Forecasting** - Estimate revenue
        - 📊 **Demand Prediction** - Forecast product demand
        - 🎯 **Risk Scoring** - Calculate continuous risk scores
        - 📈 **Stock Price Prediction** - Estimate market values
        
        ### What's Included:
        - ✅ **15+ regression algorithms** (Linear, Tree-based, Boosting, Ensemble)
        - ✅ **Smart target detection** - Auto-selects continuous target
        - ✅ **Data quality validation** - Checks dataset suitability
        - ✅ **Comprehensive metrics** - MSE, RMSE, MAE, R², MAPE
        - ✅ **Model comparison** - Interactive visualizations
        - ✅ **Feature importance** - Understand key drivers
        - ✅ **AI insights** - GPT-4 powered recommendations
        - ✅ **Export results** - CSV & Markdown reports
        """)
    
    # Data selection
    st.divider()
    st.subheader("📤 1. Select Data Source")
    
    # Check if data is already uploaded in session
    if st.session_state.data is not None:
        data_source = st.radio(
            "Choose data source:",
            ["Use uploaded data from Data Upload page", "Sample Boston Housing Dataset", "Upload new file for this analysis"],
            help="You can use the data you already uploaded, try a sample dataset, or upload a new file"
        )
        
        if data_source == "Use uploaded data from Data Upload page":
            df = st.session_state.data
            st.session_state.mlr_data = df
            st.success(f"✅ Using uploaded data: {len(df):,} rows and {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
                
        elif data_source == "Sample Boston Housing Dataset":
            if st.button("📥 Load Boston Housing Dataset", type="primary"):
                with st.status("Loading Boston Housing dataset...", expanded=True) as status:
                    try:
                        # Create synthetic Boston Housing-like dataset
                        np.random.seed(42)
                        n_samples = 506
                        df = pd.DataFrame({
                            'CRIM': np.random.exponential(3.6, n_samples),  # Crime rate
                            'RM': np.random.normal(6.3, 0.7, n_samples),  # Avg rooms
                            'AGE': np.random.uniform(0, 100, n_samples),  # Age of homes
                            'DIS': np.random.exponential(3.8, n_samples),  # Distance to employment
                            'TAX': np.random.uniform(180, 710, n_samples),  # Property tax
                            'PTRATIO': np.random.uniform(12, 22, n_samples),  # Pupil-teacher ratio
                            'LSTAT': np.random.exponential(12.7, n_samples),  # % lower status
                        })
                        # Target: Median house value (in $1000s)
                        df['MEDV'] = (
                            35 - 0.5 * df['CRIM'] + 5 * df['RM'] - 0.1 * df['AGE'] 
                            - 1 * df['DIS'] - 0.01 * df['TAX'] - 0.5 * df['PTRATIO'] 
                            - 0.5 * df['LSTAT'] + np.random.normal(0, 5, n_samples)
                        )
                        df['MEDV'] = df['MEDV'].clip(5, 50)  # Realistic range
                        
                        st.session_state.mlr_data = df
                        st.success(f"✅ Loaded Boston Housing dataset: {len(df)} rows and {len(df.columns)} columns")
                        
                        st.info("""
                        **About this dataset:**
                        - 🎯 **Target:** MEDV (Median home value in $1000s)
                        - 📊 **Features:** 7 attributes
                        - ✅ **Perfect for Regression**
                        - 🏆 **Classic dataset**
                        """)
                        
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
                        
        else:  # Upload new file
            uploaded_file = st.file_uploader(
                "Upload CSV file with features and continuous target column",
                type=['csv'],
                key="mlr_upload",
                help="Must include numerical features and continuous target column"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.mlr_data = df
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            else:
                st.info("👆 Please upload a CSV file to continue")
    else:
        data_source = st.radio(
            "Choose data source:",
            ["Sample Boston Housing Dataset", "Upload new file for this analysis"],
            help="Try the sample dataset or upload your own"
        )
        
        if data_source == "Sample Boston Housing Dataset":
            if st.button("📥 Load Boston Housing Dataset", type="primary"):
                with st.status("Loading Boston Housing dataset...", expanded=True) as status:
                    try:
                        # Same synthetic dataset code as above
                        np.random.seed(42)
                        n_samples = 506
                        df = pd.DataFrame({
                            'CRIM': np.random.exponential(3.6, n_samples),
                            'RM': np.random.normal(6.3, 0.7, n_samples),
                            'AGE': np.random.uniform(0, 100, n_samples),
                            'DIS': np.random.exponential(3.8, n_samples),
                            'TAX': np.random.uniform(180, 710, n_samples),
                            'PTRATIO': np.random.uniform(12, 22, n_samples),
                            'LSTAT': np.random.exponential(12.7, n_samples),
                        })
                        df['MEDV'] = (
                            35 - 0.5 * df['CRIM'] + 5 * df['RM'] - 0.1 * df['AGE'] 
                            - 1 * df['DIS'] - 0.01 * df['TAX'] - 0.5 * df['PTRATIO'] 
                            - 0.5 * df['LSTAT'] + np.random.normal(0, 5, n_samples)
                        )
                        df['MEDV'] = df['MEDV'].clip(5, 50)
                        
                        st.session_state.mlr_data = df
                        st.success(f"✅ Loaded Boston Housing dataset: {len(df)} rows and {len(df.columns)} columns")
                        
                        st.info("""
                        **About this dataset:**
                        - 🎯 **Target:** MEDV (Median home value in $1000s)
                        - 📊 **Features:** 7 attributes
                        - ✅ **Perfect for Regression**
                        - 🏆 **Classic dataset**
                        """)
                        
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error loading sample data: {str(e)}")
        else:  # Upload custom data
            uploaded_file = st.file_uploader(
                "Upload CSV file with features and continuous target column",
                type=['csv'],
                key="mlr_upload",
                help="Must include numerical features and continuous target column"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.mlr_data = df
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    # Configuration and training
    if 'mlr_data' in st.session_state:
        df = st.session_state.mlr_data
        
        # Check if previous quality check failed and show prominent warning
        if 'mlr_training_suitable' in st.session_state and not st.session_state.mlr_training_suitable:
            st.error("""
            ### 🚫 DATA NOT SUITABLE FOR REGRESSION
            
            Your dataset has quality issues that prevent ML training. Please review the issues below and fix your data.
            """)
            
            quality_issues = st.session_state.get('mlr_quality_issues', [])
            
            if quality_issues:
                st.write("**Issues Found:**")
                for issue in quality_issues:
                    st.write(f"• {issue}")
                
                st.info("**💡 Tip:** If your target has few unique values (< 10), use **ML Classification** instead.")
            
            st.divider()
        
        st.divider()
        st.subheader("🎯 2. Configure Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Smart target column detection for regression
            def detect_regression_target(df):
                """Detect likely continuous target column for regression."""
                # Look for numerical columns with many unique values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # High-priority patterns (specific target indicators)
                priority_patterns = ['medv', 'median_value', 'price', 'cost', 'value', 'target', 
                                    'y', 'label', 'sales', 'revenue', 'income', 'salary']
                
                # Check for high-priority pattern matches first
                for col in numeric_cols:
                    col_lower = col.lower().replace('_', '')
                    for pattern in priority_patterns:
                        pattern_clean = pattern.replace('_', '')
                        if pattern_clean == col_lower or (pattern_clean in col_lower and len(pattern_clean) > 3):
                            # Verify it's continuous (many unique values)
                            if df[col].nunique() > 10:
                                return col
                
                # Secondary: Find numerical column with most unique values (likely continuous target)
                candidates = [(col, df[col].nunique()) for col in numeric_cols if df[col].nunique() > 10]
                if candidates:
                    # Sort by number of unique values (descending)
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    return candidates[0][0]
                
                # Fallback to last numerical column
                if len(numeric_cols) > 0:
                    return numeric_cols[-1]
                
                return df.columns[-1]
            
            suggested_target = detect_regression_target(df)
            target_index = list(df.columns).index(suggested_target) if suggested_target in df.columns else 0
            
            st.info("💡 **Smart Detection:** Target column auto-selected based on your data. Change if needed.")
            
            target_col = st.selectbox(
                "Select Target Column (continuous value to predict)",
                df.columns,
                index=target_index,
                help="Column containing continuous numerical values to predict"
            )
            
            # Show target distribution and quality check
            if target_col:
                target_values = df[target_col]
                
                # Check if target is numerical
                if not pd.api.types.is_numeric_dtype(target_values):
                    st.error("⚠️ **Target must be numerical for regression!**")
                    st.session_state.mlr_training_suitable = False
                else:
                    n_unique = target_values.nunique()
                    total_samples = len(df)
                    
                    # Quality assessment - balanced approach
                    issues = []
                    warnings = []
                    
                    # Check 1: Too few unique values (suggest classification but don't block)
                    if n_unique < 5:
                        warnings.append(f"⚠️ Only {n_unique} unique values - consider using ML Classification instead")
                    elif n_unique < 10:
                        warnings.append(f"⚠️ {n_unique} unique values - if this is categorical, use ML Classification")
                    
                    # Check 2: Dataset size (warn but don't block)
                    if total_samples < 30:
                        warnings.append(f"⚠️ Small dataset ({total_samples} samples) - results may vary. Recommend 100+")
                    
                    # Check 3: Missing values (only block if ALL values are missing)
                    missing_count = target_values.isna().sum()
                    if missing_count == total_samples:
                        issues.append(f"❌ All target values are missing - cannot train")
                    elif missing_count > 0:
                        missing_pct = (missing_count / total_samples) * 100
                        if missing_pct > 50:
                            warnings.append(f"⚠️ {missing_count} missing values ({missing_pct:.1f}%) - will be dropped")
                        else:
                            warnings.append(f"⚠️ {missing_count} missing values ({missing_pct:.1f}%) - will be handled")
                    
                    # LEVEL 1: Data Source Compatibility - Only block on CRITICAL issues
                    data_compatible = len(issues) == 0
                    
                    st.session_state.mlr_data_compatible = data_compatible
                    st.session_state.mlr_quality_issues = issues
                    st.session_state.mlr_quality_warnings = warnings
                    
                    # Display quality indicator
                    if len(issues) > 0:
                        st.error("**🚨 Data Quality: NOT SUITABLE FOR REGRESSION**")
                        for issue in issues:
                            st.write(issue)
                        st.info("💡 **Tip:** Use ML Classification for categorical targets with few classes")
                    elif len(warnings) > 0:
                        st.warning("**⚠️ Data Quality: TRAINING POSSIBLE (with warnings)**")
                        for warning in warnings:
                            st.write(warning)
                    else:
                        st.success("**✅ Data Quality: EXCELLENT FOR REGRESSION**")
                        st.write(f"✓ {n_unique} unique values, continuous target")
                    
                    st.caption(f"🔍 Debug: Target='{target_col}', Unique Values={n_unique}, Samples={total_samples}")
                
                st.write("**Target Distribution:**")
                fig_dist = px.histogram(
                    df,
                    x=target_col,
                    nbins=30,
                    title=f'Distribution of {target_col}'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Model selection
            train_all = st.checkbox("Train All Models (15+ algorithms)", value=True,
                                   help="Train all models or select specific ones")
            
            if not train_all:
                from utils.ml_regression import MLRegressor
                temp_regressor = MLRegressor(df, target_col)
                all_model_names = list(temp_regressor.get_all_models().keys())
                
                selected_models = st.multiselect(
                    "Select Models to Train",
                    all_model_names,
                    default=all_model_names[:5],
                    help="Choose which models to train"
                )
            else:
                selected_models = None
            
            # LEVEL 2: Model Availability Checker - Per-model compatibility
            with st.expander("📋 Model Availability Checker", expanded=False):
                st.write("**Per-model compatibility check:**")
                
                # Get data compatibility from Level 1
                data_compatible = st.session_state.get('mlr_data_compatible', True)
                quality_issues = st.session_state.get('mlr_quality_issues', [])
                
                # Show data-level warning if incompatible
                if not data_compatible:
                    st.warning("⚠️ **Data has critical issues** - Models may fail")
                    for issue in quality_issues:
                        st.write(f"• {issue}")
                    st.divider()
                
                # Get all models
                from utils.ml_regression import MLRegressor
                temp_regressor = MLRegressor(df, target_col if target_col else df.columns[0])
                all_models = temp_regressor.get_all_models()
                
                # Check each model individually
                model_status = []
                n_samples = len(df)
                
                for model_name in all_models.keys():
                    available = True
                    reason = "✅ Ready"
                    
                    # Check 1: Library availability
                    if model_name == "XGBoost":
                        try:
                            import xgboost
                        except ImportError:
                            available = False
                            reason = "❌ XGBoost not installed"
                    elif model_name == "LightGBM":
                        try:
                            import lightgbm
                        except ImportError:
                            available = False
                            reason = "❌ LightGBM not installed"
                    elif model_name == "CatBoost":
                        try:
                            import catboost
                        except ImportError:
                            available = False
                            reason = "❌ CatBoost not installed"
                    
                    # Check 2: Data-level issues (critical ones block all models)
                    if not data_compatible and available:
                        available = False
                        reason = "❌ Data incompatible"
                    
                    # Check 3: Model-specific requirements
                    if available:
                        if n_samples < 10:
                            available = False
                            reason = f"❌ Need ≥10 samples (have {n_samples})"
                        elif n_samples < 30:
                            reason = f"⚠️ Small dataset ({n_samples} samples)"
                    
                    model_status.append({
                        'Model': model_name,
                        'Status': '✅ Available' if available else '❌ Unavailable',
                        'Notes': reason if not available else '✅ Ready'
                    })
                
                # Display as table
                status_df = pd.DataFrame(model_status)
                
                # Color code the display
                def color_status(row):
                    if '❌' in row['Status']:
                        return ['background-color: #ffebee'] * len(row)
                    elif '⚠️' in row['Notes']:
                        return ['background-color: #fff8e1'] * len(row)
                    else:
                        return ['background-color: #f1f8e9'] * len(row)
                
                styled_status = status_df.style.apply(color_status, axis=1)
                st.dataframe(styled_status, use_container_width=True, height=400)
                
                # Summary and store available models
                available_models = [m['Model'] for m in model_status if '✅' in m['Status']]
                available_count = len(available_models)
                unavailable_count = len(model_status) - available_count
                
                # Store for Level 3
                st.session_state.mlr_available_models = available_models
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Available Models", available_count, delta="Ready to train" if available_count > 0 else "None available")
                with col_b:
                    if unavailable_count > 0:
                        st.metric("Unavailable Models", unavailable_count, delta="Check notes", delta_color="inverse")
                    else:
                        st.metric("Unavailable Models", 0, delta="All ready!")
            
            # Training config
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data reserved for testing"
            )
            
            # Smart CV fold recommendation
            n_samples = len(df)
            recommended_folds, cv_reason = get_recommended_cv_folds(n_samples, None)
            
            st.info(f"💡 **Recommended:** {recommended_folds}-fold CV - {cv_reason}")
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=recommended_folds,
                max_value=10,
                value=3,
                help=f"Recommended: {recommended_folds} for your dataset ({n_samples:,} samples)"
            )
        
        # LEVEL 3: Train Button - Only enable if models available
        available_models = st.session_state.get('mlr_available_models', [])
        can_train = len(available_models) > 0
        
        if not can_train:
            st.error("❌ **No Models Available** - Check Model Availability Checker for details")
            st.button("🚀 Train Models", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Train Models", type="primary", use_container_width=True):
            from utils.ml_regression import MLRegressor
            from utils.process_manager import ProcessManager
            
            # Create process manager
            pm = ProcessManager("ML_Regression")
            
            # Show warning about not navigating
            st.warning("""
            ⚠️ **Important:** Do not navigate away from this page during training.
            Navigation is now locked to prevent data loss.
            """)
            
            # Lock navigation
            pm.lock()
            
            try:
                # Initialize regressor
                with st.status("Initializing ML Regressor...", expanded=True) as status:
                    regressor = MLRegressor(df, target_col, max_samples=10000)
                    prep_info = regressor.prepare_data(test_size=test_size/100)
                
                # Show preparation info
                st.success(f"✅ Data prepared: {prep_info['train_size']} train, {prep_info['test_size']} test samples")
                if prep_info['sampled']:
                    st.info(f"📊 Dataset sampled to 10,000 rows for performance optimization")
                
                # Progress tracking
                st.divider()
                st.subheader("⚙️ Training Progress")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                def progress_callback(current, total, model_name, result):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Training {current}/{total}: {model_name} - R²: {result['r2']:.4f}")
                    
                    # Save checkpoint every 2 models
                    if current % 2 == 0:
                        pm.save_checkpoint({
                            'completed_models': current,
                            'total_models': total,
                            'partial_results': [r['model_name'] for r in results]
                        })
                
                # FILTER to only train available models
                if selected_models is None:
                    # Train all AVAILABLE models
                    models_to_train = available_models
                else:
                    # Train only selected AND available
                    models_to_train = [m for m in selected_models if m in available_models]
                
                if len(models_to_train) == 0:
                    st.error("❌ No available models selected")
                    pm.unlock()
                    st.stop()
                
                st.info(f"📊 Training {len(models_to_train)} available model(s): {', '.join(models_to_train)}")
                
                # Check memory before starting
                memory_stats = ProcessManager.get_memory_stats()
                st.info(f"💾 Memory usage before training: {memory_stats['rss_mb']:.1f}MB ({memory_stats['percent']:.1f}%)")
                
                # Clean up old results
                ProcessManager.cleanup_large_session_state_items()
                
                # Progress callback for sequential training
                def seq_progress_callback(current, total, model_name):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Training {model_name}... ({current}/{total})")
                
                # Train models sequentially with memory management
                import gc
                results_dict = regressor.train_models_sequentially(
                    model_names=models_to_train,
                    cv_folds=cv_folds,
                    progress_callback=seq_progress_callback
                )
                
                # Convert dict results to list format for compatibility
                results = []
                for model_name, model_result in results_dict.items():
                    if 'error' not in model_result:
                        # Add model name and success flag
                        model_result['model_name'] = model_name
                        model_result['success'] = True
                        model_result['model'] = None  # Don't store model for memory
                        # Use train_time or training_time
                        if 'training_time' not in model_result:
                            model_result['training_time'] = model_result.get('train_time', 0)
                        results.append(model_result)
                    else:
                        # Add error result
                        results.append({
                            'model_name': model_name,
                            'success': False,
                            'error': model_result['error']
                        })
                
                # Store results
                st.session_state.mlr_results = results
                st.session_state.mlr_regressor = regressor
                
                # Check memory after training
                memory_stats_after = ProcessManager.get_memory_stats()
                memory_used = memory_stats_after['rss_mb'] - memory_stats['rss_mb']
                st.info(f"💾 Memory used during training: {memory_used:.1f}MB")
                
                # Force garbage collection
                gc.collect()
                
                # Final checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'total_models': len(results),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training complete!")
                
                st.success(f"🎉 Successfully trained {len(results)} models!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                # Save partial results on error
                if 'results' in locals() and len(results) > 0:
                    st.session_state.mlr_results = results
                    st.session_state.mlr_regressor = regressor
                    st.warning(f"⚠️ Saved partial results for {len(results)} models")
                    pm.save_checkpoint({
                        'error': str(e),
                        'partial_results': len(results)
                    })
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Always unlock navigation
                pm.unlock()
                st.info("✅ Navigation unlocked - you can now navigate to other pages.")
    
    # Display results
    if 'mlr_results' in st.session_state and 'mlr_regressor' in st.session_state:
        results = st.session_state.mlr_results
        regressor = st.session_state.mlr_regressor
        
        st.divider()
        st.subheader("📊 3. Model Performance Results")
        
        # Summary metrics
        successful_results = [r for r in results if r['success']]
        best_model = max(successful_results, key=lambda x: x['r2'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Trained", len(successful_results))
        with col2:
            st.metric("Best Model", best_model['model_name'])
        with col3:
            st.metric("Best R² Score", f"{best_model['r2']:.4f}")
        with col4:
            # Safely sum training times, handling non-numeric values
            total_time = 0
            for r in successful_results:
                time_val = r.get('training_time', 0)
                try:
                    if time_val is not None:
                        total_time += float(time_val)
                except (ValueError, TypeError):
                    # Skip non-numeric training times
                    continue
            st.metric("Total Time", f"{total_time:.1f}s")
        
        # Results table
        st.write("**Model Comparison:**")
        results_data = []
        for r in successful_results:
            results_data.append({
                'Model': r['model_name'],
                'R²': f"{r['r2']:.4f}",
                'RMSE': f"{r['rmse']:.2f}",
                'MAE': f"{r['mae']:.2f}",
                'MAPE': f"{r['mape']:.2f}%" if r['mape'] else 'N/A',
                'CV Mean': f"{r['cv_mean']:.4f}" if r['cv_mean'] else 'N/A',
                'Time (s)': f"{r['training_time']:.3f}"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Sort by R² (best to worst)
        results_df['_r2_sort'] = results_df['R²'].astype(float)
        results_df = results_df.sort_values('_r2_sort', ascending=False).drop('_r2_sort', axis=1).reset_index(drop=True)
        
        st.session_state.mlr_results_df = results_df
        
        # Highlight best model
        def highlight_best(row):
            if row['Model'] == best_model['model_name']:
                return ['background-color: #90EE90'] * len(row)
            return [''] * len(row)
        
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        st.divider()
        st.subheader("📈 4. Model Comparison Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["📊 R² Comparison", "📉 Error Metrics", "⏱️ Training Time"])
        
        with tab1:
            # R² comparison - sort ascending so best model appears at TOP of chart
            r2_scores = [(r['model_name'], r['r2']) for r in successful_results]
            r2_scores.sort(key=lambda x: x[1])  # Ascending order for bottom-to-top display
            
            fig_r2 = px.bar(
                x=[x[1] for x in r2_scores],
                y=[x[0] for x in r2_scores],
                orientation='h',
                title='Model Performance (R² Score - Higher is Better)',
                labels={'x': 'R² Score', 'y': 'Model'},
                color=[x[1] for x in r2_scores],
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with tab2:
            # Error metrics comparison
            error_data = pd.DataFrame([
                {'Model': r['model_name'], 'RMSE': r['rmse'], 'MAE': r['mae']}
                for r in successful_results
            ])
            
            fig_error = px.scatter(
                error_data,
                x='MAE',
                y='RMSE',
                text='Model',
                title='Error Metrics (Lower is Better)',
                labels={'MAE': 'Mean Absolute Error', 'RMSE': 'Root Mean Squared Error'}
            )
            fig_error.update_traces(textposition='top center')
            st.plotly_chart(fig_error, use_container_width=True)
        
        with tab3:
            # Training time
            time_data = [(r['model_name'], r['training_time']) for r in successful_results]
            time_data.sort(key=lambda x: x[1])
            
            colors = ['green' if x[1] < 1 else 'yellow' if x[1] < 5 else 'red' for x in time_data]
            
            fig_time = px.bar(
                y=[x[0] for x in time_data],
                x=[x[1] for x in time_data],
                orientation='h',
                title='Training Time Comparison',
                labels={'x': 'Training Time (seconds)', 'y': 'Model'},
                color=colors,
                color_discrete_map={'green': '#90EE90', 'yellow': '#FFD700', 'red': '#FFB6C1'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Best Model Details
        st.divider()
        st.subheader(f"🏆 Best Model: {best_model['model_name']}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("R² Score", f"{best_model['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{best_model['rmse']:.2f}")
        with col3:
            st.metric("MAE", f"{best_model['mae']:.2f}")
        with col4:
            if best_model['mape']:
                st.metric("MAPE", f"{best_model['mape']:.2f}%")
            else:
                st.metric("MAPE", "N/A")
        with col5:
            if best_model['cv_mean']:
                st.metric("CV R²", f"{best_model['cv_mean']:.4f}")
            else:
                st.metric("CV R²", "N/A")
        
        # Feature Importance
        if best_model.get('feature_importance'):
            st.write("**Top 10 Most Important Features:**")
            feat_imp = best_model['feature_importance']
            # Sort descending and take top 10, then reverse for proper chart display (highest at top)
            feat_imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            feat_imp_sorted = sorted(feat_imp_sorted, key=lambda x: x[1])  # Reverse for chart display
            
            fig_imp = px.bar(
                x=[x[1] for x in feat_imp_sorted],
                y=[x[0] for x in feat_imp_sorted],
                orientation='h',
                title='Top 10 Most Important Features (Highest to Lowest)',
                labels={'x': 'Importance Score', 'y': 'Feature'}
            )
            fig_imp.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # AI-Powered Insights
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'mlr_ai_insights' in st.session_state:
            st.markdown(st.session_state.mlr_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="mlr_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("Analyzing regression results and generating insights...", expanded=True) as status:
                    # Get data from session state
                    mlr_results_data = st.session_state.get('mlr_results', [])
                    mlr_regressor_data = st.session_state.get('mlr_regressor')
                    mlr_data_df = st.session_state.get('mlr_data', pd.DataFrame())
                    
                    if not mlr_results_data or mlr_regressor_data is None or mlr_data_df.empty:
                        st.error("No ML Regression results available. Please train models first.")
                        st.stop()
                    
                    # Get successful results and best model
                    successful_results_data = [r for r in mlr_results_data if r.get('r2') is not None]
                    if not successful_results_data:
                        st.error("No successful model results available.")
                        st.stop()
                    
                    best_model_data = max(successful_results_data, key=lambda x: x.get('r2', -999))
                    
                    # Prepare context
                    context = f"""
                    Machine Learning Regression Results:
                    
                    Dataset: {len(mlr_data_df)} rows, {len(mlr_data_df.columns)} columns
                    Target: {mlr_regressor_data.target_column}
                    
                    Models Trained: {len(successful_results_data)}
                    Best Model: {best_model_data['model_name']}
                    Best R² Score: {best_model_data['r2']:.4f}
                    Best RMSE: {best_model_data['rmse']:.2f}
                    Best MAE: {best_model_data['mae']:.2f}
                    
                    Top 3 Models:
                    """
                    
                    for i, r in enumerate(successful_results_data[:3], 1):
                        context += f"\n{i}. {r['model_name']}: R²={r['r2']:.4f}, RMSE={r['rmse']:.2f}, MAE={r['mae']:.2f}"
                    
                    prompt = f"""
                    As a senior data science consultant, analyze these machine learning regression results and provide:
                    
                    1. **Performance Analysis** (2-3 sentences): Why did {best_model_data['model_name']} perform best? What does the R² score tell us about model fit?
                    
                    2. **Model Comparison** (2-3 sentences): Key differences between top 3 models and when to use each. Which model offers best trade-offs?
                    
                    3. **Business Recommendations** (3-4 bullet points): Which model to deploy and why? Consider accuracy, speed, interpretability, and prediction reliability.
                    
                    4. **Improvement Suggestions** (3-4 bullet points): How to potentially improve performance? Consider feature engineering, hyperparameter tuning, ensemble methods.
                    
                    5. **Deployment Considerations** (2-3 bullet points): What to watch for in production? How to monitor model performance over time?
                    
                    {context}
                    
                    Be specific, actionable, and business-focused. Use clear language. Interpret R², RMSE, and MAE in practical terms.
                    """
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior data science consultant providing actionable ML insights for regression problems."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.mlr_ai_insights = response.choices[0].message.content
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.mlr_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
                if "openai" in str(e).lower() or "api" in str(e).lower():
                    st.info("💡 Make sure your OpenAI API key is configured in the secrets.")
        
        # Export section
        st.divider()
        st.subheader("📥 5. Export & Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export results CSV
            if 'mlr_results_df' in st.session_state:
                results_export = st.session_state.mlr_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results Table (CSV)",
                    data=results_export,
                    file_name=f"mlr_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            # Export best model report
            if best_model:
                # Format metrics safely
                mape_str = f"{best_model['mape']:.2f}%" if best_model['mape'] is not None else 'N/A'
                cv_mean_str = f"{best_model['cv_mean']:.4f}" if best_model['cv_mean'] is not None else 'N/A'
                results_table = st.session_state.mlr_results_df.to_markdown(index=False) if 'mlr_results_df' in st.session_state else 'Results not available'
                
                report = f"""
# Machine Learning Regression Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Model: {best_model['model_name']}

### Performance Metrics
- **R² Score:** {best_model['r2']:.4f}
- **RMSE:** {best_model['rmse']:.2f}
- **MAE:** {best_model['mae']:.2f}
- **MAPE:** {mape_str}
- **CV R² Mean:** {cv_mean_str}
- **Training Time:** {best_model['training_time']:.3f}s

## All Models Performance

{results_table}

## Dataset Information
- **Rows:** {len(df)}
- **Features:** {len(regressor.feature_names)}
- **Target:** {target_col}
"""
                
                # Add AI insights if available
                if 'mlr_ai_insights' in st.session_state:
                    report += f"""

## 🤖 AI-Powered Strategic Insights

{st.session_state.mlr_ai_insights}

"""
                
                report += """
---
*Report generated by DataInsights - ML Regression Module*
"""
                st.download_button(
                    label="📄 Download Best Model Report (MD)",
                    data=report,
                    file_name=f"mlr_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )



def show_anomaly_detection():
    """Anomaly & Outlier Detection page."""
    st.markdown("<h2 style='text-align: center;'>🔬 Anomaly & Outlier Detection</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Anomaly Detection?"):
        st.markdown("""
        **Anomaly Detection** identifies unusual patterns, outliers, and anomalies in your data that don't conform to expected behavior.
        
        ### Common Applications:
        
        - **Fraud Detection:** Identify suspicious transactions or behaviors
        - **Quality Control:** Detect manufacturing defects or system failures
        - **Cybersecurity:** Flag unusual network activity or intrusions
        - **Healthcare:** Identify rare diseases or abnormal patient readings
        
        ### Algorithms Available:
        
        **1. Isolation Forest** ⭐ Recommended
        - Fast and effective for high-dimensional data
        - Works by isolating anomalies using decision trees
        - Best for: General-purpose anomaly detection
        
        **2. Local Outlier Factor (LOF)**
        - Detects local outliers based on density
        - Good for datasets with varying density
        - Best for: Data with clusters of different densities
        
        **3. One-Class SVM**
        - Learns a decision boundary around normal data
        - Works well with non-linear patterns
        - Best for: Complex, non-linear data distributions
        
        ### How to Use:
        
        1. Upload your data
        2. Select numeric features to analyze
        3. Choose an algorithm and set contamination (expected % of anomalies)
        4. Review results, visualizations, and AI explanations
        """)
    
    st.markdown("""
    Identify outliers and unusual patterns in your data using machine learning algorithms.
    """)
    
    # Data source selection
    st.subheader("📤 1. Load Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Use Sample Data", "Upload Custom Data"]
        default_option = "Use Loaded Dataset"
    else:
        data_options = ["Use Sample Data", "Upload Custom Data"]
        default_option = "Use Sample Data"
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="anomaly_data_source"
    )
    
    df = None
    
    if data_source == "Use Loaded Dataset":
        st.success("✅ Using dataset from Data Upload section")
        df = st.session_state.data
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
    
    elif data_source == "Use Sample Data":
        st.info("📊 Using sample credit card transaction dataset with anomalies")
        
        # Create sample data with normal transactions and some anomalies
        np.random.seed(42)
        n_normal = 200
        n_anomalies = 10
        
        # Normal transactions
        normal_amounts = np.random.normal(loc=100, scale=30, size=n_normal)
        normal_amounts = np.clip(normal_amounts, 10, 300)  # Keep in reasonable range
        
        normal_frequency = np.random.normal(loc=15, scale=5, size=n_normal)
        normal_frequency = np.clip(normal_frequency, 1, 30)
        
        normal_time_of_day = np.random.normal(loc=14, scale=4, size=n_normal)
        normal_time_of_day = np.clip(normal_time_of_day, 0, 23)
        
        # Anomalous transactions (unusual patterns)
        anomaly_amounts = np.concatenate([
            np.random.uniform(500, 1000, size=5),  # Unusually high amounts
            np.random.uniform(1, 5, size=5)        # Unusually low amounts
        ])
        
        anomaly_frequency = np.concatenate([
            np.random.uniform(50, 100, size=5),    # Very frequent
            np.random.uniform(0.1, 0.5, size=5)    # Very rare
        ])
        
        anomaly_time_of_day = np.concatenate([
            np.random.uniform(2, 4, size=5),       # Late night transactions
            np.random.uniform(23, 24, size=5)      # Very late transactions
        ])
        
        # Combine normal and anomalous data
        amounts = np.concatenate([normal_amounts, anomaly_amounts])
        frequencies = np.concatenate([normal_frequency, anomaly_frequency])
        times = np.concatenate([normal_time_of_day, anomaly_time_of_day])
        
        # Create labels (for reference, but not used in unsupervised detection)
        labels = ['Normal'] * n_normal + ['Anomaly'] * n_anomalies
        
        # Shuffle the data
        indices = np.random.permutation(len(amounts))
        
        df = pd.DataFrame({
            'TransactionID': range(1, len(amounts) + 1),
            'Amount': amounts[indices],
            'Frequency': frequencies[indices],
            'TimeOfDay': times[indices],
            'ActualLabel': [labels[i] for i in indices]  # For reference only
        })
        
        st.success(f"✅ Loaded sample dataset: {len(df)} transactions")
        st.write("**Sample Data Preview:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info("""
        **About this dataset:**
        - 210 credit card transactions (200 normal + 10 anomalies)
        - Features: Amount, Frequency, TimeOfDay
        - Contains hidden anomalies with unusual patterns:
          - Extremely high/low transaction amounts
          - Unusual transaction frequencies
          - Odd time-of-day patterns
        - Perfect for testing anomaly detection algorithms
        - **Note:** ActualLabel column is for reference only (not used by algorithms)
        """)
    
    elif data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader(
            "Upload your data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="anomaly_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Uploaded {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            st.info("👆 Please upload a CSV or Excel file containing data")
            return
    
    if df is None:
        st.info("👆 Please select or upload data to continue")
        return
    
    # Data overview
    st.divider()
    st.subheader("📊 2. Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numeric Columns", len(numeric_cols))
    
    # Feature selection
    st.divider()
    st.subheader("🎯 3. Select Features for Analysis")
    
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found in the dataset. Anomaly detection requires numeric features.")
        return
    
    feature_cols = st.multiselect(
        "Select numeric columns to analyze:",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        help="Choose features that might contain anomalies"
    )
    
    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature to continue")
        return
    
    # Algorithm selection
    st.divider()
    st.subheader("🤖 4. Configure Detection Algorithm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Choose Algorithm:",
            ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"],
            help="Isolation Forest is recommended for most cases"
        )
    
    with col2:
        if algorithm in ["Isolation Forest", "Local Outlier Factor"]:
            contamination = st.slider(
                "Contamination (Expected % of Anomalies)",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Proportion of data expected to be anomalies"
            )
        else:  # One-Class SVM
            contamination = st.slider(
                "Nu (Upper Bound on Outliers)",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Upper bound on the fraction of outliers"
            )
    
    # Run detection button
    if st.button("🚀 Detect Anomalies", type="primary", use_container_width=True):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Anomaly_Detection")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during detection.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status(f"Running {algorithm}...", expanded=True) as status:
                from utils.anomaly_detection import AnomalyDetector
                
                # Progress tracking
                st.write("🔧 Initializing detector...")
                progress_bar = st.progress(0)
                
                progress_bar.progress(0.2)
                
                # Initialize detector
                detector = AnomalyDetector(df)
                detector.set_features(feature_cols)
                
                st.write(f"🔍 Running {algorithm} algorithm...")
                progress_bar.progress(0.5)
                
                # Run selected algorithm
                if algorithm == "Isolation Forest":
                    results = detector.run_isolation_forest(contamination)
                elif algorithm == "Local Outlier Factor":
                    results = detector.run_local_outlier_factor(contamination)
                else:  # One-Class SVM
                    results = detector.run_one_class_svm(nu=contamination)
                
                progress_bar.progress(0.9)
                st.write("💾 Storing results...")
                
                # Store results
                st.session_state.anomaly_detector = detector
                st.session_state.anomaly_results = results
                st.session_state.anomaly_algorithm = algorithm
                
                # Save checkpoint
                pm.save_checkpoint({
                    'completed': True,
                    'algorithm': algorithm,
                    'anomalies_detected': results.get('num_anomalies', len(results.get('anomaly_indices', []))),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
                progress_bar.progress(1.0)
                
                status.update(label="✅ Detection complete!", state="complete", expanded=False)
                st.success(f"✅ Anomaly detection completed using {algorithm}!")
                
        except Exception as e:
            st.error(f"❌ Error running anomaly detection: {str(e)}")
            pm.save_checkpoint({'error': str(e)})
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Always unlock navigation
            pm.unlock()
            st.info("✅ Navigation unlocked - you can now navigate to other pages.")
    
    # Show results if available
    if 'anomaly_results' in st.session_state:
        results = st.session_state.anomaly_results
        detector = st.session_state.anomaly_detector
        algorithm = st.session_state.anomaly_algorithm
        
        # Summary metrics
        st.divider()
        st.subheader("📈 5. Detection Results")
        
        stats = detector.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{stats['total_records']:,}", delta="100%")
        with col2:
            st.metric("Anomalies Detected", f"{stats['num_anomalies']:,}", 
                     delta=f"{stats['pct_anomalies']:.1f}%")
        with col3:
            st.metric("Normal Records", f"{stats['num_normal']:,}",
                     delta=f"{stats['pct_normal']:.1f}%")
        with col4:
            avg_score = results['anomaly_score'].mean()
            # Classify anomaly score
            if abs(avg_score) < 0.3:
                score_status = "Low Anomaly"
            elif abs(avg_score) < 0.6:
                score_status = "Moderate"
            else:
                score_status = "High Anomaly"
            st.metric("Avg Anomaly Score", f"{avg_score:.3f}", delta=score_status)
        
        # Results table
        st.subheader("📋 5. Detailed Results")
        
        show_filter = st.radio(
            "Display:",
            ["All Records", "Anomalies Only", "Normal Only"],
            horizontal=True,
            key="show_filter"
        )
        
        if show_filter == "Anomalies Only":
            display_df = results[results['is_anomaly']]
        elif show_filter == "Normal Only":
            display_df = results[~results['is_anomaly']]
        else:
            display_df = results
        
        st.dataframe(
            display_df[feature_cols + ['anomaly_score', 'is_anomaly', 'anomaly_type']].head(100),
            use_container_width=True
        )
        
        # Visualization
        st.divider()
        st.subheader("📊 6. Visual Analysis")
        
        use_pca = len(feature_cols) > 2
        if use_pca:
            st.info("ℹ️ Using PCA to visualize multi-dimensional data in 2D")
        
        # Only show anomalies for better performance and focus
        fig = detector.create_2d_scatter(use_pca=use_pca, show_only_anomalies=True)
        
        st.info("💡 **Chart shows only anomaly points** - hover over red points to see anomaly scores")
        
        # Display with hover enabled (few points = good performance)
        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False
            }
        )
        
        # Detailed analysis tabs
        st.divider()
        st.subheader("🔍 7. Detailed Analysis")
        
        tab1, tab2 = st.tabs(["Anomaly Profiles", "Feature Importance"])
        
        with tab1:
            st.write("**Top Anomalies Compared to Normal Data:**")
            
            profiles = detector.get_anomaly_profiles(top_n=5)
            
            if len(profiles) > 0:
                for _, row in profiles.iterrows():
                    with st.expander(f"Anomaly #{int(row['Index'])} (Score: {row['Anomaly_Score']:.3f})"):
                        comparison_data = []
                        for col in feature_cols:
                            comparison_data.append({
                                'Feature': col,
                                'Anomaly Value': row[f'{col}_value'],
                                'Normal Mean': row[f'{col}_normal_mean'],
                                'Std Deviations': row[f'{col}_deviation']
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("No anomalies detected")
        
        with tab2:
            if algorithm == "Isolation Forest":
                st.write("**Top 10 Most Important Features for Anomaly Detection:**")
                importance = detector.get_feature_importance()
                
                if importance is not None:
                    # Sort descending and take top 10, then reverse for proper chart display (highest at top)
                    importance_sorted = importance.sort_values('Importance', ascending=False).head(10)
                    importance_sorted = importance_sorted.sort_values('Importance', ascending=True)  # Reverse for chart display
                    
                    fig_importance = px.bar(
                        importance_sorted,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Features Contributing to Anomaly Detection (Highest to Lowest)'
                    )
                    fig_importance.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info(f"Feature importance is only available for Isolation Forest. Current algorithm: {algorithm}")
        
        # AI Explanation section (outside tabs to prevent tab reset on button click)
        st.divider()
        st.subheader("🤖 8. AI-Powered Anomaly Explanation")
        
        # Display saved insights if they exist
        if 'anomaly_ai_insights' in st.session_state:
            st.markdown(st.session_state.anomaly_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Explanation", key="anomaly_ai_btn", type="primary"):
            with st.status("🤖 Analyzing anomalies with AI...", expanded=True) as status:
                try:
                    from utils.ai_helper import AIHelper
                    
                    ai = AIHelper()
                    
                    st.write("Preparing anomaly data...")
                    # Prepare context
                    top_anomalies = results[results['is_anomaly']].nsmallest(5, 'anomaly_score')
                    
                    context = f"""
                    Anomaly Detection Results:
                    - Algorithm: {algorithm}
                    - Total Records: {stats['total_records']}
                    - Anomalies Found: {stats['num_anomalies']} ({stats['pct_anomalies']:.1f}%)
                    - Features Analyzed: {', '.join(feature_cols)}
                    
                    Top 5 Anomalies:
                    {top_anomalies[feature_cols + ['anomaly_score']].to_string()}
                    
                    Normal Data Statistics:
                    {results[~results['is_anomaly']][feature_cols].describe().to_string()}
                    """
                    
                    prompt = f"""
                    You are a data analyst. Analyze these anomaly detection results and provide:
                    
                    1. An explanation of what makes these points anomalous
                    2. Potential business implications or causes
                    3. Recommended actions for each type of anomaly found
                    
                    {context}
                    
                    Provide insights in a clear, business-friendly format with specific examples.
                    """
                    
                    st.write("Generating AI analysis...")
                    # Get AI response
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert data analyst specializing in anomaly detection and business insights."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.anomaly_ai_insights = response.choices[0].message.content
                    status.update(label="✅ AI analysis complete!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="❌ Analysis failed", state="error", expanded=True)
                    st.error(f"Error generating AI explanation: {str(e)}")
            
            # Display results outside status block
            if 'anomaly_ai_insights' in st.session_state and st.session_state.anomaly_ai_insights:
                st.success("✅ AI insights generated successfully!")
                st.markdown(st.session_state.anomaly_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        # Export section
        st.divider()
        st.subheader("📥 9. Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # All data with anomaly flags
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data (CSV)",
                data=csv,
                file_name=f"anomaly_detection_all_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Anomalies only
            anomalies_csv = results[results['is_anomaly']].to_csv(index=False)
            st.download_button(
                label="📥 Download Anomalies Only (CSV)",
                data=anomalies_csv,
                file_name=f"anomalies_only_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Report
            report = f"""
# Anomaly Detection Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Algorithm:** {algorithm}
- **Features:** {', '.join(feature_cols)}
- **Contamination:** {contamination:.2f}

## Results Summary
- **Total Records:** {stats['total_records']:,}
- **Anomalies Detected:** {stats['num_anomalies']:,} ({stats['pct_anomalies']:.1f}%)
- **Normal Records:** {stats['num_normal']:,} ({stats['pct_normal']:.1f}%)

## Top Anomalies
{results[results['is_anomaly']].nsmallest(10, 'anomaly_score')[feature_cols + ['anomaly_score']].to_markdown(index=False)}
"""
            
            # Add AI insights if available
            if 'anomaly_ai_insights' in st.session_state:
                report += f"""

## 🤖 AI-Powered Anomaly Explanation

{st.session_state.anomaly_ai_insights}

"""
            
            report += """
---
*Report generated by DataInsights - Anomaly Detection Module*
"""
            st.download_button(
                label="📥 Download Full Report (Markdown)",
                data=report,
                file_name=f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_time_series_forecasting():
    """Time Series Forecasting & Analysis page."""
    st.markdown("<h2 style='text-align: center;'>📈 Time Series Forecasting</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Time Series Forecasting?"):
        st.markdown("""
        **Time Series Forecasting** predicts future values based on historical time-ordered data.
        
        ### Common Applications:
        - **Sales Forecasting:** Predict future revenue and demand
        - **Stock Price Prediction:** Forecast market trends
        - **Demand Planning:** Inventory and resource optimization
        - **Weather Forecasting:** Temperature and climate prediction
        
        ### Key Concepts:
        **Trend:** Long-term increase or decrease in data  
        **Seasonality:** Regular patterns that repeat over time  
        **Stationarity:** Constant mean and variance over time  
        **ACF/PACF:** Measures of correlation between observations  
        
        ### Models Available:
        **ARIMA (Auto-Regressive Integrated Moving Average)**
        - Statistical model for univariate time series
        - Handles trends and seasonality
        - Auto-tunes parameters for best fit
        
        **Prophet (Facebook's Forecasting Tool)**
        - Robust to missing data and outliers
        - Handles multiple seasonalities
        - Good for business time series with holidays
        """)
    
    # Data source selection
    st.subheader("📤 1. Load Time Series Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Use Sample Data", "Upload Custom Data"]
        default_option = "Use Loaded Dataset"
    else:
        data_options = ["Use Sample Data", "Upload Custom Data"]
        default_option = "Use Sample Data"
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="ts_data_source"
    )
    
    df = None
    
    if data_source == "Use Loaded Dataset":
        st.success("✅ Using dataset from Data Upload section")
        df = st.session_state.data
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
    
    elif data_source == "Use Sample Data":
        st.info("📊 Using sample airline passengers dataset (1949-1960)")
        # Create sample time series data - classic airline passengers dataset
        dates = pd.date_range(start='1949-01', end='1960-12', freq='MS')
        passengers = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                     145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
                     171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
                     196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
                     204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
                     242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
                     284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
                     315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
                     340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
                     360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
                     417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432]
        
        df = pd.DataFrame({
            'Date': dates,
            'Passengers': passengers
        })
        
        st.success(f"✅ Loaded sample dataset: {len(df)} monthly observations")
        st.write("**Sample Data Preview:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info("""
        **About this dataset:**
        - Monthly totals of international airline passengers (1000s)
        - Time period: January 1949 to December 1960
        - Shows clear trend and seasonality
        - Perfect for learning time series forecasting
        """)
    
    elif data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader(
            "Upload your time series data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="ts_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Uploaded {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            st.info("👆 Please upload a CSV or Excel file containing time series data")
            return
    
    if df is None:
        st.info("👆 Please select or upload data to continue")
        return
    
    # Get smart column suggestions
    from utils.column_detector import ColumnDetector
    suggestions = ColumnDetector.get_time_series_column_suggestions(df)
    
    # Validate data suitability
    validation = ColumnDetector.validate_time_series_suitability(df)
    
    # Store validation result
    st.session_state.ts_data_suitable = validation['suitable']
    
    if not validation['suitable']:
        st.error("❌ **Dataset Not Suitable for Time Series Forecasting**")
        for warning in validation['warnings']:
            st.warning(warning)
        st.info("**💡 Recommendations:**")
        for rec in validation['recommendations']:
            st.write(f"- {rec}")
        st.write("**This dataset does not contain time series data.**")
        st.write("Time series forecasting requires sequential data with timestamps.")
        return  # Already has return - good!
    elif len(validation['warnings']) > 0:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for warning in validation['warnings']:
                st.warning(warning)
            if validation['recommendations']:
                st.info("**Recommendations:**")
                for rec in validation['recommendations']:
                    st.write(f"- {rec}")
    else:
        st.success(f"✅ **Dataset looks suitable for Time Series** (Confidence: {validation['confidence']})")
    
    # Column selection
    st.subheader("📊 2. Configure Time Series")
    st.info("💡 **Smart Detection:** Columns are auto-selected based on your data. You can change them if needed.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find index of suggested date column
        date_idx = list(df.columns).index(suggestions['date']) if suggestions['date'] in df.columns else 0
        time_col = st.selectbox(
            "Select Date/Time Column:",
            df.columns,
            index=date_idx,
            help="Column containing dates or timestamps"
        )
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.error("❌ No numeric columns found")
            return
        
        # Find index of suggested value column
        value_idx = numeric_cols.index(suggestions['value']) if suggestions['value'] in numeric_cols else 0
        value_col = st.selectbox(
            "Select Value Column:",
            numeric_cols,
            index=value_idx,
            help="Numeric column to forecast"
        )
    
    if st.button("🔍 Load Time Series", type="primary"):
        try:
            from utils.time_series import TimeSeriesAnalyzer
            
            analyzer = TimeSeriesAnalyzer(df)
            ts_data = analyzer.set_time_column(time_col, value_col)
            
            st.session_state.ts_analyzer = analyzer
            st.session_state.ts_data = ts_data
            
            st.success(f"✅ Loaded time series with {len(ts_data)} observations")
            
            # Show preview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("First Date", ts_data.index[0].strftime('%Y-%m-%d'))
            with col2:
                st.metric("Last Date", ts_data.index[-1].strftime('%Y-%m-%d'))
            with col3:
                st.metric("Observations", len(ts_data))
            
            # Plot time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Time Series'))
            fig.update_layout(title='Time Series Data', xaxis_title='Date', yaxis_title='Value', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading time series: {str(e)}")
    
    # Show analysis if data is loaded
    if 'ts_analyzer' in st.session_state:
        analyzer = st.session_state.ts_analyzer
        ts_data = st.session_state.ts_data
        
        # Analysis tabs
        st.divider()
        st.subheader("🔍 3. Time Series Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Decomposition", "Stationarity Test", "Autocorrelation"])
        
        with tab1:
            if st.button("📊 Decompose Time Series"):
                try:
                    components = analyzer.decompose_time_series()
                    fig = analyzer.create_decomposition_plot(components)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("**Decomposition** splits the time series into trend, seasonal, and residual components")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with tab2:
            if st.button("🔬 Run Stationarity Test"):
                try:
                    results = analyzer.get_stationarity_test()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Statistic", f"{results['test_statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{results['p_value']:.4f}")
                    with col3:
                        status = "✅ Stationary" if results['is_stationary'] else "❌ Non-Stationary"
                        st.metric("Status", status)
                    
                    st.write("**Critical Values:**")
                    for key, value in results['critical_values'].items():
                        st.write(f"- {key}: {value:.4f}")
                    
                    if results['is_stationary']:
                        st.success("The series is stationary (p-value < 0.05). Good for ARIMA modeling!")
                    else:
                        st.warning("The series is non-stationary (p-value >= 0.05). May need differencing.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with tab3:
            if st.button("📉 Calculate ACF/PACF"):
                try:
                    acf_vals, pacf_vals = analyzer.get_autocorrelation()
                    fig = analyzer.create_acf_pacf_plot(acf_vals, pacf_vals)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("**ACF/PACF** help identify appropriate ARIMA parameters (p, d, q)")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Forecasting section
        st.divider()
        st.subheader("🔮 4. Generate Forecasts")
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            forecast_periods = st.slider(
                "Forecast Horizon (periods):",
                min_value=7,
                max_value=365,
                value=30,
                help="Number of periods to forecast into the future"
            )
        
        with col_config2:
            use_seasonal = st.selectbox(
                "Seasonality:",
                ["Auto-detect", "Non-seasonal (Faster)", "Seasonal"],
                help="Non-seasonal is much faster for large datasets"
            )
            # Convert to boolean
            seasonal_param = None if use_seasonal == "Auto-detect" else (use_seasonal == "Seasonal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Run ARIMA Forecast", use_container_width=True):
                from utils.process_manager import ProcessManager
                
                pm = ProcessManager("ARIMA_Forecast")
                pm.lock()
                
                # Show warning BEFORE spinner to prevent text cutoff
                st.warning("⚠️ **Important:** Navigation locked during forecasting. Please do not navigate away.")
                st.info("💡 ARIMA training may take 30-60 seconds for large datasets. Please wait...")
                
                try:
                    with st.status("Running Auto-ARIMA...", expanded=True) as status:
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Training ARIMA model (this may take a minute)...")
                        progress_bar.progress(0.5)
                        
                        # Check dataset size and warn if too large
                        ts_len = len(analyzer.ts_data)
                        if ts_len > 500:
                            st.info(f"ℹ️ Large dataset ({ts_len} observations). Using last 500 for training to prevent timeouts.")
                        
                        # Run ARIMA with optimized parameters
                        results = analyzer.run_auto_arima(forecast_periods, seasonal=seasonal_param)
                        st.session_state.arima_results = results
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ ARIMA complete!")
                        
                        st.success("✅ ARIMA model trained!")
                        
                        st.write("**Model Configuration:**")
                        st.write(f"- Order (p,d,q): {results['model_order']}")
                        st.write(f"- Seasonal Order: {results['seasonal_order']}")
                        st.write(f"- AIC: {results['aic']:.2f}")
                        st.write(f"- BIC: {results['bic']:.2f}")
                        
                        fig = analyzer.create_forecast_plot('arima')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(results['forecast'].head(10), use_container_width=True)
                        
                        pm.save_checkpoint({'completed': True, 'model': 'ARIMA'})
                        
                except Exception as e:
                    st.error(f"❌ ARIMA Error: {str(e)}")
                    
                    # Common ARIMA failure reasons
                    error_str = str(e).lower()
                    ts_len = len(analyzer.ts_data)
                    
                    if 'non-stationary' in error_str or 'stationary' in error_str:
                        st.warning("⚠️ **Data may be non-stationary.** Try differencing the data first.")
                    elif 'constant' in error_str or 'variance' in error_str:
                        st.warning("⚠️ **Data has constant values or zero variance.** ARIMA requires variation in the data.")
                    elif 'too few' in error_str or 'observations' in error_str:
                        st.warning("⚠️ **Not enough data points.** ARIMA requires at least 30-50 observations.")
                    elif 'timeout' in error_str or 'memory' in error_str:
                        st.warning("⚠️ **Resource limit exceeded.** Try 'Non-seasonal (Faster)' option or reduce dataset size.")
                    else:
                        st.info("💡 **Troubleshooting Tips:**")
                        st.write("- Try **Non-seasonal (Faster)** option")
                        st.write("- Reduce forecast horizon")
                        st.write("- Check if data has trends and variation")
                        if ts_len > 500:
                            st.warning(f"⚠️ Large dataset ({ts_len} observations) may exceed cloud limits. Algorithm is already optimized but very large datasets may still timeout.")
                    
                    # Show full traceback for debugging
                    with st.expander("🔍 Full Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    
                    pm.save_checkpoint({'error': str(e)})
                finally:
                    pm.unlock()
                    st.info("✅ Navigation unlocked")
        
        with col2:
            if st.button("📊 Run Prophet Forecast", use_container_width=True):
                from utils.process_manager import ProcessManager
                
                pm = ProcessManager("Prophet_Forecast")
                pm.lock()
                
                # Show warning BEFORE spinner to prevent text cutoff
                st.warning("⚠️ **Important:** Navigation locked during forecasting. Please do not navigate away.")
                
                try:
                    with st.status("Running Prophet...", expanded=True) as status:
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Training Prophet model...")
                        progress_bar.progress(0.5)
                        
                        results = analyzer.run_prophet(forecast_periods)
                        st.session_state.prophet_results = results
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Prophet complete!")
                        
                        st.success("✅ Prophet model trained!")
                        
                        fig = analyzer.create_forecast_plot('prophet')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10),
                                   use_container_width=True)
                        
                        pm.save_checkpoint({'completed': True, 'model': 'Prophet'})
                        
                except Exception as e:
                    st.error(f"❌ Prophet Error: {str(e)}")
                    
                    # Common Prophet failure reasons
                    error_str = str(e).lower()
                    if 'dataframe' in error_str or 'ds' in error_str or 'y' in error_str:
                        st.warning("⚠️ **Data format issue.** Prophet requires columns 'ds' (dates) and 'y' (values).")
                    elif 'date' in error_str or 'datetime' in error_str:
                        st.warning("⚠️ **Date parsing error.** Ensure your date column has valid datetime values.")
                    elif 'inf' in error_str or 'nan' in error_str:
                        st.warning("⚠️ **Data contains infinity or NaN values.** Clean your data before forecasting.")
                    else:
                        st.info("💡 **Troubleshooting:** Ensure your data has valid dates and numeric values with no gaps.")
                    
                    # Show full traceback for debugging
                    with st.expander("🔍 Full Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                    
                    pm.save_checkpoint({'error': str(e)})
                finally:
                    pm.unlock()
                    st.info("✅ Navigation unlocked")
        
        # Show persistent results for each model
        if 'arima_results' in st.session_state or 'prophet_results' in st.session_state:
            st.divider()
            st.subheader("📊 Forecast Results")
            
            # Show ARIMA results if available
            if 'arima_results' in st.session_state:
                with st.expander("🤖 ARIMA Model Results", expanded=True):
                    arima_res = st.session_state.arima_results
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model Order", f"{arima_res['model_order']}")
                    with col2:
                        st.metric("AIC", f"{arima_res['aic']:.2f}")
                    with col3:
                        st.metric("BIC", f"{arima_res['bic']:.2f}")
                    with col4:
                        st.metric("Seasonal", f"{arima_res['seasonal_order']}")
                    
                    fig = analyzer.create_forecast_plot('arima')
                    st.plotly_chart(fig, use_container_width=True, key="arima_persistent_plot")
                    
                    st.write("**Forecast Data (First 10 periods):**")
                    st.dataframe(arima_res['forecast'].head(10), use_container_width=True)
            
            # Show Prophet results if available
            if 'prophet_results' in st.session_state:
                with st.expander("📈 Prophet Model Results", expanded=True):
                    prophet_res = st.session_state.prophet_results
                    
                    st.write("**Prophet Model - Automatic Trend & Seasonality Detection**")
                    
                    fig = analyzer.create_forecast_plot('prophet')
                    st.plotly_chart(fig, use_container_width=True, key="prophet_persistent_plot")
                    
                    st.write("**Forecast Data (First 10 periods):**")
                    st.dataframe(prophet_res['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10),
                               use_container_width=True)
            
            # Show comparison plot only if both models exist
            if 'arima_results' in st.session_state and 'prophet_results' in st.session_state:
                st.divider()
                st.subheader("⚖️ Model Comparison")
                
                fig = analyzer.create_comparison_plot()
                st.plotly_chart(fig, use_container_width=True, key="comparison_plot")
            
            # AI Insights section (available if either model has been run)
            st.divider()
            st.subheader("🤖 5. AI-Powered Insights")
            
            # Display saved insights if they exist
            if 'ts_ai_insights' in st.session_state:
                st.markdown(st.session_state.ts_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
            
            if st.button("🤖 Generate AI Insights", key="ts_ai_insights_btn"):
                with st.status("Analyzing forecasts...", expanded=True) as status:
                    try:
                        from utils.ai_helper import AIHelper
                        ai = AIHelper()
                        
                        # Build context based on available models
                        forecast_summary = f"- Forecast Horizon: {forecast_periods} periods\n"
                        if 'arima_results' in st.session_state:
                            forecast_summary += f"                        - ARIMA Forecast Mean: {st.session_state.arima_results['forecast']['forecast'].mean():.2f}\n"
                        if 'prophet_results' in st.session_state:
                            forecast_summary += f"                        - Prophet Forecast Mean: {st.session_state.prophet_results['forecast']['yhat'].mean():.2f}\n"
                        
                        context = f"""
                        Time Series Analysis:
                        - Time Period: {ts_data.index[0]} to {ts_data.index[-1]}
                        - Observations: {len(ts_data)}
                        - Mean: {ts_data.mean():.2f}
                        - Std Dev: {ts_data.std():.2f}
                        - Trend: {"Increasing" if ts_data.iloc[-10:].mean() > ts_data.iloc[:10].mean() else "Decreasing"}
                        
                        Forecast Summary:
{forecast_summary}                        """
                        
                        model_comparison = ""
                        if 'arima_results' in st.session_state and 'prophet_results' in st.session_state:
                            model_comparison = "2. Which model (ARIMA or Prophet) appears more reliable and why\n                        "
                        
                        prompt = f"""
                        As a business analyst, analyze this time series forecast and provide:
                        1. Interpretation of the forecast trends
                        {model_comparison}3. Business recommendations based on the forecast
                        4. Potential risks or opportunities identified
                        
                        {context}
                        
                        Provide actionable insights in business-friendly language.
                        """
                        
                        response = ai.client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are an expert business analyst specializing in time series forecasting."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        
                        # Save to session state
                        st.session_state.ts_ai_insights = response.choices[0].message.content
                        st.success("✅ AI insights generated successfully!")
                        st.markdown(st.session_state.ts_ai_insights)
                        st.info("✅ AI insights saved! These will be included in your report downloads.")
                        
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
        
        # Export section
        if 'arima_results' in st.session_state or 'prophet_results' in st.session_state:
            st.divider()
            st.subheader("📥 6. Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'arima_results' in st.session_state:
                    forecast_df = st.session_state.arima_results['forecast']
                    csv = forecast_df.to_csv()
                    st.download_button(
                        label="📥 Download ARIMA Forecast (CSV)",
                        data=csv,
                        file_name=f"arima_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if 'prophet_results' in st.session_state:
                    forecast_df = st.session_state.prophet_results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prophet Forecast (CSV)",
                        data=csv,
                        file_name=f"prophet_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Generate full report
                report = f"""# Time Series Forecasting Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
                
                # Add ARIMA results
                if 'arima_results' in st.session_state:
                    arima_res = st.session_state.arima_results
                    report += f"""## ARIMA Model Results

- **Model Order:** {arima_res.get('model_order', 'N/A')}
- **AIC:** {arima_res.get('aic', 0):.2f}
- **BIC:** {arima_res.get('bic', 0):.2f}
- **Forecast Periods:** {len(arima_res['forecast'])}

### ARIMA Forecast Summary
- **Mean Forecast:** {arima_res['forecast']['forecast'].mean():.2f}
- **Min Forecast:** {arima_res['forecast']['forecast'].min():.2f}
- **Max Forecast:** {arima_res['forecast']['forecast'].max():.2f}

"""
                
                # Add Prophet results
                if 'prophet_results' in st.session_state:
                    prophet_res = st.session_state.prophet_results
                    report += f"""## Prophet Model Results

- **Automatic Seasonality Detection:** Enabled
- **Trend and Holiday Effects:** Included
- **Forecast Periods:** {len(prophet_res['forecast'])}

### Prophet Forecast Summary
- **Mean Forecast:** {prophet_res['forecast']['yhat'].mean():.2f}
- **Min Forecast:** {prophet_res['forecast']['yhat'].min():.2f}
- **Max Forecast:** {prophet_res['forecast']['yhat'].max():.2f}

"""
                
                # Add AI insights if available
                if 'ts_ai_insights' in st.session_state:
                    report += f"""## 🤖 AI-Powered Forecast Insights

{st.session_state.ts_ai_insights}

"""
                
                report += """## Business Applications

Use these forecasts for:
- **Demand Planning:** Predict future product demand
- **Resource Allocation:** Optimize staffing and inventory
- **Budget Forecasting:** Plan financial resources
- **Capacity Planning:** Anticipate infrastructure needs

---
*Report generated by DataInsights - Time Series Forecasting Module*
"""
                
                st.download_button(
                    label="📥 Download Full Report (Markdown)",
                    data=report,
                    file_name=f"timeseries_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

def show_text_mining():
    """Text Mining & Sentiment Analysis page."""
    st.markdown("<h2 style='text-align: center;'>💬 Text Mining & Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Text Mining?"):
        st.markdown("""
        **Text Mining** extracts insights from unstructured text data using Natural Language Processing (NLP).
        
        ### Common Applications:
        - **Customer Feedback:** Analyze reviews, surveys, social media
        - **Sentiment Analysis:** Measure positive/negative opinions
        - **Topic Discovery:** Find themes in large document collections
        - **Brand Monitoring:** Track mentions and sentiment
        
        ### Features Available:
        **Sentiment Analysis (VADER)**
        - Detects positive, negative, neutral sentiment
        - Industry-standard for social media text
        - Provides detailed sentiment scores
        
        **Word Frequency Analysis**
        - Most common words and phrases
        - Word cloud visualization
        - Filters stopwords automatically
        
        **Topic Modeling (LDA)**
        - Discovers hidden themes in text
        - Groups similar documents
        - Identifies key topics
        """)
    
    # Data source selection
    st.subheader("📤 1. Load Text Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Use Sample Data", "Upload Custom Data"]
        default_option = "Use Loaded Dataset"
    else:
        data_options = ["Use Sample Data", "Upload Custom Data"]
        default_option = "Use Sample Data"
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="text_data_source"
    )
    
    df = None
    
    if data_source == "Use Loaded Dataset":
        st.success("✅ Using dataset from Data Upload section")
        df = st.session_state.data
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
    
    elif data_source == "Use Sample Data":
        st.info("📊 Using sample product reviews dataset")
        # Create sample text data - product reviews
        sample_reviews = [
            "This product is absolutely amazing! Best purchase I've made this year.",
            "Terrible quality, broke after just one week. Very disappointed.",
            "Good value for money. Works as expected, no complaints.",
            "Outstanding customer service and fast delivery. Highly recommended!",
            "Not worth the price. Found better alternatives for less money.",
            "Exactly what I needed. Great product, will buy again.",
            "Poor packaging, item arrived damaged. Refund requested.",
            "Exceeded my expectations! Love the design and functionality.",
            "Average product, nothing special but does the job.",
            "Fantastic! My family loves it. Best gift ever!",
            "Complete waste of money. Save yourself the hassle.",
            "Decent product but shipping took forever. Would prefer faster delivery.",
            "Incredible quality and attention to detail. Worth every penny!",
            "Misleading description. Product doesn't match the photos.",
            "Pretty good overall. Minor issues but nothing major.",
            "Absolutely love it! Can't imagine life without it now.",
            "Returned it immediately. Not as advertised.",
            "Solid product for the price. Recommended for beginners.",
            "Horrible experience. Customer support was unhelpful.",
            "Best in its category! Superior to all competitors.",
            "Okay for occasional use but not for daily tasks.",
            "Exceptional build quality and performance. Five stars!",
            "Disappointed with the features. Expected more functionality.",
            "Perfect fit and finish. Looks great in my home!",
            "Unreliable and inconsistent. Wouldn't buy again.",
            "Great starter product. Good for learning the basics.",
            "Premium quality at an affordable price. Impressive!",
            "Functions well but design could be improved.",
            "Absolutely horrible. One of my worst purchases ever.",
            "Satisfactory product. Met my basic requirements.",
            "Brilliant innovation! Revolutionary in its field.",
            "Cheap materials, feels flimsy. Not durable at all.",
            "Very pleased with this purchase. Smooth transaction.",
            "Awful quality control. Multiple defects found.",
            "Excellent performance and reliability. Highly satisfied!",
            "Mediocre at best. Nothing to write home about.",
            "Superb craftsmanship and elegant design. Love it!",
            "Not user-friendly. Instructions are confusing.",
            "Top-notch quality! Exceeds industry standards.",
            "Regret buying this. Should have read reviews first.",
            "Nice features but overpriced for what you get.",
            "Phenomenal product! Changed my daily routine for better.",
            "Broke within days. Very poor construction quality.",
            "Reliable and consistent. Does exactly what it promises.",
            "Worst customer service experience ever. Avoid!",
            "Beautiful aesthetics and great functionality combined.",
            "Subpar performance. Many better options available.",
            "Amazing value! Can't believe the quality at this price.",
            "Frustrating to use. Too many unnecessary complications.",
            "Impressive technology and innovative features. Recommended!"
        ]
        
        df = pd.DataFrame({
            'Review': sample_reviews,
            'ReviewID': range(1, len(sample_reviews) + 1)
        })
        
        st.success(f"✅ Loaded sample dataset: {len(df)} product reviews")
        st.write("**Sample Data Preview:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info("""
        **About this dataset:**
        - 50 product reviews with mixed sentiments
        - Contains positive, negative, and neutral opinions
        - Perfect for learning sentiment analysis and text mining
        - Includes various customer feedback patterns
        """)
    
    elif data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader(
            "Upload your text data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="text_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Uploaded {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            st.info("👆 Please upload a CSV or Excel file containing text data")
            return
    
    if df is None:
        st.info("👆 Please select or upload data to continue")
        return
    
    # Column selection with smart detection
    st.subheader("📝 2. Select Text Column")
    
    from utils.column_detector import ColumnDetector
    suggested_col = ColumnDetector.detect_text_column(df)
    
    st.info("💡 **Smart Detection:** Column is auto-selected based on your data. You can change it if needed.")
    
    # Find index of suggested column
    col_index = list(df.columns).index(suggested_col) if suggested_col in df.columns else 0
    
    text_col = st.selectbox(
        "Choose column containing text:",
        df.columns,
        index=col_index,
        help="Select the column with text data to analyze"
    )
    
    if st.button("🔍 Load Text Data", type="primary"):
        try:
            from utils.text_mining import TextAnalyzer
            
            # Initialize analyzer
            analyzer = TextAnalyzer(df[text_col])
            st.session_state.text_analyzer = analyzer
            
            st.success(f"✅ Loaded {len(analyzer.text_series)} text entries!")
            
            # Show preview
            st.write("**Sample Text:**")
            st.text(analyzer.text_series.iloc[0][:300] + "..." if len(analyzer.text_series.iloc[0]) > 300 else analyzer.text_series.iloc[0])
            
        except Exception as e:
            st.error(f"Error loading text: {str(e)}")
    
    # Show analysis if data is loaded
    if 'text_analyzer' in st.session_state:
        analyzer = st.session_state.text_analyzer
        
        # Analysis tabs
        st.divider()
        st.subheader("🔍 3. Text Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Word Frequency", "Topic Modeling", "AI Summary"])
        
        with tab1:
            st.write("**Sentiment Analysis using VADER:**")
            
            # Only show button if results don't exist
            if 'sentiment_results' not in st.session_state:
                if st.button("📊 Analyze Sentiment", key="sentiment_btn"):
                    from utils.process_manager import ProcessManager
                    
                    pm = ProcessManager("Sentiment_Analysis")
                    pm.lock()
                    
                    # Show warning BEFORE spinner to prevent text cutoff
                    st.warning("⚠️ **Important:** Navigation locked during sentiment analysis. Please do not navigate away.")
                    
                    try:
                        with st.status("Analyzing sentiment...", expanded=True) as status:
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Analyzing sentiment...")
                            progress_bar.progress(0.5)
                            
                            sentiment_df = analyzer.get_sentiment_analysis()
                            st.session_state.sentiment_results = sentiment_df
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Sentiment analysis complete!")
                            
                            pm.save_checkpoint({'completed': True, 'texts_analyzed': len(sentiment_df)})
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        pm.save_checkpoint({'error': str(e)})
                    finally:
                        pm.unlock()
                        st.info("✅ Navigation unlocked")
                        st.rerun()
            else:
                if st.button("🔄 Re-analyze Sentiment", key="sentiment_reanalyze"):
                    del st.session_state.sentiment_results
                    st.rerun()
            
            # Show results if available
            if 'sentiment_results' in st.session_state:
                sentiment_df = st.session_state.sentiment_results
                
                # Summary metrics
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                total = len(sentiment_df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Texts", total)
                with col2:
                    pos_pct = (sentiment_counts.get('Positive', 0) / total) * 100
                    st.metric("Positive", f"{pos_pct:.1f}%")
                with col3:
                    neg_pct = (sentiment_counts.get('Negative', 0) / total) * 100
                    st.metric("Negative", f"{neg_pct:.1f}%")
                with col4:
                    neu_pct = (sentiment_counts.get('Neutral', 0) / total) * 100
                    st.metric("Neutral", f"{neu_pct:.1f}%")
                
                # Visualization
                fig = analyzer.create_sentiment_plot(sentiment_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show results table
                st.write("**Detailed Results:**")
                st.dataframe(sentiment_df.head(20), use_container_width=True)
        
        with tab2:
            st.write("**Word Frequency Analysis:**")
            
            n_words = st.slider("Number of top words:", 10, 100, 50)
            
            if 'word_freq_results' not in st.session_state:
                if st.button("📈 Analyze Word Frequency", key="wordfreq_btn"):
                    with st.status("Analyzing word frequency...", expanded=True) as status:
                        try:
                            word_freq_df = analyzer.get_word_frequency(n_words)
                            st.session_state.word_freq_results = word_freq_df
                            st.session_state.n_words = n_words
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                if st.button("🔄 Re-analyze Word Frequency", key="wordfreq_reanalyze"):
                    del st.session_state.word_freq_results
                    st.rerun()
            
            # Show results if available
            if 'word_freq_results' in st.session_state:
                word_freq_df = st.session_state.word_freq_results
                
                # Bar chart
                fig = analyzer.create_word_frequency_plot(word_freq_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Word cloud
                st.write("**Word Cloud:**")
                wordcloud = analyzer.create_wordcloud(max_words=100)
                
                import matplotlib.pyplot as plt
                fig_wc, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
        
        with tab3:
            st.write("**Topic Modeling (LDA):**")
            
            num_topics = st.slider("Number of topics:", 2, 10, 5)
            
            if 'topics' not in st.session_state:
                if st.button("🔎 Discover Topics", key="topics_btn"):
                    from utils.process_manager import ProcessManager
                    
                    pm = ProcessManager("Topic_Modeling")
                    pm.lock()
                    
                    # Show warning BEFORE spinner to prevent text cutoff
                    st.warning("⚠️ **Important:** Navigation locked during topic modeling. Please do not navigate away.")
                    
                    try:
                        with st.status("Running topic modeling...", expanded=True) as status:
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text(f"Discovering {num_topics} topics...")
                            progress_bar.progress(0.5)
                            
                            topics = analyzer.get_topic_modeling(num_topics, n_words=10)
                            st.session_state.topics = topics
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Topic modeling complete!")
                            
                            pm.save_checkpoint({'completed': True, 'num_topics': num_topics})
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        pm.save_checkpoint({'error': str(e)})
                    finally:
                        pm.unlock()
                        st.info("✅ Navigation unlocked")
                        st.rerun()
            else:
                if st.button("🔄 Re-discover Topics", key="topics_reanalyze"):
                    del st.session_state.topics
                    st.rerun()
            
            # Show results if available
            if 'topics' in st.session_state:
                topics = st.session_state.topics
                
                st.write(f"**Discovered {len(topics)} Topics:**")
                
                for topic_id, words in topics.items():
                    with st.expander(f"Topic {topic_id + 1}"):
                        st.write("**Top Words:**")
                        st.write(", ".join(words))
        
        with tab4:
            st.write("**AI-Powered Text Summary:**")
            
            # Display saved insights if they exist
            if 'text_ai_insights' in st.session_state:
                st.markdown(st.session_state.text_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
            
            if st.button("🤖 Generate AI Summary", key="text_ai_summary_btn"):
                with st.status("Generating AI summary...", expanded=True) as status:
                    try:
                        from utils.ai_helper import AIHelper
                        ai = AIHelper()
                        
                        # Gather context
                        context = f"""
                        Text Analysis Summary:
                        - Total Texts: {len(analyzer.text_series)}
                        - Sample Text: {analyzer.text_series.iloc[0][:200]}...
                        """
                        
                        if 'sentiment_results' in st.session_state:
                            sentiment_df = st.session_state.sentiment_results
                            sentiment_counts = sentiment_df['sentiment'].value_counts()
                            context += f"\n\nSentiment Distribution:\n{sentiment_counts.to_string()}"
                        
                        if 'word_freq_results' in st.session_state:
                            word_freq_df = st.session_state.word_freq_results
                            context += f"\n\nTop 5 Words:\n{word_freq_df.head(5).to_string(index=False)}"
                        
                        if 'topics' in st.session_state:
                            topics = st.session_state.topics
                            context += f"\n\nDiscovered Topics:\n"
                            for topic_id, words in list(topics.items())[:3]:
                                context += f"- Topic {topic_id + 1}: {', '.join(words[:5])}\n"
                        
                        prompt = f"""
                        As a business analyst, analyze this text data and provide:
                        1. Executive summary of the text content
                        2. Key positive and negative themes identified
                        3. Actionable business recommendations
                        4. Potential areas of concern or opportunity
                        
                        {context}
                        
                        Provide clear, business-friendly insights.
                        """
                        
                        response = ai.client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are an expert business analyst specializing in text analytics and sentiment analysis."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        
                        # Save to session state
                        st.session_state.text_ai_insights = response.choices[0].message.content
                        st.success("✅ AI insights generated successfully!")
                        st.markdown(st.session_state.text_ai_insights)
                        st.info("✅ AI insights saved! These will be included in your report downloads.")
                        
                    except Exception as e:
                        st.error(f"Error generating AI summary: {str(e)}")
        
        # Export section
        if 'sentiment_results' in st.session_state or 'word_freq_results' in st.session_state:
            st.divider()
            st.subheader("📥 3. Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'sentiment_results' in st.session_state:
                    sentiment_csv = st.session_state.sentiment_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Sentiment Results (CSV)",
                        data=sentiment_csv,
                        file_name=f"sentiment_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if 'word_freq_results' in st.session_state:
                    word_freq_csv = st.session_state.word_freq_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Word Frequency (CSV)",
                        data=word_freq_csv,
                        file_name=f"word_frequency_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Generate full report
                report = f"""# Text Mining & NLP Analysis Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
                
                # Add sentiment analysis section
                if 'sentiment_results' in st.session_state:
                    sentiment_df = st.session_state.sentiment_results
                    if not sentiment_df.empty:
                        positive = (sentiment_df['sentiment'] == 'positive').sum()
                        negative = (sentiment_df['sentiment'] == 'negative').sum()
                        neutral = (sentiment_df['sentiment'] == 'neutral').sum()
                        total = len(sentiment_df)
                        
                        report += f"""## Sentiment Analysis Results

- **Total Texts Analyzed:** {total:,}
- **Positive:** {positive:,} ({positive/total*100:.1f}%)
- **Negative:** {negative:,} ({negative/total*100:.1f}%)
- **Neutral:** {neutral:,} ({neutral/total*100:.1f}%)

### Sentiment Distribution
The analysis reveals the overall sentiment tone of the text corpus.

"""
                
                # Add word frequency section
                if 'word_freq_results' in st.session_state:
                    word_freq_df = st.session_state.word_freq_results
                    if not word_freq_df.empty:
                        n_words = st.session_state.get('n_words', 20)
                        report += f"""## Word Frequency Analysis

- **Top {n_words} Words Analyzed**
- **Most Frequent Word:** {word_freq_df.iloc[0]['word']} ({word_freq_df.iloc[0]['frequency']:,} occurrences)

### Top 10 Most Frequent Words
{word_freq_df.head(10).to_markdown(index=False)}

"""
                
                # Add topic modeling section if available
                if 'topics' in st.session_state:
                    topics = st.session_state.topics
                    if topics:
                        report += f"""## Topic Modeling Results

- **Topics Discovered:** {len(topics)}

"""
                        for topic_id, words in list(topics.items())[:5]:
                            report += f"**Topic {topic_id}:** {', '.join(words)}\n\n"
                
                # Add AI insights if available
                if 'text_ai_insights' in st.session_state:
                    report += f"""## 🤖 AI-Powered Text Summary

{st.session_state.text_ai_insights}

"""
                
                report += """## Business Applications

This text mining analysis can be used for:
- **Customer Feedback Analysis:** Understand customer sentiment and concerns
- **Brand Sentiment Monitoring:** Track how customers perceive your brand
- **Content Categorization:** Organize large text collections automatically
- **Trend Identification:** Discover emerging topics and themes

---
*Report generated by DataInsights - Text Mining & NLP Module*
"""
                
                st.download_button(
                    label="📥 Download Full Report (Markdown)",
                    data=report,
                    file_name=f"textmining_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

def show_ab_testing():
    """A/B Testing page."""
    st.markdown("<h2 style='text-align: center;'>🧪 A/B Testing & Statistical Hypothesis Testing</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is A/B Testing?"):
        st.markdown("""
        **A/B Testing** is a statistical method for comparing two versions to determine which performs better.
        
        ### Key Tests Available:
        
        - **Proportion Test (Z-test):** Compare conversion rates, click-through rates, success rates
          - Example: Website variant A (5% conversion) vs. variant B (6% conversion)
        
        - **T-Test:** Compare means between two groups
          - Example: Average order value for two different pricing strategies
        
        - **Chi-Square Test:** Test independence between categorical variables
          - Example: Relationship between email template and click behavior
        
        ### Business Applications:
        - 🌐 **Website Optimization:** Landing page variants, button colors, layouts
        - 📧 **Email Marketing:** Subject lines, content variations
        - 💰 **Pricing:** Test different price points
        - 🎯 **Advertising:** Ad copy variations, targeting strategies
        
        ### Sample Ratio Mismatch (SRM) Detection:
        
        **What is SRM?** A data quality check that detects when your test groups are not split as expected (e.g., 50/50).
        
        **Why it matters:** An unexpected sample ratio often indicates:
        - 🐛 Implementation bugs in randomization code
        - 📊 Data collection/logging issues
        - ⚠️ Selection bias in how users entered the test
        
        **How we detect it:** Chi-square test comparing observed vs. expected group sizes (p < 0.01 = problem)
        
        **What to do:** If SRM is detected, **investigate before trusting results**. The test may be invalid.
        """)
    
    st.markdown("""
    Run statistical tests to validate your experiments and make data-driven decisions.
    """)
    
    # Import utilities
    from utils.ab_testing import ABTestAnalyzer
    
    # Initialize analyzer in session state
    if 'ab_analyzer' not in st.session_state:
        st.session_state.ab_analyzer = ABTestAnalyzer()
    
    analyzer = st.session_state.ab_analyzer
    
    # Data source selection
    st.subheader("📤 1. Load Test Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample A/B Test Data", "Upload Custom Data", "Manual Calculator"]
        default_option = "Use Loaded Dataset"
    else:
        data_options = ["Sample A/B Test Data", "Upload Custom Data", "Manual Calculator"]
        default_option = "Sample A/B Test Data"
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="ab_data_source"
    )
    
    # Use Loaded Dataset
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_ab_testing_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select columns for your A/B test")
        
        col1, col2 = st.columns(2)
        with col1:
            # Get suggested group column
            group_default = suggestions['group']
            group_idx = list(df.columns).index(group_default)
            
            group_col = st.selectbox(
                "Group Column (A/B variant):",
                df.columns,
                index=group_idx,
                key="ab_group_col",
                help="Column that identifies control vs treatment"
            )
            
            # Real-time validation
            if group_col:
                n_groups = df[group_col].nunique()
                groups = df[group_col].unique()
                
                issues = []
                warnings = []
                recommendations = []
                
                # Check 1: Must have exactly 2 groups (CRITICAL for > 10, WARNING for 3-10)
                if n_groups < 2:
                    issues.append(f"❌ Only {n_groups} group found. A/B testing requires exactly 2 groups")
                    recommendations.append("Select a column with control and treatment groups")
                elif n_groups > 10:
                    issues.append(f"❌ {n_groups:,} groups found. This will cause errors - A/B testing requires 2 groups")
                    recommendations.append("This column is not suitable for A/B testing. Select a different column with 2-3 groups")
                elif n_groups > 2:
                    warnings.append(f"⚠️ {n_groups} groups found. Standard A/B testing uses 2 groups")
                    recommendations.append("Consider: Filter to 2 groups or use multi-variant testing")
                
                # Check 2: Group size balance
                if n_groups == 2:
                    group_counts = df[group_col].value_counts()
                    group_sizes = group_counts.values
                    imbalance_ratio = max(group_sizes) / min(group_sizes) if min(group_sizes) > 0 else float('inf')
                    
                    if imbalance_ratio > 10:
                        warnings.append(f"⚠️ Severe group imbalance: {imbalance_ratio:.1f}:1 ratio")
                        recommendations.append("Imbalanced groups may reduce statistical power")
                    elif imbalance_ratio > 3:
                        warnings.append(f"⚠️ Group imbalance: {imbalance_ratio:.1f}:1 ratio")
                
                # Check 3: Sample size per group
                if n_groups == 2:
                    min_group_size = group_counts.min()
                    if min_group_size < 30:
                        warnings.append(f"⚠️ Smallest group has only {min_group_size} samples (recommend 100+ per group)")
                        recommendations.append("Small samples may not detect small effect sizes")
                
                # Display validation results
                data_compatible = len(issues) == 0
                
                # Store validation state in session state
                st.session_state.ab_data_compatible = data_compatible
                st.session_state.ab_issues = issues
                st.session_state.ab_warnings = warnings
                
                if len(issues) > 0:
                    st.error("**🚨 NOT SUITABLE FOR A/B TESTING**")
                    for issue in issues:
                        st.write(issue)
                    if recommendations:
                        st.info("**💡 Recommendations:**")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                elif len(warnings) > 0:
                    st.warning("**⚠️ A/B TESTING POSSIBLE (with warnings)**")
                    for warning in warnings:
                        st.write(warning)
                    if recommendations:
                        with st.expander("💡 Recommendations"):
                            for rec in recommendations:
                                st.write(f"• {rec}")
                else:
                    st.success("**✅ EXCELLENT FOR A/B TESTING**")
                    st.write(f"✓ {n_groups} groups: {list(groups)}")
                
                st.caption(f"🔍 Groups: {list(groups[:5])}")
        
        with col2:
            # Get suggested metric column
            metric_default = suggestions['metric']
            metric_idx = list(df.columns).index(metric_default)
            
            metric_col = st.selectbox(
                "Metric Column:",
                df.columns,
                index=metric_idx,
                key="ab_metric_col",
                help="Numeric column to compare (e.g., 'conversion', 'revenue', 'clicks')"
            )
        
        # Validate group column has exactly 2 groups
        # Check if data is compatible (no critical issues)
        button_disabled = not st.session_state.get('ab_data_compatible', True)
        
        if st.button("📊 Validate & Process Data", type="primary", disabled=button_disabled):
            groups = df[group_col].unique()
            
            if len(groups) != 2:
                st.error(f"""
                ❌ **Invalid Group Column**
                
                Group column must have exactly 2 unique values for A/B testing.
                Found: {len(groups)} unique values: {list(groups)}
                
                **Please select a column with 2 groups** (e.g., 'A' and 'B', 'control' and 'treatment')
                """)
                st.stop()
            
            # Check if metric is numeric
            if not pd.api.types.is_numeric_dtype(df[metric_col]):
                st.error(f"""
                ❌ **Invalid Metric Column**
                
                Metric column '{metric_col}' must be numeric!
                
                **Please select a numeric column**
                """)
                st.stop()
            
            # Store processed data
            st.session_state.ab_test_data = df[[group_col, metric_col]].copy()
            st.session_state.ab_test_groups = {
                'control': groups[0],
                'treatment': groups[1],
                'group_col': group_col,
                'metric_col': metric_col
            }
            
            # Check for Sample Ratio Mismatch (SRM)
            test_data = st.session_state.ab_test_data
            control_size = len(test_data[test_data[group_col] == groups[0]])
            treatment_size = len(test_data[test_data[group_col] == groups[1]])
            total = control_size + treatment_size
            
            # Chi-square test for 50/50 split
            expected_size = total / 2
            chi_square = ((control_size - expected_size)**2 / expected_size + 
                         (treatment_size - expected_size)**2 / expected_size)
            
            # Calculate p-value from chi-square distribution (df=1)
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi_square, df=1)
            
            # Store SRM results
            st.session_state.ab_srm_check = {
                'control_size': control_size,
                'treatment_size': treatment_size,
                'expected_ratio': 0.5,
                'chi_square': chi_square,
                'p_value': p_value,
                'has_srm': p_value < 0.01
            }
            
            st.success("✅ Data validated and ready for A/B testing!")
            st.info(f"**Groups:** {groups[0]} vs {groups[1]} | **Metric:** {metric_col}")
            
            # Show SRM warning if detected
            if p_value < 0.01:
                st.warning(f"""
                ⚠️ **Sample Ratio Mismatch (SRM) Detected!**
                
                - **Control Size:** {control_size:,} ({control_size/total*100:.1f}%)
                - **Treatment Size:** {treatment_size:,} ({treatment_size/total*100:.1f}%)
                - **Expected:** 50/50 split
                - **Chi-square:** {chi_square:.2f}
                - **P-value:** {p_value:.6f}
                
                ⚠️ This suggests a data quality issue. The traffic split is significantly different from 50/50, which may indicate:
                - Implementation bugs in randomization
                - Data collection issues
                - Sample selection bias
                
                **Recommendation:** Investigate before running tests, as results may be unreliable.
                """)
            else:
                st.info(f"✅ **No SRM Detected** - Sample sizes are balanced (p={p_value:.4f})")
            
            st.rerun()
    
    elif data_source == "Sample A/B Test Data":
        if st.button("📥 Load Sample A/B Test Data", type="primary"):
            # Generate realistic A/B test data
            np.random.seed(42)
            
            n_control = 1000
            n_treatment = 1000
            
            # Control: 10% conversion
            control_data = pd.DataFrame({
                'group': ['Control'] * n_control,
                'converted': np.random.binomial(1, 0.10, n_control),
                'revenue': np.random.gamma(shape=2, scale=50, size=n_control),
                'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_control),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_control),
                'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_control)
            })
            
            # Treatment: 12% conversion (20% lift)
            treatment_data = pd.DataFrame({
                'group': ['Treatment'] * n_treatment,
                'converted': np.random.binomial(1, 0.12, n_treatment),
                'revenue': np.random.gamma(shape=2, scale=55, size=n_treatment),
                'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_treatment),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_treatment),
                'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_treatment)
            })
            
            sample_data = pd.concat([control_data, treatment_data], ignore_index=True)
            
            st.session_state.ab_test_data = sample_data
            st.session_state.ab_test_groups = {
                'control': 'Control',
                'treatment': 'Treatment',
                'group_col': 'group',
                'metric_col': 'converted'
            }
            
            st.success(f"✅ Loaded sample A/B test data: {len(sample_data)} observations")
    
    elif data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader("Upload A/B test CSV", type=['csv'], key="ab_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Group Column", df.columns, key="ab_group_upload")
            with col2:
                metric_col = st.selectbox("Metric Column", df.columns, key="ab_metric_upload")
            
            if st.button("Process Data", type="primary", key="ab_process_upload"):
                groups = df[group_col].unique()
                if len(groups) == 2:
                    st.session_state.ab_test_data = df[[group_col, metric_col]].copy()
                    st.session_state.ab_test_groups = {
                        'control': groups[0],
                        'treatment': groups[1],
                        'group_col': group_col,
                        'metric_col': metric_col
                    }
                    st.success("✅ Data processed!")
                else:
                    st.error(f"Group column must have exactly 2 groups. Found: {len(groups)}")
    
    else:  # Manual Calculator
        st.info("💡 **Manual Mode:** Enter summary statistics for your A/B test")
        
        # Tabs for different test types
        tab1, tab2, tab3 = st.tabs(["📊 Proportion Test", "📈 T-Test", "🧮 Sample Size Calculator"])
        
        with tab1:
            st.markdown("### Proportion Test (Conversion Rates)")
            st.info("💡 Compare conversion rates between control and treatment groups")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Control Group**")
                control_n = st.number_input("Sample Size (Control)", min_value=10, value=1000, step=10, key="prop_control_n")
                control_conv = st.number_input("Conversions (Control)", min_value=0, max_value=control_n, value=100, step=1, key="prop_control_conv")
                control_rate = (control_conv / control_n * 100) if control_n > 0 else 0
                st.metric("Conversion Rate", f"{control_rate:.2f}%")
            
            with col2:
                st.markdown("**Treatment Group**")
                treatment_n = st.number_input("Sample Size (Treatment)", min_value=10, value=1000, step=10, key="prop_treatment_n")
                treatment_conv = st.number_input("Conversions (Treatment)", min_value=0, max_value=treatment_n, value=120, step=1, key="prop_treatment_conv")
                treatment_rate = (treatment_conv / treatment_n * 100) if treatment_n > 0 else 0
                st.metric("Conversion Rate", f"{treatment_rate:.2f}%")
            
            # Test settings
            alternative = st.radio("Test Type", ["two-sided", "greater", "less"], horizontal=True, key="prop_alt")
            
            if st.button("🧪 Run Proportion Test", type="primary"):
                with st.status("Running statistical test...", expanded=True) as status:
                    result = analyzer.run_proportion_test(
                        control_n, control_conv,
                        treatment_n, treatment_conv,
                        alternative=alternative
                    )
                    status.update(label="✅ Test complete!", state="complete", expanded=False)
                
                # Store results
                st.session_state.ab_test_results = result
                
                # Display results
                st.divider()
                st.subheader("📊 Test Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("P-value", f"{result['p_value']:.4f}")
                with col2:
                    st.metric("Absolute Lift", f"{result['absolute_lift']*100:.2f}%")
                with col3:
                    st.metric("Relative Lift", f"{result['relative_lift']:.1f}%")
                with col4:
                    sig_label = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
                    st.metric("Result", sig_label)
                
                # Interpretation
                if result['is_significant']:
                    st.success(f"""
                    ✅ **Statistically Significant Result!**
                    
                    The treatment group shows a statistically significant difference (p={result['p_value']:.4f} < 0.05).
                    You can confidently roll out this variant.
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **No Statistical Significance**
                    
                    The difference is not statistically significant (p={result['p_value']:.4f} ≥ 0.05).
                    Consider running the test longer or with more traffic.
                    """)
                
                # Visualization
                fig = ABTestAnalyzer.create_ab_test_visualization(result)
                st.plotly_chart(fig, use_container_width=True)
                
                # Effect size
                effect_interp = ABTestAnalyzer.interpret_effect_size(result['effect_size'], 'cohens_h')
                st.info(f"**Effect Size (Cohen's h):** {result['effect_size']:.3f} ({effect_interp})")
                
                # Confidence interval
                ci_lower, ci_upper = result['confidence_interval']
                st.write(f"**95% Confidence Interval for difference:** [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        
        with tab2:
            st.markdown("### T-Test (Compare Means)")
            st.info("💡 Upload CSV with a numeric column and a group column, or enter summary statistics")
            
            test_mode = st.radio("Input Method", ["Summary Statistics", "Upload Data"], horizontal=True, key="ttest_mode")
            
            if test_mode == "Summary Statistics":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Control Group**")
                    control_mean = st.number_input("Mean (Control)", value=100.0, key="ttest_control_mean")
                    control_std = st.number_input("Std Dev (Control)", value=15.0, min_value=0.1, key="ttest_control_std")
                    control_n_ttest = st.number_input("Sample Size (Control)", min_value=2, value=100, key="ttest_control_n")
                
                with col2:
                    st.markdown("**Treatment Group**")
                    treatment_mean = st.number_input("Mean (Treatment)", value=105.0, key="ttest_treatment_mean")
                    treatment_std = st.number_input("Std Dev (Treatment)", value=15.0, min_value=0.1, key="ttest_treatment_std")
                    treatment_n_ttest = st.number_input("Sample Size (Treatment)", min_value=2, value=100, key="ttest_treatment_n")
            
            if st.button("🧪 Run T-Test", type="primary", key="run_ttest"):
                # Generate synthetic data from summary stats
                np.random.seed(42)
                control_data = np.random.normal(control_mean, control_std, control_n_ttest)
                treatment_data = np.random.normal(treatment_mean, treatment_std, treatment_n_ttest)
                
                with st.status("Running t-test...", expanded=True) as status:
                    result = analyzer.run_ttest(control_data, treatment_data, equal_var=False)
                    status.update(label="✅ Test complete!", state="complete", expanded=False)
                
                # Store results
                st.session_state.ab_test_results = result
                
                # Display results (similar to proportion test)
                st.divider()
                st.subheader("📊 Test Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("P-value", f"{result['p_value']:.4f}")
                with col2:
                    st.metric("Mean Difference", f"{result['absolute_diff']:.2f}")
                with col3:
                    st.metric("% Difference", f"{result['relative_diff']:.1f}%")
                with col4:
                    sig_label = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
                    st.metric("Result", sig_label)
                
                # Visualization
                fig = ABTestAnalyzer.create_ab_test_visualization(result)
                st.plotly_chart(fig, use_container_width=True)
                
                # Effect size
                effect_interp = ABTestAnalyzer.interpret_effect_size(result['effect_size'], 'cohens_d')
                st.info(f"**Effect Size (Cohen's d):** {result['effect_size']:.3f} ({effect_interp})")
            
            else:
                st.info("Upload a CSV with at least two columns: a numeric metric column and a group column (e.g., 'A' and 'B')")
                uploaded_file = st.file_uploader("Upload test results CSV", type=['csv'], key="ttest_upload")
                
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head(), use_container_width=True)
                    
                    metric_col = st.selectbox("Metric Column (numeric)", df.select_dtypes(include=[np.number]).columns, key="ttest_metric")
                    group_col = st.selectbox("Group Column", df.columns, key="ttest_group")
                    
                    groups = df[group_col].unique()
                    if len(groups) == 2:
                        group_a, group_b = groups[0], groups[1]
                        data_a = df[df[group_col] == group_a][metric_col].values
                        data_b = df[df[group_col] == group_b][metric_col].values
                        
                        if st.button("🧪 Run T-Test", type="primary", key="run_ttest_upload"):
                            result = analyzer.run_ttest(data_a, data_b, equal_var=False)
                            st.session_state.ab_test_results = result
                            
                            st.subheader("📊 Test Results")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("P-value", f"{result['p_value']:.4f}")
                            with col2:
                                st.metric("Mean Difference", f"{result['absolute_diff']:.2f}")
                            with col3:
                                sig_label = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
                                st.metric("Result", sig_label)
                            
                            fig = ABTestAnalyzer.create_ab_test_visualization(result)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Group column must have exactly 2 unique values. Found: {len(groups)}")
        
        with tab3:
            st.markdown("### Sample Size Calculator")
            st.info("💡 Determine how many samples you need to detect a meaningful difference")
            
            calc_type = st.radio("Test Type", ["Proportion Test", "T-Test"], horizontal=True, key="calc_type")
            
            if calc_type == "Proportion Test":
                baseline = st.slider("Baseline Conversion Rate (%)", 1.0, 50.0, 10.0, 0.5, key="calc_baseline") / 100
                mde = st.slider("Minimum Detectable Effect (%)", 5.0, 100.0, 20.0, 5.0, key="calc_mde") / 100
                
                if st.button("📊 Calculate Sample Size", type="primary", key="calc_prop"):
                    result = analyzer.calculate_sample_size_proportion(baseline, mde)
                    
                    st.subheader("📊 Required Sample Size")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Per Group", f"{result['sample_size_per_group']:,}")
                    with col2:
                        st.metric("Total Needed", f"{result['total_sample_size']:,}")
                    with col3:
                        st.metric("Expected Lift", f"{result['relative_lift']:.1f}%")
                    
                    # Test duration calculator
                    st.divider()
                    st.markdown("### ⏱️ Test Duration")
                    daily_traffic = st.number_input("Daily visitors/users", min_value=10, value=1000, step=10, key="daily_traffic")
                    
                    duration = analyzer.calculate_test_duration(result['total_sample_size'], daily_traffic)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Days to Run", duration['days'])
                    with col2:
                        st.metric("Weeks to Run", duration['weeks'])
                    
                    if duration['days'] > 30:
                        st.warning("⚠️ Test will take over a month. Consider increasing traffic or accepting a larger minimum detectable effect.")
            
            else:  # T-Test sample size
                mean_diff = st.number_input("Expected Mean Difference", min_value=0.1, value=5.0, key="calc_mean_diff")
                std_dev = st.number_input("Standard Deviation", min_value=0.1, value=15.0, key="calc_std")
                
                if st.button("📊 Calculate Sample Size", type="primary", key="calc_ttest"):
                    result = analyzer.calculate_sample_size_means(mean_diff, std_dev)
                    
                    st.subheader("📊 Required Sample Size")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Per Group", f"{result['sample_size_per_group']:,}")
                    with col2:
                        st.metric("Total Needed", f"{result['total_sample_size']:,}")
                    with col3:
                        st.metric("Effect Size", f"{result['effect_size']:.3f}")
    
    # Analysis section (for loaded/sample/upload data)
    if 'ab_test_data' in st.session_state and data_source != "Manual Calculator":
        st.divider()
        st.subheader("📊 2. Run A/B Test Analysis")
        
        test_data = st.session_state.ab_test_data
        groups_info = st.session_state.ab_test_groups
        
        # Display loaded data preview
        with st.expander("👁️ View Loaded Data", expanded=False):
            st.dataframe(test_data.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(test_data)} rows")
        
        # Display dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", f"{len(test_data):,}")
        with col2:
            control_count = (test_data[groups_info['group_col']] == groups_info['control']).sum()
            st.metric(f"{groups_info['control']}", f"{control_count:,}")
        with col3:
            treatment_count = (test_data[groups_info['group_col']] == groups_info['treatment']).sum()
            st.metric(f"{groups_info['treatment']}", f"{treatment_count:,}")
        
        # Display SRM Check
        if 'ab_srm_check' in st.session_state:
            srm = st.session_state.ab_srm_check
            
            if srm['has_srm']:
                st.error(f"""
                🚨 **Sample Ratio Mismatch (SRM) Detected!**
                
                - **Control:** {srm['control_size']:,} ({srm['control_size']/(srm['control_size']+srm['treatment_size'])*100:.1f}%)
                - **Treatment:** {srm['treatment_size']:,} ({srm['treatment_size']/(srm['control_size']+srm['treatment_size'])*100:.1f}%)
                - **Expected:** 50/50 split
                - **Chi-square:** {srm['chi_square']:.2f}, **P-value:** {srm['p_value']:.6f}
                
                ⚠️ **This indicates a data quality issue.** Results may be unreliable. Investigate before proceeding.
                """)
            else:
                st.success(f"✅ **No SRM Detected** - Sample ratio is balanced (Chi² = {srm['chi_square']:.2f}, p = {srm['p_value']:.4f})")
        
        if st.button("🧪 Run Statistical Test", type="primary", key="run_loaded_test"):
            with st.status("Running statistical test...", expanded=True) as status:
                # Extract data for both groups
                control_data = test_data[test_data[groups_info['group_col']] == groups_info['control']][groups_info['metric_col']].values
                treatment_data = test_data[test_data[groups_info['group_col']] == groups_info['treatment']][groups_info['metric_col']].values
                
                # Run t-test
                result = analyzer.run_ttest(control_data, treatment_data, equal_var=False)
                st.session_state.ab_test_results = result
                
                status.update(label="✅ Test complete!", state="complete", expanded=False)
            
            st.success("✅ Statistical test completed!")
    
    # Display results if they exist
    if 'ab_test_results' in st.session_state:
        result = st.session_state.ab_test_results
        
        st.divider()
        st.subheader("📊 Test Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("P-value", f"{result['p_value']:.4f}")
        with col2:
            st.metric("Mean Difference", f"{result['absolute_diff']:.2f}")
        with col3:
            st.metric("% Difference", f"{result['relative_diff']:.1f}%")
        with col4:
            sig_label = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
            st.metric("Result", sig_label)
        
        # Interpretation
        if result['is_significant']:
            st.success(f"""
            ✅ **Statistically Significant Result!**
            
            The treatment group shows a statistically significant difference (p={result['p_value']:.4f} < 0.05).
            The difference is unlikely due to chance.
            """)
        else:
            st.warning(f"""
            ⚠️ **No Statistical Significance**
            
            The difference is not statistically significant (p={result['p_value']:.4f} ≥ 0.05).
            Consider collecting more data or accepting the current variation.
            """)
        
        # Visualization
        fig = ABTestAnalyzer.create_ab_test_visualization(result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Effect size
        effect_interp = ABTestAnalyzer.interpret_effect_size(result['effect_size'], 'cohens_d')
        st.info(f"**Effect Size (Cohen's d):** {result['effect_size']:.3f} ({effect_interp})")
    
    # Sequential Testing Section
    if 'ab_test_results' in st.session_state and 'control_rate' in st.session_state.ab_test_results:
        st.divider()
        st.subheader("⏱️ Sequential Testing (Early Stopping)")
        st.markdown("**Save time and resources** by stopping your test early when significance is reached.")
        
        with st.expander("ℹ️ What is Sequential Testing?"):
            st.markdown("""
            **Sequential Testing** allows you to monitor your A/B test continuously and stop early when you reach a conclusion, 
            rather than waiting for a predetermined sample size.
            
            **Benefits:**
            - ⏱️ **Reduced Test Duration**: Stop early when significance reached
            - 💰 **Cost Savings**: Collect fewer samples
            - 📊 **Controlled Error Rates**: Alpha spending functions maintain statistical rigor
            
            **Methods:**
            - **O'Brien-Fleming**: Conservative initially, more aggressive near end (recommended)
            - **Pocock**: Constant boundaries throughout test
            
            **When to Use:**
            - Long-running tests with continuous data collection
            - High opportunity cost of waiting
            - Large expected effect sizes
            """)
        
        # Sequential test configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spending_function = st.selectbox(
                "Alpha Spending Function:",
                ["obrien_fleming", "pocock"],
                format_func=lambda x: "O'Brien-Fleming" if x == "obrien_fleming" else "Pocock",
                help="O'Brien-Fleming is more conservative early on (recommended)"
            )
        
        with col2:
            num_looks = st.slider(
                "Planned Interim Analyses:",
                min_value=2,
                max_value=10,
                value=5,
                help="Total number of times you plan to check the data"
            )
        
        with col3:
            information_fraction = st.slider(
                "Current Information Fraction:",
                min_value=0.2,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Fraction of planned sample size you've collected (0.5 = 50%)"
            )
        
        # Show boundary plot
        st.markdown("### 📊 Sequential Testing Boundaries")
        fig_boundaries = ABTestAnalyzer.create_sequential_boundary_plot(
            alpha=0.05,
            spending_function=spending_function,
            num_looks=num_looks
        )
        st.plotly_chart(fig_boundaries, use_container_width=True)
        st.caption("Green: Stop for efficacy (treatment wins) | Red: Stop for futility (no effect)")
        
        # Run sequential test button
        if st.button("🔬 Run Sequential Test Analysis", type="primary"):
            result = st.session_state.ab_test_results
            
            # Get group sizes and conversions
            test_data = st.session_state.ab_test_data
            groups_info = st.session_state.ab_test_groups
            
            control_data = test_data[test_data[groups_info['group_col']] == groups_info['control']]
            treatment_data = test_data[test_data[groups_info['group_col']] == groups_info['treatment']]
            
            control_n = len(control_data)
            treatment_n = len(treatment_data)
            
            # Assuming binary metric (0/1)
            metric_col = groups_info['metric_col']
            control_conversions = int(control_data[metric_col].sum())
            treatment_conversions = int(treatment_data[metric_col].sum())
            
            # Calculate current look number
            look_number = int(information_fraction * num_looks) + 1
            
            # Run sequential test
            seq_result = analyzer.sequential_test_proportion(
                control_n=control_n,
                control_conversions=control_conversions,
                treatment_n=treatment_n,
                treatment_conversions=treatment_conversions,
                information_fraction=information_fraction,
                spending_function=spending_function,
                num_looks=num_looks,
                look_number=look_number
            )
            
            st.session_state.seq_test_result = seq_result
        
        # Display sequential test results
        if 'seq_test_result' in st.session_state:
            seq_res = st.session_state.seq_test_result
            
            st.markdown("### 📋 Sequential Test Results")
            
            # Stopping decision
            if seq_res['should_stop']:
                if seq_res['stop_for_efficacy']:
                    st.success(f"""
                    ✅ **STOP FOR EFFICACY** 
                    
                    The treatment is significantly better! You can stop the test early.
                    
                    - **Z-statistic**: {seq_res['z_statistic']:.3f}
                    - **Upper Bound**: {seq_res['upper_bound']:.3f}
                    - **Decision**: Treatment wins with statistical significance
                    """)
                elif seq_res['stop_for_futility']:
                    st.error(f"""
                    🛑 **STOP FOR FUTILITY**
                    
                    Unlikely to find a significant effect. Consider stopping the test.
                    
                    - **Z-statistic**: {seq_res['z_statistic']:.3f}
                    - **Lower Bound**: {seq_res['lower_bound']:.3f}
                    - **Decision**: No treatment effect detected
                    """)
            else:
                st.info(f"""
                ⏳ **CONTINUE TESTING**
                
                Not enough evidence yet. Keep collecting data.
                
                - **Z-statistic**: {seq_res['z_statistic']:.3f}
                - **Boundaries**: [{seq_res['lower_bound']:.3f}, {seq_res['upper_bound']:.3f}]
                - **Decision**: Continue to next interim analysis
                """)
            
            # Savings metrics
            st.markdown("### 💰 Potential Savings")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples Collected", f"{seq_res['samples_collected']:,}")
            with col2:
                st.metric("Information Fraction", f"{seq_res['information_fraction']:.0%}")
            with col3:
                st.metric("Remaining Samples", f"{seq_res['remaining_samples_needed']:,}")
            with col4:
                st.metric("Potential Savings", f"{seq_res['potential_savings_pct']:.0f}%",
                         help="Percentage of planned samples you can avoid collecting")
            
            # Interpretation
            if seq_res['should_stop']:
                savings_samples = seq_res['remaining_samples_needed']
                if savings_samples > 0:
                    st.success(f"""
                    🎯 **Resource Optimization:**
                    
                    By stopping now, you save **{savings_samples:,} samples** ({seq_res['potential_savings_pct']:.0f}% of planned).
                    
                    **Impact:**
                    - Faster decision-making
                    - Reduced opportunity cost
                    - Maintained statistical rigor with alpha spending
                    """)
            else:
                next_look = seq_res['look_number'] + 1
                next_fraction = next_look / seq_res['num_looks']
                additional_samples = int((next_fraction - seq_res['information_fraction']) * 
                                       seq_res['samples_collected'] / seq_res['information_fraction'])
                
                st.info(f"""
                📅 **Next Steps:**
                
                - **Next Analysis**: Look #{next_look} of {seq_res['num_looks']}
                - **Target Information**: {next_fraction:.0%}
                - **Additional Samples Needed**: ~{additional_samples:,}
                - **Then Re-evaluate**: Run this sequential test again
                """)
            
            # Technical details
            with st.expander("🔍 Technical Details"):
                st.markdown(f"""
                **Test Configuration:**
                - Spending Function: {seq_res['spending_function'].replace('_', ' ').title()}
                - Alpha Level: {analyzer.alpha}
                - Current Look: {seq_res['look_number']} of {seq_res['num_looks']}
                
                **Statistical Boundaries:**
                - Lower Bound (Futility): {seq_res['lower_bound']:.4f}
                - Upper Bound (Efficacy): {seq_res['upper_bound']:.4f}
                - Observed Z-statistic: {seq_res['z_statistic']:.4f}
                
                **Sample Information:**
                - Control: {seq_res['samples_collected'] // 2:,} samples
                - Treatment: {seq_res['samples_collected'] - seq_res['samples_collected'] // 2:,} samples
                - Total Collected: {seq_res['samples_collected']:,}
                """)
    
    # Segmentation Analysis Section
    if 'ab_test_data' in st.session_state and 'ab_test_groups' in st.session_state:
        st.divider()
        st.subheader("📊 Segmentation Analysis")
        st.markdown("Analyze how different user segments respond to your treatment. Identify which groups benefit most.")
        
        with st.expander("ℹ️ What is Segmentation Analysis?"):
            st.markdown("""
            **Segmentation Analysis** reveals **heterogeneous treatment effects** - how different groups respond differently to your treatment.
            
            **Why It Matters:**
            - 👥 **Personalization**: Target treatments to specific segments
            - 💰 **ROI Optimization**: Focus resources where impact is highest
            - 📈 **Better Decisions**: Avoid one-size-fits-all conclusions
            - 🎯 **Strategic Insights**: Understand WHO benefits most
            
            **Example Use Cases:**
            - **E-commerce**: Age groups responding differently to discounts
            - **Healthcare**: Treatment effectiveness by patient demographics
            - **Marketing**: Campaign performance by geographic region
            - **Product**: Feature adoption by user type
            
            **How to Use:**
            1. Select a segment column (e.g., age_group, region, user_type)
            2. Click "Analyze by Segments"
            3. Review per-segment performance
            4. Identify high and low performers
            5. Target future campaigns accordingly
            """)
        
        test_data = st.session_state.ab_test_data
        groups_info = st.session_state.ab_test_groups
        
        # Get potential segment columns (exclude group and metric columns)
        excluded_cols = {groups_info['group_col'], groups_info['metric_col']}
        segment_options = [col for col in test_data.columns if col not in excluded_cols]
        
        if len(segment_options) > 0:
            segment_col = st.selectbox(
                "Select Segment Column:",
                segment_options,
                help="Choose a column to segment by (e.g., age_group, region, user_type)"
            )
            
            # Check if selected column is suitable for segmentation
            num_segments = test_data[segment_col].nunique()
            
            # Validate segment column suitability
            segment_blocked = False
            
            if num_segments < 2:
                st.error(f"🚨 **NOT SUITABLE:** Column '{segment_col}' has only {num_segments} unique value(s). Need at least 2 segments for analysis.")
                segment_blocked = True
            elif num_segments > 50:
                st.error(f"🚨 **TOO MANY SEGMENTS:** Column '{segment_col}' has {num_segments:,} unique values. Maximum 50 segments recommended.")
                st.info("💡 **Recommendation:** Use a categorical column with 2-20 groups (e.g., age_group, region, user_type)")
                segment_blocked = True
            elif num_segments > 20:
                st.warning(f"⚠️ Column '{segment_col}' has {num_segments} segments. Results may be hard to interpret. Consider grouping into 2-20 categories.")
                st.info(f"💡 Analysis will proceed with {num_segments} segments")
            else:
                st.success(f"✅ Will analyze {num_segments} segments in '{segment_col}'")
            
            # Run segmentation analysis button (disabled if blocked)
            if st.button("📊 Analyze by Segments", type="primary", disabled=segment_blocked):
                with st.status("Analyzing treatment effects by segment...", expanded=True) as status:
                    try:
                        segment_results = analyzer.segment_analysis(
                            data=test_data,
                            group_col=groups_info['group_col'],
                            metric_col=groups_info['metric_col'],
                            segment_col=segment_col,
                            control_group=groups_info['control'],
                            treatment_group=groups_info['treatment']
                        )
                        
                        st.session_state.segment_results = segment_results
                        
                        status.update(label="✅ Segmentation complete!", state="complete", expanded=False)
                        st.success("✅ Segment analysis completed!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("💡 No additional columns available for segmentation. Upload data with demographic or categorical columns to enable segment analysis.")
    
    # Display segmentation results
    if 'segment_results' in st.session_state:
        seg_res = st.session_state.segment_results
        segments_df = seg_res['segments']
        
        # Check if DataFrame is empty
        if segments_df.empty:
            st.warning("⚠️ No segmentation results available. The selected segment column may not have enough data in each group.")
            st.stop()
        
        st.markdown("### 📋 Segment Performance")
        
        # Verify all required columns exist
        required_cols = [
            'segment', 'control_n', 'treatment_n', 
            'control_mean', 'treatment_mean', 'lift', 
            'relative_lift', 'p_value', 'significant'
        ]
        missing_cols = [col for col in required_cols if col not in segments_df.columns]
        
        if missing_cols:
            st.error(f"Error: Missing columns in segmentation results: {', '.join(missing_cols)}")
            st.info(f"Available columns: {', '.join(segments_df.columns.tolist())}")
            st.stop()
        
        # Display results table
        display_df = segments_df[required_cols].copy()
        
        # Format columns
        display_df['control_mean'] = display_df['control_mean'].apply(lambda x: f"{x:.4f}")
        display_df['treatment_mean'] = display_df['treatment_mean'].apply(lambda x: f"{x:.4f}")
        display_df['lift'] = display_df['lift'].apply(lambda x: f"{x:.4f}")
        display_df['relative_lift'] = display_df['relative_lift'].apply(lambda x: f"{x:.2f}%")
        display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
        display_df['significant'] = display_df['significant'].apply(lambda x: '✅ Yes' if x else '❌ No')
        
        st.dataframe(display_df, use_container_width=True)
        
        # Key metrics
        st.markdown("### 📊 Key Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_segment = segments_df.loc[segments_df['lift'].idxmax(), 'segment']
            best_lift = segments_df['lift'].max()
            st.metric("Best Performing Segment", best_segment,
                     delta=f"+{best_lift:.4f}",
                     help="Segment with highest treatment lift")
        
        with col2:
            worst_segment = segments_df.loc[segments_df['lift'].idxmin(), 'segment']
            worst_lift = segments_df['lift'].min()
            st.metric("Worst Performing Segment", worst_segment,
                     delta=f"{worst_lift:.4f}",
                     delta_color="inverse",
                     help="Segment with lowest treatment lift")
        
        with col3:
            sig_segments = segments_df['significant'].sum()
            total_segments = len(segments_df)
            st.metric("Significant Segments", f"{sig_segments}/{total_segments}",
                     help="Number of segments with statistically significant effects")
        
        with col4:
            lift_range = segments_df['lift'].max() - segments_df['lift'].min()
            st.metric("Effect Heterogeneity", f"{lift_range:.4f}",
                     help="Range of treatment effects across segments")
        
        # Visualization
        st.markdown("### 📈 Treatment Effect by Segment")
        
        fig_segment = ABTestAnalyzer.create_segment_comparison_plot(
            segments_df,
            seg_res['segment_col'],
            overall_lift=seg_res['overall_lift']
        )
        st.plotly_chart(fig_segment, use_container_width=True)
        st.caption("Green bars = statistically significant | Gray bars = not significant | Red dashed line = overall effect")
        
        # Strategic recommendations
        st.markdown("### 💡 Strategic Recommendations")
        
        # Identify high and low performers
        sig_positive = segments_df[(segments_df['significant']) & (segments_df['lift'] > 0)]
        sig_negative = segments_df[(segments_df['significant']) & (segments_df['lift'] < 0)]
        
        if len(sig_positive) > 0:
            st.success(f"""
            ✅ **High-Impact Segments (Focus Here)**
            
            The following segments show strong positive response:
            """)
            for _, row in sig_positive.iterrows():
                st.markdown(f"- **{row['segment']}**: +{row['lift']:.4f} lift ({row['relative_lift']:.1f}% improvement)")
            
            st.markdown("""
            **Recommendation:** Prioritize these segments in future campaigns for maximum ROI.
            """)
        
        if len(sig_negative) > 0:
            st.warning(f"""
            ⚠️ **Negative Response Segments (Caution)**
            
            The following segments show adverse reactions:
            """)
            for _, row in sig_negative.iterrows():
                st.markdown(f"- **{row['segment']}**: {row['lift']:.4f} lift ({row['relative_lift']:.1f}% change)")
            
            st.markdown("""
            **Recommendation:** Exclude these segments or develop alternative treatments specifically for them.
            """)
        
        non_sig = segments_df[~segments_df['significant']]
        if len(non_sig) > 0:
            st.info(f"""
            ℹ️ **Neutral Segments ({len(non_sig)} total)**
            
            These segments show no clear treatment effect. Consider:
            - **Lower priority** in targeting
            - **Alternative treatments** may be needed
            - **Collect more data** to increase statistical power
            """)
        
        # Heterogeneity assessment
        if lift_range > abs(seg_res['overall_lift']) * 0.5:
            st.error("""
            🚨 **High Heterogeneity Detected**
            
            Treatment effects vary significantly across segments (range > 50% of overall effect).
            
            **Action Required:**
            - **DO NOT** use a one-size-fits-all approach
            - **MUST** personalize treatment by segment
            - **CONSIDER** segment-specific strategies
            """)
        else:
            st.success("""
            ✅ **Consistent Treatment Effect**
            
            Treatment effects are relatively uniform across segments.
            
            **You can:**
            - Apply treatment broadly
            - Still prioritize high-performing segments for efficiency
            - Monitor for emerging segment differences over time
            """)
    
    # AI Insights
    if 'ab_test_results' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'ab_ai_insights' in st.session_state:
            st.markdown(st.session_state.ab_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="ab_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing A/B test results and generating strategic recommendations...", expanded=True) as status:
                    # Get data from session state
                    result = st.session_state.ab_test_results
                    
                    # Prepare detailed context based on test type
                    if 'control_rate' in result:
                        # Proportion test
                        control_size = result.get('control_size', 0)
                        treatment_size = result.get('treatment_size', 0)
                        total_size = control_size + treatment_size
                        control_conversions = int(result['control_rate'] * control_size)
                        treatment_conversions = int(result['treatment_rate'] * treatment_size)
                        
                        context = f"""
A/B Test Analysis (Conversion/Proportion Test):

Test Configuration:
- Total Sample Size: {total_size:,} observations
- Control Group: {control_size:,} observations ({control_size/total_size*100:.1f}%)
- Treatment Group: {treatment_size:,} observations ({treatment_size/total_size*100:.1f}%)

Performance Metrics:
- Control Conversion Rate: {result['control_rate']*100:.2f}% ({control_conversions} conversions)
- Treatment Conversion Rate: {result['treatment_rate']*100:.2f}% ({treatment_conversions} conversions)
- Absolute Lift: {result['absolute_lift']*100:.2f} percentage points
- Relative Lift: {result['relative_lift']:.1f}% improvement

Statistical Analysis:
- P-value: {result['p_value']:.4f}
- Statistically Significant: {'YES' if result['is_significant'] else 'NO'} (α=0.05)
- Effect Size (Cohen's h): {result['effect_size']:.3f}
- Confidence Level: 95%
"""
                    else:
                        # T-test
                        control_size = result.get('control_size', 0)
                        treatment_size = result.get('treatment_size', 0)
                        total_size = control_size + treatment_size
                        
                        context = f"""
A/B Test Analysis (Mean Comparison / T-Test):

Test Configuration:
- Total Sample Size: {total_size:,} observations
- Control Group: {control_size:,} observations
- Treatment Group: {treatment_size:,} observations

Performance Metrics:
- Control Mean: {result['control_mean']:.2f}
- Treatment Mean: {result['treatment_mean']:.2f}
- Mean Difference: {result['absolute_diff']:.2f}
- Relative Difference: {result['relative_diff']:.1f}%

Statistical Analysis:
- P-value: {result['p_value']:.4f}
- Statistically Significant: {'YES' if result['is_significant'] else 'NO'} (α=0.05)
- Effect Size (Cohen's d): {result['effect_size']:.3f}
- Confidence Level: 95%
"""
                    
                    prompt = f"""
As a senior experimentation and conversion optimization expert, analyze these A/B test results and provide:

1. **Test Outcome Summary** (3-4 sentences): Interpret the results in clear business language. What happened and why does it matter?

2. **Statistical Confidence** (3-4 sentences): Explain the p-value and effect size. How confident should we be in this result? Is the sample size adequate?

3. **Business Recommendation** (Clear GO/NO-GO decision with 2-3 sentences): Should we implement the change? Consider both statistical significance and practical significance.

4. **Implementation Strategy** (4-5 bullet points): If we proceed, how should we roll this out?
   - Rollout timeline and approach (gradual vs. immediate)
   - Monitoring metrics during implementation
   - Contingency plans if metrics decline
   - Documentation and team communication

5. **Risk Assessment** (3-4 bullet points): What could go wrong?
   - Statistical risks (false positives, sample size issues)
   - Business risks (user experience, technical debt)
   - Market timing considerations

6. **ROI Projection** (2-3 sentences): Based on the lift, estimate the business impact. If applicable, project revenue/conversion gains at scale.

{context}

Be specific, actionable, and balance statistical rigor with business pragmatism. Consider both short-term and long-term implications.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior experimentation and conversion optimization expert with 10+ years of experience running A/B tests at scale. You specialize in balancing statistical rigor with business pragmatism."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.ab_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    # Display results inside status block to avoid duplicates
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.ab_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export section
    if 'ab_test_results' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        result = st.session_state.ab_test_results
        
        # Create markdown report
        report = f"""# A/B Test Results Report

## Test Summary
- **Test Type:** {result['test_type']}
- **Significance Level:** {result['alpha']}
- **P-value:** {result['p_value']:.4f}
- **Result:** {'✅ Significant' if result['is_significant'] else '❌ Not Significant'}

## Metrics
"""
        
        if 'control_rate' in result:
            report += f"""
- **Control Conversion Rate:** {result['control_rate']*100:.2f}%
- **Treatment Conversion Rate:** {result['treatment_rate']*100:.2f}%
- **Absolute Lift:** {result['absolute_lift']*100:.2f}%
- **Relative Lift:** {result['relative_lift']:.1f}%
"""
        else:
            report += f"""
- **Control Mean:** {result['control_mean']:.2f}
- **Treatment Mean:** {result['treatment_mean']:.2f}
- **Mean Difference:** {result['absolute_diff']:.2f}
- **% Difference:** {result['relative_diff']:.1f}%
"""
        
        report += f"""
## Statistical Details
- **Effect Size:** {result['effect_size']:.3f}
- **95% Confidence Interval:** [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]
"""
        
        if 'ab_ai_insights' in st.session_state:
            report += f"\n## 🤖 AI Insights\n\n{st.session_state.ab_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - A/B Testing Module*\n"
        
        # Create CSV export data
        if 'control_rate' in result:
            # Proportion test results
            csv_data = pd.DataFrame([{
                'test_type': result['test_type'],
                'control_rate': f"{result['control_rate']:.4f}",
                'treatment_rate': f"{result['treatment_rate']:.4f}",
                'absolute_lift': f"{result['absolute_lift']:.4f}",
                'relative_lift_percent': f"{result['relative_lift']:.2f}",
                'p_value': f"{result['p_value']:.6f}",
                'significant': 'Yes' if result['is_significant'] else 'No',
                'alpha': result['alpha'],
                'effect_size': f"{result['effect_size']:.4f}",
                'ci_lower': f"{result['confidence_interval'][0]:.4f}",
                'ci_upper': f"{result['confidence_interval'][1]:.4f}"
            }])
        else:
            # T-test results
            csv_data = pd.DataFrame([{
                'test_type': result['test_type'],
                'control_mean': f"{result['control_mean']:.4f}",
                'treatment_mean': f"{result['treatment_mean']:.4f}",
                'mean_difference': f"{result['absolute_diff']:.4f}",
                'percent_difference': f"{result['relative_diff']:.2f}",
                'p_value': f"{result['p_value']:.6f}",
                'significant': 'Yes' if result['is_significant'] else 'No',
                'alpha': result['alpha'],
                'effect_size': f"{result['effect_size']:.4f}",
                'ci_lower': f"{result['confidence_interval'][0]:.4f}",
                'ci_upper': f"{result['confidence_interval'][1]:.4f}"
            }])
        
        csv_string = csv_data.to_csv(index=False)
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"ab_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"ab_test_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            use_container_width=True
        )

def show_cohort_analysis():
    """Cohort Analysis page."""
    st.markdown("<h2 style='text-align: center;'>👥 Cohort Analysis & Retention Tracking</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Cohort Analysis?"):
        st.markdown("""
        **Cohort Analysis** groups users based on shared characteristics or experiences and tracks their behavior over time.
        
        ### Key Concepts:
        
        - **Cohort:** A group of users who share a common characteristic (e.g., signed up in January)
        - **Retention Rate:** % of cohort that returns in each subsequent period
        - **Cohort Period:** Time grouping (daily, weekly, monthly)
        - **LTV (Lifetime Value):** Cumulative value generated by cohort over time
        
        ### Business Applications:
        - 📊 **Retention Analysis:** Identify when users churn
        - 💰 **Revenue Forecasting:** Predict future revenue from cohorts
        - 🎯 **Product Impact:** Measure effect of product changes
        - 📧 **Campaign Effectiveness:** Track cohort performance post-campaign
        """)
    
    st.markdown("Track customer retention and lifetime value across different cohorts.")
    
    # Import utilities
    from utils.cohort_analysis import CohortAnalyzer
    
    # Initialize analyzer
    if 'cohort_analyzer' not in st.session_state:
        st.session_state.cohort_analyzer = CohortAnalyzer()
    
    analyzer = st.session_state.cohort_analyzer
    
    # Data loading
    st.subheader("📤 1. Load User Activity Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample E-commerce Data", "Upload Custom Data"]
    else:
        data_options = ["Sample E-commerce Data", "Upload Custom Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="cohort_data_source"
    )
    
    user_data = None
    
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Show data preview
        with st.expander("👁️ Preview Dataset", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_cohort_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select user ID and date columns for cohort analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            user_default = suggestions['user_id']
            user_idx = list(df.columns).index(user_default)
            user_col = st.selectbox("User ID Column", df.columns, index=user_idx, key="cohort_user")
        with col2:
            cohort_default = suggestions['cohort_date']
            cohort_idx = list(df.columns).index(cohort_default)
            cohort_col = st.selectbox("Cohort Date (signup/first purchase)", df.columns, index=cohort_idx, key="cohort_date")
        with col3:
            activity_default = suggestions['activity_date']
            activity_idx = list(df.columns).index(activity_default)
            activity_col = st.selectbox("Activity Date", df.columns, index=activity_idx, key="cohort_activity")
        
        # Real-time validation (3-level system)
        st.divider()
        st.subheader("📋 Data Validation")
        
        issues = []
        warnings = []
        
        # Validate Cohort Date column
        try:
            cohort_dates = pd.to_datetime(df[cohort_col], errors='coerce')
            null_pct = (cohort_dates.isna().sum() / len(df)) * 100
            
            if null_pct == 100:
                issues.append(f"❌ **Cohort Date '{cohort_col}'**: Cannot be parsed as dates (0% valid dates)")
            elif null_pct > 50:
                issues.append(f"❌ **Cohort Date '{cohort_col}'**: Only {100-null_pct:.1f}% are valid dates")
            elif null_pct > 10:
                warnings.append(f"⚠️ **Cohort Date '{cohort_col}'**: {null_pct:.1f}% invalid dates will be removed")
        except:
            issues.append(f"❌ **Cohort Date '{cohort_col}'**: Not a date column")
        
        # Validate Activity Date column
        try:
            activity_dates = pd.to_datetime(df[activity_col], errors='coerce')
            null_pct = (activity_dates.isna().sum() / len(df)) * 100
            
            if null_pct == 100:
                issues.append(f"❌ **Activity Date '{activity_col}'**: Cannot be parsed as dates (0% valid dates)")
            elif null_pct > 50:
                issues.append(f"❌ **Activity Date '{activity_col}'**: Only {100-null_pct:.1f}% are valid dates")
            elif null_pct > 10:
                warnings.append(f"⚠️ **Activity Date '{activity_col}'**: {null_pct:.1f}% invalid dates will be removed")
        except:
            issues.append(f"❌ **Activity Date '{activity_col}'**: Not a date column")
        
        # Validate User ID column
        n_unique = df[user_col].nunique()
        if n_unique < 10:
            warnings.append(f"⚠️ **User ID '{user_col}'**: Only {n_unique} unique users (cohort analysis works best with 50+ users)")
        elif n_unique > len(df) * 0.9:
            warnings.append(f"⚠️ **User ID '{user_col}'**: Too many unique values ({n_unique:,}), might not be a user ID column")
        
        # Display validation results
        if len(issues) > 0:
            st.error("**🚨 CRITICAL ISSUES - CANNOT RUN COHORT ANALYSIS**")
            for issue in issues:
                st.markdown(f"- {issue}")
            with st.expander("💡 Recommendations"):
                st.markdown("""
                **Fix these issues:**
                1. Select columns that contain actual date values (e.g., InvoiceDate, OrderDate, CreatedAt)
                2. Avoid text columns like 'Description', 'Name', 'Category'
                3. Check if your date columns need to be parsed during data upload
                """)
            st.session_state.cohort_data_suitable = False
        elif len(warnings) > 0:
            st.warning("**⚠️ COHORT ANALYSIS POSSIBLE (with warnings)**")
            for warning in warnings:
                st.markdown(f"- {warning}")
            with st.expander("💡 Recommendations"):
                st.markdown("""
                **Consider:**
                - Invalid dates will be removed automatically
                - Best results require 50+ users and clean date data
                - Check your data quality in the Data Upload section
                """)
            st.session_state.cohort_data_suitable = True
        else:
            st.success("**✅ EXCELLENT FOR COHORT ANALYSIS**")
            st.markdown(f"- ✅ Cohort Date: Valid date column with clean data")
            st.markdown(f"- ✅ Activity Date: Valid date column with clean data")
            st.markdown(f"- ✅ User IDs: {n_unique:,} unique users detected")
            st.session_state.cohort_data_suitable = True
        
        # Only enable button if data is suitable
        button_disabled = len(issues) > 0
        
        if st.button("📊 Validate & Process Data", type="primary", disabled=button_disabled):
            # Validate dates
            try:
                pd.to_datetime(df[cohort_col])
                pd.to_datetime(df[activity_col])
            except:
                st.error("❌ Date columns must contain valid date values")
                st.stop()
            
            user_data = df[[user_col, cohort_col, activity_col]].copy()
            user_data.columns = ['user_id', 'signup_date', 'activity_date']
            st.session_state.cohort_data = user_data
            st.success("✅ Data processed!")
            st.info(f"📊 {user_data['user_id'].nunique()} unique users")
            st.rerun()
    
    elif data_source == "Sample E-commerce Data":
        if st.button("📥 Load Sample User Activity", type="primary"):
            # Generate sample cohort data
            np.random.seed(42)
            
            # 200 users over 12 months
            n_users = 200
            n_months = 12
            
            signup_dates = pd.date_range(end=datetime.now(), periods=n_months, freq='MS')
            
            activities = []
            for user_id in range(1, n_users + 1):
                signup_month = np.random.choice(signup_dates)
                
                # Generate activity dates (some users more active than others)
                activity_months = []
                current_month = signup_month
                
                # Retention decreases over time
                for month_num in range(12):
                    retention_prob = 0.9 * (0.85 ** month_num)  # Exponential decay
                    if np.random.random() < retention_prob:
                        activity_date = current_month + pd.DateOffset(months=month_num)
                        if activity_date <= datetime.now():
                            # Generate 1-5 activities in this month
                            n_activities = np.random.randint(1, 6)
                            for _ in range(n_activities):
                                revenue = np.random.gamma(shape=2, scale=25)
                                activities.append({
                                    'user_id': f'U{user_id:04d}',
                                    'signup_date': signup_month,
                                    'activity_date': activity_date + pd.Timedelta(days=np.random.randint(0, 28)),
                                    'revenue': round(revenue, 2)
                                })
            
            user_data = pd.DataFrame(activities)
            st.session_state.cohort_data = user_data
            
            st.success(f"✅ Loaded {len(user_data)} activities from {n_users} users!")
            st.dataframe(user_data.head(10), use_container_width=True)
    
    else:  # Upload
        uploaded_file = st.file_uploader("Upload user activity CSV", type=['csv'], key="cohort_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                user_col = st.selectbox("User ID Column", df.columns, key="cohort_user_upload")
            with col2:
                cohort_col = st.selectbox("Cohort Date", df.columns, key="cohort_date_upload")
            with col3:
                activity_col = st.selectbox("Activity Date", df.columns, key="cohort_activity_upload")
            
            if st.button("Process Data", type="primary", key="process_upload"):
                user_data = df[[user_col, cohort_col, activity_col]].copy()
                user_data.columns = ['user_id', 'signup_date', 'activity_date']
                st.session_state.cohort_data = user_data
                st.success("✅ Data processed!")
    
    # Analysis section
    if 'cohort_data' not in st.session_state:
        st.info("👆 Load user activity data to begin cohort analysis")
        return
    
    user_data = st.session_state.cohort_data
    
    # Dataset overview
    st.divider()
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", f"{user_data['user_id'].nunique():,}")
    with col2:
        st.metric("Total Activities", f"{len(user_data):,}")
    with col3:
        avg_activities = len(user_data) / user_data['user_id'].nunique()
        st.metric("Avg Activities/User", f"{avg_activities:.1f}")
    
    # Display loaded data preview
    with st.expander("👁️ View Loaded Data", expanded=False):
        st.dataframe(user_data.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(user_data)} rows")
    
    # Run cohort analysis
    st.divider()
    st.subheader("📊 2. Cohort Analysis")
    
    cohort_period = st.radio("Cohort Period", ["Monthly", "Weekly", "Daily"], horizontal=True, key="cohort_period")
    period_map = {"Monthly": "M", "Weekly": "W", "Daily": "D"}
    
    if st.button("📊 Run Cohort Analysis", type="primary"):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Cohort_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Analyzing cohorts...", expanded=True) as status:
                # Create cohorts
                cohort_data = analyzer.create_cohorts(
                    user_data,
                    user_col='user_id',
                    cohort_date_col='signup_date',
                    activity_date_col='activity_date',
                    cohort_period=period_map[cohort_period]
                )
                
                # Calculate retention
                retention_matrix = analyzer.calculate_retention(cohort_data, 'user_id')
                
                # Store results
                st.session_state.cohort_retention = retention_matrix
                st.session_state.cohort_data_processed = cohort_data
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            st.success(f"✅ Analyzed {len(retention_matrix)} cohorts!")
        except Exception as e:
            st.error(f"❌ Error during cohort analysis: {str(e)}")
        finally:
            # Always unlock navigation
            pm.unlock()
    
    # Display results if they exist
    if 'cohort_retention' in st.session_state:
        retention_matrix = st.session_state.cohort_retention
        
        st.divider()
        # Display retention heatmap
        st.subheader("🔥 Retention Heatmap")
        fig = CohortAnalyzer.create_retention_heatmap(retention_matrix)
        st.plotly_chart(fig, use_container_width=True)
        
        # Retention curves
        st.subheader("📈 Retention Curves")
        fig_curves = CohortAnalyzer.create_retention_curves(retention_matrix, top_n=5)
        st.plotly_chart(fig_curves, use_container_width=True)
        
        # Summary stats
        st.subheader("📊 Retention Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            period_1_avg = retention_matrix[1].mean() if 1 in retention_matrix.columns else 0
            st.metric("Avg Period 1 Retention", f"{period_1_avg:.1f}%")
        with col2:
            period_3_avg = retention_matrix[3].mean() if 3 in retention_matrix.columns else 0
            st.metric("Avg Period 3 Retention", f"{period_3_avg:.1f}%")
        with col3:
            last_period = retention_matrix.columns[-1]
            last_period_avg = retention_matrix[last_period].mean()
            st.metric(f"Avg Period {last_period} Retention", f"{last_period_avg:.1f}%")
        
        # Cohort Comparison Feature
        st.divider()
        st.subheader("🔬 Compare Cohorts")
        st.markdown("Select cohorts to statistically compare their retention patterns.")
        
        # Get list of cohorts
        cohorts = retention_matrix.index.tolist()
        
        if len(cohorts) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                cohort_a = st.selectbox("Cohort A", cohorts, key="cohort_compare_a")
            with col2:
                # Filter out Cohort A from Cohort B options
                cohort_b_options = [c for c in cohorts if c != cohort_a]
                cohort_b = st.selectbox("Cohort B", cohort_b_options, key="cohort_compare_b")
            
            if st.button("📊 Compare Selected Cohorts", type="primary"):
                # Get retention values for both cohorts
                cohort_a_values = retention_matrix.loc[cohort_a].dropna().values
                cohort_b_values = retention_matrix.loc[cohort_b].dropna().values
                
                # Run t-test
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_ind(cohort_a_values, cohort_b_values)
                
                # Calculate effect size (Cohen's d)
                mean_a = cohort_a_values.mean()
                mean_b = cohort_b_values.mean()
                std_a = cohort_a_values.std()
                std_b = cohort_b_values.std()
                pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
                cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
                
                # Calculate confidence intervals (95%)
                ci_a = scipy_stats.t.interval(0.95, len(cohort_a_values)-1, 
                                              loc=mean_a, 
                                              scale=scipy_stats.sem(cohort_a_values))
                ci_b = scipy_stats.t.interval(0.95, len(cohort_b_values)-1, 
                                              loc=mean_b, 
                                              scale=scipy_stats.sem(cohort_b_values))
                
                # Store comparison results
                st.session_state.cohort_comparison = {
                    'cohort_a': cohort_a,
                    'cohort_b': cohort_b,
                    'mean_a': mean_a,
                    'mean_b': mean_b,
                    'std_a': std_a,
                    'std_b': std_b,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'ci_a': ci_a,
                    'ci_b': ci_b,
                    'significant': p_value < 0.05
                }
                
                st.success("✅ Comparison complete!")
        
        # Display comparison results if they exist
        if 'cohort_comparison' in st.session_state:
            comp = st.session_state.cohort_comparison
            
            st.subheader("📊 Comparison Results")
            
            # Statistical significance
            if comp['significant']:
                st.success(f"✅ **Statistically Significant Difference** (p = {comp['p_value']:.4f})")
            else:
                st.info(f"ℹ️ **No Significant Difference** (p = {comp['p_value']:.4f})")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"**{comp['cohort_a']}** Avg Retention", f"{comp['mean_a']:.1f}%")
            with col2:
                st.metric(f"**{comp['cohort_b']}** Avg Retention", f"{comp['mean_b']:.1f}%")
            with col3:
                diff = comp['mean_a'] - comp['mean_b']
                st.metric("Difference", f"{diff:+.1f}%", delta=f"{diff:+.1f}%")
            
            # Statistical details
            st.markdown("### Statistical Details")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **T-Statistic:** {comp['t_stat']:.3f}  
                **P-value:** {comp['p_value']:.6f}  
                **Effect Size (Cohen's d):** {comp['cohens_d']:.3f}
                """)
                
                # Interpret effect size
                if abs(comp['cohens_d']) < 0.2:
                    effect_interpretation = "Small effect"
                elif abs(comp['cohens_d']) < 0.5:
                    effect_interpretation = "Medium effect"
                else:
                    effect_interpretation = "Large effect"
                st.caption(f"*{effect_interpretation}*")
            
            with col2:
                st.markdown(f"""
                **{comp['cohort_a']} CI (95%):** [{comp['ci_a'][0]:.1f}%, {comp['ci_a'][1]:.1f}%]  
                **{comp['cohort_b']} CI (95%):** [{comp['ci_b'][0]:.1f}%, {comp['ci_b'][1]:.1f}%]
                """)
                st.caption("*Confidence intervals show range of plausible values*")
            
            # Side-by-side visualization
            st.markdown("### Side-by-Side Comparison")
            
            # Create comparison plot
            import plotly.graph_objects as go
            
            cohort_a_data = retention_matrix.loc[comp['cohort_a']].dropna()
            cohort_b_data = retention_matrix.loc[comp['cohort_b']].dropna()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cohort_a_data.index.tolist(),
                y=cohort_a_data.values.tolist(),
                mode='lines+markers',
                name=str(comp['cohort_a']),
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=cohort_b_data.index.tolist(),
                y=cohort_b_data.values.tolist(),
                mode='lines+markers',
                name=str(comp['cohort_b']),
                line=dict(color='#f093fb', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"Retention Comparison: {comp['cohort_a']} vs {comp['cohort_b']}",
                xaxis_title="Period",
                yaxis_title="Retention (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Select two cohorts above and click 'Compare' to see statistical analysis")
    
    # AI Insights
    if 'cohort_retention' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'cohort_ai_insights' in st.session_state:
            st.markdown(st.session_state.cohort_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="cohort_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing cohort retention patterns and generating strategic recommendations...", expanded=True) as status:
                    # Get data from session state
                    retention_matrix = st.session_state.cohort_retention
                    
                    # Calculate comprehensive metrics
                    num_cohorts = len(retention_matrix)
                    num_periods = len(retention_matrix.columns)
                    
                    # Period-specific retention
                    period_1_retention = retention_matrix[1].mean() if 1 in retention_matrix.columns else 0
                    period_3_retention = retention_matrix[3].mean() if 3 in retention_matrix.columns else 0
                    latest_period = retention_matrix.columns[-1]
                    latest_retention = retention_matrix[latest_period].mean()
                    
                    # Cohort performance analysis
                    best_cohort_initial = retention_matrix[0].max()
                    worst_cohort_initial = retention_matrix[0].min()
                    avg_cohort_size = retention_matrix[0].mean()
                    
                    # Calculate churn rates
                    period_1_churn = 100 - period_1_retention
                    latest_churn = 100 - latest_retention
                    
                    # Retention trend
                    retention_trend = "Declining" if latest_retention < period_1_retention else "Stable" if latest_retention == period_1_retention else "Improving"
                    trend_magnitude = abs(latest_retention - period_1_retention)
                    
                    # Prepare rich context
                    context = f"""
Cohort Retention Analysis Results:

Cohort Overview:
- Total Cohorts Analyzed: {num_cohorts}
- Tracking Periods: {num_periods} periods
- Average Initial Cohort Size: {avg_cohort_size:.1f}%
- Best Performing Cohort: {best_cohort_initial:.1f}% initial retention
- Worst Performing Cohort: {worst_cohort_initial:.1f}% initial retention

Retention Performance:
- Period 1 Retention Rate: {period_1_retention:.1f}%
- Period 3 Retention Rate: {period_3_retention:.1f}%
- Period {latest_period} Retention Rate: {latest_retention:.1f}%
- Overall Retention Trend: {retention_trend} ({trend_magnitude:.1f}% change)

Churn Analysis:
- Period 1 Churn Rate: {period_1_churn:.1f}%
- Latest Period Churn Rate: {latest_churn:.1f}%
- Average Drop-off per Period: {(period_1_retention - latest_retention) / max(num_periods - 1, 1):.1f}%

Key Patterns:
- Retention Curve Shape: {'Steep early drop-off' if period_1_churn > 30 else 'Gradual decline' if period_1_churn > 15 else 'Strong retention'}
- Long-term Stability: {'Stabilizing' if latest_retention > 30 else 'High churn risk' if latest_retention < 20 else 'Moderate retention'}
"""
                    
                    prompt = f"""
As a senior retention strategist and customer lifecycle expert with 10+ years of experience optimizing subscription and SaaS retention metrics, analyze these cohort results and provide:

1. **Retention Health Assessment** (3-4 sentences): Evaluate the overall retention performance. Is this healthy for the business type? What does the retention curve shape tell us about user engagement?

2. **Churn Analysis** (4-5 bullet points): When and why are users dropping off?
   - Critical drop-off periods
   - Churn rate progression
   - At-risk cohort segments
   - Early warning indicators

3. **Cohort Segmentation Strategy** (3-4 sentences): Which cohorts should receive priority attention? How should resources be allocated across different cohort segments for maximum impact?

4. **Re-engagement Tactics** (5-6 bullet points): Specific strategies to win back and retain users:
   - Onboarding optimization (first 30 days)
   - Engagement loops and habit formation
   - Win-back campaigns for churned users
   - Retention incentives and timing
   - Communication frequency and channels
   - Feature adoption drives

5. **Product & Experience Improvements** (4-5 bullet points): What product changes would meaningfully improve retention?
   - Core feature enhancements
   - User experience friction points
   - Value delivery optimization
   - Competitive positioning

6. **ROI Impact Projection** (3-4 sentences): If retention improves by 5%, 10%, and 20%, what would be the impact on customer lifetime value and revenue? Provide realistic estimates and prioritize quick wins vs. long-term investments.

{context}

Be specific, data-driven, and focus on actionable strategies that directly impact retention metrics and customer lifetime value. Consider both immediate tactics and sustainable long-term improvements.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior retention strategist and customer lifecycle expert with 10+ years of experience optimizing subscription and SaaS retention metrics. You specialize in cohort analysis, churn prediction, and customer lifetime value maximization."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.cohort_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    # Display inside status block
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.cohort_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export
    if 'cohort_retention' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        retention_matrix = st.session_state.cohort_retention
        
        # Create report
        report = f"""# Cohort Analysis Report

## Overview
- **Cohorts Analyzed:** {len(retention_matrix)}
- **Periods Tracked:** {len(retention_matrix.columns)}
- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Retention Metrics

### Average Retention by Period
"""
        
        for period in retention_matrix.columns:
            avg_retention = retention_matrix[period].mean()
            report += f"- **Period {period}:** {avg_retention:.1f}%\n"
        
        if 'cohort_ai_insights' in st.session_state:
            report += f"\n## AI Insights\n\n{st.session_state.cohort_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - Cohort Analysis Module*\n"
        
        # Create CSV export with retention matrix
        csv_string = retention_matrix.to_csv()
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"cohort_retention_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"cohort_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_recommendation_systems():
    """Recommendation Systems page."""
    st.markdown("<h2 style='text-align: center;'>🎯 Recommendation Systems & Collaborative Filtering</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What are Recommendation Systems?"):
        st.markdown("""
        **Recommendation Systems** predict user preferences based on historical data and similarities.
        
        ### Methods Available:
        
        - **User-Based Collaborative Filtering:** Find similar users and recommend what they liked
        - **Item-Based Collaborative Filtering:** Find similar items and recommend them
        - **Cosine Similarity:** Measure similarity between users/items
        
        ### Business Applications:
        - 🛍️ **E-commerce:** Product recommendations
        - 🎬 **Media:** Movie/music suggestions
        - 📰 **Content:** Article recommendations
        - 🎯 **Marketing:** Cross-sell and up-sell
        """)
    
    st.markdown("Build personalized recommendation engines using collaborative filtering.")
    
    # Import utilities
    from utils.recommendation_engine import RecommendationEngine
    
    # Initialize engine
    if 'rec_engine' not in st.session_state:
        st.session_state.rec_engine = RecommendationEngine()
    
    engine = st.session_state.rec_engine
    
    # Data loading
    st.subheader("📤 1. Load Ratings Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample Movie Ratings", "Upload Custom Data"]
    else:
        data_options = ["Sample Movie Ratings", "Upload Custom Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="rec_data_source"
    )
    
    # Use Loaded Dataset
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_recommendation_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select user, item, and rating columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            user_default = suggestions['user']
            user_idx = list(df.columns).index(user_default)
            user_col = st.selectbox("User Column", df.columns, index=user_idx, key="rec_user")
        with col2:
            item_default = suggestions['item']
            item_idx = list(df.columns).index(item_default)
            item_col = st.selectbox("Item Column", df.columns, index=item_idx, key="rec_item")
        with col3:
            rating_default = suggestions['rating']
            rating_idx = list(df.columns).index(rating_default)
            rating_col = st.selectbox("Rating Column", df.columns, index=rating_idx, key="rec_rating")
        
        # Real-time validation
        issues = []
        warnings = []
        recommendations = []
        
        # Validate rating column is numeric
        if not pd.api.types.is_numeric_dtype(df[rating_col]):
            issues.append(f"Rating column '{rating_col}' is not numeric (type: {df[rating_col].dtype})")
            recommendations.append("Select a numeric column containing rating values")
        else:
            # Check rating range
            rating_min, rating_max = df[rating_col].min(), df[rating_col].max()
            rating_missing = df[rating_col].isnull().sum()
            
            if rating_missing > 0:
                warnings.append(f"Rating column has {rating_missing} missing values ({rating_missing/len(df)*100:.1f}%)")
            
            # Check if all same values
            if df[rating_col].nunique() == 1:
                issues.append(f"All ratings are the same value ({rating_min})")
                recommendations.append("Ratings need variation to build meaningful recommendations")
        
        # Check if user/item/rating columns are different
        if user_col == item_col or user_col == rating_col or item_col == rating_col:
            issues.append("User, Item, and Rating columns must all be different")
            recommendations.append("Select three distinct columns for user, item, and rating")
        
        # Data quality checks (if rating is numeric)
        if pd.api.types.is_numeric_dtype(df[rating_col]):
            n_users = df[user_col].nunique()
            n_items = df[item_col].nunique()
            n_ratings = len(df)
            
            # Check minimum requirements
            if n_users < 3:
                issues.append(f"Only {n_users} unique users (need at least 3 for collaborative filtering)")
                recommendations.append("Recommendation systems require multiple users to find similarities")
            elif n_users < 10:
                warnings.append(f"Only {n_users} users - recommendations may not be very diverse")
            
            if n_items < 3:
                issues.append(f"Only {n_items} unique items (need at least 3 for recommendations)")
                recommendations.append("Need multiple items to recommend")
            elif n_items < 10:
                warnings.append(f"Only {n_items} items - limited recommendation choices")
            
            if n_ratings < 10:
                issues.append(f"Only {n_ratings} ratings (need at least 10)")
                recommendations.append("More ratings improve recommendation quality")
            
            # Check sparsity
            if n_users >= 3 and n_items >= 3:
                total_possible = n_users * n_items
                sparsity = 100 * (1 - n_ratings / total_possible)
                
                if sparsity > 99:
                    warnings.append(f"Very sparse data: {sparsity:.1f}% empty (only {n_ratings} of {total_possible:,} possible ratings)")
                    recommendations.append("Collaborative filtering works better with denser rating matrices")
                
                # Check average ratings per user/item
                avg_ratings_per_user = n_ratings / n_users
                avg_ratings_per_item = n_ratings / n_items
                
                if avg_ratings_per_user < 2:
                    warnings.append(f"Average {avg_ratings_per_user:.1f} ratings per user (recommend 5+ for quality)")
                if avg_ratings_per_item < 2:
                    warnings.append(f"Average {avg_ratings_per_item:.1f} ratings per item (recommend 5+ for quality)")
        
        # Display validation results
        st.divider()
        if len(issues) > 0:
            st.error("**🚨 NOT SUITABLE FOR RECOMMENDATION SYSTEMS**")
            for issue in issues:
                st.write(f"❌ {issue}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        elif len(warnings) > 0:
            st.warning("**⚠️ RECOMMENDATION SYSTEM POSSIBLE (with warnings)**")
            for warning in warnings:
                st.write(f"⚠️ {warning}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        else:
            st.success("**✅ EXCELLENT FOR RECOMMENDATION SYSTEMS**")
            st.write("✅ Numeric rating column")
            st.write("✅ Sufficient users and items")
            st.write("✅ Good data quality")
        
        # Only show button if no critical issues
        if len(issues) == 0 and st.button("📊 Process Data", type="primary"):
            ratings_data = df[[user_col, item_col, rating_col]].copy()
            ratings_data.columns = ['user_id', 'item_id', 'rating']
            ratings_data = ratings_data.dropna()  # Remove any rows with missing values
            st.session_state.rec_ratings = ratings_data
            
            st.success("✅ Data processed!")
            st.info(f"📊 {ratings_data['user_id'].nunique()} users, {ratings_data['item_id'].nunique()} items, {len(ratings_data)} ratings")
            st.rerun()
    
    elif data_source == "Sample Movie Ratings":
        if st.button("📥 Load Sample Movie Ratings", type="primary"):
            # Generate sample movie ratings
            np.random.seed(42)
            
            movies = [f"Movie_{i}" for i in range(1, 51)]  # 50 movies
            users = [f"User_{i}" for i in range(1, 101)]  # 100 users
            
            ratings = []
            for user in users:
                # Each user rates 5-15 random movies
                n_ratings = np.random.randint(5, 16)
                rated_movies = np.random.choice(movies, n_ratings, replace=False)
                
                for movie in rated_movies:
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                    ratings.append({
                        'user_id': user,
                        'item_id': movie,
                        'rating': rating
                    })
            
            ratings_data = pd.DataFrame(ratings)
            st.session_state.rec_ratings = ratings_data
            
            st.success(f"✅ Loaded {len(ratings_data)} ratings from {len(users)} users on {len(movies)} movies!")
            st.dataframe(ratings_data.head(10), use_container_width=True)
    
    else:  # Upload
        uploaded_file = st.file_uploader("Upload ratings CSV", type=['csv'], key="rec_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                user_col = st.selectbox("User Column", df.columns, key="rec_user_upload")
            with col2:
                item_col = st.selectbox("Item Column", df.columns, key="rec_item_upload")
            with col3:
                rating_col = st.selectbox("Rating Column", df.columns, key="rec_rating_upload")
            
            if st.button("Process Data", type="primary", key="rec_process_upload"):
                ratings_data = df[[user_col, item_col, rating_col]].copy()
                ratings_data.columns = ['user_id', 'item_id', 'rating']
                st.session_state.rec_ratings = ratings_data
                st.success("✅ Data processed!")
    
    # Analysis section
    if 'rec_ratings' not in st.session_state:
        st.info("👆 Load ratings data to begin building recommendations")
        return
    
    ratings_data = st.session_state.rec_ratings
    
    # Dataset overview
    st.divider()
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ratings", f"{len(ratings_data):,}")
    with col2:
        st.metric("Unique Users", f"{ratings_data['user_id'].nunique():,}")
    with col3:
        st.metric("Unique Items", f"{ratings_data['item_id'].nunique():,}")
    with col4:
        sparsity = 1 - (len(ratings_data) / (ratings_data['user_id'].nunique() * ratings_data['item_id'].nunique()))
        st.metric("Sparsity", f"{sparsity*100:.1f}%")
    
    # Display loaded data preview
    with st.expander("👁️ View Loaded Ratings Data", expanded=False):
        st.dataframe(ratings_data.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(ratings_data)} ratings")
    
    # Build recommendation system
    st.divider()
    st.subheader("📊 2. Build Recommendation System")
    
    method = st.radio(
        "Recommendation Method:",
        ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering"],
        horizontal=True,
        key="rec_method"
    )
    
    min_support = st.slider("Minimum Support (min ratings required)", 1, 10, 3, key="rec_support")
    
    if st.button("🎯 Build Recommendations", type="primary"):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Recommendation_System")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Building recommendation model...", expanded=True) as status:
                # Get engine from session state
                engine = st.session_state.rec_engine
                
                # Build user-item matrix and calculate similarities
                # The fit method automatically calculates both user and item similarities
                engine.fit(ratings_data, user_col='user_id', item_col='item_id', rating_col='rating')
                
                # Store the appropriate similarity matrix based on method
                if method == "User-Based Collaborative Filtering":
                    st.session_state.rec_similarity = engine.user_similarity
                    st.session_state.rec_type = 'user'
                else:
                    st.session_state.rec_similarity = engine.item_similarity
                    st.session_state.rec_type = 'item'
                
                status.update(label="✅ Model built!", state="complete", expanded=False)
            
            st.success(f"✅ Built {method} model!")
        except Exception as e:
            st.error(f"❌ Error building recommendation model: {str(e)}")
        finally:
            # Always unlock navigation
            pm.unlock()
    
    # Display results if they exist
    if 'rec_similarity' in st.session_state:
        st.divider()
        
        engine = st.session_state.rec_engine
        method = st.session_state.rec_type
        ratings_data = st.session_state.rec_ratings
        
        # Cold Start Analysis
        st.subheader("🆕 Cold Start Analysis")
        st.markdown("Detecting and handling new users/items with no rating history.")
        
        # Calculate cold start metrics
        total_users = ratings_data['user_id'].nunique()
        total_items = ratings_data['item_id'].nunique()
        
        # Count users/items with minimal ratings (≤2 ratings = cold start)
        user_counts = ratings_data.groupby('user_id').size()
        item_counts = ratings_data.groupby('item_id').size()
        
        cold_start_users = (user_counts <= 2).sum()
        cold_start_items = (item_counts <= 2).sum()
        
        cold_start_user_pct = (cold_start_users / total_users) * 100
        cold_start_item_pct = (cold_start_items / total_items) * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cold Start Users", f"{cold_start_users}", 
                     delta=f"{cold_start_user_pct:.1f}%", 
                     delta_color="inverse")
        with col2:
            active_user_pct = 100 - cold_start_user_pct
            st.metric("Total Users", f"{total_users}",
                     delta=f"{active_user_pct:.1f}% active",
                     delta_color="normal")
        with col3:
            st.metric("Cold Start Items", f"{cold_start_items}", 
                     delta=f"{cold_start_item_pct:.1f}%", 
                     delta_color="inverse")
        with col4:
            active_item_pct = 100 - cold_start_item_pct
            st.metric("Total Items", f"{total_items}",
                     delta=f"{active_item_pct:.1f}% active",
                     delta_color="normal")
        
        # Cold start strategy explanation
        if cold_start_users > 0 or cold_start_items > 0:
            st.info(f"""
            🆕 **Cold Start Strategy Active**
            
            - **New users (≤2 ratings):** Recommend **popular items** based on global ratings
            - **New items (≤2 ratings):** Use global average rating as baseline
            - **Strategy:** Popularity-based fallback ensures all users get recommendations
            """)
        else:
            st.success("✅ No significant cold start issues - all users and items have sufficient rating history")
        
        # Popular Items for Cold Start
        st.subheader("⭐ Popular Items (Cold Start Fallback)")
        st.caption("These items are recommended to new users with no rating history")
        
        popular_items = engine.get_popular_items(n_items=10)
        if not popular_items.empty:
            st.dataframe(popular_items, use_container_width=True)
        else:
            st.info("No popular items available")
        
        # Sample Recommendations with Cold Start Handling
        st.divider()
        st.subheader("🎯 Sample Recommendations (Cold Start Aware)")
        
        if method == 'user':
            # Test with existing user
            sample_user = ratings_data['user_id'].iloc[0]
            recommendations, strategy = engine.recommend_with_cold_start(
                sample_user, n_recommendations=5, method='user_based'
            )
            
            # Show strategy used
            if strategy == 'popularity_fallback':
                st.warning(f"🆕 **User {sample_user}** - Cold start detected! Using **popular items** fallback.")
            elif strategy == 'collaborative_filtering':
                st.success(f"✅ **User {sample_user}** - Using **collaborative filtering** (sufficient history)")
            else:
                st.error(f"⚠️ Error: {strategy}")
            
            if not recommendations.empty:
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.info("No recommendations available for this user")
            
            # Simulate new user cold start
            st.markdown("---")
            st.markdown("**🆕 Simulating New User (Cold Start):**")
            fake_new_user = "NEW_USER_12345"
            new_user_recs, new_user_strategy = engine.recommend_with_cold_start(
                fake_new_user, n_recommendations=5, method='user_based'
            )
            
            if new_user_strategy == 'popularity_fallback':
                st.info(f"🎯 New user detected! Recommending **popular items** as fallback.")
                if not new_user_recs.empty:
                    st.dataframe(new_user_recs, use_container_width=True)
                else:
                    st.warning("No popular items available for fallback")
            else:
                st.info(f"Strategy: {new_user_strategy}")
        
        else:
            sample_user = ratings_data['user_id'].iloc[0]
            recommendations, strategy = engine.recommend_with_cold_start(
                sample_user, n_recommendations=5, method='item_based'
            )
            
            # Show strategy used
            if strategy == 'popularity_fallback':
                st.warning(f"🆕 **User {sample_user}** - Cold start detected! Using **popular items** fallback.")
            elif strategy == 'collaborative_filtering':
                st.success(f"✅ **User {sample_user}** - Using **collaborative filtering** (sufficient history)")
            else:
                st.error(f"⚠️ Error: {strategy}")
            
            if not recommendations.empty:
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.info("No recommendations available")
        
        # Diversity Metrics Section
        st.divider()
        st.subheader("📊 Diversity & Quality Metrics")
        st.markdown("Measure recommendation quality beyond accuracy: diversity, coverage, novelty, and personalization.")
        
        if st.button("🔬 Calculate Diversity Metrics", type="primary"):
            with st.status("Calculating diversity metrics for sample users...", expanded=True) as status:
                try:
                    # Generate recommendations for sample users
                    sample_size = min(50, ratings_data['user_id'].nunique())
                    sample_users = ratings_data['user_id'].unique()[:sample_size]
                    
                    recommendations_dict = {}
                    for user in sample_users:
                        if method == 'user':
                            recs = engine.recommend_items_user_based(user, n_recommendations=10)
                        else:
                            recs = engine.recommend_items_item_based(user, n_recommendations=10)
                        recommendations_dict[user] = recs
                    
                    # Calculate metrics
                    diversity_metrics = engine.calculate_diversity_metrics(recommendations_dict)
                    
                    # Calculate serendipity for first user as example
                    first_user = sample_users[0]
                    first_user_recs = recommendations_dict[first_user]
                    serendipity = engine.calculate_serendipity(first_user, first_user_recs)
                    diversity_metrics['serendipity_sample'] = serendipity
                    
                    st.session_state.diversity_metrics = diversity_metrics
                    
                    status.update(label="✅ Metrics calculated!", state="complete", expanded=False)
                    st.success("✅ Diversity metrics computed!")
                except Exception as e:
                    st.error(f"❌ Error calculating metrics: {str(e)}")
        
        # Display diversity metrics if calculated
        if 'diversity_metrics' in st.session_state:
            metrics = st.session_state.diversity_metrics
            
            st.markdown("### 📈 Quality Metrics Dashboard")
            
            # Create 4-column metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                diversity_val = metrics.get('diversity', 0)
                st.metric("Diversity", f"{diversity_val:.3f}",
                         help="Average pairwise dissimilarity (0-1). Higher = more diverse recommendations")
                if diversity_val > 0.5:
                    st.caption("🟢 High diversity")
                elif diversity_val > 0.3:
                    st.caption("🟡 Moderate diversity")
                else:
                    st.caption("🔴 Low diversity")
            
            with col2:
                coverage_val = metrics.get('coverage', 0)
                st.metric("Coverage", f"{coverage_val:.1%}",
                         help="% of catalog recommended. Higher = better catalog exploration")
                if coverage_val > 0.3:
                    st.caption("🟢 Good coverage")
                elif coverage_val > 0.1:
                    st.caption("🟡 Moderate coverage")
                else:
                    st.caption("🔴 Low coverage")
            
            with col3:
                novelty_val = metrics.get('novelty', 0)
                st.metric("Novelty", f"{novelty_val:.3f}",
                         help="Inverse popularity (0-1). Higher = recommends less popular items")
                if novelty_val > 0.6:
                    st.caption("🟢 High novelty")
                elif novelty_val > 0.4:
                    st.caption("🟡 Moderate novelty")
                else:
                    st.caption("🔴 Low novelty")
            
            with col4:
                personalization_val = metrics.get('personalization', 0)
                st.metric("Personalization", f"{personalization_val:.3f}",
                         help="How different recommendations are between users (0-1). Higher = more personalized")
                if personalization_val > 0.5:
                    st.caption("🟢 Highly personalized")
                elif personalization_val > 0.3:
                    st.caption("🟡 Moderately personalized")
                else:
                    st.caption("🔴 Low personalization")
            
            # Additional metrics
            st.markdown("### 📊 Additional Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Users Analyzed", f"{metrics.get('num_users', 0):,}")
            with col2:
                st.metric("Unique Items Recommended", f"{metrics.get('num_unique_items', 0):,}")
            with col3:
                serendipity_val = metrics.get('serendipity_sample', 0)
                st.metric("Serendipity (Sample)", f"{serendipity_val:.3f}",
                         help="Unexpectedness score for sample user (0-1)")
            
            # Interpretation guide
            st.markdown("### 💡 Interpretation Guide")
            st.markdown("""
            - **Diversity**: Measures variety within recommendation lists. High diversity means recommendations are different from each other.
            - **Coverage**: % of item catalog being recommended. Low coverage means only popular items are recommended (filter bubble).
            - **Novelty**: Recommends less popular items. High novelty helps users discover hidden gems.
            - **Personalization**: How different recommendations are between users. Low = everyone gets similar items.
            - **Serendipity**: Unexpected recommendations that might delight users (different from what they usually consume).
            
            **Goal**: Balance accuracy with diversity. Perfect accuracy but low diversity = boring, predictable recommendations.
            """)
            
            # Recommendations for improvement
            st.markdown("### 🎯 Improvement Recommendations")
            
            improvements = []
            if diversity_val < 0.3:
                improvements.append("- **Low Diversity**: Consider using diversity re-ranking or MMR (Maximal Marginal Relevance)")
            if coverage_val < 0.1:
                improvements.append("- **Low Coverage**: Implement exploration strategies to recommend long-tail items")
            if novelty_val < 0.4:
                improvements.append("- **Low Novelty**: Boost weights for less popular items in recommendations")
            if personalization_val < 0.3:
                improvements.append("- **Low Personalization**: Ensure sufficient user history and fine-tune similarity calculations")
            
            if improvements:
                for improvement in improvements:
                    st.markdown(improvement)
            else:
                st.success("✅ **Excellent!** Your recommendation system has good diversity metrics across all dimensions!")
    
    # AI Insights
    if 'rec_similarity' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'rec_ai_insights' in st.session_state:
            st.markdown(st.session_state.rec_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="rec_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing recommendation engine performance and generating optimization strategies...", expanded=True) as status:
                    # Get data from session state
                    rec_type = st.session_state.rec_type
                    ratings_data = st.session_state.rec_ratings
                    similarity_matrix = st.session_state.rec_similarity
                    
                    # Calculate comprehensive metrics
                    n_users = ratings_data['user_id'].nunique()
                    n_items = ratings_data['item_id'].nunique()
                    n_ratings = len(ratings_data)
                    sparsity = 1 - (n_ratings / (n_users * n_items))
                    
                    # Rating statistics
                    avg_rating = ratings_data['rating'].mean()
                    rating_std = ratings_data['rating'].std()
                    min_rating = ratings_data['rating'].min()
                    max_rating = ratings_data['rating'].max()
                    
                    # User/Item engagement
                    ratings_per_user = ratings_data.groupby('user_id').size()
                    ratings_per_item = ratings_data.groupby('item_id').size()
                    avg_ratings_per_user = ratings_per_user.mean()
                    avg_ratings_per_item = ratings_per_item.mean()
                    
                    # Popular vs niche analysis
                    popular_items = (ratings_per_item > ratings_per_item.quantile(0.75)).sum()
                    niche_items = (ratings_per_item < ratings_per_item.quantile(0.25)).sum()
                    
                    # Coverage metrics
                    users_with_ratings = n_users
                    items_with_ratings = n_items
                    coverage_rate = (items_with_ratings / n_items * 100) if n_items > 0 else 0
                    
                    # Prepare rich context
                    context = f"""
Recommendation System Analysis:

System Configuration:
- Algorithm: {'User-based Collaborative Filtering' if rec_type == 'user' else 'Item-based Collaborative Filtering'}
- Total Users: {n_users:,}
- Total Items: {n_items:,}
- Total Ratings: {n_ratings:,}
- Data Sparsity: {sparsity*100:.2f}% (sparse matrix)

Rating Distribution:
- Average Rating: {avg_rating:.2f} / {max_rating:.0f}
- Rating Std Dev: {rating_std:.2f}
- Rating Range: {min_rating:.1f} - {max_rating:.1f}

User Engagement:
- Avg Ratings per User: {avg_ratings_per_user:.1f}
- Most Active User: {ratings_per_user.max()} ratings
- Least Active User: {ratings_per_user.min()} ratings

Item Popularity:
- Avg Ratings per Item: {avg_ratings_per_item:.1f}
- Popular Items (top 25%): {popular_items} items
- Niche Items (bottom 25%): {niche_items} items
- Catalog Coverage: {coverage_rate:.1f}%

System Characteristics:
- Data Density: {'Very sparse' if sparsity > 0.95 else 'Sparse' if sparsity > 0.90 else 'Moderate'} ({(1-sparsity)*100:.2f}% filled)
- User Behavior: {'Diverse' if rating_std > 1.0 else 'Consistent'} (std={rating_std:.2f})
- Catalog Balance: {'Popular-heavy' if popular_items > niche_items * 1.5 else 'Balanced'}
"""
                    
                    prompt = f"""
As a senior recommendation systems architect with 10+ years building personalization engines for e-commerce and streaming platforms, analyze these results and provide:

1. **System Performance Assessment** (3-4 sentences): Evaluate the recommendation engine's quality. How effective is the current algorithm given the data characteristics? What does the sparsity level tell us about recommendation reliability?

2. **Personalization Analysis** (4-5 bullet points): How well does the system understand users?
   - User segmentation insights (active vs. casual)
   - Item coverage and long-tail discovery
   - Popularity bias vs. personalization balance
   - Cold-start challenges for new users/items

3. **Algorithm Optimization** (5-6 bullet points): Technical improvements to enhance recommendation quality:
   - Similarity metric alternatives (cosine, Pearson, adjusted cosine)
   - Neighborhood size tuning
   - Matrix factorization opportunities (SVD, NMF)
   - Hybrid approaches (content + collaborative)
   - Real-time vs. batch processing
   - Confidence scoring and ranking

4. **Business Strategy** (5-6 bullet points): Revenue and engagement opportunities:
   - Cross-sell and upsell tactics
   - Bundle recommendations
   - Seasonal and trending item promotion
   - Email/notification personalization
   - A/B testing recommendation strategies
   - Customer lifetime value optimization

5. **User Experience Improvements** (4-5 bullet points): Making recommendations more useful:
   - Explanation and transparency ("Why we recommend this")
   - Diversity vs. accuracy balance
   - Serendipity and discovery features
   - User feedback loops (likes, dislikes, adjustments)
   - Interface and presentation optimization

6. **ROI & Metrics Tracking** (3-4 sentences): What business impact can we expect? Define KPIs to track (click-through rate, conversion rate, average order value). Estimate realistic improvements in engagement and revenue from implementing these recommendations.

{context}

Be specific, data-driven, and focus on actionable improvements that balance technical sophistication with business outcomes. Prioritize quick wins vs. long-term architectural changes.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior recommendation systems architect with 10+ years of experience building personalization engines for e-commerce and streaming platforms at scale. You specialize in collaborative filtering, matrix factorization, and hybrid recommendation approaches that drive measurable business outcomes."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.rec_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    # Display inside status block
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.rec_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export
    if 'rec_similarity' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        report = f"""# Recommendation System Report

## Overview
- **Method:** {method}
- **Total Ratings:** {len(ratings_data):,}
- **Unique Users:** {ratings_data['user_id'].nunique():,}
- **Unique Items:** {ratings_data['item_id'].nunique():,}
- **Sparsity:** {sparsity*100:.1f}%

## Metrics
- **Average Rating:** {ratings_data['rating'].mean():.2f}
- **Rating Distribution:** {ratings_data['rating'].value_counts().to_dict()}
"""
        
        if 'rec_ai_insights' in st.session_state:
            report += f"\n## AI Insights\n\n{st.session_state.rec_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - Recommendation Systems Module*\n"
        
        # Create CSV exports
        engine = st.session_state.rec_engine
        rec_type = st.session_state.rec_type
        ratings_data = st.session_state.rec_ratings
        
        # Generate sample recommendations for multiple users/items
        try:
            if rec_type == 'user':
                # Get top 5 users by number of ratings
                top_users = ratings_data.groupby('user_id').size().nlargest(5).index
                recommendations_list = []
                
                for user in top_users:
                    user_recs = engine.recommend_items_user_based(user, n_recommendations=5)
                    if not user_recs.empty:
                        for idx, row in user_recs.iterrows():
                            recommendations_list.append({
                                'user_id': user,
                                'recommended_item': row['item_id'],
                                'predicted_rating': f"{row['predicted_rating']:.4f}",
                                'rank': idx + 1
                            })
                
                recommendations_df = pd.DataFrame(recommendations_list)
            else:
                # Get top 5 users by number of ratings for item-based
                top_users = ratings_data.groupby('user_id').size().nlargest(5).index
                recommendations_list = []
                
                for user in top_users:
                    user_recs = engine.recommend_items_item_based(user, n_recommendations=5)
                    if not user_recs.empty:
                        for idx, row in user_recs.iterrows():
                            recommendations_list.append({
                                'user_id': user,
                                'recommended_item': row['item_id'],
                                'predicted_rating': f"{row['predicted_rating']:.4f}",
                                'rank': idx + 1
                            })
                
                recommendations_df = pd.DataFrame(recommendations_list)
            
            csv_string = recommendations_df.to_csv(index=False)
        except Exception as e:
            # Fallback: export ratings data
            csv_string = ratings_data.to_csv(index=False)
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"recommendations_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_geospatial_analysis():
    """Geospatial Analysis page."""
    st.markdown("<h2 style='text-align: center;'>🗺️ Geospatial Analysis & Location Intelligence</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Geospatial Analysis?"):
        st.markdown("""
        **Geospatial Analysis** examines patterns and relationships based on geographic location.
        
        ### Features Available:
        
        - **Interactive Maps:** Visualize locations on scatter maps
        - **Spatial Clustering:** DBSCAN clustering for geographic patterns
        - **Distance Calculations:** Haversine distance between points
        - **Density Analysis:** Identify hotspots and patterns
        
        ### Business Applications:
        - 📍 **Retail:** Store location optimization
        - 🚚 **Logistics:** Delivery route planning
        - 🏠 **Real Estate:** Market analysis
        - 📊 **Marketing:** Geographic targeting
        """)
    
    st.markdown("Analyze geographic data to uncover location-based insights.")
    
    # Import utilities
    from utils.geospatial_analysis import GeospatialAnalyzer
    
    # Initialize analyzer
    if 'geo_analyzer' not in st.session_state:
        st.session_state.geo_analyzer = GeospatialAnalyzer()
    
    analyzer = st.session_state.geo_analyzer
    
    # Data loading
    st.subheader("📤 1. Load Geographic Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample Store Locations", "Upload Custom Data"]
    else:
        data_options = ["Sample Store Locations", "Upload Custom Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="geo_data_source"
    )
    
    # Use Loaded Dataset
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_geospatial_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select latitude and longitude columns")
        
        # Pre-filter numeric columns for lat/lon
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            lat_default = suggestions['latitude'] if suggestions['latitude'] in numeric_cols else (numeric_cols[0] if numeric_cols else df.columns[0])
            lat_idx = list(df.columns).index(lat_default) if lat_default in df.columns else 0
            lat_col = st.selectbox("Latitude Column", df.columns, index=lat_idx, key="geo_lat")
        with col2:
            lon_default = suggestions['longitude'] if suggestions['longitude'] in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else df.columns[1] if len(df.columns) > 1 else df.columns[0])
            lon_idx = list(df.columns).index(lon_default) if lon_default in df.columns else 0
            lon_col = st.selectbox("Longitude Column", df.columns, index=lon_idx, key="geo_lon")
        
        # Real-time validation (similar to A/B Testing)
        issues = []
        warnings = []
        recommendations = []
        
        # Validate latitude column
        if not pd.api.types.is_numeric_dtype(df[lat_col]):
            issues.append(f"Latitude column '{lat_col}' is not numeric (type: {df[lat_col].dtype})")
            recommendations.append("Select a numeric column containing latitude values between -90 and 90")
        else:
            lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
            if lat_min < -90 or lat_max > 90:
                issues.append(f"Latitude values out of range: {lat_min:.2f} to {lat_max:.2f} (must be -90 to 90)")
                recommendations.append("Check your data - latitude must be between -90 and 90 degrees")
            
            # Check for missing values
            lat_missing = df[lat_col].isnull().sum()
            if lat_missing > 0:
                warnings.append(f"Latitude column has {lat_missing} missing values ({lat_missing/len(df)*100:.1f}%)")
        
        # Validate longitude column
        if not pd.api.types.is_numeric_dtype(df[lon_col]):
            issues.append(f"Longitude column '{lon_col}' is not numeric (type: {df[lon_col].dtype})")
            recommendations.append("Select a numeric column containing longitude values between -180 and 180")
        else:
            lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
            if lon_min < -180 or lon_max > 180:
                issues.append(f"Longitude values out of range: {lon_min:.2f} to {lon_max:.2f} (must be -180 to 180)")
                recommendations.append("Check your data - longitude must be between -180 and 180 degrees")
            
            # Check for missing values
            lon_missing = df[lon_col].isnull().sum()
            if lon_missing > 0:
                warnings.append(f"Longitude column has {lon_missing} missing values ({lon_missing/len(df)*100:.1f}%)")
        
        # Check if both columns are the same
        if lat_col == lon_col:
            issues.append("Latitude and Longitude cannot be the same column")
            recommendations.append("Select different columns for latitude and longitude")
        
        # Check data quality
        if pd.api.types.is_numeric_dtype(df[lat_col]) and pd.api.types.is_numeric_dtype(df[lon_col]):
            valid_data = df[[lat_col, lon_col]].dropna()
            if len(valid_data) < 10:
                warnings.append(f"Only {len(valid_data)} valid locations after removing missing values")
                recommendations.append("Consider data with at least 10 valid coordinate pairs for meaningful analysis")
            
            # Check geographic spread
            if len(valid_data) > 0:
                lat_range = valid_data[lat_col].max() - valid_data[lat_col].min()
                lon_range = valid_data[lon_col].max() - valid_data[lon_col].min()
                if lat_range < 0.01 and lon_range < 0.01:
                    warnings.append("Very limited geographic spread - all points are in a small area")
                    recommendations.append("Spatial clustering works best with geographically distributed data")
        
        # Display validation results
        st.divider()
        if len(issues) > 0:
            st.error("**🚨 NOT SUITABLE FOR GEOSPATIAL ANALYSIS**")
            for issue in issues:
                st.write(f"❌ {issue}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        elif len(warnings) > 0:
            st.warning("**⚠️ GEOSPATIAL ANALYSIS POSSIBLE (with warnings)**")
            for warning in warnings:
                st.write(f"⚠️ {warning}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        else:
            st.success("**✅ EXCELLENT FOR GEOSPATIAL ANALYSIS**")
            st.write("✅ Numeric latitude and longitude columns")
            st.write("✅ Values within valid geographic ranges")
            st.write("✅ Good data quality")
        
        # Only show button if no critical issues
        if len(issues) == 0 and st.button("📊 Process Data", type="primary"):
            geo_data = df[[lat_col, lon_col]].dropna().copy()
            geo_data.columns = ['latitude', 'longitude']
            st.session_state.geo_data = geo_data
            
            st.success("✅ Data processed!")
            st.info(f"📍 {len(geo_data)} locations ready for spatial analysis")
            st.rerun()
    
    elif data_source == "Sample Store Locations":
        if st.button("📥 Load Sample Store Locations", type="primary"):
            # Generate sample geographic data (US cities)
            np.random.seed(42)
            
            cities = [
                (40.7128, -74.0060, "New York"),
                (34.0522, -118.2437, "Los Angeles"),
                (41.8781, -87.6298, "Chicago"),
                (29.7604, -95.3698, "Houston"),
                (33.4484, -112.0740, "Phoenix"),
                (39.7392, -104.9903, "Denver"),
                (47.6062, -122.3321, "Seattle"),
                (37.7749, -122.4194, "San Francisco"),
                (42.3601, -71.0589, "Boston"),
                (25.7617, -80.1918, "Miami")
            ]
            
            # Add some noise to create multiple stores per city
            locations = []
            for lat, lon, city in cities:
                for i in range(5):  # 5 stores per city
                    locations.append({
                        'latitude': lat + np.random.normal(0, 0.1),
                        'longitude': lon + np.random.normal(0, 0.1),
                        'city': city,
                        'store_id': f"{city[:3].upper()}-{i+1}"
                    })
            
            geo_data = pd.DataFrame(locations)
            st.session_state.geo_data = geo_data
            
            st.success(f"✅ Loaded {len(geo_data)} store locations across {len(cities)} cities!")
            st.dataframe(geo_data.head(10), use_container_width=True)
    
    else:  # Upload
        uploaded_file = st.file_uploader("Upload geographic data CSV", type=['csv'], key="geo_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                lat_col = st.selectbox("Latitude Column", df.select_dtypes(include=[np.number]).columns, key="geo_lat_upload")
            with col2:
                lon_col = st.selectbox("Longitude Column", df.select_dtypes(include=[np.number]).columns, key="geo_lon_upload")
            
            if st.button("Process Data", type="primary", key="geo_process_upload"):
                geo_data = df[[lat_col, lon_col]].copy()
                geo_data.columns = ['latitude', 'longitude']
                st.session_state.geo_data = geo_data
                st.success("✅ Data processed!")
    
    # Analysis section
    if 'geo_data' not in st.session_state:
        st.info("👆 Load geographic data to begin spatial analysis")
        return
    
    geo_data = st.session_state.geo_data
    
    # Display loaded data preview
    with st.expander("👁️ View Loaded Geographic Data", expanded=False):
        st.dataframe(geo_data.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(geo_data)} locations")
    
    # Dataset overview
    st.divider()
    st.subheader("📊 2. Spatial Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Locations", f"{len(geo_data):,}")
    with col2:
        lat_range = geo_data['latitude'].max() - geo_data['latitude'].min()
        st.metric("Latitude Range", f"{lat_range:.2f}°")
    with col3:
        lon_range = geo_data['longitude'].max() - geo_data['longitude'].min()
        st.metric("Longitude Range", f"{lon_range:.2f}°")
    
    # Clustering
    cluster_method = st.radio(
        "Clustering Method:",
        ["DBSCAN (Density-Based)", "K-Means"],
        horizontal=True,
        key="geo_cluster"
    )
    
    if cluster_method == "DBSCAN (Density-Based)":
        eps = st.slider("Max Distance (km)", 1, 100, 10, key="geo_eps")
        min_samples = st.slider("Min Points per Cluster", 2, 10, 3, key="geo_min_samples")
    else:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="geo_n_clusters")
    
    if st.button("🗺️ Run Spatial Analysis", type="primary"):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Geospatial_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Analyzing geographic patterns...", expanded=True) as status:
                # Get analyzer from session state
                analyzer = st.session_state.geo_analyzer
                
                # Fit analyzer on data
                analyzer.fit(geo_data, lat_col='latitude', lon_col='longitude')
                
                # Perform clustering
                if cluster_method == "DBSCAN (Density-Based)":
                    clusters = analyzer.perform_clustering(method='dbscan')
                else:
                    clusters = analyzer.perform_clustering(n_clusters=n_clusters, method='kmeans')
                
                # Store results
                geo_data_with_clusters = geo_data.copy()
                geo_data_with_clusters['cluster'] = clusters
                result = {
                    'data': geo_data_with_clusters,
                    'lat_col': 'latitude',
                    'lon_col': 'longitude',
                    'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                    'noise_points': list(clusters).count(-1) if cluster_method == "DBSCAN (Density-Based)" else 0
                }
                st.session_state.geo_results = result
                st.session_state.geo_data = geo_data_with_clusters
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            st.success("✅ Spatial clustering completed!")
        except Exception as e:
            st.error(f"❌ Error during spatial analysis: {str(e)}")
        finally:
            # Always unlock navigation
            pm.unlock()
    
    # Display results if they exist
    if 'geo_results' in st.session_state:
        result = st.session_state.geo_results
        analyzer = st.session_state.geo_analyzer
        geo_data = st.session_state.geo_data
        
        st.divider()
        st.subheader("📊 Clustering Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters Found", result['n_clusters'])
        with col2:
            if 'noise_points' in result:
                st.metric("Noise Points", result['noise_points'])
        
        # Map visualization
        st.subheader("🗺️ Interactive Map")
        fig = analyzer.create_scatter_map(
            result['data'], 
            lat_col='latitude', 
            lon_col='longitude',
            color_col='cluster',
            title='Spatial Clusters'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Expansion Analysis
        st.divider()
        st.subheader("📈 Market Expansion Analysis")
        st.markdown("Identify high-potential areas for business expansion based on current presence and value metrics.")
        
        with st.expander("ℹ️ What is Market Expansion Analysis?"):
            st.markdown("""
            **Market Expansion Analysis** identifies optimal locations for business growth by analyzing:
            
            **Analysis Components:**
            - 🎯 **Opportunity Scoring**: Areas with high potential but low current coverage
            - 📊 **Density Analysis**: Current market presence vs potential value
            - 🗺️ **Geographic Gaps**: Underserved regions with growth potential
            - 📍 **Quadrant Analysis**: Regional opportunities (NE, NW, SE, SW)
            
            **Use Cases:**
            - New store location selection
            - Service area expansion
            - Market entry strategy
            - Resource allocation planning
            
            **How It Works:**
            - **High Opportunity** = High value potential + Low current presence
            - **Low Opportunity** = Already saturated or low value
            """)
        
        # Value column selection
        numeric_cols = geo_data.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [col for col in numeric_cols if col not in [result['lat_col'], result['lon_col']]]
        
        if len(value_cols) > 0:
            value_col = st.selectbox(
                "Select Value Metric (Optional):",
                ["None"] + value_cols,
                help="Choose a business metric (e.g., revenue, customers) to weight opportunities"
            )
            
            if value_col == "None":
                value_col = None
        else:
            value_col = None
            st.info("💡 No numeric value columns found. Analysis will be based on location density only.")
        
        # Grid resolution slider
        grid_resolution = st.slider(
            "Analysis Grid Resolution:",
            min_value=10,
            max_value=30,
            value=20,
            help="Higher resolution = more detailed analysis (slower)"
        )
        
        # Run expansion analysis
        if st.button("🔍 Analyze Expansion Opportunities", type="primary"):
            with st.status("Analyzing market expansion opportunities...", expanded=True) as status:
                try:
                    expansion_results = analyzer.analyze_market_expansion(
                        value_col=value_col,
                        grid_resolution=grid_resolution
                    )
                    
                    st.session_state.expansion_results = expansion_results
                    
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    st.success("✅ Expansion analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display expansion results
        if 'expansion_results' in st.session_state:
            exp_res = st.session_state.expansion_results
            
            st.markdown("### 🎯 Top Expansion Opportunities")
            
            # Top opportunities table
            opp_df = pd.DataFrame(exp_res['top_opportunities'])
            st.dataframe(opp_df, use_container_width=True)
            
            # Market saturation metrics
            st.markdown("### 📊 Market Coverage Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Market Saturation", f"{exp_res['market_saturation_pct']:.1f}%",
                         help="Percentage of geographic area with current presence")
            with col2:
                st.metric("Covered Cells", f"{exp_res['covered_cells']:,}",
                         help="Number of grid cells with existing locations")
            with col3:
                st.metric("Uncovered Cells", f"{exp_res['total_cells'] - exp_res['covered_cells']:,}",
                         help="Number of grid cells without presence")
            with col4:
                best_q = exp_res['best_quadrant']
                st.metric("Best Quadrant", best_q,
                         help="Geographic quadrant with highest opportunity")
            
            # Saturation interpretation
            saturation = exp_res['market_saturation_pct']
            if saturation < 20:
                st.success(f"""
                ✅ **Low Saturation ({saturation:.1f}%)**
                
                Significant expansion opportunities available! Focus on:
                - Entering underserved markets
                - Building brand presence
                - Establishing market leadership
                """)
            elif saturation < 50:
                st.info(f"""
                ℹ️ **Moderate Saturation ({saturation:.1f}%)**
                
                Balanced growth potential. Consider:
                - Selective expansion in high-opportunity areas
                - Densifying presence in existing markets
                - Strategic gap filling
                """)
            else:
                st.warning(f"""
                ⚠️ **High Saturation ({saturation:.1f}%)**
                
                Market is well-covered. Options:
                - Focus on underperforming locations
                - Consider adjacent markets
                - Optimize existing footprint
                """)
            
            # Quadrant analysis
            st.markdown("### 🧭 Regional Opportunity Analysis")
            
            quadrant_df = pd.DataFrame({
                'Quadrant': list(exp_res['quadrant_scores'].keys()),
                'Opportunity Score': list(exp_res['quadrant_scores'].values())
            }).sort_values('Opportunity Score', ascending=False)
            
            import plotly.express as px
            fig_quad = px.bar(
                quadrant_df,
                x='Quadrant',
                y='Opportunity Score',
                title='Opportunity by Geographic Quadrant',
                color='Opportunity Score',
                color_continuous_scale='YlOrRd'
            )
            fig_quad.update_layout(height=400)
            st.plotly_chart(fig_quad, use_container_width=True)
            
            # Expansion heatmap
            st.markdown("### 🗺️ Opportunity Heatmap")
            
            lat_range = (geo_data[result['lat_col']].min(), geo_data[result['lat_col']].max())
            lon_range = (geo_data[result['lon_col']].min(), geo_data[result['lon_col']].max())
            
            fig_heatmap = GeospatialAnalyzer.create_expansion_heatmap(
                exp_res,
                lat_range,
                lon_range
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.caption("🌟 Stars indicate top 5 expansion opportunities. Brighter colors = higher opportunity.")
            
            # Strategic recommendations
            st.markdown("### 💡 Strategic Recommendations")
            
            top_3 = exp_res['top_opportunities'][:3]
            
            st.markdown("**Priority Expansion Targets:**")
            for opp in top_3:
                st.markdown(f"""
                **#{opp['rank']} - ({opp['latitude']:.4f}, {opp['longitude']:.4f})**
                - Opportunity Score: {opp['opportunity_score']:.3f}
                - Current Presence: {opp['current_density']:.1f} locations
                - Value Potential: {opp['value_potential']:.2f}
                """)
            
            # Resource allocation
            if saturation < 50:
                st.success("""
                🎯 **Recommended Strategy: Expansion Focus**
                
                - **Allocate 70%** of resources to new market entry
                - **Allocate 30%** to optimizing existing locations
                - **Target**: Top 5 opportunities for maximum ROI
                - **Timeline**: Phased rollout starting with #1 ranked location
                """)
            else:
                st.info("""
                🎯 **Recommended Strategy: Optimization Focus**
                
                - **Allocate 40%** to selective expansion (top opportunities only)
                - **Allocate 60%** to improving existing footprint
                - **Target**: Fill strategic gaps between current locations
                - **Timeline**: Focus on quick wins in partially covered areas
                """)
    
    # AI Insights
    if 'geo_results' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'geo_ai_insights' in st.session_state:
            st.markdown(st.session_state.geo_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="geo_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing geographic patterns and generating location intelligence strategies...", expanded=True) as status:
                    # Get data from session state
                    result = st.session_state.geo_results
                    geo_data = st.session_state.geo_data
                    
                    # Calculate comprehensive metrics
                    total_locations = len(geo_data)
                    n_clusters = result['n_clusters']
                    method = result.get('method', 'Unknown')
                    
                    # Geographic bounds
                    lat_min = geo_data['latitude'].min()
                    lat_max = geo_data['latitude'].max()
                    lon_min = geo_data['longitude'].min()
                    lon_max = geo_data['longitude'].max()
                    lat_range = lat_max - lat_min
                    lon_range = lon_max - lon_min
                    lat_center = (lat_min + lat_max) / 2
                    lon_center = (lon_min + lon_max) / 2
                    
                    # Calculate approximate area (rough estimate in km²)
                    lat_km = lat_range * 111  # 1° latitude ≈ 111 km
                    lon_km = lon_range * 111 * abs(np.cos(np.radians(lat_center)))  # longitude varies with latitude
                    area_km2 = lat_km * lon_km
                    
                    # Cluster analysis
                    clustered_data = result.get('data', geo_data)
                    if 'cluster' in clustered_data.columns:
                        cluster_sizes = clustered_data['cluster'].value_counts()
                        avg_cluster_size = cluster_sizes.mean()
                        largest_cluster = cluster_sizes.max()
                        smallest_cluster = cluster_sizes.min()
                        cluster_size_std = cluster_sizes.std()
                    else:
                        avg_cluster_size = largest_cluster = smallest_cluster = cluster_size_std = 0
                    
                    # Density analysis
                    density = total_locations / max(area_km2, 1)
                    locations_per_cluster = total_locations / max(n_clusters, 1)
                    
                    # Noise analysis (for DBSCAN)
                    noise_points = result.get('noise_points', 0)
                    noise_percentage = (noise_points / total_locations * 100) if total_locations > 0 else 0
                    
                    # Spatial distribution
                    lat_std = geo_data['latitude'].std()
                    lon_std = geo_data['longitude'].std()
                    dispersion = np.sqrt(lat_std**2 + lon_std**2)
                    
                    # Prepare rich context
                    context = f"""
Geospatial Analysis Results:

Study Area Overview:
- Total Locations Analyzed: {total_locations:,}
- Geographic Bounds: {lat_min:.4f}° to {lat_max:.4f}° latitude, {lon_min:.4f}° to {lon_max:.4f}° longitude
- Geographic Spread: {lat_range:.2f}° × {lon_range:.2f}° ({lat_km:.1f} km × {lon_km:.1f} km)
- Approximate Coverage Area: {area_km2:.1f} km²
- Geographic Center: {lat_center:.4f}°N, {lon_center:.4f}°E

Clustering Results:
- Method Used: {method}
- Clusters Identified: {n_clusters}
- Average Locations per Cluster: {locations_per_cluster:.1f}
- Largest Cluster Size: {largest_cluster} locations
- Smallest Cluster Size: {smallest_cluster} locations
- Cluster Size Variation (std): {cluster_size_std:.1f}

Density Analysis:
- Overall Location Density: {density:.2f} locations per km²
- Spatial Dispersion Index: {dispersion:.4f}
- Distribution Pattern: {'Highly concentrated' if dispersion < 0.5 else 'Moderately distributed' if dispersion < 2.0 else 'Widely dispersed'}
"""
                    
                    # Add noise analysis for DBSCAN
                    if noise_points > 0:
                        context += f"""
DBSCAN Noise Analysis:
- Noise Points (Outliers): {noise_points} ({noise_percentage:.1f}%)
- Clustered Locations: {total_locations - noise_points} ({100 - noise_percentage:.1f}%)
- Clustering Efficiency: {'High' if noise_percentage < 10 else 'Moderate' if noise_percentage < 25 else 'Low'}
"""
                    
                    context += f"""
Spatial Characteristics:
- Geographic Scale: {'Local' if area_km2 < 100 else 'Regional' if area_km2 < 10000 else 'National/Continental'}
- Cluster Compactness: {'Tight' if avg_cluster_size / locations_per_cluster > 0.7 else 'Moderate' if avg_cluster_size / locations_per_cluster > 0.4 else 'Loose'}
- Market Saturation Level: {'High density' if density > 10 else 'Moderate density' if density > 1 else 'Low density'}
"""
                    
                    prompt = f"""
As a senior geospatial analyst and location intelligence strategist with 10+ years of experience in site selection, territory optimization, and spatial economics, analyze these geospatial clustering results and provide:

1. **Spatial Pattern Assessment** (3-4 sentences): Evaluate the geographic distribution and clustering patterns. What does the cluster structure reveal about market concentration, accessibility, and coverage? How does the density and dispersion inform our understanding of the spatial dynamics?

2. **Cluster Characteristics** (4-5 bullet points): Detailed insights about each major cluster:
   - Dominant geographic areas (city, region, terrain)
   - Cluster size and density implications
   - Coverage gaps and overlap
   - Competitive positioning in each cluster
   - Accessibility and transportation considerations

3. **Market Expansion Strategy** (5-6 bullet points): Data-driven recommendations for growth:
   - Underserved geographic areas with high potential
   - Optimal locations for new sites/facilities/stores
   - Market penetration vs. geographic expansion trade-offs
   - Competitor clustering and white space opportunities
   - Demographic and economic considerations by cluster
   - Priority ranking of expansion territories

4. **Operational Optimization** (4-5 bullet points): Improve efficiency using spatial insights:
   - Service territory redesign and optimization
   - Resource allocation across clusters
   - Logistics and distribution routing
   - Field team territory assignments
   - Capacity planning by geographic zone

5. **Risk & Constraints** (3-4 bullet points): Geographic challenges to address:
   - Over-concentration risks in certain areas
   - Underserved markets with high opportunity cost
   - Geographic barriers (terrain, infrastructure)
   - Competitive saturation in specific clusters

6. **ROI Projection** (3-4 sentences): Quantify the business impact of implementing these location strategies. What improvements in market coverage, operational efficiency, and revenue can be expected? Consider both quick wins (existing cluster optimization) and long-term gains (strategic expansion).

{context}

Be specific, data-driven, and focus on actionable location intelligence strategies that balance market opportunity with operational feasibility. Consider both tactical improvements and strategic positioning.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior geospatial analyst and location intelligence strategist with 10+ years of experience in site selection, territory optimization, and spatial economics. You specialize in translating spatial data into actionable business strategies for retail, logistics, and service organizations."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.geo_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    # Display inside status block
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.geo_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export
    if 'geo_results' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        result = st.session_state.geo_results
        geo_data = st.session_state.geo_data
        
        # Calculate metrics for report
        lat_range = geo_data['latitude'].max() - geo_data['latitude'].min()
        lon_range = geo_data['longitude'].max() - geo_data['longitude'].min()
        
        report = f"""# Geospatial Analysis Report

## Overview
- **Total Locations:** {len(geo_data):,}
- **Clusters Found:** {result['n_clusters']}
- **Geographic Spread:** {lat_range:.2f}° × {lon_range:.2f}°

## Cluster Summary
{result.get('cluster_summary', 'N/A')}
"""
        
        if 'geo_ai_insights' in st.session_state:
            report += f"\n## AI Insights\n\n{st.session_state.geo_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - Geospatial Analysis Module*\n"
        
        # Create CSV export with location data and cluster assignments
        csv_string = geo_data.to_csv(index=False)
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"geospatial_clusters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"geospatial_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_survival_analysis():
    """Survival Analysis page."""
    st.markdown("<h2 style='text-align: center;'>⏱️ Survival Analysis & Time-to-Event Modeling</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Survival Analysis?"):
        st.markdown("""
        **Survival Analysis** models time-to-event data and predicts the probability of an event occurring over time.
        
        ### Methods Available:
        
        - **Kaplan-Meier Estimator:** Non-parametric survival curve estimation
        - **Log-Rank Test:** Compare survival between groups
        - **Median Survival Time:** Half-life of population
        
        ### Business Applications:
        - 📉 **Customer Retention:** Churn prediction
        - ⚙️ **Maintenance:** Equipment failure forecasting
        - 💳 **Subscriptions:** Cancellation modeling
        - 🎯 **Marketing:** Time-to-conversion analysis
        """)
    
    st.markdown("Analyze time-to-event data and predict survival probabilities.")
    
    # Import utilities
    from utils.survival_analysis import SurvivalAnalyzer
    
    # Initialize analyzer
    if 'survival_analyzer' not in st.session_state:
        try:
            st.session_state.survival_analyzer = SurvivalAnalyzer()
        except ImportError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 The `lifelines` package is being installed on Streamlit Cloud. Please refresh the page in a few moments.")
            st.stop()
    
    analyzer = st.session_state.survival_analyzer
    
    # Data loading
    st.subheader("📤 1. Load Survival Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample Customer Churn Data", "Upload Custom Data"]
    else:
        data_options = ["Sample Customer Churn Data", "Upload Custom Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="surv_data_source"
    )
    
    # Use Loaded Dataset
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_survival_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select time (duration), event, and optional group columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            time_default = suggestions['time']
            time_idx = list(df.columns).index(time_default)
            time_col = st.selectbox("Time/Duration Column", df.columns, index=time_idx, key="surv_time")
        with col2:
            event_default = suggestions['event']
            event_idx = list(df.columns).index(event_default)
            event_col = st.selectbox("Event Column (1=event occurred, 0=censored)", df.columns, index=event_idx, key="surv_event")
        with col3:
            group_default = suggestions['group']
            group_options = ["None"] + df.columns.tolist()
            group_idx = group_options.index(group_default) if group_default and group_default in group_options else 0
            group_col = st.selectbox("Group Column (optional)", group_options, index=group_idx, key="surv_group")
        
        # Real-time validation
        issues = []
        warnings = []
        recommendations = []
        
        # Validate time/duration column is numeric
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            issues.append(f"Time column '{time_col}' is not numeric (type: {df[time_col].dtype})")
            recommendations.append("Select a numeric column containing time durations (days, months, etc.)")
        else:
            time_min, time_max = df[time_col].min(), df[time_col].max()
            time_missing = df[time_col].isnull().sum()
            
            # Check for negative or zero durations
            if time_min <= 0:
                issues.append(f"Time column has non-positive values (min: {time_min})")
                recommendations.append("Time durations must be positive (> 0)")
            
            # Check for missing values
            if time_missing > 0:
                warnings.append(f"Time column has {time_missing} missing values ({time_missing/len(df)*100:.1f}%)")
            
            # Check if all durations are the same
            if df[time_col].nunique() == 1:
                warnings.append(f"All durations are the same value ({time_min})")
                recommendations.append("Survival analysis works better with variation in time durations")
        
        # Validate event column is binary
        try:
            event_values = df[event_col].dropna()
            unique_events = event_values.unique()
            
            # Check if binary
            if len(unique_events) > 2:
                issues.append(f"Event column has {len(unique_events)} unique values (need 2: event/censored)")
                recommendations.append("Event column must be binary: 1/0, True/False, or Yes/No")
            elif len(unique_events) == 1:
                warnings.append(f"Event column only has one value ({unique_events[0]}) - all events or all censored")
                recommendations.append("Survival analysis works better with both events and censored observations")
            else:
                # Check if values are valid binary
                valid_binary = all(val in [0, 1, True, False, 'Yes', 'No', 'yes', 'no'] for val in unique_events)
                if not valid_binary:
                    issues.append(f"Event column values are not binary: {unique_events.tolist()}")
                    recommendations.append("Use 1/0, True/False, or Yes/No for event indicator")
            
            # Check event missing values
            event_missing = df[event_col].isnull().sum()
            if event_missing > 0:
                warnings.append(f"Event column has {event_missing} missing values ({event_missing/len(df)*100:.1f}%)")
        except Exception as e:
            issues.append(f"Cannot validate event column: {str(e)}")
        
        # Check for sufficient observations
        n_obs = len(df)
        if n_obs < 10:
            issues.append(f"Only {n_obs} observations (need at least 10 for survival analysis)")
            recommendations.append("Survival curves require sufficient observations for reliability")
        elif n_obs < 30:
            warnings.append(f"Only {n_obs} observations - survival estimates may be unstable")
        
        # Check event rate if valid
        if pd.api.types.is_numeric_dtype(df[time_col]):
            try:
                event_rate = event_values.astype(int).mean()
                if event_rate < 0.05:
                    warnings.append(f"Very low event rate ({event_rate*100:.1f}%) - mostly censored data")
                    recommendations.append("Survival analysis works better with some observed events")
                elif event_rate > 0.95:
                    warnings.append(f"Very high event rate ({event_rate*100:.1f}%) - few censored observations")
            except:
                pass
        
        # Validate group column if selected
        if group_col != "None":
            n_groups = df[group_col].nunique()
            if n_groups < 2:
                warnings.append(f"Group column '{group_col}' only has {n_groups} unique value(s)")
                recommendations.append("Group comparison requires at least 2 groups")
            elif n_groups > 10:
                warnings.append(f"Group column has {n_groups} groups - survival curves may be cluttered")
        
        # Check if columns are the same
        if time_col == event_col:
            issues.append("Time and Event columns must be different")
            recommendations.append("Select distinct columns for duration and event indicator")
        
        # Display validation results
        st.divider()
        if len(issues) > 0:
            st.error("**🚨 NOT SUITABLE FOR SURVIVAL ANALYSIS**")
            for issue in issues:
                st.write(f"❌ {issue}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        elif len(warnings) > 0:
            st.warning("**⚠️ SURVIVAL ANALYSIS POSSIBLE (with warnings)**")
            for warning in warnings:
                st.write(f"⚠️ {warning}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        else:
            st.success("**✅ EXCELLENT FOR SURVIVAL ANALYSIS**")
            st.write("✅ Numeric time/duration column")
            st.write("✅ Binary event column")
            st.write("✅ Sufficient observations")
        
        # Only show button if no critical issues
        if len(issues) == 0 and st.button("📊 Process Data", type="primary"):
            surv_data = df[[time_col, event_col]].copy()
            surv_data.columns = ['duration', 'event']
            
            # Convert event to binary if needed
            if surv_data['event'].dtype == 'object':
                surv_data['event'] = surv_data['event'].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0, True: 1, False: 0})
            surv_data['event'] = surv_data['event'].astype(int)
            
            if group_col != "None":
                surv_data['group'] = df[group_col]
            
            # Remove rows with missing values
            surv_data = surv_data.dropna()
            
            st.session_state.surv_data = surv_data
            
            st.success("✅ Data processed!")
            st.info(f"📊 {len(surv_data)} observations, {surv_data['event'].sum()} events")
            st.rerun()
    
    elif data_source == "Sample Customer Churn Data":
        if st.button("📥 Load Sample Churn Data", type="primary"):
            # Generate sample survival/churn data
            np.random.seed(42)
            
            n_customers = 500
            
            # Simulate customer tenure and churn
            data = []
            for i in range(n_customers):
                segment = np.random.choice(['Basic', 'Premium'], p=[0.6, 0.4])
                
                # Premium customers have lower churn hazard
                if segment == 'Premium':
                    duration = np.random.exponential(24)  # Average 24 months
                    churn_prob = 0.3
                else:
                    duration = np.random.exponential(12)  # Average 12 months
                    churn_prob = 0.5
                
                churned = np.random.binomial(1, churn_prob)
                
                data.append({
                    'customer_id': f'C{i+1:04d}',
                    'duration': max(1, int(duration)),
                    'event': churned,
                    'segment': segment
                })
            
            surv_data = pd.DataFrame(data)
            surv_data['group'] = surv_data['segment']
            st.session_state.surv_data = surv_data
            
            st.success(f"✅ Loaded {len(surv_data)} customer records!")
            st.dataframe(surv_data.head(10), use_container_width=True)
    
    else:  # Upload
        uploaded_file = st.file_uploader("Upload survival data CSV", type=['csv'], key="surv_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                time_col = st.selectbox("Time/Duration Column", df.select_dtypes(include=[np.number]).columns, key="surv_time_upload")
            with col2:
                event_col = st.selectbox("Event Column", df.columns, key="surv_event_upload")
            with col3:
                group_col = st.selectbox("Group Column (optional)", ["None"] + list(df.columns), key="surv_group_upload")
            
            if st.button("Process Data", type="primary", key="surv_process_upload"):
                surv_data = df[[time_col, event_col]].copy()
                surv_data.columns = ['duration', 'event']
                if group_col != "None":
                    surv_data['group'] = df[group_col]
                st.session_state.surv_data = surv_data
                st.success("✅ Data processed!")
    
    # Analysis section
    if 'surv_data' not in st.session_state:
        st.info("👆 Load survival data to begin analysis")
        return
    
    surv_data = st.session_state.surv_data
    
    # Display loaded data preview
    with st.expander("👁️ View Loaded Survival Data", expanded=False):
        st.dataframe(surv_data.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(surv_data)} observations")
    
    # Dataset overview
    st.divider()
    st.subheader("📊 2. Survival Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{len(surv_data):,}")
    with col2:
        st.metric("Events", f"{surv_data['event'].sum():,}")
    with col3:
        st.metric("Censored", f"{(~surv_data['event'].astype(bool)).sum():,}")
    with col4:
        event_rate = surv_data['event'].mean() * 100
        st.metric("Event Rate", f"{event_rate:.1f}%")
    
    # Run analysis
    if st.button("⏱️ Run Survival Analysis", type="primary"):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Survival_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Fitting survival models...", expanded=True) as status:
                # Get analyzer from session state  
                analyzer = st.session_state.survival_analyzer
                
                if 'group' in surv_data.columns:
                    # Perform log-rank test
                    logrank = analyzer.perform_logrank_test(
                        surv_data, 
                        duration_col='duration', 
                        event_col='event', 
                        group_col='group'
                    )
                    result = {
                        'has_groups': True,
                        'log_rank_p': logrank['p_value'],
                        'log_rank_stat': logrank['test_statistic'],
                        'groups': surv_data['group'].unique().tolist(),
                        'median_survival': {}
                    }
                    # Calculate median survival for each group
                    for group in result['groups']:
                        group_data = surv_data[surv_data['group'] == group]
                        analyzer.fit_kaplan_meier(group_data, 'duration', 'event', label=str(group))
                        result['median_survival'][group] = analyzer.get_median_survival_time()
                else:
                    # Overall survival
                    analyzer.fit_kaplan_meier(surv_data, 'duration', 'event')
                    result = {
                        'has_groups': False,
                        'median_survival': analyzer.get_median_survival_time()
                    }
                
                st.session_state.surv_results = result
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            st.success("✅ Survival analysis completed!")
        except Exception as e:
            st.error(f"❌ Error during survival analysis: {str(e)}")
        finally:
            # Always unlock navigation
            pm.unlock()
    
    # Display results if they exist
    if 'surv_results' in st.session_state:
        result = st.session_state.surv_results
        analyzer = st.session_state.survival_analyzer
        surv_data = st.session_state.surv_data
        
        st.divider()
        # Display results
        st.subheader("📈 Survival Curve")
        fig = analyzer.create_kaplan_meier_plot(
            surv_data,
            duration_col='duration',
            event_col='event',
            group_col='group' if 'group' in surv_data.columns else None
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Median survival time
        st.subheader("📊 Key Metrics")
        if result.get('has_groups', False):
            col1, col2 = st.columns(2)
            groups = result['groups']
            with col1:
                st.metric(f"Median Survival ({groups[0]})", f"{result['median_survival'][groups[0]]:.1f} time units")
            with col2:
                if len(groups) > 1:
                    st.metric(f"Median Survival ({groups[1]})", f"{result['median_survival'][groups[1]]:.1f} time units")
            
            # Log-rank test
            if 'log_rank_p' in result:
                if result['log_rank_p'] < 0.05:
                    st.success(f"✅ **Significant difference** between groups (p={result['log_rank_p']:.4f})")
                else:
                    st.info(f"ℹ️ No significant difference between groups (p={result['log_rank_p']:.4f})")
        else:
            st.metric("Median Survival Time", f"{result['median_survival']:.1f} time units")
        
        # Cox Proportional Hazards Model
        st.divider()
        st.subheader("🔬 Cox Proportional Hazards Model")
        st.markdown("Identify which factors affect survival time using regression analysis.")
        
        # Get potential covariates (numeric columns except duration and event)
        numeric_cols = surv_data.select_dtypes(include=[np.number]).columns.tolist()
        potential_covariates = [col for col in numeric_cols if col not in ['duration', 'event']]
        
        # Check if group column exists (categorical covariate)
        if 'group' in surv_data.columns:
            st.info("💡 **Tip:** The 'group' column will be converted to a numeric indicator for Cox regression.")
            
            # Convert group to numeric
            surv_data_cox = surv_data.copy()
            group_values = surv_data['group'].unique()
            if len(group_values) == 2:
                # Binary encoding
                surv_data_cox['group_indicator'] = (surv_data['group'] == group_values[1]).astype(int)
                potential_covariates.append('group_indicator')
        else:
            surv_data_cox = surv_data.copy()
        
        if len(potential_covariates) > 0:
            # Covariate selection
            selected_covariates = st.multiselect(
                "Select Covariates (independent variables):",
                potential_covariates,
                default=potential_covariates[:min(3, len(potential_covariates))],
                help="Select variables that might affect survival time"
            )
            
            if len(selected_covariates) > 0 and st.button("📊 Run Cox Regression", type="primary"):
                try:
                    with st.status("Running Cox proportional hazards regression...", expanded=True) as status:
                        # Fit Cox model
                        cox_results = analyzer.fit_cox_model(
                            surv_data_cox,
                            duration_col='duration',
                            event_col='event',
                            covariate_cols=selected_covariates
                        )
                        
                        st.session_state.cox_results = cox_results
                        st.session_state.cox_covariates = selected_covariates
                        
                        status.update(label="✅ Cox model fitted!", state="complete", expanded=False)
                    
                    st.success("✅ Cox regression complete!")
                except Exception as e:
                    st.error(f"❌ Error fitting Cox model: {str(e)}")
                    st.info("💡 Make sure covariates have sufficient variation and no missing values.")
            
            # Display Cox results if they exist
            if 'cox_results' in st.session_state:
                cox_res = st.session_state.cox_results
                cox_covs = st.session_state.cox_covariates
                
                st.markdown("### 📊 Hazard Ratios")
                st.markdown("**Interpretation:** HR > 1 = increased risk, HR < 1 = decreased risk, HR = 1 = no effect")
                
                # Create hazard ratio table
                hr_df = pd.DataFrame({
                    'Covariate': cox_res['hazard_ratios'].index,
                    'Hazard Ratio': cox_res['hazard_ratios'].values,
                    'CI Lower (95%)': cox_res['confidence_intervals'].iloc[:, 0].values,
                    'CI Upper (95%)': cox_res['confidence_intervals'].iloc[:, 1].values,
                    'P-value': cox_res['summary']['p'].values
                })
                
                # Add significance indicator
                hr_df['Significant'] = hr_df['P-value'].apply(lambda p: '✅ Yes' if p < 0.05 else '❌ No')
                
                # Add interpretation
                def interpret_hr(hr):
                    if hr > 1.5:
                        return "🔴 Strong increase in risk"
                    elif hr > 1.1:
                        return "🟡 Moderate increase in risk"
                    elif hr > 0.9:
                        return "⚪ Minimal effect"
                    elif hr > 0.67:
                        return "🟢 Moderate decrease in risk"
                    else:
                        return "🟢🟢 Strong decrease in risk"
                
                hr_df['Interpretation'] = hr_df['Hazard Ratio'].apply(interpret_hr)
                
                st.dataframe(hr_df, use_container_width=True)
                
                # Model performance
                st.markdown("### 📈 Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Concordance Index", f"{cox_res['concordance_index']:.3f}",
                             help="C-index measures discrimination ability (0.5=random, 1.0=perfect)")
                with col2:
                    st.metric("Log Likelihood", f"{cox_res['log_likelihood']:.2f}",
                             help="Higher values indicate better fit")
                
                if cox_res['concordance_index'] > 0.7:
                    st.success("✅ **Good discrimination** - Model can distinguish between high and low risk")
                elif cox_res['concordance_index'] > 0.6:
                    st.info("ℹ️ **Moderate discrimination** - Model has some predictive ability")
                else:
                    st.warning("⚠️ **Weak discrimination** - Model may need better predictors")
                
                # Forest plot
                st.markdown("### 🌲 Forest Plot - Hazard Ratios with 95% CI")
                fig_forest = analyzer.create_hazard_ratio_plot(cox_res)
                st.plotly_chart(fig_forest, use_container_width=True)
                
                st.caption("*Reference line at HR=1.0 (no effect). Error bars show 95% confidence intervals.*")
                
                # Key insights
                st.markdown("### 💡 Key Insights")
                
                significant_vars = hr_df[hr_df['P-value'] < 0.05]
                if len(significant_vars) > 0:
                    st.markdown("**Significant Predictors:**")
                    for _, row in significant_vars.iterrows():
                        hr_val = row['Hazard Ratio']
                        effect = "increases" if hr_val > 1 else "decreases"
                        magnitude = abs(hr_val - 1) * 100
                        st.markdown(f"- **{row['Covariate']}**: {effect} risk by {magnitude:.1f}% (HR={hr_val:.3f}, p={row['P-value']:.4f})")
                else:
                    st.info("No statistically significant predictors found at p < 0.05 level")
        else:
            st.info("💡 No numeric covariates available for Cox regression. Add numeric variables or use 'group' column for comparison.")
    
    # AI Insights
    if 'surv_results' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'surv_ai_insights' in st.session_state:
            st.markdown(st.session_state.surv_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="surv_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing survival patterns and generating risk mitigation strategies...", expanded=True) as status:
                    # Get data from session state
                    result = st.session_state.surv_results
                    surv_data = st.session_state.surv_data
                    
                    # Calculate comprehensive metrics
                    total_obs = len(surv_data)
                    events = surv_data['event'].sum()
                    censored = (~surv_data['event'].astype(bool)).sum()
                    event_rate = (events / total_obs) * 100
                    censoring_rate = (censored / total_obs) * 100
                    
                    # Duration statistics
                    mean_duration = surv_data['duration'].mean()
                    median_duration = surv_data['duration'].median()
                    min_duration = surv_data['duration'].min()
                    max_duration = surv_data['duration'].max()
                    duration_range = max_duration - min_duration
                    
                    # Event timing analysis
                    event_data = surv_data[surv_data['event'] == 1]
                    if len(event_data) > 0:
                        mean_time_to_event = event_data['duration'].mean()
                        median_time_to_event = event_data['duration'].median()
                        early_events = (event_data['duration'] < event_data['duration'].quantile(0.25)).sum()
                        late_events = (event_data['duration'] > event_data['duration'].quantile(0.75)).sum()
                    else:
                        mean_time_to_event = median_time_to_event = early_events = late_events = 0
                    
                    # Prepare rich context
                    context = f"""
Survival Analysis Results:

Study Overview:
- Total Observations: {total_obs:,}
- Events Occurred: {events:,} ({event_rate:.1f}%)
- Censored (Did Not Experience Event): {censored:,} ({censoring_rate:.1f}%)
- Follow-up Duration Range: {min_duration:.1f} - {max_duration:.1f} time units ({duration_range:.1f} span)

Duration Statistics:
- Mean Duration: {mean_duration:.1f} time units
- Median Duration: {median_duration:.1f} time units

Event Timing Analysis:
- Mean Time to Event: {mean_time_to_event:.1f} time units
- Median Time to Event: {median_time_to_event:.1f} time units
- Early Events (1st quartile): {early_events} ({early_events/max(events, 1)*100:.1f}%)
- Late Events (4th quartile): {late_events} ({late_events/max(events, 1)*100:.1f}%)
"""
                    
                    # Add group-specific analysis if present
                    if 'group' in surv_data.columns:
                        groups = surv_data['group'].unique()
                        context += f"\nGroup Comparison Analysis:\n"
                        context += f"- Number of Groups: {len(groups)}\n"
                        
                        for group in groups:
                            group_data = surv_data[surv_data['group'] == group]
                            group_events = group_data['event'].sum()
                            group_event_rate = (group_events / len(group_data)) * 100
                            group_median = result['median_survival'].get(group, 0)
                            context += f"- {group}: {len(group_data)} obs, {group_events} events ({group_event_rate:.1f}%), median survival = {group_median:.1f}\n"
                        
                        if 'log_rank_p' in result:
                            significance = "SIGNIFICANT" if result['log_rank_p'] < 0.05 else "NOT significant"
                            context += f"\nLog-Rank Test:\n"
                            context += f"- P-value: {result['log_rank_p']:.4f}\n"
                            context += f"- Result: {significance} difference between groups (α=0.05)\n"
                    else:
                        overall_median = result.get('median_survival', 0)
                        context += f"\nOverall Survival:\n"
                        context += f"- Median Survival Time: {overall_median:.1f} time units\n"
                    
                    context += f"""
Risk Profile:
- Event Occurrence: {'High risk' if event_rate > 50 else 'Moderate risk' if event_rate > 25 else 'Low risk'} ({event_rate:.1f}% event rate)
- Censoring Level: {'High censoring' if censoring_rate > 40 else 'Moderate censoring' if censoring_rate > 20 else 'Low censoring'} ({censoring_rate:.1f}%)
- Event Timing: {'Early events dominate' if early_events > late_events else 'Late events dominate' if late_events > early_events else 'Evenly distributed'}
"""
                    
                    prompt = f"""
As a senior biostatistician and survival analysis expert with 10+ years of experience in clinical research and predictive risk modeling, analyze these survival results and provide:

1. **Survival Curve Interpretation** (3-4 sentences): Evaluate the overall survival pattern. What does the curve shape tell us about risk over time? Are there critical periods where events cluster? How does the censoring rate affect our confidence in the results?

2. **Risk Stratification** (4-5 bullet points): Identify high-risk patterns and segments:
   - Time periods with highest event rates
   - Population segments at greatest risk
   - Early vs. late event patterns
   - Protective factors (if groups show differences)
   - Hazard ratio implications (if group comparison)

3. **Predictive Insights** (3-4 sentences): Based on these survival patterns, when should we expect future events? What is the likelihood of survival at key time points (25%, 50%, 75% of max duration)? Are there inflection points in the survival curve?

4. **Intervention Strategies** (5-6 bullet points): Evidence-based recommendations to improve survival:
   - Primary prevention tactics (reduce event occurrence)
   - Early detection and monitoring (identify at-risk individuals)
   - Timing of interventions (when to act for maximum impact)
   - Risk-based segmentation (targeted vs. population-wide approaches)
   - Follow-up and surveillance protocols
   - Resource allocation priorities

5. **Statistical Considerations** (3-4 bullet points): Important methodological insights:
   - Censoring impact on interpretation
   - Sample size adequacy for conclusions
   - Proportional hazards assumption
   - Confidence in median survival estimates

6. **Expected Impact** (3-4 sentences): If interventions are implemented, what realistic improvements in survival can we expect? Quantify potential reductions in event rates and increases in median survival time. Consider both short-term wins and long-term strategic goals.

{context}

Be specific, evidence-based, and focus on actionable risk mitigation strategies that can be realistically implemented. Balance statistical rigor with practical clinical or business applicability.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior biostatistician and survival analysis expert with 10+ years of experience in clinical research, epidemiology, and predictive risk modeling. You specialize in Kaplan-Meier analysis, Cox proportional hazards models, and translating survival curves into actionable risk mitigation strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.surv_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                    
                    # Display inside status block
                    st.success("✅ AI insights generated successfully!")
                    st.markdown(st.session_state.surv_ai_insights)
                    st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export
    if 'surv_results' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        result = st.session_state.surv_results
        
        report = f"""# Survival Analysis Report

## Overview
- **Total Observations:** {len(surv_data):,}
- **Events:** {surv_data['event'].sum():,}
- **Censored:** {(~surv_data['event'].astype(bool)).sum():,}
- **Event Rate:** {event_rate:.1f}%

## Results
"""
        if 'group' in surv_data.columns:
            report += "### Group Comparison\n"
            for group, median in result['median_survival'].items():
                report += f"- **{group}:** Median survival = {median:.1f} time units\n"
            if 'log_rank_p' in result:
                report += f"\n**Log-Rank Test:** p-value = {result['log_rank_p']:.4f}\n"
        else:
            report += f"**Median Survival Time:** {result['median_survival']:.1f} time units\n"
        
        if 'surv_ai_insights' in st.session_state:
            report += f"\n## AI Insights\n\n{st.session_state.surv_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - Survival Analysis Module*\n"
        
        # Create CSV export with survival data
        surv_data = st.session_state.surv_data
        csv_string = surv_data.to_csv(index=False)
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"survival_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"survival_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_network_analysis():
    """Network Analysis page."""
    st.markdown("<h2 style='text-align: center;'>🕸️ Network Analysis & Graph Theory</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Network Analysis?"):
        st.markdown("""
        **Network Analysis** examines relationships and connections between entities (nodes) in a network (graph).
        
        ### Metrics Available:
        
        - **Centrality Measures:** Identify important nodes (degree, betweenness, closeness)
        - **Community Detection:** Find groups of highly connected nodes
        - **Network Statistics:** Density, clustering coefficient, average path length
        
        ### Business Applications:
        - 👥 **Social Networks:** Influencer identification
        - 🔍 **Fraud Detection:** Unusual transaction patterns
        - 📊 **Organizational:** Team structure analysis
        - 🌐 **Supply Chain:** Network optimization
        """)
    
    st.markdown("Analyze networks to discover relationships, communities, and key influencers.")
    
    # Import utilities
    from utils.network_analysis import NetworkAnalyzer
    
    # Initialize analyzer (refresh if missing new methods)
    if 'network_analyzer' not in st.session_state or not hasattr(st.session_state.network_analyzer, 'predict_links'):
        st.session_state.network_analyzer = NetworkAnalyzer()
    
    analyzer = st.session_state.network_analyzer
    
    # Data loading
    st.subheader("📤 1. Load Network Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    if has_loaded_data:
        data_options = ["Use Loaded Dataset", "Sample Social Network", "Upload Custom Data"]
    else:
        data_options = ["Sample Social Network", "Upload Custom Data"]
    
    data_source = st.radio(
        "Choose data source:",
        data_options,
        index=0,
        key="net_data_source"
    )
    
    # Use Loaded Dataset
    if data_source == "Use Loaded Dataset" and has_loaded_data:
        df = st.session_state.data
        st.success("✅ Using dataset from Data Upload section")
        st.write(f"**Dataset:** {len(df)} rows, {len(df.columns)} columns")
        
        # Smart column detection using ColumnDetector
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_network_column_suggestions(df)
        
        st.info("💡 **Smart Detection:** Select source (from) and target (to) columns for network edges")
        
        col1, col2 = st.columns(2)
        with col1:
            source_default = suggestions['source']
            source_idx = list(df.columns).index(source_default)
            source_col = st.selectbox("Source (From) Column", df.columns, index=source_idx, key="net_source")
        with col2:
            target_default = suggestions['target']
            target_idx = list(df.columns).index(target_default)
            target_col = st.selectbox("Target (To) Column", df.columns, index=target_idx, key="net_target")
        
        # Real-time validation
        issues = []
        warnings = []
        recommendations = []
        
        # Validate columns are different
        if source_col == target_col:
            issues.append("Source and Target columns must be different")
            recommendations.append("Select two distinct columns representing 'from' and 'to' nodes")
        
        # Check for missing values
        source_missing = df[source_col].isnull().sum()
        target_missing = df[target_col].isnull().sum()
        
        if source_missing > 0:
            warnings.append(f"Source column has {source_missing} missing values ({source_missing/len(df)*100:.1f}%)")
        if target_missing > 0:
            warnings.append(f"Target column has {target_missing} missing values ({target_missing/len(df)*100:.1f}%)")
        
        # Data quality checks
        if source_col != target_col:
            # Count potential edges after removing nulls
            valid_edges = df[[source_col, target_col]].dropna()
            n_edges = len(valid_edges)
            n_unique_nodes = len(set(valid_edges[source_col].unique()) | set(valid_edges[target_col].unique()))
            
            # Check minimum requirements
            if n_edges < 3:
                issues.append(f"Only {n_edges} valid edges (need at least 3 for network analysis)")
                recommendations.append("Networks require multiple connections to analyze")
            
            if n_unique_nodes < 3:
                issues.append(f"Only {n_unique_nodes} unique nodes (need at least 3)")
                recommendations.append("Network analysis requires multiple distinct nodes")
            elif n_unique_nodes < 5:
                warnings.append(f"Only {n_unique_nodes} nodes - network metrics may be limited")
            
            # Check for self-loops (source == target in same row)
            if n_edges > 0:
                self_loops = (valid_edges[source_col] == valid_edges[target_col]).sum()
                if self_loops > 0:
                    warnings.append(f"{self_loops} self-loops detected (edges where source = target)")
                    recommendations.append("Self-loops are often removed in network analysis")
            
            # Check network density
            if n_unique_nodes >= 3 and n_edges > 0:
                max_possible_edges = n_unique_nodes * (n_unique_nodes - 1)  # Directed graph
                density = n_edges / max_possible_edges
                
                if density > 0.8:
                    warnings.append(f"Very dense network ({density*100:.1f}% of possible connections)")
                    recommendations.append("Dense networks may have less interesting community structure")
                elif density < 0.01:
                    warnings.append(f"Very sparse network ({density*100:.1f}% of possible connections)")
                    recommendations.append("Sparse networks may have disconnected components")
                
                # Check average degree
                avg_degree = n_edges / n_unique_nodes
                if avg_degree < 1.5:
                    warnings.append(f"Low average degree ({avg_degree:.1f} connections per node)")
                    recommendations.append("Networks with higher connectivity provide richer analysis")
            
            # Check for isolated patterns
            if n_edges > 0 and n_unique_nodes > 0:
                # Count unique sources and targets
                n_sources = valid_edges[source_col].nunique()
                n_targets = valid_edges[target_col].nunique()
                
                if n_sources < n_unique_nodes * 0.3:
                    warnings.append(f"Only {n_sources} nodes have outgoing edges (of {n_unique_nodes} total)")
                if n_targets < n_unique_nodes * 0.3:
                    warnings.append(f"Only {n_targets} nodes have incoming edges (of {n_unique_nodes} total)")
        
        # Display validation results
        st.divider()
        if len(issues) > 0:
            st.error("**🚨 NOT SUITABLE FOR NETWORK ANALYSIS**")
            for issue in issues:
                st.write(f"❌ {issue}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        elif len(warnings) > 0:
            st.warning("**⚠️ NETWORK ANALYSIS POSSIBLE (with warnings)**")
            for warning in warnings:
                st.write(f"⚠️ {warning}")
            if recommendations:
                with st.expander("💡 Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        else:
            st.success("**✅ EXCELLENT FOR NETWORK ANALYSIS**")
            st.write("✅ Distinct source and target columns")
            st.write("✅ Sufficient nodes and edges")
            st.write("✅ Good network connectivity")
        
        # Only show button if no critical issues
        if len(issues) == 0 and st.button("📊 Process Data", type="primary"):
            edge_data = df[[source_col, target_col]].copy()
            edge_data.columns = ['source', 'target']
            # Remove rows with missing values
            edge_data = edge_data.dropna()
            st.session_state.net_data = edge_data
            
            st.success("✅ Data processed!")
            st.info(f"🕸️ {len(edge_data)} edges ready for analysis")
            st.rerun()
    
    elif data_source == "Sample Social Network":
        if st.button("📥 Load Sample Social Network", type="primary"):
            # Generate sample social network (follower/friend relationships)
            np.random.seed(42)
            
            n_users = 50
            users = [f"User_{i}" for i in range(1, n_users + 1)]
            
            # Create edges (friendships/follows)
            edges = []
            for user in users:
                # Each user follows 2-8 random other users
                n_follows = np.random.randint(2, 9)
                follows = np.random.choice([u for u in users if u != user], n_follows, replace=False)
                
                for follow in follows:
                    edges.append({
                        'source': user,
                        'target': follow
                    })
            
            edge_data = pd.DataFrame(edges)
            st.session_state.net_data = edge_data
            
            st.success(f"✅ Loaded social network with {len(users)} users and {len(edges)} connections!")
            st.dataframe(edge_data.head(10), use_container_width=True)
    
    else:  # Upload
        uploaded_file = st.file_uploader("Upload network edges CSV", type=['csv'], key="net_upload")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                source_col = st.selectbox("Source (From) Column", df.columns, key="net_source_upload")
            with col2:
                target_col = st.selectbox("Target (To) Column", df.columns, key="net_target_upload")
            
            if st.button("Process Data", type="primary", key="net_process_upload"):
                edge_data = df[[source_col, target_col]].copy()
                edge_data.columns = ['source', 'target']
                st.session_state.net_data = edge_data
                st.success("✅ Data processed!")
    
    # Analysis section
    if 'net_data' not in st.session_state:
        st.info("👆 Load network data to begin graph analysis")
        return
    
    edge_data = st.session_state.net_data
    
    # Display loaded data preview
    with st.expander("👁️ View Loaded Network Data", expanded=False):
        st.dataframe(edge_data.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(edge_data)} edges")
    
    # Dataset overview
    st.divider()
    st.subheader("📊 2. Network Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Edges", f"{len(edge_data):,}")
    with col2:
        n_nodes = len(set(edge_data['source'].tolist() + edge_data['target'].tolist()))
        st.metric("Total Nodes", f"{n_nodes:,}")
    with col3:
        avg_degree = len(edge_data) / n_nodes
        st.metric("Avg Degree", f"{avg_degree:.2f}")
    
    # Run analysis
    if st.button("🕸️ Analyze Network", type="primary"):
        from utils.process_manager import ProcessManager
        
        # Create process manager
        pm = ProcessManager("Network_Analysis")
        
        # Show warning about not navigating
        st.warning("""
        ⚠️ **Important:** Do not navigate away from this page during analysis.
        Navigation is now locked to prevent data loss.
        """)
        
        # Lock navigation
        pm.lock()
        
        try:
            with st.status("Computing network metrics...", expanded=True) as status:
                # Get analyzer from session state
                analyzer = st.session_state.network_analyzer
                
                # Build graph
                analyzer.build_graph(edge_data, source_col='source', target_col='target')
                
                # Get network statistics
                stats = analyzer.get_network_stats()
                
                # Calculate centrality measures
                degree_cent = analyzer.calculate_centrality('degree')
                between_cent = analyzer.calculate_centrality('betweenness')
                close_cent = analyzer.calculate_centrality('closeness')
                
                # Package results
                result = {
                    'n_nodes': stats['num_nodes'],
                    'n_edges': stats['num_edges'],
                    'density': stats['density'],
                    'avg_clustering': stats.get('avg_clustering', 0),
                    'n_components': stats['num_components'],
                    'diameter': stats.get('diameter'),
                    'top_degree': degree_cent.head(10),
                    'top_betweenness': between_cent.head(10),
                    'top_closeness': close_cent.head(10)
                }
                
                st.session_state.net_results = result
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            st.success("✅ Network analysis completed!")
        except Exception as e:
            st.error(f"❌ Error during network analysis: {str(e)}")
        finally:
            # Always unlock navigation
            pm.unlock()
    
    # Display results if they exist
    if 'net_results' in st.session_state:
        result = st.session_state.net_results
        analyzer = st.session_state.network_analyzer
        
        st.divider()
        # Display metrics
        st.subheader("📈 Network Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Density", f"{result['density']:.4f}")
        with col2:
            st.metric("Avg Clustering", f"{result['avg_clustering']:.4f}")
        with col3:
            st.metric("Components", result['n_components'])
        with col4:
            if 'diameter' in result:
                st.metric("Diameter", result['diameter'])
        
        # Top nodes by centrality
        st.subheader("🌟 Top 10 Most Central Nodes")
        
        tab1, tab2, tab3 = st.tabs(["Degree Centrality", "Betweenness", "Closeness"])
        
        with tab1:
            st.dataframe(result['top_degree'], use_container_width=True)
        with tab2:
            st.dataframe(result['top_betweenness'], use_container_width=True)
        with tab3:
            st.dataframe(result['top_closeness'], use_container_width=True)
        
        # Network visualization
        st.subheader("🕸️ Network Visualization")
        try:
            fig = analyzer.create_network_visualization(analyzer.graph, title='Network Graph')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error creating network visualization: {str(e)}")
            st.info("💡 The network metrics are still available above. Visualization may fail for very large or disconnected networks.")
        
        # Influence Propagation Section
        st.divider()
        st.subheader("📢 Influence Propagation & Viral Spread")
        st.markdown("Simulate how information spreads through your network. Identify key influencers for viral marketing campaigns.")
        
        # Model selection
        col1, col2 = st.columns(2)
        with col1:
            propagation_model = st.selectbox(
                "Propagation Model:",
                ["independent_cascade", "linear_threshold"],
                format_func=lambda x: "Independent Cascade" if x == "independent_cascade" else "Linear Threshold",
                help="Independent Cascade: Each active node tries to activate neighbors once. Linear Threshold: Node activates when enough neighbors are active."
            )
        with col2:
            propagation_prob = st.slider(
                "Propagation Probability:", 
                min_value=0.01, 
                max_value=0.5, 
                value=0.1, 
                step=0.01,
                help="Probability that influence spreads along an edge (IC model) or activation threshold (LT model)"
            )
        
        # Find optimal seed nodes button
        if st.button("🎯 Find Optimal Influencers", type="primary"):
            if analyzer.graph is None or analyzer.graph.number_of_nodes() == 0:
                st.error("❌ Network graph not available. Please run the network analysis first.")
            else:
                with st.status("Finding optimal seed nodes for influence maximization...", expanded=True) as status:
                    try:
                        # Find top 5 seed nodes
                        num_seeds = min(5, analyzer.graph.number_of_nodes())
                        seed_results = analyzer.find_optimal_seed_nodes(
                            num_seeds=num_seeds,
                            propagation_prob=propagation_prob,
                            model=propagation_model,
                            simulations=5  # Reduced for speed
                        )
                        
                        st.session_state.influence_seeds = seed_results
                        
                        status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                        st.success("✅ Optimal influencers identified!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display seed nodes if calculated
        if 'influence_seeds' in st.session_state:
            seeds_df = st.session_state.influence_seeds
            
            st.markdown("### 🎯 Top Influencers for Viral Campaigns")
            st.dataframe(seeds_df, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                top_reach = seeds_df.iloc[0]['reach_pct'] if len(seeds_df) > 0 else 0
                st.metric("Top Influencer Reach", f"{top_reach:.1f}%",
                         help="Expected reach of top influencer")
            with col2:
                avg_reach = seeds_df['reach_pct'].mean() if len(seeds_df) > 0 else 0
                st.metric("Average Seed Reach", f"{avg_reach:.1f}%",
                         help="Average reach across all seed nodes")
            with col3:
                total_nodes = result['n_nodes']
                st.metric("Total Network Nodes", f"{total_nodes:,}")
            
            st.markdown("### 💡 Campaign Strategy Recommendations")
            
            if top_reach > 50:
                st.success(f"""
                ✅ **Excellent Reach Potential!**
                
                The top influencer can reach over 50% of your network. This is ideal for:
                - Viral marketing campaigns
                - Product launches
                - Information dissemination
                
                **Strategy**: Focus resources on top 2-3 influencers for maximum ROI.
                """)
            elif top_reach > 30:
                st.info(f"""
                ℹ️ **Good Reach Potential**
                
                Your top influencers can reach 30-50% of the network.
                
                **Strategy**: Use all 5 seed nodes in combination. Consider incentivizing them to share your message.
                """)
            else:
                st.warning(f"""
                ⚠️ **Moderate Reach Potential**
                
                Network structure limits single-influencer reach (<30%).
                
                **Strategy**: 
                - Use multiple seed nodes from different communities
                - Consider increasing propagation probability (more compelling content)
                - Target nodes with high betweenness centrality
                """)
        
        # Simulate spread with custom seeds
        st.markdown("### 🧪 Custom Spread Simulation")
        
        # Check if graph exists and get list of nodes for selection
        if analyzer.graph is not None and analyzer.graph.number_of_nodes() > 0:
            node_list = list(analyzer.graph.nodes())[:100]  # Limit to first 100 for dropdown
            
            selected_seeds = st.multiselect(
                "Select Seed Nodes:",
                node_list,
                default=node_list[:min(3, len(node_list))],
                help="Choose initial nodes to start the influence cascade"
            )
        else:
            st.warning("⚠️ Network graph not available. Please run the network analysis first.")
            selected_seeds = []
        
        if len(selected_seeds) > 0 and st.button("▶️ Run Simulation"):
            with st.spinner("Simulating influence propagation..."):
                try:
                    spread_result = analyzer.simulate_influence_spread(
                        seed_nodes=selected_seeds,
                        propagation_prob=propagation_prob,
                        model=propagation_model
                    )
                    
                    st.session_state.spread_simulation = spread_result
                    st.success("✅ Simulation complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display simulation results
        if 'spread_simulation' in st.session_state:
            sim = st.session_state.spread_simulation
            
            st.markdown("### 📊 Spread Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes Activated", f"{sim['num_activated']:,}")
            with col2:
                st.metric("Final Reach", f"{sim['final_reach_pct']:.1f}%")
            with col3:
                st.metric("Iterations", f"{sim['iterations']}")
            with col4:
                amplification = sim['num_activated'] / len(selected_seeds) if selected_seeds else 0
                st.metric("Amplification Factor", f"{amplification:.1f}x",
                         help="How many nodes activated per seed node")
            
            # Spread over time chart
            st.markdown("### 📈 Spread Over Time")
            
            spread_df = pd.DataFrame({
                'Iteration': range(len(sim['spread_over_time'])),
                'Activated Nodes': sim['spread_over_time']
            })
            
            import plotly.express as px
            fig = px.line(spread_df, x='Iteration', y='Activated Nodes',
                         title='Influence Propagation Over Time')
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if sim['final_reach_pct'] > 50:
                st.success("🎯 **High Viral Potential**: Information spreads to majority of network!")
            elif sim['final_reach_pct'] > 25:
                st.info("📢 **Moderate Viral Potential**: Good spread but not reaching full network")
            else:
                st.warning("⚠️ **Limited Spread**: Consider increasing propagation probability or choosing better seed nodes")
        
        # Link Prediction Section
        st.divider()
        st.subheader("🔗 Link Prediction & Future Connections")
        st.markdown("Predict which connections are likely to form next in your network using machine learning algorithms.")
        
        with st.expander("ℹ️ What is Link Prediction?"):
            st.markdown("""
            **Link Prediction** identifies node pairs likely to form connections in the future based on current network structure.
            
            **Algorithms:**
            - **Common Neighbors**: Count of shared connections
            - **Adamic-Adar**: Weighted common neighbors (rare connections weighted higher)
            - **Jaccard Coefficient**: Similarity of neighbor sets
            - **Preferential Attachment**: "Rich get richer" - product of degrees
            - **Resource Allocation**: Information flow between nodes
            
            **Applications:**
            - **Social Networks**: Friend recommendations
            - **Business**: Partnership opportunities
            - **Research**: Collaboration predictions
            - **E-commerce**: Product co-purchase patterns
            
            **How It Works:**
            Analyzes patterns in existing connections to predict future ones without requiring temporal data.
            """)
        
        # Method selection
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_method = st.selectbox(
                "Prediction Algorithm:",
                ['common_neighbors', 'adamic_adar', 'jaccard', 
                 'preferential_attachment', 'resource_allocation'],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Algorithm for scoring potential connections"
            )
        
        with col2:
            top_k_predictions = st.slider(
                "Number of Predictions:",
                min_value=10,
                max_value=50,
                value=20,
                help="Top K most likely future connections"
            )
        
        # Predict links button
        if st.button("🔮 Predict Future Links", type="primary"):
            if analyzer.graph is None or analyzer.graph.number_of_nodes() == 0:
                st.error("❌ Network graph not available. Please run the network analysis first.")
            else:
                with st.status("Analyzing network structure and predicting links...", expanded=True) as status:
                    try:
                        # Run link prediction
                        predictions = analyzer.predict_links(
                            method=prediction_method,
                            top_k=top_k_predictions
                        )
                        
                        # Analyze link formation patterns
                        patterns = analyzer.analyze_link_formation_patterns()
                        
                        st.session_state.link_predictions = predictions
                        st.session_state.link_patterns = patterns
                        
                        status.update(label="✅ Predictions complete!", state="complete", expanded=False)
                        st.success("✅ Link predictions generated!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Display predictions
        if 'link_predictions' in st.session_state:
            preds = st.session_state.link_predictions
            
            st.markdown("### 🎯 Top Predicted Connections")
            
            if not preds.empty:
                st.dataframe(preds, use_container_width=True)
                
                # Prediction metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predictions Generated", len(preds))
                with col2:
                    avg_score = preds['score'].mean()
                    st.metric("Average Score", f"{avg_score:.4f}")
                with col3:
                    max_score = preds['score'].max()
                    st.metric("Highest Score", f"{max_score:.4f}")
                
                # Top 3 predictions
                st.markdown("### 🌟 Top 3 Connection Recommendations")
                
                for idx, row in preds.head(3).iterrows():
                    st.markdown(f"""
                    **#{row['rank']}: {row['node_1']} ↔ {row['node_2']}**
                    - Prediction Score: {row['score']:.4f}
                    - Algorithm: {row['method'].replace('_', ' ').title()}
                    - Confidence: {"High" if row['score'] > avg_score else "Moderate"}
                    """)
                
                # Algorithm explanation
                st.markdown("### 💡 Algorithm Interpretation")
                
                if prediction_method == 'common_neighbors':
                    st.info("""
                    **Common Neighbors**: Measures shared connections.
                    
                    Higher scores = More mutual friends/connections. Based on the principle that people with 
                    many friends in common are likely to know each other.
                    """)
                elif prediction_method == 'adamic_adar':
                    st.info("""
                    **Adamic-Adar Index**: Weighted common neighbors.
                    
                    Rare shared connections are weighted higher. A shared connection to a low-degree node 
                    (specialist) is more significant than one to a high-degree node (hub).
                    """)
                elif prediction_method == 'jaccard':
                    st.info("""
                    **Jaccard Coefficient**: Similarity measure.
                    
                    Ratio of shared neighbors to total neighbors. Measures how similar the two nodes' 
                    neighborhoods are, independent of network size.
                    """)
                elif prediction_method == 'preferential_attachment':
                    st.info("""
                    **Preferential Attachment**: "Rich get richer".
                    
                    Product of node degrees. High-degree nodes (hubs) are more likely to form new connections. 
                    Models real-world network growth patterns.
                    """)
                elif prediction_method == 'resource_allocation':
                    st.info("""
                    **Resource Allocation**: Information flow.
                    
                    Simulates resources flowing through common neighbors. More efficient resource transfer 
                    indicates higher connection likelihood.
                    """)
            else:
                st.warning("No predictions generated. Network may be fully connected or too sparse.")
        
        # Network formation patterns
        if 'link_patterns' in st.session_state:
            patterns = st.session_state.link_patterns
            
            st.markdown("### 📊 Network Formation Patterns")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Clustering Coefficient", f"{patterns['clustering_coefficient']:.3f}",
                         help="Tendency to form triangles")
            with col2:
                st.metric("Transitivity", f"{patterns['transitivity']:.3f}",
                         help="Global clustering measure")
            with col3:
                st.metric("Degree Assortativity", f"{patterns['degree_assortativity']:.3f}",
                         help="Homophily - similar-degree nodes connect")
            with col4:
                density = patterns['density']
                st.metric("Network Density", f"{density:.3f}",
                         help="Fraction of possible connections")
            
            # Pattern interpretation
            st.markdown("### 🔍 Pattern Analysis")
            
            clustering = patterns['clustering_coefficient']
            assortativity = patterns['degree_assortativity']
            
            insights = []
            
            if clustering > 0.3:
                insights.append("✅ **High Clustering**: Network shows strong community structure. Connections tend to form closed triangles.")
            elif clustering > 0.1:
                insights.append("ℹ️ **Moderate Clustering**: Some community structure present. Mixed local and global connections.")
            else:
                insights.append("⚠️ **Low Clustering**: Sparse local structure. Connections are more random or tree-like.")
            
            if assortativity > 0.1:
                insights.append("✅ **Positive Assortativity**: Similar nodes connect (homophily). Hubs connect to hubs, periphery to periphery.")
            elif assortativity < -0.1:
                insights.append("⚠️ **Negative Assortativity**: Dissimilar nodes connect. Hubs connect to low-degree nodes (hub-and-spoke).")
            else:
                insights.append("ℹ️ **Neutral Assortativity**: Random mixing. Node degree doesn't predict connection patterns.")
            
            if density < 0.1:
                insights.append("📌 **Sparse Network**: Many potential connections remain. High prediction opportunity.")
            elif density < 0.5:
                insights.append("📌 **Moderate Density**: Balanced between existing and potential connections.")
            else:
                insights.append("📌 **Dense Network**: Highly connected. Limited room for new connections.")
            
            for insight in insights:
                st.markdown(insight)
            
            # Business recommendations
            st.markdown("### 🎯 Business Recommendations")
            
            if clustering > 0.3 and density < 0.3:
                st.success("""
                **Strategy: Community-Based Growth**
                
                - Focus on connecting communities (bridge building)
                - Introduce cross-cluster collaborations
                - Leverage existing strong communities
                """)
            elif assortativity > 0.1:
                st.info("""
                **Strategy: Tier-Based Networking**
                
                - Facilitate connections within experience levels
                - Create peer networking opportunities
                - Support natural clustering patterns
                """)
            else:
                st.info("""
                **Strategy: Hub Development**
                
                - Identify and empower connector nodes
                - Create central resources/people
                - Facilitate hub-to-periphery connections
                """)
    
    # AI Insights
    if 'net_results' in st.session_state:
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'net_ai_insights' in st.session_state:
            st.markdown(st.session_state.net_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="net_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing network topology and generating strategic insights...", expanded=True) as status:
                    # Get data from session state
                    result = st.session_state.net_results
                    edge_data = st.session_state.net_data
                    
                    # Calculate comprehensive metrics
                    n_nodes = result['n_nodes']
                    n_edges = len(edge_data)
                    density = result['density']
                    avg_clustering = result['avg_clustering']
                    n_components = result['n_components']
                    
                    # Centrality analysis
                    top_degree = result['top_degree']
                    top_betweenness = result['top_betweenness']
                    top_closeness = result['top_closeness']
                    
                    top_node_degree = top_degree.iloc[0]['node']
                    top_node_degree_score = top_degree.iloc[0]['centrality']
                    top_node_betweenness = top_betweenness.iloc[0]['node']
                    top_node_betweenness_score = top_betweenness.iloc[0]['centrality']
                    top_node_closeness = top_closeness.iloc[0]['node']
                    top_node_closeness_score = top_closeness.iloc[0]['centrality']
                    
                    # Network characteristics
                    avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
                    possible_edges = (n_nodes * (n_nodes - 1)) / 2
                    sparsity = 1 - density
                    
                    # Community/component analysis
                    is_connected = n_components == 1
                    largest_component_size = n_nodes if is_connected else int(n_nodes * 0.8)  # estimate
                    
                    # Network type classification
                    if density > 0.5:
                        network_type = "Dense network (high connectivity)"
                    elif density > 0.1:
                        network_type = "Moderate density network"
                    else:
                        network_type = "Sparse network (low connectivity)"
                    
                    if avg_clustering > 0.5:
                        clustering_level = "High clustering (strong communities)"
                    elif avg_clustering > 0.2:
                        clustering_level = "Moderate clustering"
                    else:
                        clustering_level = "Low clustering (weak communities)"
                    
                    # Prepare rich context
                    context = f"""
Network Analysis Results:

Network Overview:
- Total Nodes (Entities): {n_nodes:,}
- Total Edges (Connections): {n_edges:,}
- Network Density: {density:.4f} ({density*100:.2f}% of possible connections)
- Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% missing connections)
- Average Degree (Connections per Node): {avg_degree:.2f}

Network Topology:
- Network Type: {network_type}
- Clustering Coefficient: {avg_clustering:.4f}
- Clustering Level: {clustering_level}
- Connected Components: {n_components}
- Network Connectivity: {'Fully connected' if is_connected else f'Fragmented ({n_components} separate groups)'}
- Largest Component: ~{largest_component_size} nodes

Key Influencers (Centrality Analysis):
- Top Node by Degree (Most Connected): {top_node_degree} (score: {top_node_degree_score:.4f})
- Top Node by Betweenness (Bridges/Brokers): {top_node_betweenness} (score: {top_node_betweenness_score:.4f})
- Top Node by Closeness (Information Speed): {top_node_closeness} (score: {top_node_closeness_score:.4f})

Network Characteristics:
- Hub Concentration: {'High' if top_degree.iloc[0]['centrality'] > 0.5 else 'Moderate' if top_degree.iloc[0]['centrality'] > 0.2 else 'Distributed'}
- Information Flow: {'Centralized' if top_betweenness.iloc[0]['centrality'] > 0.3 else 'Decentralized'}
- Network Efficiency: {'High' if avg_clustering > 0.5 and density > 0.1 else 'Moderate' if avg_clustering > 0.2 else 'Low'}
- Vulnerability: {'High risk' if n_components > 1 or top_betweenness.iloc[0]['centrality'] > 0.5 else 'Moderate risk' if not is_connected else 'Resilient'}
"""
                    
                    prompt = f"""
As a senior network scientist and graph analytics expert with 10+ years of experience in social network analysis, organizational network mapping, and complex systems, analyze these network results and provide:

1. **Network Topology Assessment** (3-4 sentences): Evaluate the overall network structure and connectivity. What does the density, clustering, and component structure tell us about how information flows? Is this a centralized hub-and-spoke network or a distributed mesh? What are the strengths and vulnerabilities?

2. **Key Node Analysis** (4-5 bullet points): Identify and interpret critical nodes:
   - High-degree nodes (super-connectors and their role)
   - High-betweenness nodes (gatekeepers and bridge positions)
   - High-closeness nodes (information broadcasters)
   - Structural holes and broker positions
   - Single points of failure or bottlenecks

3. **Community & Clustering Insights** (3-4 sentences): Analyze the community structure and clustering patterns. Are there distinct subgroups? How strong are community boundaries? What does the clustering coefficient reveal about local connectivity vs. global reach?

4. **Influencer Strategy** (5-6 bullet points): Leverage network structure for maximum impact:
   - Primary influencer targets (who to engage first)
   - Cascade amplification strategies (viral spread tactics)
   - Bridge nodes for cross-community reach
   - Peripheral engagement (reaching isolated nodes)
   - Network seeding and diffusion optimization
   - Trust and credibility pathways

5. **Network Optimization** (4-5 bullet points): Improve network health and efficiency:
   - Strengthen weak ties and bridge gaps
   - Reduce over-dependence on central nodes
   - Increase redundancy and resilience
   - Foster community growth and cohesion
   - Remove bottlenecks and improve flow

6. **Business Applications & ROI** (3-4 sentences): Translate network insights into business outcomes. How can understanding this network structure drive better marketing, collaboration, fraud detection, or organizational design? Estimate the impact of network-based interventions on engagement, spread, and efficiency.

{context}

Be specific, data-driven, and focus on actionable network strategies that leverage graph structure for competitive advantage. Balance theoretical network science with practical business applications.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior network scientist and graph analytics expert with 10+ years of experience in social network analysis, organizational network mapping, and complex systems. You specialize in translating network topology into actionable business strategies for influence, optimization, and resilience."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.net_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                
                # Display outside status block so it persists when collapsed
                st.success("✅ AI insights generated successfully!")
                st.markdown(st.session_state.net_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
    
    # Export
    if 'net_results' in st.session_state:
        st.divider()
        st.subheader("📥 Export Results")
        
        result = st.session_state.net_results
        
        report = f"""# Network Analysis Report

## Overview
- **Total Nodes:** {n_nodes:,}
- **Total Edges:** {len(edge_data):,}
- **Network Density:** {result['density']:.4f}
- **Average Clustering Coefficient:** {result['avg_clustering']:.4f}
- **Connected Components:** {result['n_components']}

## Top Central Nodes

### By Degree Centrality
{result['top_degree'].head(5).to_markdown(index=False)}

### By Betweenness Centrality
{result['top_betweenness'].head(5).to_markdown(index=False)}
"""
        
        if 'net_ai_insights' in st.session_state:
            report += f"\n## AI Insights\n\n{st.session_state.net_ai_insights}\n"
        
        report += "\n---\n*Report generated by DataInsights - Network Analysis Module*\n"
        
        # Create CSV export with node metrics
        # Combine all centrality measures into one DataFrame
        node_metrics = result['top_degree'].copy()
        node_metrics.rename(columns={'centrality': 'degree_centrality'}, inplace=True)
        
        # Add betweenness
        betweenness = result['top_betweenness'].copy()
        betweenness.rename(columns={'centrality': 'betweenness_centrality'}, inplace=True)
        node_metrics = node_metrics.merge(betweenness, on='node', how='outer')
        
        # Add closeness
        closeness = result['top_closeness'].copy()
        closeness.rename(columns={'centrality': 'closeness_centrality'}, inplace=True)
        node_metrics = node_metrics.merge(closeness, on='node', how='outer')
        
        # Fill NaN values with 0 and sort by degree
        node_metrics = node_metrics.fillna(0).sort_values('degree_centrality', ascending=False)
        
        csv_string = node_metrics.to_csv(index=False)
        
        # 2-column layout for exports
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name=f"network_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"network_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_churn_prediction():
    """Churn Prediction page."""
    st.markdown("<h2 style='text-align: center;'>🔄 Predictive Churn Modeling & Retention</h2>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("ℹ️ What is Predictive Churn Modeling?"):
        st.markdown("""
        **Predictive Churn Modeling** uses machine learning to identify customers at risk of leaving and develops targeted retention strategies.
        
        **Key Capabilities:**
        - 🎯 **Automated Feature Engineering** - Create predictive features from raw transaction data
        - 🤖 **ML-Powered Predictions** - Train Random Forest, Gradient Boosting, or Logistic Regression models
        - 📊 **Risk Segmentation** - Classify customers into High/Medium/Low risk categories
        - 💡 **Retention Strategies** - Get personalized action plans for each risk segment
        - 📈 **Business Impact** - Reduce churn, increase CLV, improve retention ROI
        
        **Common Use Cases:**
        - **SaaS**: Subscription cancellation prediction
        - **E-commerce**: Purchase recency and frequency analysis
        - **Telecom**: Service discontinuation forecasting
        - **Banking**: Account closure and product churn
        - **Streaming**: Content engagement and cancellation risk
        
        **How It Works:**
        1. **Feature Engineering**: Automatically creates 15+ predictive features from transaction data (recency, frequency, monetary, engagement trends)
        2. **Model Training**: Trains multiple ML models to predict churn probability for each customer
        3. **Risk Scoring**: Classifies customers into High (>60%), Medium (30-60%), or Low (<30%) risk categories
        4. **Retention Planning**: Generates personalized action plans for each risk segment
        
        **Benefits:**
        - Early identification of at-risk customers
        - Data-driven retention strategies
        - Resource optimization (focus on high-risk customers)
        - Improved customer lifetime value
        - Reduced customer acquisition costs
        """)
    
    # Import utilities
    from utils.churn_prediction import ChurnPredictor
    
    # Initialize predictor
    if 'churn_predictor' not in st.session_state:
        st.session_state.churn_predictor = ChurnPredictor()
    
    predictor = st.session_state.churn_predictor
    
    # Data loading
    st.divider()
    st.subheader("📤 1. Load Customer Data")
    
    # Check if data is already loaded
    has_loaded_data = st.session_state.data is not None
    
    data_source = st.radio(
        "Choose data source:",
        ["Use Loaded Dataset", "Use Sample Data", "Upload Custom Data"],
        disabled=not has_loaded_data and False,  # Always enable for sample
        help="Sample data includes synthetic customer transaction history"
    )
    
    churn_data = None
    
    if data_source == "Use Loaded Dataset":
        if has_loaded_data:
            churn_data = st.session_state.data.copy()
            st.success(f"✅ Using loaded dataset ({len(churn_data):,} rows)")
        else:
            st.warning("⚠️ No data loaded. Please upload data first or use sample data.")
        
    elif data_source == "Use Sample Data":
        # Create sample churn data
        np.random.seed(42)
        n_customers = 200
        n_transactions = 1500
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
        
        # Generate transactions
        transactions = []
        base_date = pd.Timestamp('2024-01-01')
        
        for _ in range(n_transactions):
            customer = np.random.choice(customer_ids)
            days_offset = np.random.randint(0, 365)
            amount = np.random.gamma(2, 50)
            
            transactions.append({
                'customer_id': customer,
                'transaction_date': base_date + pd.Timedelta(days=days_offset),
                'amount': round(amount, 2)
            })
        
        churn_data = pd.DataFrame(transactions)
        
        # Add churn labels (customers with no activity in last 90 days)
        max_date = churn_data['transaction_date'].max()
        last_trans = churn_data.groupby('customer_id')['transaction_date'].max()
        churn_labels = (max_date - last_trans).dt.days > 90
        
        churn_map = churn_labels.astype(int).to_dict()
        churn_data['churned'] = churn_data['customer_id'].map(churn_map)
        
        st.success(f"✅ Sample data loaded ({len(churn_data):,} transactions, {n_customers} customers)")
        st.info("💡 This dataset contains synthetic customer transactions with churn labels for demonstration")
        
    else:  # Upload Custom Data
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded:
            churn_data = pd.read_csv(uploaded)
            st.success(f"✅ File uploaded ({len(churn_data):,} rows)")
    
    # Display dataset preview
    if churn_data is not None:
        with st.expander("👀 Preview Data"):
            st.dataframe(churn_data.head(20), use_container_width=True)
            st.caption(f"Showing first 20 rows of {len(churn_data):,} total")
    
    # Feature Engineering Section
    if churn_data is not None:
        st.divider()
        st.subheader("🔧 2. Configure Feature Engineering")
        
        with st.expander("ℹ️ What is Feature Engineering?"):
            st.markdown("""
            **Feature Engineering** creates predictive variables from raw transaction data.
            
            **Automated Features Created:**
            - **Recency**: Days since last transaction (recent = engaged)
            - **Frequency**: Total number of transactions (high = loyal)
            - **Monetary**: Total/average/std of transaction values
            - **Lifetime**: Customer tenure in days
            - **Engagement Trend**: Recent vs. historical activity ratio
            - **Time-Based**: Activity in last 30/60/90 days
            - **Risk Indicators**: Dormancy, declining activity flags
            
            **Why It Matters:**
            These features capture customer behavior patterns that predict churn better than raw transactions.
            """)
        
        # Smart column detection
        from utils.column_detector import ColumnDetector
        suggestions = ColumnDetector.get_churn_column_suggestions(churn_data)
        
        st.info("💡 **Smart Detection:** Automatically detected relevant columns based on naming patterns")
        
        # Column selection
        col1, col2 = st.columns(2)
        
        with col1:
            customer_default = suggestions['customer_id']
            customer_idx = list(churn_data.columns).index(customer_default) if customer_default in churn_data.columns else 0
            customer_col = st.selectbox(
                "Customer ID Column:",
                churn_data.columns.tolist(),
                index=customer_idx,
                help="Column identifying unique customers"
            )
        
        with col2:
            date_default = suggestions['date']
            date_idx = list(churn_data.columns).index(date_default) if date_default in churn_data.columns else (1 if len(churn_data.columns) > 1 else 0)
            date_col = st.selectbox(
                "Transaction Date Column:",
                churn_data.columns.tolist(),
                index=date_idx,
                help="Column with transaction timestamps"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            value_col_options = ["None"] + [col for col in churn_data.columns if col not in [customer_col, date_col]]
            value_default = suggestions['value']
            value_idx = value_col_options.index(value_default) if value_default and value_default in value_col_options else 0
            value_col = st.selectbox(
                "Transaction Value Column (Optional):",
                value_col_options,
                index=value_idx,
                help="Column with monetary values (optional)"
            )
            value_col = None if value_col == "None" else value_col
        
        with col4:
            churn_col_options = ["None"] + [col for col in churn_data.columns if col not in [customer_col, date_col]]
            churn_default = suggestions['churn']
            churn_idx = churn_col_options.index(churn_default) if churn_default and churn_default in churn_col_options else 0
            churn_col = st.selectbox(
                "Churn Label Column (Optional):",
                churn_col_options,
                index=churn_idx,
                help="Column indicating if customer churned (0/1)"
            )
            churn_col = None if churn_col == "None" else churn_col
        
        # Validate date column selection
        validation_msg = []
        try:
            test_parse = pd.to_datetime(churn_data[date_col].head(5), errors='coerce')
            if test_parse.isna().all():
                validation_msg.append(f"⚠️ Column '{date_col}' doesn't appear to contain valid dates. Please select a different column.")
        except:
            validation_msg.append(f"⚠️ Unable to parse '{date_col}' as dates. Please select a date/datetime column.")
        
        if validation_msg:
            for msg in validation_msg:
                st.warning(msg)
        
        # Engineer features button
        if st.button("🔧 Engineer Features", type="primary"):
            with st.status("Creating predictive features...", expanded=True) as status:
                try:
                    features = predictor.engineer_features(
                        churn_data,
                        customer_id_col=customer_col,
                        date_col=date_col,
                        value_col=value_col,
                        churn_col=churn_col
                    )
                    
                    st.session_state.churn_features = features
                    st.session_state.churn_has_labels = churn_col is not None
                    
                    status.update(label="✅ Features created!", state="complete", expanded=False)
                    st.success(f"✅ Engineered {len(features.columns)-1} features for {len(features):,} customers!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display engineered features
    if 'churn_features' in st.session_state:
        features = st.session_state.churn_features
        
        st.markdown("### 📊 Engineered Features")
        st.dataframe(features.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(features):,}")
        with col2:
            avg_recency = features['recency_days'].mean()
            st.metric("Avg Recency (days)", f"{avg_recency:.1f}")
        with col3:
            avg_frequency = features['frequency'].mean()
            st.metric("Avg Frequency", f"{avg_frequency:.1f}")
    
    # Model Training Section
    if 'churn_features' in st.session_state and st.session_state.churn_has_labels:
        st.divider()
        st.subheader("🤖 3. Train Churn Prediction Model")
        
        with st.expander("ℹ️ About the Models"):
            st.markdown("""
            **Random Forest** (Recommended):
            - Ensemble of decision trees
            - Handles non-linear relationships
            - Provides feature importance
            - Robust to outliers
            
            **Gradient Boosting**:
            - Sequential tree building
            - Often highest accuracy
            - Can overfit on small datasets
            
            **Logistic Regression**:
            - Fast and interpretable
            - Linear decision boundaries
            - Good for large datasets
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model:",
                ['random_forest', 'gradient_boosting', 'logistic'],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Random Forest recommended for most cases"
            )
        
        with col2:
            test_size = st.slider(
                "Test Set Size:",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Proportion of data reserved for testing"
            )
        
        if st.button("🚀 Train Model", type="primary"):
            with st.status("Training churn prediction model...", expanded=True) as status:
                try:
                    results = predictor.train_model(
                        features=st.session_state.churn_features,
                        target_col='churned',
                        model_type=model_type,
                        test_size=test_size
                    )
                    
                    st.session_state.churn_results = results
                    
                    status.update(label="✅ Model trained!", state="complete", expanded=False)
                    st.success("✅ Model training complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display model results
    if 'churn_results' in st.session_state:
        results = st.session_state.churn_results
        
        st.markdown("### 📈 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}",
                     help="Overall correctness")
        with col2:
            st.metric("Precision", f"{results['precision']:.3f}",
                     help="Of predicted churners, how many actually churned")
        with col3:
            st.metric("Recall", f"{results['recall']:.3f}",
                     help="Of actual churners, how many we caught")
        with col4:
            st.metric("ROC-AUC", f"{results['roc_auc']:.3f}",
                     help="Overall discriminative power")
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        
        cm = results['confusion_matrix']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No Churn', 'Predicted Churn'],
                y=['Actual No Churn', 'Actual Churn'],
                text=cm,
                texttemplate='%{text}',
                colorscale='RdYlGn_r',
                showscale=False
            ))
            fig_cm.update_layout(
                title='Prediction Results',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("**Interpretation:**")
            st.markdown(f"- ✅ **True Negatives**: {cm[0,0]} (correctly predicted no churn)")
            st.markdown(f"- ❌ **False Positives**: {cm[0,1]} (false alarms)")
            st.markdown(f"- ❌ **False Negatives**: {cm[1,0]} (missed churners - costly!)")
            st.markdown(f"- ✅ **True Positives**: {cm[1,1]} (correctly predicted churn)")
            
            st.markdown("**Business Impact:**")
            saved_customers = cm[1,1]
            missed_customers = cm[1,0]
            st.markdown(f"- 🎯 Can save up to **{saved_customers}** at-risk customers")
            st.markdown(f"- ⚠️ Missing **{missed_customers}** churners - room for improvement")
        
        # Feature Importance
        if predictor.feature_importance is not None:
            st.markdown("### 🔍 Top Churn Drivers")
            
            fig_imp = ChurnPredictor.create_feature_importance_plot(
                predictor.feature_importance,
                top_n=10
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("**Insights:**")
            top_feature = predictor.feature_importance.iloc[0]
            st.info(f"💡 **{top_feature['feature']}** is the strongest churn predictor (importance: {top_feature['importance']:.3f})")
    
    # Prediction Section
    if 'churn_results' in st.session_state:
        st.divider()
        st.subheader("🎯 4. Predict Churn Risk")
        
        if st.button("🔮 Generate Churn Predictions", type="primary"):
            with st.status("Scoring customers...", expanded=True) as status:
                try:
                    predictions = predictor.predict_churn_risk(
                        st.session_state.churn_features
                    )
                    
                    st.session_state.churn_predictions = predictions
                    
                    status.update(label="✅ Predictions complete!", state="complete", expanded=False)
                    st.success("✅ Churn risk scores generated!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display predictions
    if 'churn_predictions' in st.session_state:
        predictions = st.session_state.churn_predictions
        
        st.markdown("### 📊 Customer Risk Scores")
        
        # Risk distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_dist = ChurnPredictor.create_risk_distribution_plot(predictions)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            risk_counts = predictions['risk_category'].value_counts()
            
            for risk, count in risk_counts.items():
                pct = (count / len(predictions)) * 100
                color = '🔴' if risk == 'High Risk' else '🟠' if risk == 'Medium Risk' else '🟢'
                st.metric(f"{color} {risk}", f"{count}", delta=f"{pct:.1f}%")
        
        # Top at-risk customers
        st.markdown("### 🚨 Top 20 At-Risk Customers")
        
        top_risk = predictions.head(20)[[
            'customer_id', 'churn_probability', 'risk_category'
        ]].copy()
        
        top_risk['churn_probability'] = top_risk['churn_probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(top_risk, use_container_width=True)
        
        # Retention Strategies
        st.divider()
        st.subheader("💡 5. Retention Strategies")
        
        if st.button("🎯 Generate Retention Strategies", type="primary"):
            with st.status("Developing retention strategies...", expanded=True) as status:
                try:
                    strategies = predictor.get_retention_strategies(
                        predictions,
                        st.session_state.churn_features
                    )
                    
                    st.session_state.churn_strategies = strategies
                    
                    status.update(label="✅ Strategies ready!", state="complete", expanded=False)
                    st.success("✅ Retention strategies generated!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display strategies
        if 'churn_strategies' in st.session_state:
            strategies = st.session_state.churn_strategies
            
            for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
                if risk_level in strategies and len(strategies[risk_level]) > 0:
                    strategy = strategies[risk_level][0]
                    
                    color = '🔴' if risk_level == 'High Risk' else '🟠' if risk_level == 'Medium Risk' else '🟢'
                    
                    with st.expander(f"{color} {risk_level} Segment ({strategy['customer_count']} customers)", expanded=risk_level=='High Risk'):
                        st.markdown(f"**Average Churn Probability:** {strategy['avg_churn_probability']:.1%}")
                        
                        st.markdown("**Segment Characteristics:**")
                        chars = strategy['characteristics']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Recency", f"{chars['avg_recency_days']:.0f} days")
                        with col2:
                            st.metric("Avg Frequency", f"{chars['avg_frequency']:.1f}")
                        with col3:
                            st.metric("% Dormant", f"{chars['pct_dormant']:.0f}%")
                        with col4:
                            st.metric("% Declining", f"{chars['pct_declining']:.0f}%")
                        
                        st.markdown("**Recommended Actions:**")
                        
                        for action in strategy['recommended_actions']:
                            st.markdown(f"**{action['action']}**")
                            st.markdown(f"*Why:* {action['reason']}")
                            st.markdown("*Tactics:*")
                            for tactic in action['tactics']:
                                st.markdown(f"  - {tactic}")
                            st.markdown("")
        
        # AI Insights Section
        st.divider()
        st.subheader("✨ AI-Powered Insights")
        
        # Display saved insights if they exist
        if 'churn_ai_insights' in st.session_state:
            st.markdown(st.session_state.churn_ai_insights)
            st.info("✅ AI insights saved! These will be included in your report downloads.")
        
        if st.button("🤖 Generate AI Insights", key="churn_ai_insights_btn", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.status("🤖 Analyzing churn patterns and generating retention strategies...", expanded=True) as status:
                    # Get data from session state
                    features = st.session_state.churn_features
                    predictions = st.session_state.churn_predictions
                    results = st.session_state.churn_results
                    
                    # Calculate comprehensive metrics
                    total_customers = len(predictions)
                    high_risk = len(predictions[predictions['risk_category'] == 'High Risk'])
                    medium_risk = len(predictions[predictions['risk_category'] == 'Medium Risk'])
                    low_risk = len(predictions[predictions['risk_category'] == 'Low Risk'])
                    
                    high_risk_pct = (high_risk / total_customers) * 100
                    medium_risk_pct = (medium_risk / total_customers) * 100
                    low_risk_pct = (low_risk / total_customers) * 100
                    
                    avg_churn_prob = predictions['churn_probability'].mean()
                    
                    # Model performance
                    accuracy = results['accuracy']
                    precision = results['precision']
                    recall = results['recall']
                    f1 = results['f1_score']
                    roc_auc = results['roc_auc']
                    
                    # Feature importance (top 5)
                    if results.get('feature_importance') is not None and len(results['feature_importance']) > 0:
                        top_features = results['feature_importance'].head(5)
                        top_feature_str = "\n".join([f"  - {row['feature']}: {row['importance']:.3f}" 
                                                    for _, row in top_features.iterrows()])
                    else:
                        top_feature_str = "  - Feature importance not available for this model type"
                    
                    # Customer behavior patterns
                    avg_recency = features['recency_days'].mean()
                    avg_frequency = features['frequency'].mean()
                    pct_dormant = (features['is_dormant'].sum() / len(features)) * 100
                    pct_declining = (features['is_declining'].sum() / len(features)) * 100
                    
                    # Prepare rich context
                    context = f"""
Churn Prediction Analysis Results:

Model Performance:
- Algorithm: {results.get('model_type', 'N/A')}
- Accuracy: {accuracy:.1%}
- Precision: {precision:.1%} (correctness of churn predictions)
- Recall: {recall:.1%} (coverage of actual churners)
- F1 Score: {f1:.3f} (balance of precision and recall)
- ROC-AUC: {roc_auc:.3f} (model discrimination power)

Customer Risk Distribution:
- Total Customers: {total_customers:,}
- 🔴 High Risk: {high_risk:,} ({high_risk_pct:.1f}%) - Churn probability >60%
- 🟠 Medium Risk: {medium_risk:,} ({medium_risk_pct:.1f}%) - Churn probability 30-60%
- 🟢 Low Risk: {low_risk:,} ({low_risk_pct:.1f}%) - Churn probability <30%
- Average Churn Probability: {avg_churn_prob:.1%}

Top Churn Drivers (Feature Importance):
{top_feature_str}

Customer Behavior Patterns:
- Average Recency: {avg_recency:.1f} days since last transaction
- Average Frequency: {avg_frequency:.1f} transactions per customer
- Dormant Customers: {pct_dormant:.1f}% (>90 days inactive)
- Declining Engagement: {pct_declining:.1f}% (recent activity < historical)

"""
                    
                    prompt = f"""As a senior customer retention strategist and churn analytics expert with 15+ years of experience in SaaS, e-commerce, and subscription businesses, analyze this churn prediction model and provide actionable insights.

Generate a comprehensive churn analysis with these sections:

1. **Executive Summary** (3-4 sentences): Key findings about churn risk across the customer base. What's the overall health? What percentage are at risk? What's the urgency level?

2. **Model Performance Assessment** (3-4 sentences): Evaluate the predictive model's quality. Is it reliable for business decisions? What do the precision/recall metrics tell us about false positives vs. false negatives? Can we trust the high-risk predictions?

3. **Churn Driver Analysis** (4-5 bullet points): Deep dive into the top features driving churn:
   - What do these features reveal about customer behavior?
   - Are customers churning due to disengagement (recency), low usage (frequency), or value issues?
   - Which behaviors are early warning signs?
   - What patterns separate churners from loyal customers?

4. **Risk Segment Insights** (one paragraph per segment - High/Medium/Low):
   - **High Risk Segment**: Profile these customers. Why are they leaving? What immediate actions can save them?
   - **Medium Risk Segment**: What's pushing them toward churn? How do we prevent escalation?
   - **Low Risk Segment**: What are they doing right? How do we keep them engaged?

5. **Prioritized Retention Strategy** (6-8 bullet points ranked by impact):
   - Immediate interventions for high-risk customers (win-back)
   - Preventive measures for medium-risk (re-engagement)
   - Loyalty programs for low-risk (advocacy)
   - Product/service improvements based on churn drivers
   - Communication timing and channels
   - Incentive structures and personalization tactics

6. **Business Impact & ROI** (3-4 sentences): Translate predictions into financial outcomes. If we successfully reduce high-risk churn by 20-30%, what's the revenue impact? How should we allocate retention budget across segments? What's the expected customer lifetime value improvement?

{context}

Be data-driven, specific, and focus on actionable strategies that directly address the identified churn drivers. Connect patterns to business outcomes and prioritize by impact potential.
"""
                    
                    response = ai.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a world-class customer retention strategist and churn analytics expert with deep expertise in predictive modeling, customer lifecycle management, and retention economics. You specialize in translating churn predictions into high-impact retention strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    # Save to session state
                    st.session_state.churn_ai_insights = response.choices[0].message.content
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                
                # Display outside status block so it persists when collapsed
                st.success("✅ AI insights generated successfully!")
                st.markdown(st.session_state.churn_ai_insights)
                st.info("✅ AI insights saved! These will be included in your report downloads.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
        
        # Export Section
        st.divider()
        st.subheader("📥 7. Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export predictions
            predictions_csv = predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=predictions_csv,
                file_name=f"churn_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export features
            features_csv = st.session_state.churn_features.to_csv(index=False)
            st.download_button(
                label="📥 Download Features (CSV)",
                data=features_csv,
                file_name=f"churn_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
