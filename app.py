import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime
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
            ["Home", "Data Upload", "Data Analysis & Cleaning", "Anomaly Detection", "Insights", "Market Basket Analysis", "RFM Analysis", "Time Series Forecasting", "Text Mining & NLP", "ML Classification", "Monte Carlo Simulation", "Reports"],
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
    elif page == "Anomaly Detection":
        show_anomaly_detection()
    elif page == "Time Series Forecasting":
        show_time_series_forecasting()
    elif page == "Text Mining & NLP":
        show_text_mining()

def show_home():
    st.header("Welcome to DataInsight AI! 👋")
    
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
            <p>Ask questions in natural language and get intelligent answers</p>
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
    st.header("📊 Data Analysis")
    
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
    
    # Tabs organized by best practice workflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧹 Quick Clean", "📈 Statistics", "📊 Visualizations", "🤖 AI Insights", "🔧 Advanced Cleaning"])
    
    with tab1:
        st.subheader("🧹 Quick Data Cleaning")
        st.write("Apply automatic data cleaning with one click, or customize the cleaning options.")
        
        # Cleaning options
        with st.form("cleaning_form"):
            st.write("**Select Cleaning Steps:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                normalize_cols = st.checkbox("Normalize column names", value=True, 
                                            help="Convert to lowercase with underscores")
                convert_numeric = st.checkbox("Convert to numeric", value=True,
                                             help="Convert compatible columns to numbers")
                remove_dups = st.checkbox("Remove duplicate rows", value=True)
            
            with col2:
                fill_missing = st.checkbox("Fill missing values", value=True)
                missing_strategy = st.selectbox("Missing value strategy:", 
                                               ["median", "mean", "mode"],
                                               help="Strategy for filling missing values")
                drop_high_missing = st.checkbox("Drop columns with >80% missing", value=False)
            
            submitted = st.form_submit_button("🚀 Clean Data Now", type="primary", use_container_width=True)
        
        if submitted:
            with st.spinner("🧹 Cleaning data..."):
                try:
                    from utils.data_cleaning import DataCleaner
                    
                    cleaner = DataCleaner(df)
                    result = cleaner.clean_pipeline(
                        normalize_cols=normalize_cols,
                        convert_numeric=convert_numeric,
                        remove_dups=remove_dups,
                        fill_missing=fill_missing,
                        missing_strategy=missing_strategy,
                        drop_high_missing_cols=drop_high_missing,
                        col_threshold=0.8
                    )
                    
                    # Store cleaned data and results
                    st.session_state.data = result['cleaned_df']
                    st.session_state.original_data = result['original_df']
                    st.session_state.cleaning_stats = result['stats']
                    st.session_state.cleaning_quality_score = result['quality_score']
                    
                    # Clear cached analysis
                    for key in ['profile', 'issues', 'viz_suggestions', 'ai_insights', 'cleaning_suggestions']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success("✅ Data cleaned successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during cleaning: {str(e)}")
        
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
            
            # Quality score
            quality_score = st.session_state.get('cleaning_quality_score', 0)
            st.metric("Data Quality Score", f"{quality_score:.1f}/100",
                     delta="Improved" if quality_score > 70 else "Needs more cleaning")
    
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

    with tab5:
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
                
                st.info("💡 Review each suggestion below. You can apply fixes directly or copy the code to run manually.")
                
                for i, suggestion in enumerate(suggestions):
                    with st.expander(f"💡 Suggestion {i+1}: {suggestion.get('issue', 'N/A')}", expanded=False):
                        st.write("**What to do:**", suggestion.get('suggestion', 'N/A'))
                        st.write("**Why:**", suggestion.get('reason', 'N/A'))
                        
                        if suggestion.get('code'):
                            st.code(suggestion['code'], language='python')
                            
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                if st.button(f"✅ Apply This Fix", key=f"apply_fix_{i}", type="primary"):
                                    try:
                                        # Create a copy of the dataframe to apply fix
                                        df_copy = df.copy()
                                        
                                        # Execute the cleaning code
                                        # Make df available in the exec context
                                        exec_globals = {'df': df_copy, 'pd': pd, 'np': np}
                                        exec(suggestion['code'], exec_globals)
                                        
                                        # Update the dataframe in session state
                                        st.session_state.data = exec_globals.get('df', df_copy)
                                        
                                        # Clear cached analysis to force refresh
                                        if 'profile' in st.session_state:
                                            del st.session_state.profile
                                        if 'issues' in st.session_state:
                                            del st.session_state.issues
                                        if 'cleaning_suggestions' in st.session_state:
                                            del st.session_state.cleaning_suggestions
                                        
                                        st.success(f"✅ Fix applied successfully! Data updated.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error applying fix: {str(e)}")
                            
                            with col2:
                                if st.button(f"👁️ Preview Impact", key=f"preview_{i}"):
                                    try:
                                        df_copy = df.copy()
                                        exec_globals = {'df': df_copy, 'pd': pd, 'np': np}
                                        exec(suggestion['code'], exec_globals)
                                        df_cleaned = exec_globals.get('df', df_copy)
                                        
                                        st.write("**Before:**")
                                        st.write(f"- Rows: {len(df)}, Columns: {len(df.columns)}")
                                        st.write(f"- Missing values: {df.isnull().sum().sum()}")
                                        
                                        st.write("**After:**")
                                        st.write(f"- Rows: {len(df_cleaned)}, Columns: {len(df_cleaned.columns)}")
                                        st.write(f"- Missing values: {df_cleaned.isnull().sum().sum()}")
                                    except Exception as e:
                                        st.error(f"Error previewing: {str(e)}")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Regenerate All Suggestions", use_container_width=True):
                        del st.session_state.cleaning_suggestions
                        st.rerun()
                
                with col2:
                    if st.button("📥 Download Cleaning Script", use_container_width=True):
                        # Combine all code into a single script
                        script = "# Data Cleaning Script\n"
                        script += "# Generated by DataInsight AI\n\n"
                        script += "import pandas as pd\nimport numpy as np\n\n"
                        script += "# Load your data\n"
                        script += "# df = pd.read_csv('your_data.csv')\n\n"
                        
                        for i, suggestion in enumerate(suggestions):
                            if suggestion.get('code'):
                                script += f"# Fix {i+1}: {suggestion.get('issue', 'N/A')}\n"
                                script += f"# {suggestion.get('suggestion', 'N/A')}\n"
                                script += suggestion['code'] + "\n\n"
                        
                        script += "# Save cleaned data\n"
                        script += "# df.to_csv('cleaned_data.csv', index=False)\n"
                        
                        st.download_button(
                            label="Download Python Script",
                            data=script,
                            file_name="data_cleaning_script.py",
                            mime="text/plain",
                            use_container_width=True
                        )

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
                
                # Rerun to show new result
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
        
        if st.button("🔄 Process Loaded Data", type="primary"):
            with st.spinner("Processing transactions..."):
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
        
        if st.button("🔄 Process Loaded Data for RFM", type="primary"):
            with st.spinner("Processing RFM data..."):
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

def show_ml_classification():
    """Comprehensive ML Classification with 15 models and full evaluation."""
    st.header("🤖 Machine Learning - Classification Models")
    
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
            ["Use uploaded data from Data Upload page", "Upload new file for this analysis"],
            help="You can use the data you already uploaded or upload a new file"
        )
        
        if data_source == "Use uploaded data from Data Upload page":
            df = st.session_state.data
            st.session_state.ml_data = df
            st.success(f"✅ Using uploaded data: {len(df):,} rows and {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file with features and target column",
                type=['csv'],
                key="ml_upload",
                help="Must include predictor features and target column"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.ml_data = df
                    st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            else:
                st.info("👆 Please upload a CSV file to continue")
    else:
        st.info("💡 **Tip:** Upload data from the 'Data Upload' page first, or upload a file below")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with features and target column",
            type=['csv'],
            key="ml_upload",
            help="Must include predictor features and target column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.ml_data = df
                st.success(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
                
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Configuration and training
    if 'ml_data' in st.session_state:
        df = st.session_state.ml_data
        
        st.divider()
        st.subheader("🎯 2. Configure Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Select Target Column (what to predict)",
                df.columns,
                help="Column containing the categories/classes to predict"
            )
            
            # Show class distribution
            if target_col:
                class_counts = df[target_col].value_counts()
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
            
            # Training config
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data reserved for testing"
            )
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=3,
                help="Number of folds for cross-validation"
            )
        
        # Train button
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            from utils.ml_training import MLTrainer
            
            try:
                # Initialize trainer
                with st.spinner("Initializing ML Trainer..."):
                    trainer = MLTrainer(df, target_col, max_samples=10000)
                    prep_info = trainer.prepare_data(test_size=test_size/100)
                
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
                    status_text.text(f"Training {current}/{total}: {model_name} - F1: {result['f1']:.4f}")
                
                # Train models
                results = trainer.train_all_models(
                    selected_models=selected_models,
                    cv_folds=cv_folds,
                    progress_callback=progress_callback
                )
                
                # Store results
                st.session_state.ml_results = results
                st.session_state.ml_trainer = trainer
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training complete!")
                
                st.success(f"🎉 Successfully trained {len(results)} models!")
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    if 'ml_results' in st.session_state and 'ml_trainer' in st.session_state:
        results = st.session_state.ml_results
        trainer = st.session_state.ml_trainer
        
        st.divider()
        st.subheader("📊 3. Model Performance Results")
        
        # Summary metrics
        successful_results = [r for r in results if r['success']]
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
            total_time = sum(r['training_time'] for r in results)
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
                    st.write("**Feature Importance:**")
                    
                    feat_imp = best_details['feature_importance']
                    importance_df = pd.DataFrame({
                        'Feature': feat_imp['features'],
                        'Importance': feat_imp['importances']
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Features'
                    )
                    fig_imp.update_layout(height=400)
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
        
        if st.button("🤖 Generate AI Insights", use_container_width=True):
            try:
                from utils.ai_helper import AIHelper
                ai = AIHelper()
                
                with st.spinner("Analyzing results and generating insights..."):
                    # Prepare context
                    context = f"""
                    Machine Learning Classification Results:
                    
                    Dataset: {len(df)} rows, {len(df.columns)} columns
                    Target: {target_col}
                    Classes: {', '.join(trainer.class_names)}
                    
                    Models Trained: {len(successful_results)}
                    Best Model: {best_model['model_name']}
                    Best F1 Score: {best_model['f1']:.4f}
                    
                    Top 3 Models:
                    """
                    
                    for i, r in enumerate(successful_results[:3], 1):
                        context += f"\n{i}. {r['model_name']}: F1={r['f1']:.4f}, Accuracy={r['accuracy']:.4f}"
                    
                    prompt = f"""
                    As a senior data science consultant, analyze these machine learning results and provide:
                    
                    1. **Performance Analysis** (2-3 sentences): Why did {best_model['model_name']} perform best?
                    
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
                    
                    st.markdown(response.choices[0].message.content)
                    
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")
        
        # Export section
        st.divider()
        st.subheader("📥 5. Export & Download")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export results
            results_export = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results Table (CSV)",
                data=results_export,
                file_name=f"ml_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export best model report
            if best_model:
                report = f"""
# Machine Learning Classification Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Model: {best_model['model_name']}

### Performance Metrics
- **Accuracy:** {best_model['accuracy']:.4f}
- **Precision:** {best_model['precision']:.4f}
- **Recall:** {best_model['recall']:.4f}
- **F1 Score:** {best_model['f1']:.4f}
- **ROC-AUC:** {best_model['roc_auc']:.4f if best_model['roc_auc'] else 'N/A'}
- **CV Mean:** {best_model['cv_mean']:.4f if best_model['cv_mean'] else 'N/A'}
- **Training Time:** {best_model['training_time']:.3f}s

## All Models Performance

{results_df.to_markdown(index=False)}

## Dataset Information
- **Rows:** {len(df)}
- **Features:** {len(trainer.feature_names)}
- **Target:** {target_col}
- **Classes:** {', '.join(trainer.class_names)}

---
*Report generated by DataInsight AI - Machine Learning Module*
"""
                st.download_button(
                    label="📄 Download Best Model Report (MD)",
                    data=report,
                    file_name=f"ml_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

def show_anomaly_detection():
    """Anomaly & Outlier Detection page."""
    st.header("🔬 Anomaly & Outlier Detection")
    
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
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.info("👆 Please upload data from the **Data Upload** page first")
        return
    
    df = st.session_state.data
    
    # Data overview
    st.subheader("📊 1. Dataset Overview")
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
    st.subheader("🎯 2. Select Features for Analysis")
    
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
    st.subheader("🤖 3. Configure Detection Algorithm")
    
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
        with st.spinner(f"Running {algorithm}..."):
            try:
                from utils.anomaly_detection import AnomalyDetector
                
                # Initialize detector
                detector = AnomalyDetector(df)
                detector.set_features(feature_cols)
                
                # Run selected algorithm
                if algorithm == "Isolation Forest":
                    results = detector.run_isolation_forest(contamination)
                elif algorithm == "Local Outlier Factor":
                    results = detector.run_local_outlier_factor(contamination)
                else:  # One-Class SVM
                    results = detector.run_one_class_svm(nu=contamination)
                
                # Store results
                st.session_state.anomaly_detector = detector
                st.session_state.anomaly_results = results
                st.session_state.anomaly_algorithm = algorithm
                
                st.success(f"✅ Anomaly detection completed using {algorithm}!")
                
            except Exception as e:
                st.error(f"Error running anomaly detection: {str(e)}")
    
    # Show results if available
    if 'anomaly_results' in st.session_state:
        results = st.session_state.anomaly_results
        detector = st.session_state.anomaly_detector
        algorithm = st.session_state.anomaly_algorithm
        
        # Summary metrics
        st.divider()
        st.subheader("📈 4. Detection Results")
        
        stats = detector.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{stats['total_records']:,}")
        with col2:
            st.metric("Anomalies Detected", f"{stats['num_anomalies']:,}", 
                     delta=f"{stats['pct_anomalies']:.1f}%")
        with col3:
            st.metric("Normal Records", f"{stats['num_normal']:,}",
                     delta=f"{stats['pct_normal']:.1f}%")
        with col4:
            avg_score = results['anomaly_score'].mean()
            st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
        
        # Results table
        st.subheader("📋 5. Detailed Results")
        
        show_filter = st.radio(
            "Display:",
            ["All Records", "Anomalies Only", "Normal Only"],
            horizontal=True
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
        
        tab1, tab2, tab3 = st.tabs(["Anomaly Profiles", "Feature Importance", "AI Explanation"])
        
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
                st.write("**Feature Importance for Anomaly Detection:**")
                importance = detector.get_feature_importance()
                
                if importance is not None:
                    fig_importance = px.bar(
                        importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Contribution to Anomaly Detection'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info(f"Feature importance is only available for Isolation Forest. Current algorithm: {algorithm}")
        
        with tab3:
            st.write("**AI-Powered Anomaly Explanation:**")
            
            if st.button("🤖 Generate AI Explanation", type="primary"):
                with st.spinner("Analyzing anomalies with AI..."):
                    try:
                        from utils.ai_helper import AIHelper
                        
                        ai = AIHelper()
                        
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
                        
                        explanation = response.choices[0].message.content
                        st.markdown(explanation)
                        
                    except Exception as e:
                        st.error(f"Error generating AI explanation: {str(e)}")
        
        # Export section
        st.divider()
        st.subheader("📥 8. Export Results")
        
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

---
*Report generated by DataInsight AI - Anomaly Detection Module*
"""
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

def show_time_series_forecasting():
    """Time Series Forecasting & Analysis page."""
    st.header("📈 Time Series Forecasting")
    
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
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.info("👆 Please upload data from the **Data Upload** page first")
        return
    
    df = st.session_state.data
    
    # Get smart column suggestions
    from utils.column_detector import ColumnDetector
    suggestions = ColumnDetector.get_time_series_column_suggestions(df)
    
    # Validate data suitability
    validation = ColumnDetector.validate_time_series_suitability(df)
    
    if not validation['suitable']:
        st.error("❌ **Dataset Not Suitable for Time Series Forecasting**")
        for warning in validation['warnings']:
            st.warning(warning)
        st.info("**💡 Recommendations:**")
        for rec in validation['recommendations']:
            st.write(f"- {rec}")
        return
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
    st.subheader("📊 1. Configure Time Series")
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
        st.subheader("🔍 2. Time Series Analysis")
        
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
        st.subheader("🔮 3. Generate Forecasts")
        
        forecast_periods = st.slider(
            "Forecast Horizon (periods):",
            min_value=7,
            max_value=365,
            value=30,
            help="Number of periods to forecast into the future"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Run ARIMA Forecast", use_container_width=True):
                with st.spinner("Running Auto-ARIMA..."):
                    try:
                        results = analyzer.run_auto_arima(forecast_periods)
                        st.session_state.arima_results = results
                        
                        st.success("✅ ARIMA model trained!")
                        
                        st.write("**Model Configuration:**")
                        st.write(f"- Order (p,d,q): {results['model_order']}")
                        st.write(f"- Seasonal Order: {results['seasonal_order']}")
                        st.write(f"- AIC: {results['aic']:.2f}")
                        st.write(f"- BIC: {results['bic']:.2f}")
                        
                        fig = analyzer.create_forecast_plot('arima')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(results['forecast'].head(10), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📊 Run Prophet Forecast", use_container_width=True):
                with st.spinner("Running Prophet..."):
                    try:
                        results = analyzer.run_prophet(forecast_periods)
                        st.session_state.prophet_results = results
                        
                        st.success("✅ Prophet model trained!")
                        
                        fig = analyzer.create_forecast_plot('prophet')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10),
                                   use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Model comparison
        if 'arima_results' in st.session_state and 'prophet_results' in st.session_state:
            st.divider()
            st.subheader("⚖️ 4. Model Comparison")
            
            fig = analyzer.create_comparison_plot()
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            if st.button("🤖 Generate AI Insights"):
                with st.spinner("Analyzing forecasts..."):
                    try:
                        from utils.ai_helper import AIHelper
                        ai = AIHelper()
                        
                        context = f"""
                        Time Series Analysis:
                        - Time Period: {ts_data.index[0]} to {ts_data.index[-1]}
                        - Observations: {len(ts_data)}
                        - Mean: {ts_data.mean():.2f}
                        - Std Dev: {ts_data.std():.2f}
                        - Trend: {"Increasing" if ts_data.iloc[-10:].mean() > ts_data.iloc[:10].mean() else "Decreasing"}
                        
                        Forecast Summary:
                        - Forecast Horizon: {forecast_periods} periods
                        - ARIMA Forecast Mean: {st.session_state.arima_results['forecast']['forecast'].mean():.2f}
                        - Prophet Forecast Mean: {st.session_state.prophet_results['forecast']['yhat'].mean():.2f}
                        """
                        
                        prompt = f"""
                        As a business analyst, analyze this time series forecast and provide:
                        1. Interpretation of the forecast trends
                        2. Which model (ARIMA or Prophet) appears more reliable and why
                        3. Business recommendations based on the forecast
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
                        
                        st.markdown(response.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
        
        # Export section
        if 'arima_results' in st.session_state or 'prophet_results' in st.session_state:
            st.divider()
            st.subheader("📥 5. Export Results")
            
            col1, col2 = st.columns(2)
            
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

def show_text_mining():
    """Text Mining & Sentiment Analysis page."""
    st.header("💬 Text Mining & Sentiment Analysis")
    
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
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.info("👆 Please upload data from the **Data Upload** page first")
        return
    
    df = st.session_state.data
    
    # Column selection with smart detection
    st.subheader("📝 1. Select Text Column")
    
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
        st.subheader("🔍 2. Text Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Word Frequency", "Topic Modeling", "AI Summary"])
        
        with tab1:
            st.write("**Sentiment Analysis using VADER:**")
            
            # Only show button if results don't exist
            if 'sentiment_results' not in st.session_state:
                if st.button("📊 Analyze Sentiment", key="sentiment_btn"):
                    with st.spinner("Analyzing sentiment..."):
                        try:
                            sentiment_df = analyzer.get_sentiment_analysis()
                            st.session_state.sentiment_results = sentiment_df
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
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
                    with st.spinner("Analyzing word frequency..."):
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
                    with st.spinner("Running topic modeling..."):
                        try:
                            topics = analyzer.get_topic_modeling(num_topics, n_words=10)
                            st.session_state.topics = topics
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
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
            
            if st.button("🤖 Generate AI Summary"):
                with st.spinner("Generating AI summary..."):
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
                        
                        st.markdown(response.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Error generating AI summary: {str(e)}")
        
        # Export section
        if 'sentiment_results' in st.session_state or 'word_freq_results' in st.session_state:
            st.divider()
            st.subheader("📥 3. Export Results")
            
            col1, col2 = st.columns(2)
            
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

if __name__ == "__main__":
    main()
