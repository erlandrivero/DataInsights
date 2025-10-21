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
            ["Home", "Data Upload", "Analysis", "Insights", "Reports"],
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

def show_home():
    st.header("Welcome to DataInsight AI! 👋")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📤 Upload Data</h3>
            <p>Upload CSV or Excel files and get instant data profiling</p>
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
            <p>Interactive charts and dashboards generated automatically</p>
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
            
            # Navigation hint
            st.info("👉 Navigate to **Analysis** to explore your data further!")
            
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

if __name__ == "__main__":
    main()
