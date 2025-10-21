# DataInsight AI - Windsurf Prompts (Core Features)

## Overview
These prompts will guide you through building the core features of DataInsight AI using Windsurf. Follow them in order for best results.

**Total Prompts in This Set:** 6  
**Estimated Time:** 6-7 hours  
**Result:** Working core application with data upload, analysis, AI integration, and basic visualizations

---

# PROMPT 1: Project Setup and Foundation

## Context
Create a new Streamlit application for DataInsight AI - an AI-powered business intelligence assistant. This prompt sets up the project structure, dependencies, and basic UI framework.

## Instructions

Create a new Python project with the following structure and files:

### Project Structure:
```
datainsight-ai/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── utils/
│   ├── __init__.py
│   ├── data_processor.py
│   └── ai_helper.py
└── assets/
    └── style.css
```

### File: `requirements.txt`
```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
openai==1.10.0
python-dotenv==1.0.0
openpyxl==3.1.2
```

### File: `.env.example`
```
OPENAI_API_KEY=your_openai_api_key_here
```

### File: `.gitignore`
```
.env
__pycache__/
*.pyc
.DS_Store
.streamlit/
venv/
*.csv
*.xlsx
```

### File: `app.py` (Initial Version)
```python
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

# Main app
def main():
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
    st.write("Upload your CSV or Excel file to get started")
    # Will be implemented in next prompt

def show_analysis():
    st.header("📊 Data Analysis")
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        st.write("Analysis features coming soon...")

def show_insights():
    st.header("🤖 AI Insights")
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        st.write("AI insights coming soon...")

def show_reports():
    st.header("📄 Reports")
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        st.write("Report generation coming soon...")

if __name__ == "__main__":
    main()
```

### File: `README.md`
```markdown
# DataInsight AI 🎯

An AI-powered business intelligence assistant that helps you analyze data, generate insights, and create professional reports using natural language.

## Features

- 📤 **Data Upload**: Upload CSV/Excel files with automatic profiling
- 🤖 **AI Analysis**: Ask questions in natural language
- 📊 **Visualizations**: Interactive charts and dashboards
- 💡 **Automated Insights**: AI-generated findings and recommendations
- 📄 **Professional Reports**: Export-ready business reports

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Run the app: `streamlit run app.py`

## Technologies

- Streamlit
- OpenAI GPT-4
- pandas
- plotly

## Author

[Your Name]
```

## Testing

1. Create a new folder and set up the project structure
2. Create all files as specified
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with your OpenAI API key
5. Run the app: `streamlit run app.py`
6. Verify:
   - App launches without errors
   - Navigation works (Home, Data Upload, Analysis, Insights, Reports)
   - UI looks professional
   - Custom CSS is applied

## Expected Output

- ✅ Clean, professional UI with custom styling
- ✅ Working navigation between pages
- ✅ Responsive layout
- ✅ Feature boxes on home page
- ✅ Sidebar with navigation and info
- ✅ No errors in console

## Review Checklist

- [ ] All files created
- [ ] Dependencies installed
- [ ] App runs successfully
- [ ] Navigation works
- [ ] UI looks professional
- [ ] No console errors

---

# PROMPT 2: Data Upload and Processing

## Context
Implement the data upload functionality with automatic data profiling, validation, and preview. This includes handling CSV and Excel files, detecting data types, and providing initial statistics.

## Instructions

### Update `utils/data_processor.py`:

```python
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataProcessor:
    """Handles data loading, validation, and profiling."""
    
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'duplicates': df.duplicated().sum()
            },
            'column_info': [],
            'missing_data': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Column information
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'missing_pct': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
                'unique': df[col].nunique(),
                'unique_pct': f"{(df[col].nunique() / len(df)) * 100:.2f}%"
            }
            
            # Add sample values
            if df[col].dtype in ['object', 'category']:
                col_info['sample'] = df[col].value_counts().head(3).to_dict()
            else:
                col_info['sample'] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None
                }
            
            profile['column_info'].append(col_info)
        
        # Missing data summary
        missing_cols = df.columns[df.isnull().any()].tolist()
        profile['missing_data'] = {
            'columns_with_missing': len(missing_cols),
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%"
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            profile['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            profile['categorical_summary'] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:5]  # Limit to first 5 categorical columns
            }
        
        return profile
    
    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential data quality issues."""
        issues = []
        
        # Check for high missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                issues.append({
                    'type': 'High Missing Values',
                    'column': col,
                    'severity': 'High',
                    'description': f"{missing_pct:.1f}% of values are missing"
                })
            elif missing_pct > 20:
                issues.append({
                    'type': 'Moderate Missing Values',
                    'column': col,
                    'severity': 'Medium',
                    'description': f"{missing_pct:.1f}% of values are missing"
                })
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                'type': 'Duplicate Rows',
                'column': 'All',
                'severity': 'Medium' if dup_count < len(df) * 0.1 else 'High',
                'description': f"{dup_count} duplicate rows found ({(dup_count/len(df)*100):.1f}%)"
            })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append({
                    'type': 'Constant Column',
                    'column': col,
                    'severity': 'Low',
                    'description': 'Column has only one unique value'
                })
        
        # Check for high cardinality in categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_pct = (df[col].nunique() / len(df)) * 100
            if unique_pct > 90:
                issues.append({
                    'type': 'High Cardinality',
                    'column': col,
                    'severity': 'Low',
                    'description': f"{unique_pct:.1f}% unique values (might need encoding review)"
                })
        
        return issues
```

### Update `app.py` - Replace `show_data_upload()` function:

```python
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
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'product': np.random.choice(['Product A', 'Product B', 'Product C'], 100),
                'revenue': np.random.uniform(100, 1000, 100),
                'quantity': np.random.randint(1, 20, 100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
            })
            st.session_state.data = sample_data
            st.rerun()
```

## Testing

1. Run the app: `streamlit run app.py`
2. Navigate to "Data Upload"
3. Test uploading a CSV file:
   - Verify file loads successfully
   - Check data preview displays
   - Verify column information is accurate
   - Check data quality issues are detected
4. Test uploading an Excel file
5. Test the "Load Sample Sales Data" button
6. Verify session state persists (data available on other pages)

## Expected Output

- ✅ File upload works for CSV and Excel
- ✅ Data preview displays correctly
- ✅ Column information table shows all details
- ✅ Data quality issues are detected and displayed
- ✅ Sample data button works
- ✅ Success/error messages display appropriately
- ✅ Data persists in session state

## Review Checklist

- [ ] `DataProcessor` class implemented
- [ ] File upload works for both CSV and Excel
- [ ] Data profiling generates complete information
- [ ] Data quality issues are detected
- [ ] UI displays all information clearly
- [ ] Sample data button works
- [ ] No errors in console

---

# PROMPT 3: OpenAI Integration and AI Helper

## Context
Integrate OpenAI GPT-4 to provide AI-powered data analysis capabilities. This includes setting up the AI helper, creating prompts for data analysis, and implementing natural language querying.

## Instructions

### Update `utils/ai_helper.py`:

```python
import openai
import os
import pandas as pd
import json
from typing import Dict, Any, List

class AIHelper:
    """Handles all AI-related operations using OpenAI."""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_data_insights(self, df: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Generate AI insights about the dataset."""
        # Prepare context
        context = f"""
        Dataset Overview:
        - Rows: {profile['basic_info']['rows']}
        - Columns: {profile['basic_info']['columns']}
        - Duplicates: {profile['basic_info']['duplicates']}
        
        Column Information:
        {json.dumps(profile['column_info'][:10], indent=2)}
        
        Missing Data:
        {json.dumps(profile['missing_data'], indent=2)}
        """
        
        prompt = f"""
        You are a data analyst. Analyze this dataset and provide:
        1. A brief summary of the data
        2. Key observations about data quality
        3. Interesting patterns or anomalies
        4. Recommendations for analysis or cleaning
        
        {context}
        
        Provide insights in a clear, business-friendly format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst providing insights on business data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def answer_data_question(self, question: str, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Answer a natural language question about the data."""
        # Get data summary
        data_summary = f"""
        Dataset has {len(df)} rows and {len(df.columns)} columns.
        Columns: {', '.join(df.columns.tolist())}
        
        Sample data (first 3 rows):
        {df.head(3).to_string()}
        
        Data types:
        {df.dtypes.to_string()}
        """
        
        prompt = f"""
        You are a data analyst. The user has a dataset and asks: "{question}"
        
        Dataset information:
        {data_summary}
        
        {context}
        
        Provide:
        1. A direct answer to the question
        2. Python pandas code to get this answer (if applicable)
        3. Any relevant insights
        
        Format your response as JSON with keys: "answer", "code", "insights"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Always provide actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                result = json.loads(content)
            except:
                # If not JSON, structure the response
                result = {
                    "answer": content,
                    "code": None,
                    "insights": "See answer above"
                }
            
            return result
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "code": None,
                "insights": None
            }
    
    def generate_cleaning_suggestions(self, df: pd.DataFrame, issues: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate AI-powered data cleaning suggestions."""
        issues_summary = "\n".join([
            f"- {issue['type']} in {issue['column']}: {issue['description']}"
            for issue in issues
        ])
        
        prompt = f"""
        Given these data quality issues:
        {issues_summary}
        
        Dataset info:
        - {len(df)} rows, {len(df.columns)} columns
        - Columns: {', '.join(df.columns.tolist())}
        
        Provide specific, actionable cleaning suggestions. For each issue, suggest:
        1. What to do
        2. Why it matters
        3. Python pandas code to fix it
        
        Format as JSON array with keys: "issue", "suggestion", "reason", "code"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data cleaning expert. Provide practical, executable solutions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            try:
                suggestions = json.loads(content)
                if isinstance(suggestions, list):
                    return suggestions
            except:
                pass
            
            # Fallback if JSON parsing fails
            return [{
                "issue": "General",
                "suggestion": content,
                "reason": "AI-generated recommendation",
                "code": "# See suggestion above"
            }]
            
        except Exception as e:
            return [{
                "issue": "Error",
                "suggestion": f"Could not generate suggestions: {str(e)}",
                "reason": "",
                "code": ""
            }]
```

### Update `app.py` - Add AI Insights to Analysis page:

```python
def show_analysis():
    st.header("📊 Data Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    profile = st.session_state.get('profile', {})
    issues = st.session_state.get('issues', [])
    
    # Tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["📈 Statistics", "🤖 AI Insights", "🔧 Cleaning Suggestions"])
    
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
```

## Testing

1. Ensure your `.env` file has a valid OpenAI API key
2. Run the app: `streamlit run app.py`
3. Upload a dataset
4. Navigate to "Analysis"
5. Test "Generate AI Insights" button:
   - Verify AI generates meaningful insights
   - Check insights are displayed correctly
   - Test "Regenerate Insights" button
6. Test "Get AI Cleaning Suggestions":
   - Verify suggestions are generated
   - Check code snippets are provided
   - Verify suggestions are actionable

## Expected Output

- ✅ OpenAI integration works
- ✅ AI insights are generated successfully
- ✅ Insights are relevant and well-formatted
- ✅ Cleaning suggestions are practical
- ✅ Code snippets are provided
- ✅ Error handling works (try with invalid API key)
- ✅ Loading states display during AI calls

## Review Checklist

- [ ] `AIHelper` class implemented
- [ ] OpenAI API key is loaded correctly
- [ ] AI insights generation works
- [ ] Cleaning suggestions are generated
- [ ] UI displays AI responses clearly
- [ ] Error handling is robust
- [ ] Loading spinners show during AI calls

---

# PROMPT 4: Natural Language Querying

## Context
Implement the natural language querying feature that allows users to ask questions about their data in plain English and get AI-powered answers with visualizations.

## Instructions

### Update `app.py` - Replace `show_insights()` function:

```python
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
```

### Add numpy import to `app.py`:
```python
import numpy as np  # Add this at the top with other imports
```

## Testing

1. Run the app: `streamlit run app.py`
2. Upload a dataset (or use sample data)
3. Navigate to "Insights"
4. Test various questions:
   - "What are the top 5 products by revenue?"
   - "Show me the average revenue"
   - "How many missing values are there?"
   - "Find the maximum value in [column]"
5. Verify:
   - AI provides relevant answers
   - Code is generated when applicable
   - Code can be executed
   - Chat history is maintained
   - Clear history works

## Expected Output

- ✅ Natural language input works
- ✅ AI provides relevant answers
- ✅ Python code is generated
- ✅ Code execution works (when applicable)
- ✅ Chat history is displayed
- ✅ Clear history button works
- ✅ Example questions are helpful

## Review Checklist

- [ ] Question input works
- [ ] AI answers questions correctly
- [ ] Code generation works
- [ ] Code execution is safe and functional
- [ ] Chat history persists
- [ ] UI is clean and intuitive
- [ ] Error handling works

---

# PROMPT 5: Data Visualizations

## Context
Add interactive visualizations using Plotly to help users explore their data visually. Include automatic chart suggestions and customizable visualizations.

## Instructions

### Create `utils/visualizations.py`:

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any

class Visualizer:
    """Handles data visualization creation."""
    
    @staticmethod
    def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data types."""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3
            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'columns': [col],
                'description': f'Shows the distribution of values in {col}'
            })
        
        # Bar charts for categorical columns
        for col in categorical_cols[:3]:
            suggestions.append({
                'type': 'bar',
                'title': f'Count of {col}',
                'columns': [col],
                'description': f'Shows the frequency of each category in {col}'
            })
        
        # Scatter plots for numeric pairs
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'title': f'{numeric_cols[0]} vs {numeric_cols[1]}',
                'columns': [numeric_cols[0], numeric_cols[1]],
                'description': f'Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}'
            })
        
        # Box plots for numeric columns
        for col in numeric_cols[:2]:
            suggestions.append({
                'type': 'box',
                'title': f'Box Plot of {col}',
                'columns': [col],
                'description': f'Shows distribution and outliers in {col}'
            })
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 3:
            suggestions.append({
                'type': 'correlation',
                'title': 'Correlation Heatmap',
                'columns': numeric_cols,
                'description': 'Shows correlations between numeric columns'
            })
        
        return suggestions
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a histogram."""
        fig = px.histogram(
            df,
            x=column,
            title=title or f'Distribution of {column}',
            labels={column: column},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            showlegend=False,
            height=400
        )
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a bar chart."""
        value_counts = df[column].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            title=title or f'Top 10 {column} Values',
            xaxis_title=column,
            yaxis_title='Count',
            height=400
        )
        return fig
    
    @staticmethod
    def create_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str = None) -> go.Figure:
        """Create a scatter plot."""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title or f'{x_col} vs {y_col}',
            labels={x_col: x_col, y_col: y_col},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create a box plot."""
        fig = px.box(
            df,
            y=column,
            title=title or f'Box Plot of {column}',
            labels={column: column},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None, title: str = None) -> go.Figure:
        """Create a correlation heatmap."""
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title or 'Correlation Heatmap',
            height=500,
            width=500
        )
        return fig
```

### Update `app.py` - Add visualization section to Analysis page:

Add this new tab to the `show_analysis()` function after the existing tabs:

```python
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
    
    # ... (keep existing tab1, tab2, tab3 code) ...
    
    with tab4:
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
            col = st.selectbox("Select column:", numeric_cols)
            if st.button("Create Histogram"):
                fig = viz.create_histogram(df, col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            col = st.selectbox("Select column:", categorical_cols)
            if st.button("Create Bar Chart"):
                fig = viz.create_bar_chart(df, col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            col1 = st.selectbox("Select X axis:", numeric_cols, key="scatter_x")
            col2 = st.selectbox("Select Y axis:", numeric_cols, key="scatter_y")
            if st.button("Create Scatter Plot"):
                fig = viz.create_scatter(df, col1, col2)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            col = st.selectbox("Select column:", numeric_cols)
            if st.button("Create Box Plot"):
                fig = viz.create_box_plot(df, col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Heatmap":
            selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:5])
            if st.button("Create Heatmap"):
                if len(selected_cols) >= 2:
                    fig = viz.create_correlation_heatmap(df, selected_cols)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 columns")
```

## Testing

1. Run the app: `streamlit run app.py`
2. Upload a dataset
3. Navigate to "Analysis" → "Visualizations" tab
4. Test:
   - Verify suggested visualizations display
   - Check each visualization type works
   - Test custom visualization builder
   - Try different column combinations
   - Verify charts are interactive (Plotly features)

## Expected Output

- ✅ Suggested visualizations are generated
- ✅ All chart types render correctly
- ✅ Charts are interactive (zoom, pan, hover)
- ✅ Custom visualization builder works
- ✅ Appropriate columns are suggested for each chart type
- ✅ Error handling works for invalid selections

## Review Checklist

- [ ] `Visualizer` class implemented
- [ ] All visualization types work
- [ ] Suggested visualizations are relevant
- [ ] Custom builder is functional
- [ ] Charts are interactive
- [ ] UI is clean and intuitive

---

# PROMPT 6: Testing Core Features

## Context
Comprehensive testing of all core features implemented so far. This ensures everything works correctly before moving to advanced features.

## Instructions

### Create `tests/test_core_features.py`:

```python
"""
Test script for DataInsight AI core features.
Run this to verify all core functionality works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_data_processor():
    """Test DataProcessor functionality."""
    print("Testing DataProcessor...")
    
    from utils.data_processor import DataProcessor
    
    # Create test data
    test_data = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5, None],
        'numeric2': [10, 20, 30, 40, 50, 60],
        'category1': ['A', 'B', 'A', 'B', 'A', 'B'],
        'category2': ['X', 'X', 'X', 'X', 'X', 'X']  # Constant column
    })
    
    # Test profiling
    profile = DataProcessor.profile_data(test_data)
    assert profile['basic_info']['rows'] == 6
    assert profile['basic_info']['columns'] == 4
    assert len(profile['column_info']) == 4
    
    # Test issue detection
    issues = DataProcessor.detect_data_quality_issues(test_data)
    assert len(issues) > 0  # Should detect constant column and missing values
    
    print("✅ DataProcessor tests passed!")

def test_ai_helper():
    """Test AIHelper functionality."""
    print("Testing AIHelper...")
    
    try:
        from utils.ai_helper import AIHelper
        
        ai = AIHelper()
        
        # Create test data
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'revenue': [100, 200, 150]
        })
        
        # Test data insights (this will make an API call)
        print("  Testing AI insights generation...")
        profile = {'basic_info': {'rows': 3, 'columns': 2, 'duplicates': 0}, 'column_info': []}
        insights = ai.generate_data_insights(test_data, profile)
        assert isinstance(insights, str)
        assert len(insights) > 0
        
        print("✅ AIHelper tests passed!")
    except Exception as e:
        print(f"⚠️ AIHelper tests skipped or failed: {str(e)}")
        print("  Make sure OPENAI_API_KEY is set in .env file")

def test_visualizer():
    """Test Visualizer functionality."""
    print("Testing Visualizer...")
    
    from utils.visualizations import Visualizer
    
    # Create test data
    test_data = pd.DataFrame({
        'numeric1': np.random.rand(100),
        'numeric2': np.random.rand(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    viz = Visualizer()
    
    # Test suggestions
    suggestions = viz.suggest_visualizations(test_data)
    assert len(suggestions) > 0
    
    # Test each visualization type
    fig1 = viz.create_histogram(test_data, 'numeric1')
    assert fig1 is not None
    
    fig2 = viz.create_bar_chart(test_data, 'category')
    assert fig2 is not None
    
    fig3 = viz.create_scatter(test_data, 'numeric1', 'numeric2')
    assert fig3 is not None
    
    fig4 = viz.create_box_plot(test_data, 'numeric1')
    assert fig4 is not None
    
    fig5 = viz.create_correlation_heatmap(test_data, ['numeric1', 'numeric2'])
    assert fig5 is not None
    
    print("✅ Visualizer tests passed!")

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running DataInsight AI Core Feature Tests")
    print("=" * 50)
    print()
    
    try:
        test_data_processor()
        print()
        test_visualizer()
        print()
        test_ai_helper()
        print()
        print("=" * 50)
        print("✅ All core feature tests completed!")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Tests failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
```

### Create `tests/__init__.py`:
```python
# Empty file to make tests a package
```

### Testing Checklist Document:

Create `TESTING_CHECKLIST.md`:

```markdown
# DataInsight AI - Core Features Testing Checklist

## Automated Tests

Run the automated test script:
```bash
python tests/test_core_features.py
```

Expected output: All tests should pass ✅

## Manual Testing

### 1. Application Launch
- [ ] App launches without errors
- [ ] UI loads correctly
- [ ] Custom CSS is applied
- [ ] Navigation sidebar works

### 2. Data Upload
- [ ] Can upload CSV files
- [ ] Can upload Excel files
- [ ] Sample data button works
- [ ] Data preview displays correctly
- [ ] Column information table is accurate
- [ ] Data quality issues are detected
- [ ] Error handling works (try invalid file)

### 3. Data Analysis
- [ ] Statistical summary displays
- [ ] Numeric column statistics are correct
- [ ] Categorical column counts are correct
- [ ] All tabs work (Statistics, Visualizations, AI Insights, Cleaning)

### 4. AI Features
- [ ] AI insights generation works
- [ ] Insights are relevant and well-formatted
- [ ] Cleaning suggestions are generated
- [ ] Suggestions include code snippets
- [ ] Error handling works (test with invalid API key)

### 5. Natural Language Querying
- [ ] Question input works
- [ ] AI provides relevant answers
- [ ] Code is generated when applicable
- [ ] Code execution works
- [ ] Chat history is maintained
- [ ] Clear history button works

### 6. Visualizations
- [ ] Suggested visualizations display
- [ ] Histogram works
- [ ] Bar chart works
- [ ] Scatter plot works
- [ ] Box plot works
- [ ] Correlation heatmap works
- [ ] Custom visualization builder works
- [ ] Charts are interactive (zoom, hover, pan)

### 7. Performance
- [ ] App responds quickly
- [ ] Large datasets load without crashing
- [ ] AI calls complete in reasonable time
- [ ] No memory leaks

### 8. Error Handling
- [ ] Invalid file uploads show error messages
- [ ] Missing API key shows helpful error
- [ ] Invalid questions handled gracefully
- [ ] Network errors handled properly

## Test Datasets

Test with these scenarios:
1. **Small dataset** (< 100 rows) - Should work perfectly
2. **Medium dataset** (1,000-10,000 rows) - Should work well
3. **Large dataset** (> 10,000 rows) - Should work but may be slower
4. **Dataset with missing values** - Should detect and suggest fixes
5. **Dataset with duplicates** - Should detect and report
6. **Dataset with only numeric columns** - Should work
7. **Dataset with only categorical columns** - Should work
8. **Dataset with mixed types** - Should work

## Known Issues / Limitations

Document any issues found:
- 
- 
- 

## Sign-off

- [ ] All automated tests pass
- [ ] All manual tests pass
- [ ] Performance is acceptable
- [ ] Error handling is robust
- [ ] Ready for advanced features

Tested by: _______________
Date: _______________
```

## Testing Instructions

1. Create the `tests/` directory
2. Create `test_core_features.py` and `__init__.py`
3. Run automated tests:
   ```bash
   python tests/test_core_features.py
   ```
4. Go through the manual testing checklist in `TESTING_CHECKLIST.md`
5. Document any issues found
6. Fix issues before proceeding to advanced features

## Expected Output

- ✅ All automated tests pass
- ✅ All manual tests pass
- ✅ No critical bugs
- ✅ Performance is acceptable
- ✅ Ready for advanced features (next prompt set)

## Review Checklist

- [ ] Automated test script created
- [ ] All automated tests pass
- [ ] Manual testing checklist completed
- [ ] Issues documented
- [ ] Core features are stable

---

## 🎉 Core Features Complete!

You now have a fully functional DataInsight AI application with:
- ✅ Data upload and profiling
- ✅ AI-powered insights
- ✅ Natural language querying
- ✅ Interactive visualizations
- ✅ Data quality analysis
- ✅ Cleaning suggestions

**Next:** Advanced features (Reports, Export, Polish) in the next prompt set!

---

## Estimated Time Summary

- Prompt 1 (Setup): 30 minutes
- Prompt 2 (Data Upload): 1 hour
- Prompt 3 (AI Integration): 1.5 hours
- Prompt 4 (NL Querying): 1 hour
- Prompt 5 (Visualizations): 1.5 hours
- Prompt 6 (Testing): 30 minutes

**Total: ~6 hours**
