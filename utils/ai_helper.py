"""AI integration utilities for DataInsights using OpenAI GPT-4.

This module provides comprehensive AI capabilities including data insights generation,
natural language question answering, and automated data cleaning suggestions.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import openai
import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Union, Optional


class AIHelper:
    """Handles all AI-related operations using OpenAI GPT-4.
    
    This class provides methods for generating intelligent insights about datasets,
    answering natural language questions about data, and suggesting data cleaning
    operations using OpenAI's GPT-4 model.
    
    Attributes:
        api_key (str): OpenAI API key from Streamlit secrets or environment
        client: OpenAI client instance for API calls
    
    Example:
        >>> # Initialize AI helper
        >>> ai = AIHelper()
        >>> 
        >>> # Generate insights
        >>> insights = ai.generate_data_insights(df, profile)
        >>> st.write(insights)
        >>> 
        >>> # Answer questions
        >>> response = ai.answer_data_question("What are the top 5 customers?", df)
        >>> st.write(response['answer'])
    
    Note:
        Requires OPENAI_API_KEY in Streamlit secrets or environment variables
    """
    
    @staticmethod
    def convert_to_json_serializable(obj: Any) -> Any:
        """Convert numpy/pandas types to native Python types for JSON serialization.
        
        Recursively converts non-serializable types (numpy arrays, pandas Series,
        numpy numeric types) to standard Python types that can be JSON serialized.
        
        Args:
            obj: Object to convert (can be nested dicts, lists, or numpy/pandas types)
        
        Returns:
            JSON-serializable version of the object
        
        Example:
            >>> import numpy as np
            >>> data = {'count': np.int64(100), 'values': np.array([1, 2, 3])}
            >>> serializable = AIHelper.convert_to_json_serializable(data)
            >>> json.dumps(serializable)  # Works without errors
        
        Note:
            - Handles nested structures recursively
            - Converts numpy int64/int32 → int
            - Converts numpy float64/float32 → float
            - Converts numpy arrays → lists
            - Converts pandas Series → lists
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            # Convert pandas Timestamp to ISO string
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif hasattr(obj, 'timestamp'):
            # Handle datetime.datetime, datetime.date objects
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: AIHelper.convert_to_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [AIHelper.convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def __init__(self):
        """Initialize AIHelper with OpenAI API credentials.
        
        Attempts to retrieve API key from Streamlit secrets first (for cloud
        deployment), then falls back to environment variable (for local development).
        
        Raises:
            ValueError: If OPENAI_API_KEY is not found in secrets or environment
        
        Example:
            >>> try:
            >>>     ai = AIHelper()
            >>>     st.success("AI features enabled!")
            >>> except ValueError as e:
            >>>     st.error(f"AI not available: {e}")
        
        Note:
            - Cloud deployment: Set OPENAI_API_KEY in Streamlit secrets.toml
            - Local development: Set OPENAI_API_KEY in .env file
            - API key is never exposed in logs or error messages
        """
        # Try to get API key from Streamlit secrets first (for cloud deployment)
        # Then fall back to environment variable (for local development)
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                self.api_key = st.secrets['OPENAI_API_KEY']
            else:
                self.api_key = os.getenv('OPENAI_API_KEY')
        except:
            self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in "
                "Streamlit secrets or .env file"
            )
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_data_insights(
        self, 
        df: pd.DataFrame, 
        profile: Dict[str, Any]
    ) -> str:
        """Generate AI-powered insights about the dataset using GPT-4.
        
        Analyzes dataset characteristics including size, structure, data quality,
        and patterns to provide business-friendly insights and recommendations.
        
        Args:
            df: DataFrame to analyze
            profile: Data profile dictionary from DataProcessor.profile_data()
        
        Returns:
            Markdown-formatted string containing AI-generated insights including:
                - Brief data summary
                - Data quality observations
                - Interesting patterns or anomalies
                - Recommendations for analysis or cleaning
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> ai = AIHelper()
            >>> insights = ai.generate_data_insights(df, profile)
            >>> 
            >>> st.markdown("### AI Insights")
            >>> st.write(insights)
        
        Note:
            - Uses GPT-4 model (higher quality than GPT-3.5)
            - Temperature set to 0.7 for balanced creativity
            - Max 1000 tokens output
            - Automatically handles JSON serialization of profile
            - Returns error message on API failure
        """
        # Convert profile to JSON-serializable format
        safe_profile = self.convert_to_json_serializable(profile)
        
        # Prepare context
        context = f"""
        Dataset Overview:
        - Rows: {safe_profile['basic_info']['rows']}
        - Columns: {safe_profile['basic_info']['columns']}
        - Duplicates: {safe_profile['basic_info']['duplicates']}
        
        Column Information:
        {json.dumps(safe_profile['column_info'][:10], indent=2)}
        
        Missing Data:
        {json.dumps(safe_profile['missing_data'], indent=2)}
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
                    {
                        "role": "system", 
                        "content": "You are an expert data analyst providing insights on business data."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def answer_data_question(
        self, 
        question: str, 
        df: pd.DataFrame, 
        context: str = ""
    ) -> Dict[str, Optional[str]]:
        """Answer a natural language question about the data using GPT-4.
        
        Interprets user questions in natural language and provides direct answers
        along with executable Python code to reproduce the analysis.
        
        Args:
            question: Natural language question about the data
                     (e.g., "What are the top 5 customers by revenue?")
            df: DataFrame to query
            context: Optional additional context about the data
        
        Returns:
            Dictionary with keys:
                - answer (str): Direct answer to the question
                - code (str|None): Python pandas code to get this answer
                - insights (str|None): Brief business insight (1-2 sentences)
        
        Example:
            >>> ai = AIHelper()
            >>> response = ai.answer_data_question(
            >>>     "What are the top 5 products by sales?", 
            >>>     df
            >>> )
            >>> 
            >>> st.write("**Answer:**", response['answer'])
            >>> if response['code']:
            >>>     st.code(response['code'], language='python')
            >>> st.info(response['insights'])
        
        Note:
            - GPT-4 interprets ambiguous questions intelligently
            - "top 5" in numeric columns → 5 highest values
            - "top 5" in text columns → 5 most frequent values
            - Returns executable pandas code when possible
            - Max 1500 tokens for detailed answers
        """
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
        You are a data analyst. The user asks: "{question}"
        
        Dataset information:
        {data_summary}
        
        {context}
        
        IMPORTANT: Answer the question DIRECTLY without over-explaining. If they ask for "top 5 values" in a numeric column, show the 5 highest values. If they ask for "top 5" in a text column, show the 5 most frequent values. Don't explain ambiguity - just pick the most logical interpretation and answer.
        
        Provide:
        1. A clear, direct answer (no run-around explanations)
        2. Working Python pandas code to get this answer
        3. Brief business insight (1-2 sentences max)
        
        Format your response as JSON with keys: "answer", "code", "insights"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": ("You are an expert data analyst. Answer questions directly "
                                  "and concisely. No over-explaining or philosophy - just give "
                                  "the user what they asked for.")
                    },
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
    
    def generate_cleaning_suggestions(
        self, 
        df: pd.DataFrame, 
        issues: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate AI-powered data cleaning suggestions for quality issues.
        
        Analyzes detected data quality issues and generates specific, actionable
        cleaning suggestions with executable Python code.
        
        Args:
            df: DataFrame with quality issues
            issues: List of issue dictionaries from DataProcessor.detect_data_quality_issues()
                   Each should have: type, column, severity, description
        
        Returns:
            List of suggestion dictionaries, each containing:
                - issue (str): Problem description
                - suggestion (str): Recommended action
                - reason (str): Why this matters
                - code (str): Python pandas code to implement fix
        
        Example:
            >>> issues = DataProcessor.detect_data_quality_issues(df)
            >>> ai = AIHelper()
            >>> suggestions = ai.generate_cleaning_suggestions(df, issues)
            >>> 
            >>> for i, sug in enumerate(suggestions, 1):
            >>>     st.write(f"**{i}. {sug['issue']}**")
            >>>     st.write(f"Action: {sug['suggestion']}")
            >>>     st.write(f"Reason: {sug['reason']}")
            >>>     st.code(sug['code'], language='python')
        
        Note:
            - Uses GPT-4 for intelligent suggestions
            - Provides executable pandas code
            - Explains business rationale for each suggestion
            - Max 2000 tokens for comprehensive guidance
            - Handles API errors gracefully with fallback responses
        """
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
                    {
                        "role": "system", 
                        "content": "You are a data cleaning expert. Provide practical, executable solutions."
                    },
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
