import openai
import os
import pandas as pd
import json
from typing import Dict, Any, List

class AIHelper:
    """Handles all AI-related operations using OpenAI."""
    
    def __init__(self):
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
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets or .env file")
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
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
