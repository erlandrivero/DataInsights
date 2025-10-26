"""Business report generation utilities for DataInsights with complete type hints.

This module provides comprehensive report generation capabilities for creating
professional business intelligence reports with data profiling, quality assessment,
and actionable recommendations.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from datetime import datetime


class ReportGenerator:
    """Handles professional business report generation.
    
    This class provides static methods for generating comprehensive reports
    including executive summaries, data profiles, quality assessments, and
    recommendations in markdown format.
    
    Attributes:
        None - All methods are static
    
    Example:
        >>> # Generate complete report
        >>> report = ReportGenerator.generate_full_report(
        >>>     df, profile, issues, insights, suggestions
        >>> )
        >>> st.download_button("Download", report.encode(), "report.md")
    """
    
    @staticmethod
    def generate_executive_summary(
        df: pd.DataFrame, 
        profile: Dict[str, Any], 
        insights: str
    ) -> str:
        """Generate executive summary section of business report.
        
        Creates a high-level overview including dataset statistics, key findings,
        data quality metrics, and recommendations.
        
        Args:
            df: Source DataFrame being analyzed
            profile: Data profile dictionary from DataProcessor.profile_data()
            insights: AI-generated insights text from analysis
        
        Returns:
            Formatted markdown string containing executive summary
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> insights = "Key finding: Revenue increased 15% YoY..."
            >>> summary = ReportGenerator.generate_executive_summary(
            >>>     df, profile, insights
            >>> )
            >>> print(summary)
        
        Note:
            - Includes timestamp in readable format
            - Calculates data completeness automatically
            - Formats numbers with thousand separators
        """
        summary = f"""
## Executive Summary

**Report Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

### Dataset Overview
This report analyzes a dataset containing **{profile['basic_info']['rows']:,} records** across **{profile['basic_info']['columns']} variables**. 

### Key Findings
{insights}

### Data Quality Snapshot
- **Missing Data:** {profile['missing_data']['missing_percentage']} of values are missing
- **Duplicate Records:** {profile['basic_info']['duplicates']} duplicate rows identified
- **Data Completeness:** {100 - float(profile['missing_data']['missing_percentage'].rstrip('%')):.1f}% complete

*See detailed Data Quality Assessment and Recommendations sections below.*
"""
        return summary
    
    @staticmethod
    def generate_data_profile_section(profile: Dict[str, Any]) -> str:
        """Generate data profile section with column information table.
        
        Creates a detailed breakdown of all columns including data types,
        missing value counts, and unique value statistics.
        
        Args:
            profile: Data profile dictionary containing column_info list
        
        Returns:
            Formatted markdown string with column information table
        
        Example:
            >>> profile = DataProcessor.profile_data(df)
            >>> section = ReportGenerator.generate_data_profile_section(profile)
            >>> report += section
        
        Note:
            - Uses markdown table format
            - Shows all columns from profile['column_info']
            - Includes type, missing %, and unique values
        """
        section = """
## Data Profile

### Column Information

| Column | Type | Missing | Unique Values |
|--------|------|---------|---------------|
"""
        for col in profile['column_info']:
            section += f"| {col['name']} | {col['dtype']} | {col['missing_pct']} | {col['unique']} |\n"
        
        return section
    
    @staticmethod
    def generate_quality_issues_section(issues: List[Dict[str, Any]]) -> str:
        """Generate data quality issues section with severity indicators.
        
        Creates detailed documentation of all detected quality issues with
        color-coded severity levels (🔴 High, 🟡 Medium, 🟢 Low).
        
        Args:
            issues: List of issue dictionaries from DataProcessor.detect_data_quality_issues()
                   Each dict should have: type, column, severity, description
        
        Returns:
            Formatted markdown string documenting all quality issues
        
        Example:
            >>> issues = DataProcessor.detect_data_quality_issues(df)
            >>> section = ReportGenerator.generate_quality_issues_section(issues)
            >>> 
            >>> # For clean data
            >>> if not issues:
            >>>     section  # Returns "No significant issues detected"
        
        Note:
            - Returns positive message if no issues found
            - Groups issues by severity with emoji indicators
            - Includes column name and detailed description
        """
        if not issues:
            return """
## Data Quality Assessment

✅ **No significant data quality issues detected.**

The dataset appears to be well-maintained with:
- No excessive missing values
- No duplicate records
- Appropriate data types
- Reasonable value distributions
"""
        
        section = """
## Data Quality Assessment

The following data quality issues were identified:

"""
        for issue in issues:
            severity_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            section += f"""
### {severity_emoji[issue['severity']]} {issue['type']} - {issue['column']}
**Severity:** {issue['severity']}  
**Description:** {issue['description']}

"""
        return section
    
    @staticmethod
    def generate_recommendations_section(
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """Generate recommendations section with actionable code examples.
        
        Creates detailed recommendations for addressing data quality issues
        including rationale and implementation code in Python.
        
        Args:
            suggestions: List of recommendation dictionaries, each containing:
                - issue (str): Problem description
                - suggestion (str): Recommended action
                - reason (str): Rationale for recommendation
                - code (str): Python code to implement fix
        
        Returns:
            Formatted markdown string with numbered recommendations
        
        Example:
            >>> suggestions = [
            >>>     {
            >>>         'issue': 'High missing values in age',
            >>>         'suggestion': 'Impute with median',
            >>>         'reason': 'Preserves distribution',
            >>>         'code': 'df["age"].fillna(df["age"].median(), inplace=True)'
            >>>     }
            >>> ]
            >>> section = ReportGenerator.generate_recommendations_section(suggestions)
        
        Note:
            - Includes code blocks with syntax highlighting
            - Numbered for easy reference
            - Returns default recommendations if list is empty
        """
        if not suggestions:
            return """
## Recommendations

The dataset is in good condition. Consider:
1. Regular data quality monitoring
2. Documenting data collection processes
3. Implementing data validation rules
4. Creating automated quality checks
"""
        
        section = """
## Recommendations

Based on the analysis, we recommend the following actions:

"""
        for i, suggestion in enumerate(suggestions, 1):
            section += f"""
### {i}. {suggestion.get('issue', 'General')}

**Action:** {suggestion.get('suggestion', 'N/A')}

**Rationale:** {suggestion.get('reason', 'N/A')}

**Implementation:**
```python
{suggestion.get('code', '# No code provided')}
```

"""
        return section
    
    @staticmethod
    def generate_full_report(
        df: pd.DataFrame,
        profile: Dict[str, Any],
        issues: List[Dict[str, Any]],
        insights: str,
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """Generate complete business intelligence report.
        
        Combines all report sections into a comprehensive markdown document
        including executive summary, data profile, quality assessment, and
        recommendations.
        
        Args:
            df: Source DataFrame being analyzed
            profile: Data profile from DataProcessor.profile_data()
            issues: Quality issues from DataProcessor.detect_data_quality_issues()
            insights: AI-generated insights text
            suggestions: List of recommendation dictionaries
        
        Returns:
            Complete markdown report ready for download or display
        
        Example:
            >>> # Full workflow
            >>> df = DataProcessor.load_data(uploaded_file)
            >>> profile = DataProcessor.profile_data(df)
            >>> issues = DataProcessor.detect_data_quality_issues(df)
            >>> insights = AIHelper.generate_insights(df)
            >>> suggestions = [{...}]  # Your recommendations
            >>> 
            >>> report = ReportGenerator.generate_full_report(
            >>>     df, profile, issues, insights, suggestions
            >>> )
            >>> 
            >>> st.download_button(
            >>>     "Download Report",
            >>>     report.encode('utf-8'),
            >>>     "business_report.md",
            >>>     mime="text/markdown"
            >>> )
        
        Note:
            - Returns complete markdown document with sections
            - Includes header, footer, and metadata
            - Properly formatted for download
            - All sections separated by horizontal rules
        """
        report = f"""
# DataInsights - Business Intelligence Report

---

{ReportGenerator.generate_executive_summary(df, profile, insights)}

---

{ReportGenerator.generate_data_profile_section(profile)}

{ReportGenerator.generate_quality_issues_section(issues)}

---

{ReportGenerator.generate_recommendations_section(suggestions)}

---

## Conclusion

This analysis provides a comprehensive overview of the dataset's structure, quality, and potential areas for improvement. 
The recommendations outlined above should be prioritized based on business requirements and data usage scenarios.

For questions or additional analysis, please contact the data team.

---

**Report Generated by:** DataInsights  
**Date:** {datetime.now().strftime('%B %d, %Y')}  
**Version:** 1.0
"""
        return report
    
    @staticmethod
    def format_report_metadata(
        dataset_name: Optional[str] = None,
        analysis_type: Optional[str] = None,
        analyst_name: Optional[str] = None
    ) -> str:
        """Format additional metadata section for reports.
        
        Creates optional metadata header for reports including dataset name,
        analysis type, and analyst information.
        
        Args:
            dataset_name: Name of the dataset being analyzed
            analysis_type: Type of analysis performed (e.g., "Exploratory", "Quality Check")
            analyst_name: Name of person/team conducting analysis
        
        Returns:
            Formatted markdown metadata section
        
        Example:
            >>> metadata = ReportGenerator.format_report_metadata(
            >>>     dataset_name="Customer Transactions Q4 2024",
            >>>     analysis_type="Quality Assessment",
            >>>     analyst_name="Data Science Team"
            >>> )
            >>> report = metadata + main_report
        
        Note:
            - All parameters are optional
            - Only includes provided metadata
            - Returns empty string if no metadata provided
        """
        if not any([dataset_name, analysis_type, analyst_name]):
            return ""
        
        metadata = "\n## Report Metadata\n\n"
        
        if dataset_name:
            metadata += f"**Dataset:** {dataset_name}  \n"
        if analysis_type:
            metadata += f"**Analysis Type:** {analysis_type}  \n"
        if analyst_name:
            metadata += f"**Analyst:** {analyst_name}  \n"
        
        metadata += "\n---\n"
        return metadata
