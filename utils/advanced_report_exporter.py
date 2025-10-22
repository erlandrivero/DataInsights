"""
Advanced Report Exporter
Handles multiple export formats: HTML with embedded charts, Markdown+Images, and PDF
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import base64
import io
import zipfile
import os
import tempfile


class AdvancedReportExporter:
    """Handles advanced report exports with visualizations."""
    
    @staticmethod
    def plotly_to_html_div(fig: go.Figure, include_plotlyjs: str = 'cdn') -> str:
        """
        Convert Plotly figure to HTML div string.
        
        Args:
            fig: Plotly figure object
            include_plotlyjs: 'cdn', 'inline', or False
            
        Returns:
            HTML string of the figure
        """
        if fig is None:
            return ""
        return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False, div_id=None)
    
    @staticmethod
    def plotly_to_png_base64(fig: go.Figure, width: int = 1200, height: int = 600) -> str:
        """
        Convert Plotly figure to base64-encoded PNG.
        
        Args:
            fig: Plotly figure object
            width: Image width
            height: Image height
            
        Returns:
            Base64-encoded PNG string
        """
        if fig is None:
            return ""
        
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            # Fallback if kaleido not available
            return ""
    
    @staticmethod
    def plotly_to_png_bytes(fig: go.Figure, width: int = 1200, height: int = 600) -> bytes:
        """
        Convert Plotly figure to PNG bytes.
        
        Args:
            fig: Plotly figure object
            width: Image width
            height: Image height
            
        Returns:
            PNG bytes
        """
        if fig is None:
            return b""
        
        try:
            return fig.to_image(format="png", width=width, height=height, engine="kaleido")
        except Exception as e:
            # Fallback if kaleido not available
            return b""
    
    @staticmethod
    def create_html_report(
        markdown_content: str,
        charts: List[Tuple[str, go.Figure]] = None,
        title: str = "Data Analysis Report"
    ) -> str:
        """
        Create HTML report with embedded interactive Plotly charts.
        
        Args:
            markdown_content: Markdown text content
            charts: List of (chart_title, plotly_figure) tuples
            title: Report title
            
        Returns:
            Complete HTML string
        """
        # Convert markdown to HTML (simple conversion)
        html_content = AdvancedReportExporter._markdown_to_html(markdown_content)
        
        # Build chart HTML sections
        chart_sections = ""
        if charts:
            chart_sections = "<div class='charts-section'>"
            for idx, (chart_title, fig) in enumerate(charts):
                if fig is not None:
                    chart_html = AdvancedReportExporter.plotly_to_html_div(
                        fig, 
                        include_plotlyjs='cdn' if idx == 0 else False
                    )
                    chart_sections += f"""
                    <div class='chart-container'>
                        <h3>{chart_title}</h3>
                        {chart_html}
                    </div>
                    """
            chart_sections += "</div>"
        
        # Complete HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c5282;
            margin-top: 30px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #4a5568;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }}
        th {{
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            border: 1px solid #e2e8f0;
            padding: 10px;
        }}
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f9fafb;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .chart-container h3 {{
            margin-top: 0;
            color: #1f77b4;
        }}
        code {{
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #1e293b;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            color: #e2e8f0;
        }}
        .metadata {{
            background-color: #f0f9ff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 20px 0;
        }}
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .print-button:hover {{
            background-color: #1565c0;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .print-button {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">🖨️ Print / Save as PDF</button>
    <div class="container">
        <div class="metadata">
            <strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br>
            <strong>Format:</strong> Interactive HTML Report
        </div>
        
        {html_content}
        
        {chart_sections}
        
        <hr style="margin-top: 40px;">
        <p style="text-align: center; color: #718096; font-size: 14px;">
            Generated by DataInsight AI • {datetime.now().strftime('%Y')}
        </p>
    </div>
</body>
</html>
"""
        return html_template
    
    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown_text
        
        # Headers
        html = html.replace('\n# ', '\n<h1>').replace('\n## ', '\n<h2>').replace('\n### ', '\n<h3>')
        html = html.replace('</h1>', '</h1>\n').replace('</h2>', '</h2>\n').replace('</h3>', '</h3>\n')
        
        # Bold
        import re
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        
        # Lists
        html = re.sub(r'\n- (.+)', r'\n<li>\1</li>', html)
        html = re.sub(r'(<li>.*</li>\n)+', r'<ul>\n\g<0></ul>\n', html)
        
        # Code blocks
        html = re.sub(r'```python\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        
        # Paragraphs
        html = re.sub(r'\n\n', '</p>\n<p>', html)
        html = f'<p>{html}</p>'
        
        # Tables (already in markdown format)
        lines = html.split('\n')
        in_table = False
        html_lines = []
        for line in lines:
            if '|' in line and not in_table:
                html_lines.append('<table>')
                in_table = True
            if in_table and '|' not in line:
                html_lines.append('</table>')
                in_table = False
            
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if all(c.replace('-', '').strip() == '' for c in cells):
                    continue  # Skip separator row
                if '---' not in line:
                    if 'Column' in line or 'Feature' in line or 'Model' in line:
                        html_lines.append('<tr>' + ''.join(f'<th>{cell}</th>' for cell in cells) + '</tr>')
                    else:
                        html_lines.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
            else:
                html_lines.append(line)
        
        if in_table:
            html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    @staticmethod
    def create_markdown_with_images_zip(
        markdown_content: str,
        charts: List[Tuple[str, go.Figure]] = None,
        filename_prefix: str = "report"
    ) -> bytes:
        """
        Create ZIP file containing markdown report and PNG images.
        
        Args:
            markdown_content: Markdown text
            charts: List of (chart_title, plotly_figure) tuples
            filename_prefix: Prefix for filenames
            
        Returns:
            ZIP file bytes
        """
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add markdown content with image references
            enhanced_markdown = markdown_content
            
            if charts:
                enhanced_markdown += "\n\n---\n\n## Visualizations\n\n"
                
                for idx, (chart_title, fig) in enumerate(charts):
                    if fig is not None:
                        # Generate image filename
                        img_filename = f"chart_{idx+1}_{chart_title.replace(' ', '_').replace('/', '_')[:30]}.png"
                        
                        # Add reference in markdown
                        enhanced_markdown += f"\n### {chart_title}\n\n"
                        enhanced_markdown += f"![{chart_title}]({img_filename})\n\n"
                        
                        # Convert chart to PNG and add to ZIP
                        try:
                            img_bytes = AdvancedReportExporter.plotly_to_png_bytes(fig)
                            if img_bytes:
                                zip_file.writestr(f"images/{img_filename}", img_bytes)
                        except Exception as e:
                            enhanced_markdown += f"*Chart generation failed: {str(e)}*\n\n"
            
            # Add markdown file
            zip_file.writestr(f"{filename_prefix}.md", enhanced_markdown)
            
            # Add README
            readme = """# DataInsight AI Report Package

This package contains:
- `report.md` - Full markdown report
- `images/` - Folder with chart images

## How to View
1. Open `report.md` in any markdown viewer (VS Code, Typora, GitHub, etc.)
2. Images are referenced from the `images/` folder

Generated by DataInsight AI
"""
            zip_file.writestr("README.txt", readme)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    @staticmethod
    def create_pdf_report(
        markdown_content: str,
        charts: List[Tuple[str, go.Figure]] = None,
        title: str = "Data Analysis Report"
    ) -> bytes:
        """
        Create PDF report with embedded charts.
        
        Args:
            markdown_content: Markdown text
            charts: List of (chart_title, plotly_figure) tuples
            title: Report title
            
        Returns:
            PDF bytes
        """
        try:
            # Try using weasyprint (requires installation)
            from weasyprint import HTML, CSS
            
            # Create HTML first
            html_content = AdvancedReportExporter.create_html_report(
                markdown_content, charts, title
            )
            
            # Convert to PDF
            pdf_bytes = HTML(string=html_content).write_pdf()
            return pdf_bytes
            
        except ImportError:
            # Fallback: Create simple PDF with reportlab
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#1f77b4'),
                    spaceAfter=30
                )
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Metadata
                meta_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
                story.append(Paragraph(meta_text, styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
                
                # Content (simple text conversion)
                for line in markdown_content.split('\n'):
                    if line.startswith('# '):
                        story.append(Paragraph(line[2:], styles['Heading1']))
                    elif line.startswith('## '):
                        story.append(Paragraph(line[3:], styles['Heading2']))
                    elif line.startswith('### '):
                        story.append(Paragraph(line[4:], styles['Heading3']))
                    elif line.strip():
                        story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
                
                # Add charts
                if charts:
                    story.append(PageBreak())
                    story.append(Paragraph("Visualizations", styles['Heading1']))
                    story.append(Spacer(1, 0.3*inch))
                    
                    for chart_title, fig in charts:
                        if fig is not None:
                            story.append(Paragraph(chart_title, styles['Heading3']))
                            try:
                                img_bytes = AdvancedReportExporter.plotly_to_png_bytes(fig, width=800, height=400)
                                if img_bytes:
                                    img = Image(io.BytesIO(img_bytes), width=6*inch, height=3*inch)
                                    story.append(img)
                            except:
                                story.append(Paragraph("Chart generation failed", styles['Normal']))
                            story.append(Spacer(1, 0.3*inch))
                
                # Build PDF
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
                
            except ImportError:
                # Neither library available - return error message as text
                error_msg = """PDF generation requires additional libraries.

Please install one of:
1. weasyprint: pip install weasyprint
2. reportlab: pip install reportlab

Use HTML or Markdown+Images export instead."""
                return error_msg.encode('utf-8')
