# Advanced Report Exports with Visualizations

## Overview

DataInsight AI now supports **three advanced export formats** that include visualizations from your analyses:

1. **🌐 HTML with Interactive Charts** - Web page with embedded Plotly charts
2. **📦 Markdown + Images (ZIP)** - Markdown report with PNG chart images  
3. **📄 PDF Report** - Print-ready PDF document with charts

---

## Export Formats

### 1. HTML with Interactive Charts ⭐ RECOMMENDED

**What you get:**
- Single self-contained HTML file
- Interactive Plotly charts (zoom, hover, pan)
- Professional styling with print support
- Works in any web browser

**Use cases:**
- Presentations and demos
- Sharing with stakeholders
- Interactive data exploration
- Web publishing

**How to use:**
1. Go to **Reports** section
2. Generate your comprehensive report
3. Click **"🌐 HTML with Charts"**
4. Download and open in browser

**No additional libraries required!** ✅

---

### 2. Markdown + Images (ZIP)

**What you get:**
- ZIP package containing:
  - `report.md` - Full markdown report with image references
  - `images/` - Folder with PNG chart images
  - `README.txt` - Instructions

**Use cases:**
- GitHub/GitLab documentation
- Version control systems
- Markdown editors (VS Code, Typora)
- Documentation websites

**How to use:**
1. Go to **Reports** section
2. Generate your comprehensive report
3. Click **"📦 Markdown + Images"**
4. Extract ZIP and view in markdown viewer

**Optional library for image generation:**
```bash
pip install kaleido
```

Without kaleido, you'll get text-only markdown. The app will notify you if it's missing.

---

### 3. PDF Report

**What you get:**
- Print-ready PDF document
- Static chart images embedded
- Professional page layout
- Compatible with all PDF readers

**Use cases:**
- Formal reports
- Printing and archiving
- Email attachments
- Corporate presentations

**How to use:**
1. Go to **Reports** section
2. Generate your comprehensive report  
3. Click **"📄 PDF Report"**
4. Download PDF file

**Required libraries** (choose one):
```bash
# Option 1: weasyprint (recommended - better formatting)
pip install weasyprint

# Option 2: reportlab (simpler, fewer dependencies)
pip install reportlab
```

The app will detect which library is available and use it automatically.

---

## Charts Included in Exports

The advanced exports automatically detect and include charts from completed analyses:

| Analysis Module | Charts Included |
|----------------|-----------------|
| **ML Classification** | Model comparison (accuracy, precision, recall, F1) |
| **ML Regression** | Model comparison (R², RMSE, MAE) |
| **Market Basket Analysis** | Association rules scatter plot (support vs confidence) |
| **RFM Analysis** | Customer segmentation distribution |
| **Anomaly Detection** | Anomaly vs normal distribution |
| **Time Series** | Forecast plots (coming soon) |
| **Monte Carlo** | Simulation results (coming soon) |

The number of available charts is displayed at the bottom of the export section.

---

## Installation Guide

### For HTML Export (No Installation Needed)
HTML exports work out-of-the-box! Just click and download.

### For Markdown + Images
```bash
pip install kaleido
```

### For PDF Export

**Windows:**
```bash
# Option 1: weasyprint (recommended)
pip install weasyprint

# Option 2: reportlab (simpler)
pip install reportlab
```

**Mac/Linux:**
```bash
# weasyprint
pip install weasyprint

# reportlab
pip install reportlab
```

---

## Troubleshooting

### "Kaleido not installed" warning
**Solution:** Install kaleido to enable PNG image generation:
```bash
pip install kaleido
```
Without it, Markdown exports will still work but won't include chart images.

### "PDF generation requires additional libraries"
**Solution:** Install one of the PDF libraries:
```bash
pip install weasyprint  # recommended
# OR
pip install reportlab
```

### HTML export works but charts don't appear
**Possible causes:**
- No analyses have been run yet (no charts to export)
- Browser JavaScript is disabled
- Internet connection required for Plotly CDN (first time only)

**Solution:** 
- Run some analyses first to generate charts
- Enable JavaScript in browser
- For offline use, download the HTML file once while online

### Charts look different in PDF vs HTML
**Expected behavior:** 
- HTML charts are **interactive** (zoom, hover, pan)
- PDF charts are **static images** (print-friendly)
- This is by design for different use cases

---

## Best Practices

### For Presentations
✅ Use **HTML with Charts** for interactive demos  
✅ Use **PDF Report** for printed handouts

### For Documentation
✅ Use **Markdown + Images** for GitHub/GitLab  
✅ Commit both the `.md` file and `images/` folder

### For Archiving
✅ Use **PDF Report** for long-term storage  
✅ PDFs are more stable and readable across systems

### For Sharing
✅ Use **HTML with Charts** for technical audiences  
✅ Use **PDF Report** for executive summaries

---

## Technical Details

### HTML Export
- Uses Plotly.js CDN for chart rendering
- Self-contained HTML5 document
- Responsive design (works on mobile)
- Print styles included (Ctrl+P to print)

### Markdown + Images Export
- Charts converted to PNG (1200x600px default)
- Markdown uses relative paths (`![Chart](images/chart_1.png)`)
- ZIP structure: `report.md`, `images/`, `README.txt`

### PDF Export
- **weasyprint**: HTML → PDF (better formatting, CSS support)
- **reportlab**: Direct PDF generation (simpler, fewer dependencies)
- Charts rendered as static PNGs at 800x400px

---

## API Reference

For developers integrating the export functionality:

```python
from utils.advanced_report_exporter import AdvancedReportExporter
import plotly.graph_objects as go

# Create HTML report
html = AdvancedReportExporter.create_html_report(
    markdown_content="# My Report\nContent here...",
    charts=[("Chart Title", plotly_figure)],
    title="My Analysis Report"
)

# Create Markdown + Images ZIP
zip_data = AdvancedReportExporter.create_markdown_with_images_zip(
    markdown_content="# My Report\nContent here...",
    charts=[("Chart Title", plotly_figure)],
    filename_prefix="report"
)

# Create PDF report
pdf_data = AdvancedReportExporter.create_pdf_report(
    markdown_content="# My Report\nContent here...",
    charts=[("Chart Title", plotly_figure)],
    title="My Analysis Report"
)
```

---

## Changelog

### Version 2.0 (October 22, 2025)
- ✨ Added HTML export with interactive Plotly charts
- ✨ Added Markdown + Images ZIP export
- ✨ Added PDF report generation
- ✨ Auto-detection of available charts from analyses
- ✨ Graceful fallbacks when optional libraries missing
- 📝 Added comprehensive documentation

### Version 1.0 (Previous)
- Basic Markdown and Text exports only
- No visualization support

---

## Support

For issues or feature requests, please contact the development team or check the main README.md file.

**Tip:** Start with the HTML export - it works immediately without any additional setup!
