# DataInsight AI 🎯

An AI-powered business intelligence assistant that helps you analyze data, generate insights, and create professional reports using natural language.

## Features

- 📤 **Data Upload**: Upload CSV/Excel files with automatic profiling
- 📊 **Data Analysis**: Comprehensive statistical analysis and quality checks
- 🤖 **AI Insights**: Natural language querying and automated insights
- 📈 **Visualizations**: Interactive charts (histograms, scatter, box plots, heatmaps)
- 📄 **Professional Reports**: Generate and export business-ready reports
- 📥 **Export Options**: Download data, reports, and analysis in multiple formats
- 🧺 **Market Basket Analysis**: Apriori algorithm for discovering item associations

## Market Basket Analysis Module 🧺

DataInsight AI now includes a comprehensive **Market Basket Analysis** module for discovering patterns in transactional data.

### Features:
- 🧺 Apriori algorithm implementation
- 📊 Interactive threshold controls (support, confidence, lift)
- 📈 Multiple visualizations (scatter plot, network graph, top items)
- 💡 AI-generated business insights
- 📥 Export rules and comprehensive reports
- 🔍 Search and filter functionality
- 📚 Built-in educational guide

### Quick Start:
1. Navigate to "Market Basket Analysis" page
2. Load sample groceries data (9,835 transactions) or upload your own
3. Adjust thresholds (support, confidence, lift)
4. Click "Run Market Basket Analysis"
5. Explore interactive visualizations and insights
6. Download rules and reports

See [MBA_GUIDE.md](MBA_GUIDE.md) for detailed instructions and business applications.

## Live Demo

🚀 [Try DataInsight AI](https://your-app-name.streamlit.app) _(Update after deployment)_

## Local Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/datainsight-ai.git
cd datainsight-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

## Usage Guide

### 1. Upload Data
- Navigate to **Data Upload** page
- Upload CSV or Excel file (or try sample data)
- View automatic data profiling and quality checks

### 2. Analyze Data
- Go to **Analysis** tab
- View statistical summaries
- Generate AI-powered insights
- Get cleaning suggestions with code

### 3. Ask Questions
- Navigate to **Insights** page
- Ask questions in natural language
- View AI-generated answers and code
- Execute code to see results

### 4. Create Visualizations
- In **Analysis** → **Visualizations** tab
- Browse suggested visualizations
- Create custom charts

### 5. Generate Reports
- Go to **Reports** page
- Configure report options
- Generate and download professional reports

### 6. Export Results
- Use sidebar **Quick Export** section
- Download data in CSV, Excel, or JSON
- Export data dictionary and analysis summary

## Deployment to Streamlit Cloud

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

**Quick Steps:**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add OpenAI API key to secrets
5. Deploy!

## Technologies

- **Framework:** Streamlit 1.31.0
- **AI:** OpenAI GPT-4
- **Data Processing:** pandas, numpy
- **Visualizations:** Plotly
- **Deployment:** Streamlit Cloud

## Project Structure

```
datainsight-ai/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── DEPLOYMENT_GUIDE.md        # Deployment instructions
├── TESTING_CHECKLIST.md       # Testing checklist
├── BUSINESS_REPORT_TEMPLATE.md # Report template
├── utils/
│   ├── __init__.py
│   ├── data_processor.py      # Data processing utilities
│   ├── ai_helper.py           # AI integration
│   ├── visualizations.py      # Chart generation
│   ├── report_generator.py    # Report creation
│   └── export_helper.py       # Export utilities
├── assets/
│   └── style.css              # Custom styling
└── .streamlit/
    ├── config.toml            # Streamlit configuration
    └── secrets.toml.example   # Secrets template
```

## Features Breakdown

### Data Upload & Processing
- Supports CSV and Excel formats
- Automatic data type detection
- Comprehensive profiling (rows, columns, memory usage)
- Missing value analysis
- Duplicate detection
- Data quality issue identification

### AI-Powered Analysis
- Natural language question answering
- Automated insight generation
- Context-aware data cleaning suggestions
- Executable Python code generation
- Chat history with code execution

### Interactive Visualizations
- Histograms for distributions
- Bar charts for categorical data
- Scatter plots for relationships
- Box plots for outliers
- Correlation heatmaps
- Custom visualization builder

### Professional Reports
- Executive summaries
- Data profile sections
- Quality assessments
- Actionable recommendations
- Downloadable formats (Markdown, Text)

### Export Capabilities
- CSV, Excel, JSON formats
- Data dictionaries
- Analysis summaries
- Report downloads

## Testing

Run through the [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) to verify all features.

## Business Report

Use [BUSINESS_REPORT_TEMPLATE.md](BUSINESS_REPORT_TEMPLATE.md) for your final project submission.

## Troubleshooting

**Error: OpenAI API key not found**
- Ensure `.env` file exists with `OPENAI_API_KEY=your-key`
- For Streamlit Cloud, add key in Secrets section

**Error: Module not found**
- Run `pip install -r requirements.txt`
- Ensure using Python 3.8+

**App runs slowly**
- Consider smaller datasets for testing
- Check internet connection for AI features
- Optimize large datasets before upload

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for your own purposes.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Powered by [OpenAI GPT-4](https://openai.com)
- Data processing with [pandas](https://pandas.pydata.org)
- Visualizations with [Plotly](https://plotly.com)

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Made with ❤️ for Data Mining Capstone Project**
