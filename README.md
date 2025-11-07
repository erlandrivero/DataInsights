# DataInsights 🎯

**A Comprehensive AI-Powered Data Mining & Business Intelligence Platform**

DataInsights is an all-in-one data analytics platform that combines 19 specialized modules for data mining, machine learning, and business intelligence. Built with Streamlit and powered by AI, it provides enterprise-grade analytics capabilities with an intuitive interface.

---

## 🌟 Key Features

### Core Capabilities
- 📤 **Smart Data Upload**: CSV/Excel support with automatic profiling
- 📊 **Advanced Analysis**: Statistical analysis, cleaning, and quality checks
- 🤖 **AI-Powered Insights**: Natural language querying with GPT-4
- 📈 **Interactive Visualizations**: 10+ chart types with Plotly
- 📄 **Professional Reports**: Business-ready documentation
- 📥 **Flexible Exports**: CSV, Excel, JSON, and more

### 19 Specialized Modules

#### Data Mining & Pattern Discovery
- 🧺 **Market Basket Analysis**: Apriori algorithm for association rules
- 👥 **RFM Analysis**: Customer segmentation with K-Means
- 🕸️ **Network Analysis**: Relationship and connection analysis
- 👥 **Cohort Analysis**: User behavior tracking over time

#### Machine Learning
- 🤖 **ML Classification**: 15 algorithms with SHAP interpretability
- 📈 **ML Regression**: 15 algorithms for continuous prediction
- 🔄 **Churn Prediction**: Specialized customer retention models
- 🎯 **Recommendation Systems**: Collaborative & content-based filtering

#### Advanced Analytics
- 🔍 **Anomaly Detection**: Isolation Forest, LOF, One-Class SVM
- 📈 **Time Series Forecasting**: ARIMA & Prophet
- 💬 **Text Mining & NLP**: Sentiment analysis, topic modeling, NER
- 🧪 **A/B Testing**: Statistical significance testing
- ⏱️ **Survival Analysis**: Time-to-event modeling
- 🗺️ **Geospatial Analysis**: Location-based insights
- 🎲 **Monte Carlo Simulation**: Risk analysis and forecasting

---

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user manual (150+ pages)
- **[MBA_GUIDE.md](MBA_GUIDE.md)** - Market Basket Analysis guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Deployment instructions
- **[TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)** - Feature verification
- **[BUSINESS_REPORT_TEMPLATE.md](BUSINESS_REPORT_TEMPLATE.md)** - Report template

---

## 🚀 Quick Start

### Option 1: Use Sample Data
1. Launch the application
2. Navigate to any module
3. Click "Load Sample Data"
4. Explore features immediately

### Option 2: Upload Your Data
1. Go to **📤 Data Upload**
2. Upload CSV or Excel file
3. Review automatic data profiling
4. Navigate to any analysis module

---

## 💻 Local Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Or Google AI API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/erlandrivero/DataInsights.git
cd DataInsights
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API Key**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-key-here
# OR
# GOOGLE_API_KEY=your-google-key-here
```

4. **Run Application**
```bash
streamlit run app.py
```

5. **Access Application**
- Open browser to `http://localhost:8501`
- Start analyzing data!

---

## 🎯 Featured Modules

### Market Basket Analysis 🧺
Discover purchasing patterns and product associations using the Apriori algorithm.

**Use Cases:**
- Product placement optimization
- Cross-selling strategies
- Bundle recommendations
- Inventory management

**See:** [MBA_GUIDE.md](MBA_GUIDE.md)

### ML Classification 🤖
Train and compare 15 machine learning algorithms with comprehensive evaluation.

**Features:**
- AI-powered model recommendations
- SHAP interpretability
- Cross-validation
- Hyperparameter tuning
- Export predictions

**Algorithms:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, and more

### Time Series Forecasting 📈
Predict future values using ARIMA and Prophet algorithms.

**Applications:**
- Sales forecasting
- Demand planning
- Inventory optimization
- Resource allocation

### Text Mining & NLP 💬
Extract insights from text data with sentiment analysis and topic modeling.

**Features:**
- Sentiment classification
- Named Entity Recognition
- Topic modeling (LDA)
- Word clouds
- N-gram analysis

---

## 🌐 Live Demo

🚀 **[Try DataInsights on Streamlit Cloud](https://datainsights.streamlit.app)**

Experience all features with sample datasets - no installation required!

---

## 📖 Complete Module List

| Module | Purpose | Key Features |
|--------|---------|--------------|
| 📤 Data Upload | Import & profile data | CSV/Excel support, auto-profiling |
| 📊 Analysis & Cleaning | Explore & clean data | Statistics, visualizations, cleaning tools |
| 🤖 AI Insights | Natural language Q&A | GPT-4 powered, code generation |
| 📄 Reports | Generate documentation | Professional reports, multiple formats |
| 🧺 Market Basket Analysis | Association rules | Apriori, network graphs, AI insights |
| 👥 RFM Analysis | Customer segmentation | RFM scoring, K-Means clustering |
| 🎲 Monte Carlo | Risk analysis | Financial forecasting, simulations |
| 🤖 ML Classification | Predict categories | 15 algorithms, SHAP, cross-validation |
| 📈 ML Regression | Predict numbers | 15 algorithms, SHAP, residual analysis |
| 🔍 Anomaly Detection | Find outliers | 3 algorithms, visualization |
| 📈 Time Series | Forecast future values | ARIMA, Prophet, confidence intervals |
| 💬 Text Mining | Analyze text | Sentiment, NER, topic modeling |
| 🧪 A/B Testing | Statistical testing | Significance tests, effect size |
| 👥 Cohort Analysis | User behavior | Retention, heatmaps, trends |
| 🎯 Recommendations | Personalization | Collaborative, content-based, hybrid |
| 🗺️ Geospatial | Location analysis | Interactive maps, clustering |
| ⏱️ Survival Analysis | Time-to-event | Kaplan-Meier, Cox models |
| 🕸️ Network Analysis | Relationships | Centrality, communities, paths |
| 🔄 Churn Prediction | Customer retention | Specialized models, risk scoring |

---

## 🎓 For Students & Educators

DataInsights is perfect for:
- **Data Mining Courses**: Hands-on experience with 19 algorithms
- **Capstone Projects**: Enterprise-grade analytics platform
- **Research**: Comprehensive analysis and visualization tools
- **Learning**: Built-in guides and AI-powered explanations

**Academic Features:**
- Sample datasets included
- Educational guides
- Step-by-step workflows
- Export results for reports
- Professional documentation

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

**Quick Steps:**
1. Fork/clone this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add API key to Secrets:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   # OR
   GOOGLE_API_KEY = "your-google-key-here"
   ```
5. Deploy!

### Local Development

Perfect for customization and testing:
```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API key

# Run
streamlit run app.py
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit 1.31.0 |
| **AI/ML** | OpenAI GPT-4, Google Gemini, scikit-learn |
| **Data Processing** | pandas, numpy, scipy |
| **Visualizations** | Plotly, matplotlib, seaborn |
| **ML Libraries** | XGBoost, LightGBM, CatBoost, Prophet |
| **NLP** | NLTK, spaCy, TextBlob |
| **Deployment** | Streamlit Cloud, Docker-ready |

---

## 📁 Project Structure

```
DataInsights/
├── app.py                          # Main application (20,000+ lines)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── USER_GUIDE.md                   # Complete user manual
├── MBA_GUIDE.md                    # Market Basket Analysis guide
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
├── TESTING_CHECKLIST.md            # Feature verification
├── BUSINESS_REPORT_TEMPLATE.md     # Report template
├── QUICK_SETUP_GOOGLE_AI.md        # Google AI setup
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── ai_helper.py               # AI integration
│   ├── anomaly_detection.py       # Anomaly detection algorithms
│   ├── churn_prediction.py        # Churn prediction models
│   ├── data_processor.py          # Data processing utilities
│   ├── export_helper.py           # Export functionality
│   ├── market_basket.py           # MBA implementation
│   ├── monte_carlo.py             # Monte Carlo simulations
│   ├── rfm_analysis.py            # RFM analysis
│   ├── text_mining.py             # NLP utilities
│   ├── time_series.py             # Time series forecasting
│   ├── visualizations.py          # Chart generation
│   └── report_generator.py        # Report creation
├── assets/
│   └── style.css                  # Custom styling
├── .streamlit/
│   ├── config.toml                # Streamlit configuration
│   └── secrets.toml.example       # Secrets template
└── tests/                         # Unit tests
    └── README.md                  # Testing documentation
```

---

## ✨ Key Highlights

### Enterprise-Grade Features
- ✅ 19 specialized analytics modules
- ✅ 45+ machine learning algorithms
- ✅ AI-powered insights and recommendations
- ✅ SHAP interpretability for ML models
- ✅ Comprehensive data quality checks
- ✅ Professional report generation
- ✅ Multiple export formats

### User Experience
- 🎨 Modern, intuitive interface
- 📱 Responsive design
- 🚀 Fast performance with caching
- 💾 Session state management
- 📊 Interactive visualizations
- 🔍 Built-in search and filtering

### Educational Value
- 📚 Comprehensive documentation
- 🎓 Sample datasets included
- 💡 AI-powered explanations
- 📖 Step-by-step guides
- 🧪 Experiment-friendly environment

---

## 📊 Use Cases

### Business Analytics
- Customer segmentation and profiling
- Churn prediction and retention
- Market basket analysis
- Sales forecasting
- Risk assessment

### Data Science
- Exploratory data analysis
- Feature engineering
- Model comparison and selection
- Hyperparameter tuning
- Model interpretability

### Academic Research
- Data mining projects
- Machine learning experiments
- Statistical analysis
- Visualization creation
- Report generation

---

## 🧪 Testing & Quality

**Comprehensive Testing:**
- Unit tests for core functionality
- Integration tests for modules
- Performance benchmarks
- User acceptance testing

**See:** [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)

---

## 📝 Documentation

| Document | Description |
|----------|-------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Complete 150+ page user manual |
| [MBA_GUIDE.md](MBA_GUIDE.md) | Market Basket Analysis guide |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Deployment instructions |
| [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) | Feature verification |
| [BUSINESS_REPORT_TEMPLATE.md](BUSINESS_REPORT_TEMPLATE.md) | Report template |
| [QUICK_SETUP_GOOGLE_AI.md](QUICK_SETUP_GOOGLE_AI.md) | Google AI setup guide |

---

## ❓ Troubleshooting

### Common Issues

**API Key Not Found**
```
Error: OpenAI API key not found
```
- **Solution:** Ensure `.env` file exists with `OPENAI_API_KEY=your-key`
- **Streamlit Cloud:** Add key in Secrets section

**Module Not Found**
```
Error: ModuleNotFoundError
```
- **Solution:** Run `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

**Slow Performance**
- Use smaller datasets for testing
- Check internet connection for AI features
- Clear browser cache
- Restart Streamlit server

**Memory Issues**
- Reduce dataset size
- Close other applications
- Use sampling for large datasets
- Clear session state

**For More Help:**
- See [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive troubleshooting section
- Check [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Feature verification
- Review error messages carefully
- Try sample datasets first

---

## 🤝 Contributing

Contributions are welcome! This project is open for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- New module additions
- Performance optimizations

**How to Contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a Pull Request

---

## 📜 License

**MIT License** - Free to use for personal, academic, and commercial purposes.

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Built With:**
- [Streamlit](https://streamlit.io) - Web framework
- [OpenAI GPT-4](https://openai.com) - AI capabilities
- [Google Gemini](https://ai.google.dev/) - Alternative AI provider
- [pandas](https://pandas.pydata.org) - Data processing
- [Plotly](https://plotly.com) - Interactive visualizations
- [scikit-learn](https://scikit-learn.org) - Machine learning
- [XGBoost](https://xgboost.ai), [LightGBM](https://lightgbm.readthedocs.io), [CatBoost](https://catboost.ai) - Gradient boosting
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting
- [SHAP](https://shap.readthedocs.io) - Model interpretability

**Special Thanks:**
- Streamlit team for the amazing framework
- Open-source community for excellent libraries
- Data mining community for inspiration

---

## 📧 Contact & Support

**Project Repository:**
- GitHub: [erlandrivero/DataInsights](https://github.com/erlandrivero/DataInsights)
- Issues: [Report a bug](https://github.com/erlandrivero/DataInsights/issues)

**For Questions:**
- Check [USER_GUIDE.md](USER_GUIDE.md) first
- Review [Troubleshooting](#-troubleshooting) section
- Open a GitHub issue
- Contact: erlandrivero@example.com

---

## 🎓 Academic Information

**Course:** CAP 4767 - Data Mining  
**Project:** Capstone Project - Data Mining Platform  
**Institution:** Florida International University  
**Year:** 2024

**Project Highlights:**
- ✅ 19 specialized analytics modules
- ✅ 45+ machine learning algorithms  
- ✅ 20,000+ lines of code
- ✅ Comprehensive documentation
- ✅ Enterprise-grade features
- ✅ Production-ready deployment

---

<div align="center">

## ⭐ Star this repository if you find it helpful!

**Made with ❤️ for Data Mining Capstone Project**

[Documentation](USER_GUIDE.md) • [Live Demo](https://datainsights.streamlit.app) • [Report Issues](https://github.com/erlandrivero/DataInsights/issues)

</div>
