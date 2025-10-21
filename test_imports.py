"""Quick test to verify all imports work"""
import sys
print("Testing imports...")

try:
    # Test core imports
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from dotenv import load_dotenv
    print("✓ Core packages imported")
    
    # Test utility modules
    from utils.data_processor import DataProcessor
    from utils.ai_helper import AIHelper
    from utils.visualizations import Visualizer
    from utils.report_generator import ReportGenerator
    from utils.export_helper import ExportHelper
    print("✓ Utility modules imported")
    
    print("\n✅ All imports successful! No syntax errors detected.")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
