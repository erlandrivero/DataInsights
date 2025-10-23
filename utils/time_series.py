"""Time Series Analysis and Forecasting Utilities.

This module provides comprehensive tools for time series analysis including decomposition,
stationarity testing, autocorrelation analysis, and forecasting with ARIMA and Prophet models.
Optimized for Streamlit Cloud with intelligent sampling and parameter constraints.

Typical usage example:
    analyzer = TimeSeriesAnalyzer(df)
    ts_data = analyzer.set_time_column('Date', 'Sales')
    components = analyzer.decompose_time_series(model='additive')
    arima_results = analyzer.run_auto_arima(forecast_periods=30)
    fig = analyzer.create_forecast_plot('arima')

Classes:
    TimeSeriesAnalyzer: Main class for time series operations.

Notes:
    - ARIMA uses pmdarima (unavailable on Python 3.13)
    - Prophet requires separate installation
    - Auto-samples datasets >500 obs for cloud performance
    - Cloud-optimized ARIMA: max_p=2, max_q=2
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# pmdarima is optional
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    pm = None
    PMDARIMA_AVAILABLE = False

# Prophet is optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

from typing import Dict, Any, Tuple, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Handles time series data analysis and forecasting."""
    
    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize TimeSeriesAnalyzer with input dataframe.
        
        Prepares a new analyzer instance for time series analysis on the provided
        DataFrame. Initializes all internal state variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing time series data with at least
                one datetime column and one numeric value column.
        
        Examples:
            >>> df = pd.read_csv('sales_data.csv')
            >>> analyzer = TimeSeriesAnalyzer(df)
        
        Notes:
            - Makes a copy of the input DataFrame
            - Call set_time_column() before analysis methods
            - All models and forecasts stored as instance attributes
        """
        self.df = df.copy()
        self.time_col = None
        self.value_col = None
        self.ts_data = None
        self.decomposition = None
        self.arima_model = None
        self.prophet_model = None
        self.forecast_arima = None
        self.forecast_prophet = None
    
    def set_time_column(self, time_col: str, value_col: str) -> pd.Series:
        """
        Set the time and value columns for analysis.
        
        Args:
            time_col: Name of the datetime column
            value_col: Name of the value column to analyze
            
        Returns:
            Time series data
        """
        if time_col not in self.df.columns:
            raise ValueError(f"Column '{time_col}' not found in dataframe")
        if value_col not in self.df.columns:
            raise ValueError(f"Column '{value_col}' not found in dataframe")
        
        self.time_col = time_col
        self.value_col = value_col
        
        # Convert to datetime
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        
        # Sort by time
        self.df = self.df.sort_values(time_col)
        
        # Set time as index
        self.ts_data = self.df.set_index(time_col)[value_col]
        
        # Remove duplicates and handle missing values
        self.ts_data = self.ts_data[~self.ts_data.index.duplicated(keep='first')]
        self.ts_data = self.ts_data.fillna(method='ffill').fillna(method='bfill')
        
        return self.ts_data
    
    def decompose_time_series(self, model: str = 'additive', period: int = None) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            model: 'additive' or 'multiplicative'
            period: Period for seasonal component (auto-detected if None)
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        if len(self.ts_data) < 4:
            raise ValueError("Time series is too short for decomposition (need at least 4 observations)")
        
        # Auto-detect period if not provided
        if period is None:
            # Try to infer frequency
            freq = pd.infer_freq(self.ts_data.index)
            if freq:
                if 'D' in freq:
                    period = 7  # Weekly seasonality for daily data
                elif 'M' in freq:
                    period = 12  # Yearly seasonality for monthly data
                elif 'H' in freq:
                    period = 24  # Daily seasonality for hourly data
            
            if period is None or len(self.ts_data) < 2 * period:
                period = min(12, len(self.ts_data) // 2)
        
        try:
            self.decomposition = seasonal_decompose(
                self.ts_data,
                model=model,
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                'trend': self.decomposition.trend,
                'seasonal': self.decomposition.seasonal,
                'residual': self.decomposition.resid,
                'observed': self.decomposition.observed
            }
        except Exception as e:
            raise ValueError(f"Decomposition failed: {str(e)}")
    
    def get_stationarity_test(self) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Returns:
            Dictionary with test results
        """
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        # Perform ADF test
        result = adfuller(self.ts_data.dropna(), autolag='AIC')
        
        # Interpret results
        is_stationary = result[1] < 0.05
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': is_stationary,
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary'
        }
    
    def get_autocorrelation(self, nlags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ACF and PACF.
        
        Args:
            nlags: Number of lags to calculate
            
        Returns:
            Tuple of (acf_values, pacf_values)
        """
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        nlags = min(nlags, len(self.ts_data) // 2 - 1)
        
        acf_values = acf(self.ts_data.dropna(), nlags=nlags)
        pacf_values = pacf(self.ts_data.dropna(), nlags=nlags)
        
        return acf_values, pacf_values
    
    def run_auto_arima(self, forecast_periods: int = 30, seasonal: Optional[bool] = None) -> Dict[str, Any]:
        """Run Auto-ARIMA model selection and generate forecasts.
        
        This method automatically selects the best ARIMA(p,d,q)(P,D,Q)m model using
        stepwise search algorithm. Cloud-optimized with reduced search space (max_p=2, max_q=2)
        and automatic sampling for datasets >500 observations.
        
        Args:
            forecast_periods (int): Number of future time periods to forecast. Default is 30.
            seasonal (Optional[bool]): Seasonality setting:
                - True: Force seasonal ARIMA model
                - False: Force non-seasonal ARIMA model
                - None: Auto-detect based on data characteristics
                Default is None (auto-detect).
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - model_order (tuple): ARIMA (p, d, q) order
                - seasonal_order (tuple): Seasonal (P, D, Q, m) order
                - aic (float): Akaike Information Criterion
                - bic (float): Bayesian Information Criterion
                - forecast (pd.DataFrame): Forecast with confidence intervals
                - summary (str): Full model summary
        
        Raises:
            ValueError: If pmdarima not installed or ts_data not set.
        
        Examples:
            >>> analyzer = TimeSeriesAnalyzer(sales_df)
            >>> analyzer.set_time_column('Date', 'Revenue')
            >>> results = analyzer.run_auto_arima(forecast_periods=60, seasonal=True)
            >>> print(f"Best model: ARIMA{results['model_order']}")
            >>> print(f"AIC: {results['aic']:.2f}")
        
        Notes:
            - Uses last 500 observations for datasets >500 rows
            - Cloud-optimized: max_p=2, max_q=2, max_order=4
            - Seasonal m auto-detected: 12 (monthly), 7 (daily)
            - Stepwise search for performance
            - 95% confidence intervals included in forecast
            - pmdarima unavailable on Python 3.13
        """
        if not PMDARIMA_AVAILABLE:
            raise ValueError("pmdarima is not installed. This feature is temporarily unavailable on Python 3.13.")
        
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        # Sample large datasets to prevent cloud timeouts
        ts_to_fit = self.ts_data
        if len(self.ts_data) > 500:
            # Use last 500 observations for training to reduce computation
            ts_to_fit = self.ts_data.iloc[-500:]
            warnings.warn(f"Dataset has {len(self.ts_data)} observations. Using last 500 for faster training.")
        
        # Auto-detect if seasonal model is needed
        if seasonal is None:
            # Only use seasonal if dataset is large enough (need at least 2 seasons)
            inferred_freq = pd.infer_freq(ts_to_fit.index)
            if inferred_freq and len(ts_to_fit) >= 24:
                seasonal = True
                m = 12 if 'M' in str(inferred_freq) else 7 if 'D' in str(inferred_freq) else 12
            else:
                seasonal = False
                m = 1
        else:
            m = 12 if seasonal else 1
        
        # Fit auto ARIMA with cloud-friendly parameters
        # Heavily optimized for Streamlit Cloud resource limits
        self.arima_model = pm.auto_arima(
            ts_to_fit,
            start_p=1, start_q=1,  # Start with simple model
            max_p=2, max_q=2,  # Very conservative limits for cloud
            max_order=4,  # Strict complexity limit
            seasonal=seasonal,
            m=m,
            d=None,
            D=None,
            start_P=0, start_Q=0,  # Seasonal starting points
            max_P=1, max_Q=1,  # Minimal seasonal terms
            max_d=2,
            max_D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,  # Fast stepwise search
            n_jobs=1,
            information_criterion='aic',
            with_intercept='auto',
            maxiter=50  # Limit iterations per model
        )
        
        # Generate forecast
        forecast, conf_int = self.arima_model.predict(
            n_periods=forecast_periods,
            return_conf_int=True
        )
        
        # Create forecast index
        last_date = self.ts_data.index[-1]
        freq = pd.infer_freq(self.ts_data.index) or 'D'
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit='D'),
            periods=forecast_periods,
            freq=freq
        )
        
        self.forecast_arima = pd.DataFrame({
            'forecast': forecast,
            'lower_bound': conf_int[:, 0],
            'upper_bound': conf_int[:, 1]
        }, index=forecast_index)
        
        return {
            'model_order': self.arima_model.order,
            'seasonal_order': self.arima_model.seasonal_order,
            'aic': self.arima_model.aic(),
            'bic': self.arima_model.bic(),
            'forecast': self.forecast_arima,
            'summary': str(self.arima_model.summary())
        }
    
    def run_prophet(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """Run Facebook Prophet model for time series forecasting.
        
        This method fits a Prophet model with automatic seasonality detection,
        including weekly and yearly patterns. Prophet is robust to missing data,
        outliers, and trend changes.
        
        Args:
            forecast_periods (int): Number of future time periods to forecast. Default is 30.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - forecast (pd.DataFrame): Future forecast only with yhat, yhat_lower, yhat_upper
                - full_forecast (pd.DataFrame): Historical + future forecast
                - components (Dict): Decomposed components:
                    - trend: Overall trend component
                    - weekly: Weekly seasonality (if detected)
                    - yearly: Yearly seasonality (if detected)
        
        Raises:
            ValueError: If Prophet not installed or ts_data not set.
        
        Examples:
            >>> analyzer = TimeSeriesAnalyzer(traffic_df)
            >>> analyzer.set_time_column('Timestamp', 'Visitors')
            >>> results = analyzer.run_prophet(forecast_periods=90)
            >>> forecast = results['forecast']
            >>> print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
        
        Notes:
            - Automatically detects weekly and yearly seasonality
            - Daily seasonality disabled by default
            - Handles missing values and outliers well
            - Returns 80% confidence intervals (yhat_lower, yhat_upper)
            - Requires: pip install prophet
            - Works better with longer time series (>2 cycles)
        """
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet is not installed. Install it with: pip install prophet")
        
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': self.ts_data.index,
            'y': self.ts_data.values
        })
        
        # Initialize and fit model
        self.prophet_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.prophet_model.fit(prophet_df)
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=forecast_periods)
        
        # Generate forecast
        forecast = self.prophet_model.predict(future)
        
        # Extract forecast for future periods only
        self.forecast_prophet = forecast[forecast['ds'] > self.ts_data.index[-1]]
        
        return {
            'forecast': self.forecast_prophet,
            'full_forecast': forecast,
            'components': {
                'trend': forecast[['ds', 'trend']],
                'weekly': forecast[['ds', 'weekly']] if 'weekly' in forecast.columns else None,
                'yearly': forecast[['ds', 'yearly']] if 'yearly' in forecast.columns else None
            }
        }
    
    def create_decomposition_plot(self, components: Dict[str, pd.Series]) -> go.Figure:
        """Create decomposition visualization."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Observed
        fig.add_trace(
            go.Scatter(x=components['observed'].index, y=components['observed'].values,
                      mode='lines', name='Observed', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=components['trend'].index, y=components['trend'].values,
                      mode='lines', name='Trend', line=dict(color='red')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=components['seasonal'].index, y=components['seasonal'].values,
                      mode='lines', name='Seasonal', line=dict(color='green')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=components['residual'].index, y=components['residual'].values,
                      mode='lines', name='Residual', line=dict(color='purple')),
            row=4, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Time Series Decomposition")
        return fig
    
    def create_acf_pacf_plot(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> go.Figure:
        """Create ACF and PACF plots."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)')
        )
        
        # ACF
        fig.add_trace(
            go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
            row=1, col=1
        )
        
        # PACF
        fig.add_trace(
            go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
            row=1, col=2
        )
        
        # Add confidence intervals
        conf_int = 1.96 / np.sqrt(len(self.ts_data))
        for i in [1, 2]:
            fig.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=i)
            fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=i)
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def create_forecast_plot(self, forecast_type: str = 'arima') -> go.Figure:
        """Create forecast visualization."""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.ts_data.index,
            y=self.ts_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        if forecast_type == 'arima' and self.forecast_arima is not None:
            # ARIMA forecast
            fig.add_trace(go.Scatter(
                x=self.forecast_arima.index,
                y=self.forecast_arima['forecast'],
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=self.forecast_arima.index,
                y=self.forecast_arima['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=self.forecast_arima.index,
                y=self.forecast_arima['lower_bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
        
        elif forecast_type == 'prophet' and self.forecast_prophet is not None:
            # Prophet forecast
            fig.add_trace(go.Scatter(
                x=self.forecast_prophet['ds'],
                y=self.forecast_prophet['yhat'],
                mode='lines',
                name='Prophet Forecast',
                line=dict(color='green', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=self.forecast_prophet['ds'],
                y=self.forecast_prophet['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=self.forecast_prophet['ds'],
                y=self.forecast_prophet['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0, 255, 0, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Time Series Forecast ({forecast_type.upper()})',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_comparison_plot(self) -> go.Figure:
        """Create comparison plot of both forecasts."""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.ts_data.index,
            y=self.ts_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # ARIMA forecast
        if self.forecast_arima is not None:
            fig.add_trace(go.Scatter(
                x=self.forecast_arima.index,
                y=self.forecast_arima['forecast'],
                mode='lines',
                name='ARIMA',
                line=dict(color='red', dash='dash')
            ))
        
        # Prophet forecast
        if self.forecast_prophet is not None:
            fig.add_trace(go.Scatter(
                x=self.forecast_prophet['ds'],
                y=self.forecast_prophet['yhat'],
                mode='lines',
                name='Prophet',
                line=dict(color='green', dash='dot')
            ))
        
        fig.update_layout(
            title='Forecast Comparison: ARIMA vs Prophet',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=500
        )
        
        return fig
