"""
Time Series Analysis and Forecasting utilities.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# Prophet is optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Handles time series data analysis and forecasting."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the TimeSeriesAnalyzer with a dataframe.
        
        Args:
            df: Input dataframe
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
    
    def run_auto_arima(self, forecast_periods: int = 30) -> Dict[str, Any]:
        """
        Run auto-ARIMA to find best model and generate forecast.
        
        Args:
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with model info and forecast
        """
        if self.ts_data is None:
            raise ValueError("Time series data must be set first using set_time_column()")
        
        # Fit auto ARIMA
        self.arima_model = pm.auto_arima(
            self.ts_data,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=True,
            m=12,  # Monthly seasonality
            d=None,
            D=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
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
        """
        Run Prophet model for forecasting.
        
        Args:
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with model info and forecast
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
