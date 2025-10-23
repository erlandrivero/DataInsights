"""
Monte Carlo Simulation utilities for financial forecasting and risk analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

class MonteCarloSimulator:
    """Handles Monte Carlo simulations for financial data."""
    
    def __init__(self):
        self.stock_data = None
        self.returns = None
        self.simulations = None
    
    @staticmethod
    def fetch_stock_data(ticker: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for historical data
            end_date: End date (default: today)
            
        Returns:
            DataFrame with stock price data
        """
        if end_date is None:
            end_date = datetime.now()
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker '{ticker}'")
            
            return data
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns from price data."""
        returns = prices.pct_change().dropna()
        self.returns = returns
        return returns
    
    def get_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical measures for returns."""
        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'variance': returns.var(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min': returns.min(),
            'max': returns.max(),
            'median': returns.median()
        }
    
    def run_simulation(
        self,
        start_price: float,
        mean_return: float,
        std_return: float,
        days: int,
        num_simulations: int = 1000,
        random_seed: int = 42
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation.
        
        Args:
            start_price: Starting price
            mean_return: Mean daily return
            std_return: Standard deviation of returns
            days: Number of days to simulate
            num_simulations: Number of simulation paths
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of simulation paths (num_simulations x days)
        """
        np.random.seed(random_seed)
        
        # Generate random returns
        random_returns = np.random.normal(
            mean_return,
            std_return,
            size=(num_simulations, days)
        )
        
        # Calculate price paths
        price_paths = np.zeros((num_simulations, days + 1))
        price_paths[:, 0] = start_price
        
        for t in range(1, days + 1):
            price_paths[:, t] = price_paths[:, t-1] * (1 + random_returns[:, t-1])
        
        self.simulations = price_paths
        return price_paths
    
    def calculate_confidence_intervals(
        self,
        simulations: np.ndarray,
        confidence_levels: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
    ) -> Dict[float, np.ndarray]:
        """
        Calculate confidence intervals from simulations.
        
        Args:
            simulations: Simulation results
            confidence_levels: List of confidence levels (e.g., 0.05 = 5th percentile)
            
        Returns:
            Dictionary mapping confidence levels to percentile paths
        """
        intervals = {}
        for level in confidence_levels:
            intervals[level] = np.percentile(simulations, level * 100, axis=0)
        
        return intervals
    
    def get_risk_metrics(self, final_prices: np.ndarray, initial_price: float) -> Dict[str, Any]:
        """
        Calculate risk metrics from final simulation prices.
        
        Args:
            final_prices: Array of final prices from all simulations
            initial_price: Starting price
            
        Returns:
            Dictionary of risk metrics
        """
        returns = (final_prices - initial_price) / initial_price * 100
        
        return {
            'expected_return': np.mean(returns),
            'expected_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_dev': np.std(returns),
            'var_95': np.percentile(returns, 5),  # Value at Risk (95% confidence)
            'var_99': np.percentile(returns, 1),  # Value at Risk (99% confidence)
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),  # Conditional VaR
            'probability_profit': np.sum(final_prices > initial_price) / len(final_prices) * 100,
            'probability_loss': np.sum(final_prices < initial_price) / len(final_prices) * 100,
            'min_price': np.min(final_prices),
            'max_price': np.max(final_prices)
        }
    
    def create_simulation_plot(
        self,
        simulations: np.ndarray,
        intervals: Dict[float, np.ndarray],
        ticker: str,
        days: int
    ) -> go.Figure:
        """Create interactive plot of Monte Carlo simulations."""
        fig = go.Figure()
        
        # Plot sample simulation paths (show 100 random paths)
        num_paths_to_show = min(100, simulations.shape[0])
        indices = np.random.choice(simulations.shape[0], num_paths_to_show, replace=False)
        
        for i in indices:
            fig.add_trace(go.Scatter(
                x=list(range(days + 1)),
                y=simulations[i],
                mode='lines',
                line=dict(width=0.5, color='lightblue'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Plot confidence intervals
        colors = {
            0.05: 'red',
            0.25: 'orange',
            0.50: 'green',
            0.75: 'orange',
            0.95: 'red'
        }
        
        for level, path in intervals.items():
            fig.add_trace(go.Scatter(
                x=list(range(days + 1)),
                y=path,
                mode='lines',
                line=dict(width=2, color=colors.get(level, 'blue')),
                name=f'{int(level*100)}th Percentile'
            ))
        
        fig.update_layout(
            title=f'Monte Carlo Simulation: {ticker} ({days} Days, {simulations.shape[0]} Paths)',
            xaxis_title='Days',
            yaxis_title='Price ($)',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def create_distribution_plot(self, final_prices: np.ndarray, initial_price: float) -> go.Figure:
        """Create distribution plot of final prices."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=final_prices,
            nbinsx=50,
            name='Final Price Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add vertical line for initial price
        fig.add_vline(
            x=initial_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Initial: ${initial_price:.2f}",
            annotation_position="top"
        )
        
        # Add vertical line for mean
        mean_price = np.mean(final_prices)
        fig.add_vline(
            x=mean_price,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: ${mean_price:.2f}",
            annotation_position="bottom"
        )
        
        fig.update_layout(
            title='Distribution of Final Prices',
            xaxis_title='Price ($)',
            yaxis_title='Frequency',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_returns_distribution(self, returns: pd.Series) -> go.Figure:
        """Create distribution plot of historical returns."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='lightgreen',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Distribution of Historical Daily Returns',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def generate_insights(risk_metrics: Dict[str, Any], ticker: str, days: int) -> List[str]:
        """Generate business insights from risk metrics."""
        insights = []
        
        # Expected return insight
        expected_return = risk_metrics['expected_return']
        if expected_return > 0:
            insights.append(
                f"📈 **Positive Outlook:** Expected return of **{expected_return:.2f}%** "
                f"over {days} days (${risk_metrics['expected_price']:.2f} average final price)"
            )
        else:
            insights.append(
                f"📉 **Negative Outlook:** Expected return of **{expected_return:.2f}%** "
                f"over {days} days (${risk_metrics['expected_price']:.2f} average final price)"
            )
        
        # Probability insight
        prob_profit = risk_metrics['probability_profit']
        insights.append(
            f"🎲 **Probability:** {prob_profit:.1f}% chance of profit, "
            f"{risk_metrics['probability_loss']:.1f}% chance of loss"
        )
        
        # Risk insight
        var_95 = risk_metrics['var_95']
        insights.append(
            f"⚠️ **Risk (VaR 95%):** 5% chance of losing more than **{abs(var_95):.2f}%** "
            f"(worst-case scenario: {abs(risk_metrics['var_99']):.2f}% loss)"
        )
        
        # Volatility insight
        std_dev = risk_metrics['std_dev']
        if std_dev > 20:
            insights.append(
                f"🔥 **High Volatility:** Standard deviation of {std_dev:.2f}% indicates high risk/reward"
            )
        elif std_dev > 10:
            insights.append(
                f"⚡ **Moderate Volatility:** Standard deviation of {std_dev:.2f}% indicates moderate risk"
            )
        else:
            insights.append(
                f"✅ **Low Volatility:** Standard deviation of {std_dev:.2f}% indicates lower risk"
            )
        
        # Price range insight
        insights.append(
            f"📊 **Price Range:** Simulations show prices from **${risk_metrics['min_price']:.2f}** "
            f"to **${risk_metrics['max_price']:.2f}**"
        )
        
        return insights
