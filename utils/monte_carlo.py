"""Monte Carlo Simulation utilities for financial forecasting and risk analysis.

This module provides comprehensive Monte Carlo simulation capabilities for
financial data analysis, including stock price forecasting, risk assessment,
and probability calculations.

Author: DataInsights Team
Phase 2 Enhancement: Oct 23, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


class MonteCarloSimulator:
    """Handles Monte Carlo simulations for financial data and risk analysis.
    
    This class provides complete functionality for running Monte Carlo simulations
    on stock prices, calculating risk metrics, and visualizing potential future
    price paths with confidence intervals.
    
    Attributes:
        stock_data (Optional[pd.DataFrame]): Historical stock price data
        returns (Optional[pd.Series]): Calculated daily returns
        simulations (Optional[np.ndarray]): Simulation results matrix
    
    Example:
        >>> # Basic Monte Carlo workflow
        >>> simulator = MonteCarloSimulator()
        >>> 
        >>> # Fetch historical data
        >>> start_date = datetime.now() - timedelta(days=365)
        >>> data = simulator.fetch_stock_data('AAPL', start_date)
        >>> 
        >>> # Calculate returns
        >>> returns = simulator.calculate_returns(data['Close'])
        >>> stats = simulator.get_statistics(returns)
        >>> 
        >>> # Run simulation
        >>> simulations = simulator.run_simulation(
        >>>     start_price=data['Close'].iloc[-1],
        >>>     mean_return=stats['mean'],
        >>>     std_return=stats['std'],
        >>>     days=30,
        >>>     num_simulations=10000
        >>> )
        >>> 
        >>> # Analyze results
        >>> risk_metrics = simulator.get_risk_metrics(
        >>>     simulations[:, -1], 
        >>>     data['Close'].iloc[-1]
        >>> )
    
    Note:
        - Uses geometric Brownian motion for price simulation
        - Assumes returns follow normal distribution
        - All simulations use configurable random seed for reproducibility
    """
    
    def __init__(self):
        """Initialize MonteCarloSimulator with empty state.
        
        All attributes are set to None initially and populated through
        the simulation workflow methods.
        """
        self.stock_data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.Series] = None
        self.simulations: Optional[np.ndarray] = None
    
    @staticmethod
    def fetch_stock_data(
        ticker: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Fetch historical stock data and company name from Yahoo Finance.
        
        Downloads complete historical pricing data including open, high, low,
        close, volume, and adjusted close prices, plus company name.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
            start_date: Start date for historical data
            end_date: End date for data (default: today)
        
        Returns:
            Tuple containing:
                - DataFrame with columns:
                    - Open, High, Low, Close: Daily prices
                    - Volume: Trading volume
                    - Dividends, Stock Splits: Corporate actions
                - Company name (str) or None if unavailable
        
        Raises:
            ValueError: If ticker symbol is invalid or no data found
            Exception: If API request fails
        
        Example:
            >>> # Get 1 year of Apple stock data
            >>> start = datetime(2023, 1, 1)
            >>> end = datetime(2024, 1, 1)
            >>> data, name = MonteCarloSimulator.fetch_stock_data('AAPL', start, end)
            >>> print(f"Fetched {len(data)} days of data for {name}")
        
        Note:
            - Uses yfinance library for data access
            - Data is adjusted for splits and dividends
            - Free API with no authentication required
            - Rate limits may apply for excessive requests
            - Returns company name in same call to avoid rate limiting
        """
        if end_date is None:
            end_date = datetime.now()
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for ticker '{ticker}'")
            
            # Get company name from same ticker object to avoid extra API call
            company_name = None
            try:
                info = stock.info
                company_name = info.get('longName', info.get('shortName', None))
            except:
                pass  # If company info fails, continue without it
            
            return data, company_name
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily percentage returns from price data.
        
        Computes daily returns using percentage change formula:
        return[t] = (price[t] - price[t-1]) / price[t-1]
        
        Args:
            prices: Series of price data (typically Close prices)
        
        Returns:
            Series of daily returns with NaN values removed
        
        Example:
            >>> prices = pd.Series([100, 102, 101, 105])
            >>> simulator = MonteCarloSimulator()
            >>> returns = simulator.calculate_returns(prices)
            >>> print(returns.values)
            [0.02, -0.0098, 0.0396]
        
        Note:
            - First value is always NaN (no prior price) and is dropped
            - Returns are decimal values (0.02 = 2% return)
            - Stored in self.returns for later use
        """
        returns = prices.pct_change().dropna()
        self.returns = returns
        return returns
    
    def get_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistical measures for returns distribution.
        
        Computes key statistics needed for Monte Carlo simulation and
        risk assessment.
        
        Args:
            returns: Series of daily returns
        
        Returns:
            Dictionary containing:
                - mean (float): Average daily return
                - std (float): Standard deviation of returns
                - variance (float): Variance of returns
                - skewness (float): Skewness (asymmetry of distribution)
                - kurtosis (float): Kurtosis (tail heaviness)
                - min (float): Minimum return observed
                - max (float): Maximum return observed
                - median (float): Median return
        
        Example:
            >>> stats = simulator.get_statistics(returns)
            >>> print(f"Mean return: {stats['mean']*100:.3f}%")
            >>> print(f"Volatility (std): {stats['std']*100:.3f}%")
        
        Note:
            - Mean and std are primary inputs for simulation
            - Skewness: 0 = symmetric, >0 = right tail, <0 = left tail
            - Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails
        """
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
        """Run Monte Carlo simulation for stock price forecasting.
        
        Generates multiple possible future price paths using geometric Brownian
        motion with parameters estimated from historical returns.
        
        Args:
            start_price: Initial price for simulation (typically last close price)
            mean_return: Expected daily return (from historical data)
            std_return: Standard deviation of returns (volatility measure)
            days: Number of days to simulate forward
            num_simulations: Number of random paths to generate (default: 1000)
            random_seed: Seed for reproducibility (default: 42)
        
        Returns:
            NumPy array of shape (num_simulations, days + 1)
                - Rows: Different simulation paths
                - Columns: Time steps (0 to days)
                - Values: Simulated prices
        
        Example:
            >>> # Simulate 30 days forward with 10,000 paths
            >>> simulations = simulator.run_simulation(
            >>>     start_price=150.0,
            >>>     mean_return=0.001,  # 0.1% daily
            >>>     std_return=0.02,     # 2% volatility
            >>>     days=30,
            >>>     num_simulations=10000
            >>> )
            >>> print(f"Shape: {simulations.shape}")  # (10000, 31)
        
        Note:
            - Uses geometric Brownian motion: S(t+1) = S(t) * (1 + return(t))
            - Returns sampled from normal distribution N(mean, std)
            - More simulations = smoother confidence intervals
            - Results stored in self.simulations
        """
        np.random.seed(random_seed)
        
        # Generate random returns from normal distribution
        random_returns = np.random.normal(
            mean_return,
            std_return,
            size=(num_simulations, days)
        )
        
        # Calculate price paths using geometric Brownian motion
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
        """Calculate confidence intervals from simulation results.
        
        Computes percentile paths showing range of possible outcomes at
        different confidence levels.
        
        Args:
            simulations: Simulation results array (from run_simulation)
            confidence_levels: List of percentiles to calculate (0.0 to 1.0)
                              - 0.05 = 5th percentile (pessimistic)
                              - 0.50 = 50th percentile (median)
                              - 0.95 = 95th percentile (optimistic)
        
        Returns:
            Dictionary mapping confidence level to price path array
            - Keys: Confidence levels (e.g., 0.05, 0.50, 0.95)
            - Values: Arrays of prices at that percentile over time
        
        Example:
            >>> intervals = simulator.calculate_confidence_intervals(
            >>>     simulations,
            >>>     confidence_levels=[0.05, 0.50, 0.95]
            >>> )
            >>> 
            >>> # Get median forecast
            >>> median_path = intervals[0.50]
            >>> print(f"Median price in 30 days: ${median_path[-1]:.2f}")
        
        Note:
            - 0.05 and 0.95 show 90% confidence range
            - 0.50 is the median (middle) path
            - Useful for visualizing uncertainty
        """
        intervals = {}
        for level in confidence_levels:
            intervals[level] = np.percentile(simulations, level * 100, axis=0)
        
        return intervals
    
    def get_risk_metrics(
        self, 
        final_prices: np.ndarray, 
        initial_price: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics from final simulation prices.
        
        Analyzes distribution of final prices to compute expected returns,
        downside risk measures, and probability estimates.
        
        Args:
            final_prices: Array of final prices from all simulations
            initial_price: Starting price for return calculations
        
        Returns:
            Dictionary containing:
                - expected_return (float): Mean return percentage
                - expected_price (float): Mean final price
                - median_price (float): Median final price
                - std_dev (float): Standard deviation of returns
                - var_95 (float): Value at Risk at 95% confidence
                - var_99 (float): Value at Risk at 99% confidence
                - cvar_95 (float): Conditional VaR (expected shortfall)
                - probability_profit (float): % chance of gain
                - probability_loss (float): % chance of loss
                - min_price (float): Worst-case price
                - max_price (float): Best-case price
        
        Example:
            >>> final_prices = simulations[:, -1]
            >>> metrics = simulator.get_risk_metrics(final_prices, 150.0)
            >>> 
            >>> print(f"Expected return: {metrics['expected_return']:.2f}%")
            >>> print(f"VaR 95%: {metrics['var_95']:.2f}%")
            >>> print(f"Probability of profit: {metrics['probability_profit']:.1f}%")
        
        Note:
            - VaR 95%: 5% chance of losing more than this amount
            - CVaR: Expected loss given VaR threshold is exceeded
            - All returns expressed as percentages
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
        """Create interactive Plotly visualization of Monte Carlo simulations.
        
        Shows sample price paths and confidence interval bands for visual
        interpretation of forecast uncertainty.
        
        Args:
            simulations: Full simulation results array
            intervals: Confidence intervals from calculate_confidence_intervals()
            ticker: Stock ticker symbol for title
            days: Number of days simulated
        
        Returns:
            Plotly Figure object ready for display
        
        Example:
            >>> intervals = simulator.calculate_confidence_intervals(simulations)
            >>> fig = simulator.create_simulation_plot(
            >>>     simulations, intervals, 'AAPL', 30
            >>> )
            >>> st.plotly_chart(fig, use_container_width=True)
        
        Note:
            - Shows up to 100 random sample paths (light blue)
            - Confidence intervals in distinct colors
            - Interactive hover for detailed values
        """
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
    
    def create_distribution_plot(
        self, 
        final_prices: np.ndarray, 
        initial_price: float
    ) -> go.Figure:
        """Create histogram showing distribution of final simulated prices.
        
        Visualizes the spread and likelihood of different price outcomes.
        
        Args:
            final_prices: Array of final prices from all simulations
            initial_price: Starting price for reference line
        
        Returns:
            Plotly Figure object with histogram and reference lines
        
        Example:
            >>> fig = simulator.create_distribution_plot(
            >>>     simulations[:, -1],
            >>>     150.0
            >>> )
            >>> st.plotly_chart(fig)
        
        Note:
            - Includes vertical lines for initial and mean prices
            - 50 bins for detailed distribution view
        """
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
        """Create histogram of historical returns with normal distribution overlay.
        
        Helps assess whether returns follow normal distribution (key assumption).
        
        Args:
            returns: Series of historical daily returns
        
        Returns:
            Plotly Figure with histogram and fitted normal curve
        
        Example:
            >>> fig = simulator.create_returns_distribution(returns)
            >>> st.plotly_chart(fig)
        
        Note:
            - Red line shows theoretical normal distribution
            - Deviations indicate non-normal behavior
            - Useful for validating simulation assumptions
        """
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
    def generate_insights(
        risk_metrics: Dict[str, Any], 
        ticker: str, 
        days: int
    ) -> List[str]:
        """Generate actionable business insights from simulation risk metrics.
        
        Interprets quantitative results into plain-language recommendations
        for investment decisions.
        
        Args:
            risk_metrics: Dictionary from get_risk_metrics()
            ticker: Stock ticker symbol
            days: Number of days simulated
        
        Returns:
            List of formatted insight strings with markdown
        
        Example:
            >>> insights = MonteCarloSimulator.generate_insights(
            >>>     risk_metrics, 'AAPL', 30
            >>> )
            >>> for insight in insights:
            >>>     st.markdown(insight)
        
        Note:
            - Categorizes volatility as high/moderate/low
            - Highlights key risks and opportunities
            - Suitable for non-technical stakeholders
        """
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
        
        # Risk insight (Value at Risk)
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
