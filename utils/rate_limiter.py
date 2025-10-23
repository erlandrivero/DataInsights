"""
Rate Limiter Utility for DataInsights

Provides rate limiting for API calls to prevent abuse and manage costs,
with input sanitization and usage tracking.

Author: DataInsights Team
Created: Oct 23, 2025
"""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import streamlit as st
import time
import logging

logger = logging.getLogger('DataInsights.RateLimiter')


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Rate limiter for API calls with per-user tracking.
    
    Attributes:
        max_calls: Maximum number of calls allowed
        period_seconds: Time period in seconds
        call_history: Dictionary tracking calls by user/session
    """
    
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        
        # Initialize session state for tracking
        if 'rate_limit_history' not in st.session_state:
            st.session_state.rate_limit_history = {}
        
        if 'api_usage_stats' not in st.session_state:
            st.session_state.api_usage_stats = {
                'total_calls': 0,
                'blocked_calls': 0,
                'total_tokens': 0,
                'estimated_cost': 0.0
            }
    
    def check_rate_limit(self, identifier: str = "default") -> bool:
        """
        Check if rate limit has been exceeded.
        
        Args:
            identifier: Unique identifier for the caller (e.g., user ID, session ID)
            
        Returns:
            True if within rate limit, False if exceeded
        """
        now = datetime.now()
        
        # Get call history for this identifier
        if identifier not in st.session_state.rate_limit_history:
            st.session_state.rate_limit_history[identifier] = []
        
        call_times = st.session_state.rate_limit_history[identifier]
        
        # Remove calls outside the time window
        cutoff_time = now - timedelta(seconds=self.period_seconds)
        call_times = [t for t in call_times if t > cutoff_time]
        
        # Update the history
        st.session_state.rate_limit_history[identifier] = call_times
        
        # Check if limit exceeded
        if len(call_times) >= self.max_calls:
            return False
        
        return True
    
    def record_call(self, identifier: str = "default") -> None:
        """
        Record a successful API call.
        
        Args:
            identifier: Unique identifier for the caller
        """
        now = datetime.now()
        
        if identifier not in st.session_state.rate_limit_history:
            st.session_state.rate_limit_history[identifier] = []
        
        st.session_state.rate_limit_history[identifier].append(now)
        st.session_state.api_usage_stats['total_calls'] += 1
    
    def get_remaining_calls(self, identifier: str = "default") -> int:
        """
        Get number of remaining calls in the current period.
        
        Args:
            identifier: Unique identifier for the caller
            
        Returns:
            Number of remaining calls
        """
        # First check rate limit to clean up old entries
        self.check_rate_limit(identifier)
        
        if identifier not in st.session_state.rate_limit_history:
            return self.max_calls
        
        used_calls = len(st.session_state.rate_limit_history[identifier])
        return max(0, self.max_calls - used_calls)
    
    def get_time_until_reset(self, identifier: str = "default") -> Optional[int]:
        """
        Get seconds until rate limit resets.
        
        Args:
            identifier: Unique identifier for the caller
            
        Returns:
            Seconds until reset, or None if not at limit
        """
        if identifier not in st.session_state.rate_limit_history:
            return None
        
        call_times = st.session_state.rate_limit_history[identifier]
        if not call_times:
            return None
        
        oldest_call = min(call_times)
        reset_time = oldest_call + timedelta(seconds=self.period_seconds)
        
        now = datetime.now()
        if reset_time > now:
            return int((reset_time - now).total_seconds())
        
        return 0
    
    def enforce_rate_limit(self, identifier: str = "default") -> None:
        """
        Enforce rate limit, raising exception if exceeded.
        
        Args:
            identifier: Unique identifier for the caller
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self.check_rate_limit(identifier):
            st.session_state.api_usage_stats['blocked_calls'] += 1
            
            time_until_reset = self.get_time_until_reset(identifier)
            
            raise RateLimitExceeded(
                f"Rate limit exceeded. Maximum {self.max_calls} calls per "
                f"{self.period_seconds} seconds. Try again in {time_until_reset} seconds."
            )
    
    def display_rate_limit_warning(self, identifier: str = "default") -> None:
        """
        Display rate limit warning in Streamlit if approaching limit.
        
        Args:
            identifier: Unique identifier for the caller
        """
        remaining = self.get_remaining_calls(identifier)
        
        if remaining <= 2 and remaining > 0:
            st.warning(f"⚠️ Rate Limit Warning: {remaining} API calls remaining in this period")
        elif remaining == 0:
            time_until_reset = self.get_time_until_reset(identifier)
            st.error(f"❌ Rate limit reached. Please wait {time_until_reset} seconds before making more requests.")


class InputSanitizer:
    """Sanitize user inputs to prevent injection and abuse."""
    
    @staticmethod
    def sanitize_text_query(query: str, max_length: int = 2000) -> str:
        """
        Sanitize text query for AI API calls.
        
        Args:
            query: Raw user input
            max_length: Maximum allowed length
            
        Returns:
            Sanitized query
            
        Raises:
            ValueError: If input is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Strip whitespace
        query = query.strip()
        
        if len(query) == 0:
            raise ValueError("Query cannot be empty")
        
        # Check length
        if len(query) > max_length:
            raise ValueError(f"Query too long. Maximum {max_length} characters allowed.")
        
        # Remove potentially dangerous patterns (basic sanitization)
        dangerous_patterns = ['<script>', '</script>', 'javascript:', 'eval(', 'exec(']
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValueError("Query contains potentially unsafe content")
        
        return query
    
    @staticmethod
    def sanitize_column_name(column_name: str) -> str:
        """
        Sanitize column name selection.
        
        Args:
            column_name: Column name from user input
            
        Returns:
            Sanitized column name
            
        Raises:
            ValueError: If column name is invalid
        """
        if not column_name or not isinstance(column_name, str):
            raise ValueError("Column name must be a non-empty string")
        
        column_name = column_name.strip()
        
        if len(column_name) == 0:
            raise ValueError("Column name cannot be empty")
        
        # Check for SQL injection patterns
        dangerous_sql = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', '--', ';']
        column_upper = column_name.upper()
        for pattern in dangerous_sql:
            if pattern in column_upper:
                raise ValueError("Column name contains invalid SQL keywords")
        
        return column_name
    
    @staticmethod
    def validate_numeric_input(value: Any, min_value: Optional[float] = None, 
                              max_value: Optional[float] = None) -> float:
        """
        Validate and sanitize numeric input.
        
        Args:
            value: Input value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated numeric value
            
        Raises:
            ValueError: If value is invalid
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError("Value must be numeric")
        
        if not isinstance(numeric_value, (int, float)) or not (-float('inf') < numeric_value < float('inf')):
            raise ValueError("Value must be a valid number")
        
        if min_value is not None and numeric_value < min_value:
            raise ValueError(f"Value must be at least {min_value}")
        
        if max_value is not None and numeric_value > max_value:
            raise ValueError(f"Value must be at most {max_value}")
        
        return numeric_value


class APIUsageTracker:
    """Track API usage and costs."""
    
    # OpenAI GPT-4 pricing (approximate, as of 2024)
    PRICING = {
        'gpt-4': {
            'input': 0.03 / 1000,   # $0.03 per 1K tokens
            'output': 0.06 / 1000   # $0.06 per 1K tokens
        },
        'gpt-3.5-turbo': {
            'input': 0.001 / 1000,
            'output': 0.002 / 1000
        }
    }
    
    @staticmethod
    def track_usage(model: str, input_tokens: int, output_tokens: int) -> None:
        """
        Track API usage and update cost estimates.
        
        Args:
            model: Model name (e.g., 'gpt-4')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        if 'api_usage_stats' not in st.session_state:
            st.session_state.api_usage_stats = {
                'total_calls': 0,
                'blocked_calls': 0,
                'total_tokens': 0,
                'estimated_cost': 0.0
            }
        
        # Update token count
        total_tokens = input_tokens + output_tokens
        st.session_state.api_usage_stats['total_tokens'] += total_tokens
        
        # Calculate cost
        if model in APIUsageTracker.PRICING:
            pricing = APIUsageTracker.PRICING[model]
            cost = (input_tokens * pricing['input']) + (output_tokens * pricing['output'])
            st.session_state.api_usage_stats['estimated_cost'] += cost
    
    @staticmethod
    def get_usage_stats() -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        if 'api_usage_stats' not in st.session_state:
            return {
                'total_calls': 0,
                'blocked_calls': 0,
                'total_tokens': 0,
                'estimated_cost': 0.0
            }
        
        return st.session_state.api_usage_stats.copy()
    
    @staticmethod
    def display_usage_stats() -> None:
        """Display API usage statistics in Streamlit."""
        stats = APIUsageTracker.get_usage_stats()
        
        with st.expander("📊 API Usage Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Calls", stats['total_calls'])
            
            with col2:
                st.metric("Blocked Calls", stats['blocked_calls'])
            
            with col3:
                st.metric("Total Tokens", f"{stats['total_tokens']:,}")
            
            with col4:
                st.metric("Est. Cost", f"${stats['estimated_cost']:.4f}")


def rate_limited(max_calls: int = 10, period_seconds: int = 60, 
                identifier_func: Optional[Callable[[], str]] = None):
    """
    Decorator for rate-limiting functions.
    
    Args:
        max_calls: Maximum calls allowed in period
        period_seconds: Time period in seconds
        identifier_func: Function to get unique identifier
        
    Example:
        @rate_limited(max_calls=5, period_seconds=60)
        def call_openai_api(prompt):
            return openai.ChatCompletion.create(...)
    """
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter(max_calls, period_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get identifier
            if identifier_func:
                identifier = identifier_func()
            else:
                # Use session-based identifier
                if 'session_id' not in st.session_state:
                    st.session_state.session_id = str(time.time())
                identifier = st.session_state.session_id
            
            # Check rate limit
            try:
                limiter.enforce_rate_limit(identifier)
            except RateLimitExceeded as e:
                st.error(f"❌ {str(e)}")
                limiter.display_rate_limit_warning(identifier)
                return None
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                limiter.record_call(identifier)
                limiter.display_rate_limit_warning(identifier)
                return result
            except Exception as e:
                logger.error(f"Error in rate-limited function: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Convenience function for API key validation
def validate_api_key(api_key_name: str = "OPENAI_API_KEY") -> bool:
    """
    Validate that API key is configured.
    
    Args:
        api_key_name: Name of the API key environment variable
        
    Returns:
        True if valid, False otherwise (displays error)
    """
    import os
    
    api_key = os.getenv(api_key_name)
    
    # Also check Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get(api_key_name)
        except Exception:
            pass
    
    if not api_key:
        st.error(f"❌ **API Configuration Error**\n\n"
                f"{api_key_name} is not configured. Please add it to:\n"
                f"- Streamlit secrets (for deployed apps)\n"
                f"- `.env` file (for local development)")
        
        with st.expander("ℹ️ How to Configure API Key", expanded=False):
            st.markdown(f"""
            **For Local Development:**
            1. Create a `.env` file in the project root
            2. Add: `{api_key_name}=your_api_key_here`
            
            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Navigate to "Secrets"
            3. Add: `{api_key_name} = "your_api_key_here"`
            """)
        
        st.stop()
        return False
    
    return True
