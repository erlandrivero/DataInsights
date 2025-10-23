"""
Error Handler Utility for DataInsights

Provides centralized error handling with user-friendly messages,
technical details toggling, and error logging.

Author: DataInsights Team
Created: Oct 23, 2025
"""

from typing import Optional, Callable, Any, Dict
import streamlit as st
import traceback
import logging
from functools import wraps
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('datainsights_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DataInsights')


class ErrorCategory:
    """Error category constants."""
    DATA_ERROR = "Data Error"
    MODEL_ERROR = "Model Training Error"
    API_ERROR = "API Error"
    VALIDATION_ERROR = "Validation Error"
    FILE_ERROR = "File Error"
    COMPUTATION_ERROR = "Computation Error"
    UNKNOWN_ERROR = "Unexpected Error"


class ErrorHandler:
    """Centralized error handler for DataInsights application."""
    
    @staticmethod
    def handle_error(error: Exception, 
                    category: str = ErrorCategory.UNKNOWN_ERROR,
                    user_message: Optional[str] = None,
                    show_details: bool = True,
                    log_error: bool = True) -> None:
        """
        Handle an error with user-friendly display and logging.
        
        Args:
            error: The exception that occurred
            category: Category of the error
            user_message: Optional custom user-friendly message
            show_details: Whether to show technical details toggle
            log_error: Whether to log the error
        """
        # Log the error
        if log_error:
            logger.error(f"{category}: {str(error)}", exc_info=True)
        
        # Display user-friendly message
        if user_message:
            st.error(f"❌ **{category}**\n\n{user_message}")
        else:
            st.error(f"❌ **{category}**\n\n{ErrorHandler._get_friendly_message(error, category)}")
        
        # Show technical details in expander
        if show_details:
            with st.expander("🔍 Show Technical Details", expanded=False):
                st.code(str(error))
                st.text("Full Traceback:")
                st.code(traceback.format_exc())
        
        # Show helpful actions
        ErrorHandler._show_helpful_actions(error, category)
    
    @staticmethod
    def _get_friendly_message(error: Exception, category: str) -> str:
        """
        Generate a user-friendly error message based on the error type.
        
        Args:
            error: The exception
            category: Error category
            
        Returns:
            User-friendly error message
        """
        error_messages = {
            ErrorCategory.DATA_ERROR: (
                "There's an issue with your data. Please check that:\n"
                "- Your dataset has the expected columns\n"
                "- Data types are correct (numbers for numeric columns, dates for date columns)\n"
                "- There are no unexpected missing values"
            ),
            ErrorCategory.MODEL_ERROR: (
                "The machine learning model encountered an error during training. This could be due to:\n"
                "- Insufficient data samples\n"
                "- Data quality issues\n"
                "- Incompatible data types\n"
                "- Extreme class imbalance"
            ),
            ErrorCategory.API_ERROR: (
                "There was a problem connecting to the external service. Please:\n"
                "- Check your internet connection\n"
                "- Verify your API key is configured correctly\n"
                "- Try again in a few moments"
            ),
            ErrorCategory.VALIDATION_ERROR: (
                "Your input didn't pass validation checks. Please review the requirements and try again."
            ),
            ErrorCategory.FILE_ERROR: (
                "There was a problem reading or writing a file. Please check:\n"
                "- The file exists and is accessible\n"
                "- You have the necessary permissions\n"
                "- The file format is supported"
            ),
            ErrorCategory.COMPUTATION_ERROR: (
                "A calculation error occurred. This might be due to:\n"
                "- Numerical overflow or underflow\n"
                "- Division by zero\n"
                "- Invalid mathematical operations"
            )
        }
        
        base_message = error_messages.get(category, 
            "An unexpected error occurred. Please try again or contact support if the problem persists."
        )
        
        return base_message
    
    @staticmethod
    def _show_helpful_actions(error: Exception, category: str) -> None:
        """
        Show helpful actions based on error type.
        
        Args:
            error: The exception
            category: Error category
        """
        # Data error suggestions
        if category == ErrorCategory.DATA_ERROR:
            st.info("💡 **Quick Fixes:**\n"
                   "- Go to 'Data Analysis & Cleaning' to inspect your data\n"
                   "- Check for missing values and handle them appropriately\n"
                   "- Ensure numeric columns contain only numbers")
        
        # Model error suggestions
        elif category == ErrorCategory.MODEL_ERROR:
            st.info("💡 **Quick Fixes:**\n"
                   "- Ensure you have at least 50 samples for classification\n"
                   "- Check that your target column has at least 2 classes with multiple samples\n"
                   "- Try removing highly correlated or constant features")
        
        # API error suggestions
        elif category == ErrorCategory.API_ERROR:
            st.info("💡 **Quick Fixes:**\n"
                   "- Verify your OpenAI API key in Streamlit secrets or .env file\n"
                   "- Check if you have available API credits\n"
                   "- The service might be temporarily unavailable - try again later")
        
        # File error suggestions
        elif category == ErrorCategory.FILE_ERROR:
            st.info("💡 **Quick Fixes:**\n"
                   "- Ensure the file is in CSV or Excel format\n"
                   "- Check that the file is not corrupted\n"
                   "- Try re-uploading the file")
        
        # Generic suggestion
        else:
            st.info("💡 **Need Help?**\n"
                   "- Check the 'About' page for documentation\n"
                   "- Review the example datasets in 'Data Upload'\n"
                   "- Ensure your data meets the minimum requirements")
    
    @staticmethod
    def safe_execute(func: Callable, 
                    error_category: str = ErrorCategory.UNKNOWN_ERROR,
                    user_message: Optional[str] = None,
                    fallback_value: Any = None) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            error_category: Category for errors
            user_message: Custom error message
            fallback_value: Value to return on error
            
        Returns:
            Function result or fallback value on error
        """
        try:
            return func()
        except Exception as e:
            ErrorHandler.handle_error(e, error_category, user_message)
            return fallback_value
    
    @staticmethod
    def handle_api_error(error: Exception, 
                        service_name: str = "API",
                        fallback_message: Optional[str] = None) -> None:
        """
        Handle API-specific errors with fallback messaging.
        
        Args:
            error: The API exception
            service_name: Name of the API service
            fallback_message: Optional fallback message to display
        """
        logger.error(f"{service_name} Error: {str(error)}", exc_info=True)
        
        st.error(f"❌ **{service_name} Error**\n\n"
                f"Unable to connect to {service_name}. "
                "This service is temporarily unavailable.")
        
        if fallback_message:
            st.warning(f"⚠️ **Fallback Mode**\n\n{fallback_message}")
        
        with st.expander("🔍 Error Details", expanded=False):
            st.code(str(error))
    
    @staticmethod
    def handle_data_error(error: Exception, 
                         context: Optional[str] = None) -> None:
        """
        Handle data-related errors with specific guidance.
        
        Args:
            error: The exception
            context: Optional context about what was being done
        """
        context_msg = f" while {context}" if context else ""
        
        ErrorHandler.handle_error(
            error,
            category=ErrorCategory.DATA_ERROR,
            user_message=f"There's an issue with your data{context_msg}. "
                        "Please check the data quality and try again."
        )
    
    @staticmethod
    def handle_validation_error(message: str, 
                               recommendation: Optional[str] = None) -> None:
        """
        Handle validation errors with recommendations.
        
        Args:
            message: Validation error message
            recommendation: Optional recommendation
        """
        st.error(f"❌ **Validation Error**\n\n{message}")
        
        if recommendation:
            st.info(f"💡 **Recommendation:** {recommendation}")
        
        st.stop()


def error_boundary(error_category: str = ErrorCategory.UNKNOWN_ERROR, 
                  user_message: Optional[str] = None,
                  show_details: bool = True):
    """
    Decorator for wrapping functions with error handling.
    
    Args:
        error_category: Category of errors to expect
        user_message: Custom user message
        show_details: Whether to show technical details
        
    Example:
        @error_boundary(ErrorCategory.MODEL_ERROR, "Failed to train model")
        def train_model(X, y):
            # model training code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(
                    e, 
                    error_category, 
                    user_message,
                    show_details
                )
                return None
        return wrapper
    return decorator


class SafeOperations:
    """Safe wrappers for common operations."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, 
                   default: float = 0.0) -> float:
        """
        Safely divide two numbers.
        
        Args:
            numerator: Number to divide
            denominator: Number to divide by
            default: Default value if division fails
            
        Returns:
            Result of division or default value
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except Exception:
            return default
    
    @staticmethod
    def safe_percentage(part: float, total: float, 
                       decimals: int = 2) -> str:
        """
        Safely calculate percentage.
        
        Args:
            part: Part value
            total: Total value
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            if total == 0:
                return "0.0%"
            pct = (part / total) * 100
            return f"{pct:.{decimals}f}%"
        except Exception:
            return "N/A"
    
    @staticmethod
    def safe_mean(values: list, default: float = 0.0) -> float:
        """
        Safely calculate mean of values.
        
        Args:
            values: List of numbers
            default: Default value if calculation fails
            
        Returns:
            Mean value or default
        """
        try:
            if not values or len(values) == 0:
                return default
            return sum(values) / len(values)
        except Exception:
            return default


# Global error tracking
class ErrorTracker:
    """Track errors for analytics and debugging."""
    
    _errors: Dict[str, int] = {}
    
    @classmethod
    def log_error(cls, category: str) -> None:
        """Log an error occurrence."""
        cls._errors[category] = cls._errors.get(category, 0) + 1
    
    @classmethod
    def get_error_counts(cls) -> Dict[str, int]:
        """Get error counts by category."""
        return cls._errors.copy()
    
    @classmethod
    def reset(cls) -> None:
        """Reset error counts."""
        cls._errors.clear()
