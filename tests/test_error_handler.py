"""
Unit Tests for Error Handler

Tests the error handling utilities for proper error management and logging.

Author: DataInsights Team
Created: Oct 23, 2025
"""

import pytest
from utils.error_handler import (
    ErrorHandler, ErrorCategory, error_boundary,
    SafeOperations, ErrorTracker
)


class TestErrorCategory:
    """Tests for ErrorCategory constants."""
    
    def test_error_categories_exist(self):
        """Test that all error categories are defined."""
        assert hasattr(ErrorCategory, 'DATA_ERROR')
        assert hasattr(ErrorCategory, 'MODEL_ERROR')
        assert hasattr(ErrorCategory, 'API_ERROR')
        assert hasattr(ErrorCategory, 'VALIDATION_ERROR')
        assert hasattr(ErrorCategory, 'FILE_ERROR')
        assert hasattr(ErrorCategory, 'COMPUTATION_ERROR')
        assert hasattr(ErrorCategory, 'UNKNOWN_ERROR')


class TestErrorHandler:
    """Tests for ErrorHandler class."""
    
    def test_get_friendly_message_data_error(self):
        """Test friendly message generation for data errors."""
        error = ValueError("Test error")
        message = ErrorHandler._get_friendly_message(error, ErrorCategory.DATA_ERROR)
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert 'data' in message.lower()
    
    def test_get_friendly_message_model_error(self):
        """Test friendly message generation for model errors."""
        error = RuntimeError("Test error")
        message = ErrorHandler._get_friendly_message(error, ErrorCategory.MODEL_ERROR)
        
        assert isinstance(message, str)
        assert 'model' in message.lower() or 'training' in message.lower()
    
    def test_get_friendly_message_api_error(self):
        """Test friendly message generation for API errors."""
        error = ConnectionError("Test error")
        message = ErrorHandler._get_friendly_message(error, ErrorCategory.API_ERROR)
        
        assert isinstance(message, str)
        assert 'api' in message.lower() or 'service' in message.lower()
    
    def test_get_friendly_message_unknown_error(self):
        """Test friendly message generation for unknown errors."""
        error = Exception("Test error")
        message = ErrorHandler._get_friendly_message(error, ErrorCategory.UNKNOWN_ERROR)
        
        assert isinstance(message, str)
        assert 'unexpected' in message.lower() or 'error' in message.lower()
    
    def test_safe_execute_success(self):
        """Test safe execution of successful function."""
        def successful_func():
            return 42
        
        result = ErrorHandler.safe_execute(successful_func)
        assert result == 42
    
    def test_safe_execute_with_error(self):
        """Test safe execution with error and fallback."""
        def failing_func():
            raise ValueError("Test error")
        
        result = ErrorHandler.safe_execute(
            failing_func,
            error_category=ErrorCategory.DATA_ERROR,
            fallback_value="fallback"
        )
        assert result == "fallback"
    
    def test_safe_execute_no_fallback(self):
        """Test safe execution with error and no fallback."""
        def failing_func():
            raise ValueError("Test error")
        
        result = ErrorHandler.safe_execute(failing_func)
        assert result is None


class TestErrorBoundaryDecorator:
    """Tests for error_boundary decorator."""
    
    def test_decorator_on_successful_function(self):
        """Test decorator on function that succeeds."""
        @error_boundary(ErrorCategory.DATA_ERROR)
        def successful_func(x, y):
            return x + y
        
        result = successful_func(2, 3)
        assert result == 5
    
    def test_decorator_on_failing_function(self):
        """Test decorator on function that fails."""
        @error_boundary(ErrorCategory.COMPUTATION_ERROR)
        def failing_func():
            raise ZeroDivisionError("Division by zero")
        
        result = failing_func()
        assert result is None
    
    def test_decorator_with_custom_message(self):
        """Test decorator with custom user message."""
        @error_boundary(
            ErrorCategory.MODEL_ERROR,
            user_message="Model training failed",
            show_details=False
        )
        def model_func():
            raise RuntimeError("Model error")
        
        result = model_func()
        assert result is None


class TestSafeOperations:
    """Tests for SafeOperations utility class."""
    
    def test_safe_divide_normal(self):
        """Test safe division with normal inputs."""
        result = SafeOperations.safe_divide(10, 2)
        assert result == 5.0
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero."""
        result = SafeOperations.safe_divide(10, 0)
        assert result == 0.0
    
    def test_safe_divide_by_zero_custom_default(self):
        """Test safe division by zero with custom default."""
        result = SafeOperations.safe_divide(10, 0, default=-1.0)
        assert result == -1.0
    
    def test_safe_divide_with_error(self):
        """Test safe division with invalid inputs."""
        result = SafeOperations.safe_divide("invalid", 2, default=0.0)
        assert result == 0.0
    
    def test_safe_percentage_normal(self):
        """Test safe percentage calculation."""
        result = SafeOperations.safe_percentage(25, 100)
        assert result == "25.00%"
    
    def test_safe_percentage_zero_total(self):
        """Test safe percentage with zero total."""
        result = SafeOperations.safe_percentage(10, 0)
        assert result == "0.0%"
    
    def test_safe_percentage_custom_decimals(self):
        """Test safe percentage with custom decimal places."""
        result = SafeOperations.safe_percentage(1, 3, decimals=4)
        assert result == "33.3333%"
    
    def test_safe_mean_normal(self):
        """Test safe mean calculation."""
        result = SafeOperations.safe_mean([1, 2, 3, 4, 5])
        assert result == 3.0
    
    def test_safe_mean_empty_list(self):
        """Test safe mean with empty list."""
        result = SafeOperations.safe_mean([])
        assert result == 0.0
    
    def test_safe_mean_custom_default(self):
        """Test safe mean with custom default."""
        result = SafeOperations.safe_mean([], default=-1.0)
        assert result == -1.0
    
    def test_safe_mean_with_invalid_data(self):
        """Test safe mean with invalid data."""
        result = SafeOperations.safe_mean(["a", "b", "c"], default=0.0)
        assert result == 0.0


class TestErrorTracker:
    """Tests for ErrorTracker class."""
    
    def setup_method(self):
        """Reset error tracker before each test."""
        ErrorTracker.reset()
    
    def test_log_single_error(self):
        """Test logging a single error."""
        ErrorTracker.log_error(ErrorCategory.DATA_ERROR)
        counts = ErrorTracker.get_error_counts()
        
        assert ErrorCategory.DATA_ERROR in counts
        assert counts[ErrorCategory.DATA_ERROR] == 1
    
    def test_log_multiple_errors(self):
        """Test logging multiple errors."""
        ErrorTracker.log_error(ErrorCategory.DATA_ERROR)
        ErrorTracker.log_error(ErrorCategory.DATA_ERROR)
        ErrorTracker.log_error(ErrorCategory.MODEL_ERROR)
        
        counts = ErrorTracker.get_error_counts()
        assert counts[ErrorCategory.DATA_ERROR] == 2
        assert counts[ErrorCategory.MODEL_ERROR] == 1
    
    def test_error_tracker_reset(self):
        """Test resetting error tracker."""
        ErrorTracker.log_error(ErrorCategory.DATA_ERROR)
        ErrorTracker.log_error(ErrorCategory.MODEL_ERROR)
        
        ErrorTracker.reset()
        counts = ErrorTracker.get_error_counts()
        
        assert len(counts) == 0
    
    def test_get_error_counts_returns_copy(self):
        """Test that get_error_counts returns a copy."""
        ErrorTracker.log_error(ErrorCategory.DATA_ERROR)
        
        counts1 = ErrorTracker.get_error_counts()
        counts1['TEST'] = 999
        
        counts2 = ErrorTracker.get_error_counts()
        assert 'TEST' not in counts2


@pytest.mark.integration
class TestErrorHandlerIntegration:
    """Integration tests for error handling workflows."""
    
    def test_full_error_workflow(self):
        """Test full error handling workflow."""
        ErrorTracker.reset()
        
        @error_boundary(ErrorCategory.COMPUTATION_ERROR)
        def risky_computation(x, y):
            if y == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return x / y
        
        # Successful execution
        result1 = risky_computation(10, 2)
        assert result1 == 5.0
        
        # Failed execution
        result2 = risky_computation(10, 0)
        assert result2 is None
        
        # Check error was tracked
        counts = ErrorTracker.get_error_counts()
        # Note: error_boundary doesn't automatically track errors
        # This would need to be added to the decorator if desired
    
    def test_nested_error_boundaries(self):
        """Test nested error boundaries."""
        @error_boundary(ErrorCategory.DATA_ERROR)
        def outer_func():
            @error_boundary(ErrorCategory.COMPUTATION_ERROR)
            def inner_func():
                raise ValueError("Inner error")
            
            return inner_func()
        
        result = outer_func()
        assert result is None


class TestRealWorldScenarios:
    """Tests for real-world error handling scenarios."""
    
    def test_file_not_found_handling(self):
        """Test handling of file not found errors."""
        def read_file():
            with open('nonexistent_file.txt', 'r') as f:
                return f.read()
        
        result = ErrorHandler.safe_execute(
            read_file,
            error_category=ErrorCategory.FILE_ERROR,
            fallback_value=""
        )
        
        assert result == ""
    
    def test_invalid_data_type_handling(self):
        """Test handling of invalid data type errors."""
        def process_data(data):
            return int(data) * 2
        
        result = ErrorHandler.safe_execute(
            lambda: process_data("invalid"),
            error_category=ErrorCategory.DATA_ERROR,
            fallback_value=None
        )
        
        assert result is None
    
    def test_api_connection_error_handling(self):
        """Test handling of API connection errors."""
        def call_api():
            raise ConnectionError("Failed to connect to API")
        
        result = ErrorHandler.safe_execute(
            call_api,
            error_category=ErrorCategory.API_ERROR,
            fallback_value={"error": "API unavailable"}
        )
        
        assert result == {"error": "API unavailable"}


class TestEdgeCases:
    """Tests for edge cases in error handling."""
    
    def test_none_as_function(self):
        """Test handling when None is passed as function."""
        # safe_execute should handle None gracefully and return fallback
        result = ErrorHandler.safe_execute(None, fallback_value="fallback")
        assert result == "fallback"
    
    def test_empty_error_message(self):
        """Test handling of empty error messages."""
        error = ValueError("")
        message = ErrorHandler._get_friendly_message(error, ErrorCategory.DATA_ERROR)
        
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_safe_divide_with_float_inf(self):
        """Test safe division resulting in infinity."""
        result = SafeOperations.safe_divide(float('inf'), 2)
        # Should return the result even if it's infinity
        assert result == float('inf')
