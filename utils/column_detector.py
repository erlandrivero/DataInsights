"""
Smart column detection utilities for data analysis modules.
"""

import pandas as pd
from typing import Optional, List


class ColumnDetector:
    """Detects and suggests appropriate columns for different analysis types."""
    
    @staticmethod
    def detect_transaction_id_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect transaction/order ID column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely transaction ID column name, or first column if no match
        """
        # Common transaction ID column name patterns (case-insensitive)
        patterns = [
            'invoice', 'transaction', 'order', 'receipt', 'ticket', 
            'trans_id', 'order_id', 'invoice_no', 'invoiceno'
        ]
        
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # Default to first column if no match
        return df.columns[0] if len(df.columns) > 0 else None
    
    @staticmethod
    def detect_item_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect item/product description column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely item column name
        """
        # Common item column name patterns
        patterns = [
            'description', 'item', 'product', 'sku', 'name', 
            'itemdescription', 'productname', 'item_name'
        ]
        
        # First try pattern matching
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # Look for text columns with reasonable variety
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                # Good item columns have many unique values but not 100% unique
                if 0.01 < unique_ratio < 0.9:
                    return col
        
        # Default to second column or first object column
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        return object_cols[0] if object_cols else df.columns[1] if len(df.columns) > 1 else None
    
    @staticmethod
    def detect_customer_id_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect customer ID column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely customer ID column name
        """
        patterns = [
            'customer', 'client', 'user', 'member', 'account',
            'customerid', 'customer_id', 'userid', 'user_id'
        ]
        
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # Look for columns with moderate cardinality (not too few, not too many)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if 0.001 < unique_ratio < 0.5:  # 0.1% to 50% unique
                return col
        
        return df.columns[0] if len(df.columns) > 0 else None
    
    @staticmethod
    def detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect date/datetime column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely date column name
        """
        # First look for datetime types
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            return datetime_cols[0]
        
        # Then look for date-like column names
        patterns = [
            'date', 'time', 'timestamp', 'created', 'order_date',
            'invoice_date', 'purchase', 'transaction_date'
        ]
        
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # Try to find parseable date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100))
                    return col
                except:
                    continue
        
        return None
    
    @staticmethod
    def detect_amount_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect amount/price/revenue column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely amount column name
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Look for amount-like column names
        patterns = [
            'price', 'amount', 'total', 'revenue', 'value', 'cost',
            'unitprice', 'unit_price', 'sales', 'payment'
        ]
        
        for col in numeric_cols:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # Return first numeric column with reasonable positive values
        for col in numeric_cols:
            if df[col].min() >= 0 and df[col].max() > 0:
                return col
        
        # Default to first numeric column
        return numeric_cols[0]
    
    @staticmethod
    def get_mba_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Market Basket Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'transaction_id' and 'item' suggestions
        """
        return {
            'transaction_id': ColumnDetector.detect_transaction_id_column(df),
            'item': ColumnDetector.detect_item_column(df)
        }
    
    @staticmethod
    def get_rfm_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for RFM Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'customer_id', 'date', and 'amount' suggestions
        """
        return {
            'customer_id': ColumnDetector.detect_customer_id_column(df),
            'date': ColumnDetector.detect_date_column(df),
            'amount': ColumnDetector.detect_amount_column(df)
        }
    
    @staticmethod
    def get_time_series_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Time Series Forecasting.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'date' and 'value' suggestions
        """
        return {
            'date': ColumnDetector.detect_date_column(df),
            'value': ColumnDetector.detect_amount_column(df)  # Reuse amount detection for numeric values
        }
    
    @staticmethod
    def validate_mba_suitability(df: pd.DataFrame) -> dict:
        """
        Validate if dataset is suitable for Market Basket Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'suitable' (bool), 'warnings' (list), 'recommendations' (list)
        """
        warnings = []
        recommendations = []
        suitable = True
        
        # Check if we can find transaction and item columns
        suggestions = ColumnDetector.get_mba_column_suggestions(df)
        
        if not suggestions['transaction_id']:
            warnings.append("⚠️ No clear transaction ID column detected")
            recommendations.append("MBA requires a column that groups items into transactions (e.g., OrderID, InvoiceNo)")
            suitable = False
        
        if not suggestions['item']:
            warnings.append("⚠️ No clear item/product column detected")
            recommendations.append("MBA requires a column with product names or descriptions")
            suitable = False
        
        # Check for sufficient data
        if len(df) < 100:
            warnings.append("⚠️ Dataset has fewer than 100 rows")
            recommendations.append("MBA works best with at least 1,000+ transactions for meaningful patterns")
        
        # Check for item variety
        if suggestions['item']:
            unique_items = df[suggestions['item']].nunique()
            if unique_items < 5:
                warnings.append(f"⚠️ Only {unique_items} unique items detected")
                recommendations.append("MBA requires multiple items to find associations (ideally 20+ items)")
                suitable = False
        
        # Check for transaction variety
        if suggestions['transaction_id']:
            unique_transactions = df[suggestions['transaction_id']].nunique()
            if unique_transactions < 50:
                warnings.append(f"⚠️ Only {unique_transactions} unique transactions detected")
                recommendations.append("MBA requires many transactions to find patterns (ideally 500+ transactions)")
        
        return {
            'suitable': suitable,
            'warnings': warnings,
            'recommendations': recommendations,
            'confidence': 'high' if suitable and len(warnings) == 0 else 'medium' if suitable else 'low'
        }
    
    @staticmethod
    def validate_rfm_suitability(df: pd.DataFrame) -> dict:
        """
        Validate if dataset is suitable for RFM Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'suitable' (bool), 'warnings' (list), 'recommendations' (list)
        """
        warnings = []
        recommendations = []
        suitable = True
        
        # Check if we can find required columns
        suggestions = ColumnDetector.get_rfm_column_suggestions(df)
        
        if not suggestions['customer_id']:
            warnings.append("⚠️ No clear customer ID column detected")
            recommendations.append("RFM requires a column that identifies unique customers")
            suitable = False
        
        if not suggestions['date']:
            warnings.append("⚠️ No date column detected")
            recommendations.append("RFM requires transaction dates to calculate Recency")
            suitable = False
        
        if not suggestions['amount']:
            warnings.append("⚠️ No numeric amount/price column detected")
            recommendations.append("RFM requires transaction amounts to calculate Monetary value")
            suitable = False
        
        # Check for sufficient customers
        if suggestions['customer_id']:
            unique_customers = df[suggestions['customer_id']].nunique()
            if unique_customers < 10:
                warnings.append(f"⚠️ Only {unique_customers} unique customers detected")
                recommendations.append("RFM works best with 100+ customers for meaningful segmentation")
                suitable = False
            elif unique_customers < 100:
                warnings.append(f"⚠️ Only {unique_customers} customers - results may not be very meaningful")
                recommendations.append("For better insights, RFM typically needs 500+ customers")
        
        # Check for sufficient transactions
        if len(df) < 100:
            warnings.append("⚠️ Dataset has fewer than 100 transactions")
            recommendations.append("RFM requires transaction history to calculate Frequency accurately")
        
        # Check date range
        if suggestions['date']:
            try:
                dates = pd.to_datetime(df[suggestions['date']])
                date_range_days = (dates.max() - dates.min()).days
                if date_range_days < 30:
                    warnings.append(f"⚠️ Transaction history spans only {date_range_days} days")
                    recommendations.append("RFM works best with 6+ months of transaction history")
                elif date_range_days < 90:
                    warnings.append(f"⚠️ Transaction history spans only {date_range_days} days")
                    recommendations.append("For better Recency analysis, 6+ months of data is recommended")
            except:
                pass
        
        return {
            'suitable': suitable,
            'warnings': warnings,
            'recommendations': recommendations,
            'confidence': 'high' if suitable and len(warnings) == 0 else 'medium' if suitable else 'low'
        }
    
    @staticmethod
    def validate_time_series_suitability(df: pd.DataFrame) -> dict:
        """
        Validate if dataset is suitable for Time Series Forecasting.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'suitable' (bool), 'warnings' (list), 'recommendations' (list)
        """
        warnings = []
        recommendations = []
        suitable = True
        
        # Check if we can find required columns
        suggestions = ColumnDetector.get_time_series_column_suggestions(df)
        
        if not suggestions['date']:
            warnings.append("⚠️ No date/time column detected")
            recommendations.append("Time series requires a date or datetime column")
            suitable = False
        
        if not suggestions['value']:
            warnings.append("⚠️ No numeric value column detected")
            recommendations.append("Time series requires a numeric column to forecast")
            suitable = False
        
        # Check for sufficient data points
        if len(df) < 30:
            warnings.append(f"⚠️ Only {len(df)} data points - very limited for forecasting")
            recommendations.append("Time series forecasting works best with 100+ data points")
            suitable = False
        elif len(df) < 50:
            warnings.append(f"⚠️ Only {len(df)} data points - limited forecast accuracy")
            recommendations.append("For better forecasts, 100+ data points recommended")
        
        # Check date range if date column exists
        if suggestions['date']:
            try:
                dates = pd.to_datetime(df[suggestions['date']])
                date_range_days = (dates.max() - dates.min()).days
                if date_range_days < 30:
                    warnings.append(f"⚠️ Time series spans only {date_range_days} days")
                    recommendations.append("Longer time periods (6+ months) produce better forecasts")
                elif date_range_days < 90:
                    warnings.append(f"⚠️ Time series spans {date_range_days} days")
                    recommendations.append("6+ months of data recommended for seasonal patterns")
            except:
                pass
        
        return {
            'suitable': suitable,
            'warnings': warnings,
            'recommendations': recommendations,
            'confidence': 'high' if suitable and len(warnings) == 0 else 'medium' if suitable else 'low'
        }
