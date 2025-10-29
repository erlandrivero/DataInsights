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
    
    @staticmethod
    def detect_text_column(df: pd.DataFrame) -> Optional[str]:
        """
        Detect text/description column for text mining.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Most likely text column name
        """
        patterns = [
            'description', 'text', 'comment', 'review', 'feedback', 
            'message', 'content', 'body', 'notes', 'summary'
        ]
        
        # Check for pattern matches
        for col in df.columns:
            if df[col].dtype == 'object':  # Text columns
                col_lower = col.lower().replace('_', '').replace(' ', '')
                for pattern in patterns:
                    if pattern in col_lower:
                        return col
        
        # Find column with longest average text length
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_cols}
            return max(avg_lengths, key=avg_lengths.get)
        
        return df.columns[0] if len(df.columns) > 0 else None
    
    @staticmethod
    def get_ab_testing_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for A/B Testing.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'group' and 'metric' suggestions
        """
        # Detect group column - prefer 2-3 unique values with relevant keywords
        keyword_matches = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['group', 'variant', 'test', 'ab', 'treatment', 'control', 'cohort', 'segment'])]
        suitable_group_cols = [col for col in keyword_matches if 2 <= df[col].nunique() <= 3]
        
        # If no keyword matches, look for ANY columns with 2-3 unique values
        if not suitable_group_cols:
            suitable_group_cols = [col for col in df.columns if 2 <= df[col].nunique() <= 3]
        
        # Exclude clearly non-group columns
        suitable_group_cols = [col for col in suitable_group_cols if not any(exclude in col.lower() 
                              for exclude in ['id', 'invoice', 'order', 'date', 'time', 'price', 'quantity', 'amount'])]
        
        group_col = suitable_group_cols[0] if suitable_group_cols else df.columns[0]
        
        # Detect metric column - prefer numeric with relevant keywords
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        metric_keywords = [col for col in numeric_cols if any(keyword in col.lower() 
                          for keyword in ['conversion', 'revenue', 'sales', 'clicks', 'ctr', 'rate', 'value', 'metric', 'score', 'amount'])]
        
        # Exclude ID columns
        metric_keywords = [col for col in metric_keywords if not any(exclude in col.lower() 
                          for exclude in ['id', 'invoice', 'order_id', 'transaction_id', 'number'])]
        
        if metric_keywords:
            metric_col = metric_keywords[0]
        elif numeric_cols:
            clean_numeric = [col for col in numeric_cols if not any(exclude in col.lower() 
                            for exclude in ['id', 'invoice', 'number'])]
            metric_col = clean_numeric[0] if clean_numeric else numeric_cols[0]
        else:
            metric_col = df.columns[0]
        
        return {
            'group': group_col,
            'metric': metric_col
        }
    
    @staticmethod
    def get_cohort_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Cohort Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'user_id', 'cohort_date', and 'activity_date' suggestions
        """
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        
        # Detect User ID
        id_suggestions = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['id', 'user', 'customer', 'client'])]
        user_col = id_suggestions[0] if id_suggestions else df.columns[0]
        
        # Detect Cohort Date (signup, registration, first purchase)
        cohort_suggestions = [col for col in date_cols if any(keyword in col.lower() 
                             for keyword in ['signup', 'register', 'created', 'first', 'join', 'date', 'invoice', 'order'])]
        cohort_suggestions = [col for col in cohort_suggestions if not any(exclude in col.lower() 
                             for exclude in ['description', 'name', 'country', 'status'])]
        cohort_col = cohort_suggestions[0] if cohort_suggestions else (date_cols[0] if date_cols else df.columns[0])
        
        # Detect Activity Date
        activity_suggestions = [col for col in date_cols if any(keyword in col.lower() 
                               for keyword in ['activity', 'purchase', 'order', 'transaction', 'date', 'time', 'invoice'])]
        activity_suggestions = [col for col in activity_suggestions if not any(exclude in col.lower() 
                               for exclude in ['description', 'name', 'country', 'status'])]
        activity_col = activity_suggestions[0] if activity_suggestions else (date_cols[1] if len(date_cols) > 1 else (date_cols[0] if date_cols else df.columns[0]))
        
        return {
            'user_id': user_col,
            'cohort_date': cohort_col,
            'activity_date': activity_col
        }
    
    @staticmethod
    def get_recommendation_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Recommendation Systems.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'user', 'item', and 'rating' suggestions
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Detect user column
        user_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                           for keyword in ['user', 'customer', 'userid', 'customerid', 'client'])]
        user_suggestions = [col for col in user_suggestions if not any(exclude in col.lower() 
                           for exclude in ['item', 'product', 'movie', 'name', 'description'])]
        user_col = user_suggestions[0] if user_suggestions else all_cols[0]
        
        # Detect item column
        item_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                           for keyword in ['item', 'product', 'movie', 'title', 'stock', 'sku', 'article'])]
        item_suggestions = [col for col in item_suggestions if not any(exclude in col.lower() 
                           for exclude in ['user', 'customer', 'client', 'rating', 'score'])]
        item_col = item_suggestions[0] if item_suggestions else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
        
        # Detect rating column
        rating_suggestions = [col for col in numeric_cols if any(keyword in col.lower() 
                             for keyword in ['rating', 'score', 'stars', 'review'])]
        if not rating_suggestions:
            rating_suggestions = [col for col in numeric_cols if not any(exclude in col.lower() 
                                 for exclude in ['id', 'index', 'count', 'number'])]
        rating_col = rating_suggestions[0] if rating_suggestions else (numeric_cols[0] if numeric_cols else all_cols[0])
        
        return {
            'user': user_col,
            'item': item_col,
            'rating': rating_col
        }
    
    @staticmethod
    def get_geospatial_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Geospatial Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'latitude' and 'longitude' suggestions
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Detect latitude
        lat_suggestions = [col for col in numeric_cols if any(keyword in col.lower() 
                          for keyword in ['lat', 'latitude', 'y', 'coord'])]
        lat_suggestions = [col for col in lat_suggestions if not any(exclude in col.lower() 
                          for exclude in ['long', 'lon', 'price', 'quantity', 'amount', 'id', 'index'])]
        lat_col = lat_suggestions[0] if lat_suggestions else (numeric_cols[0] if numeric_cols else df.columns[0])
        
        # Detect longitude
        lon_suggestions = [col for col in numeric_cols if any(keyword in col.lower() 
                          for keyword in ['lon', 'long', 'longitude', 'x', 'coord'])]
        lon_suggestions = [col for col in lon_suggestions if not any(exclude in col.lower() 
                          for exclude in ['lat', 'latitude', 'price', 'quantity', 'amount', 'id', 'index'])]
        lon_col = lon_suggestions[0] if lon_suggestions else (numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else df.columns[0]))
        
        return {
            'latitude': lat_col,
            'longitude': lon_col
        }
    
    @staticmethod
    def get_survival_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Survival Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'time', 'event', and 'group' suggestions
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Detect time/duration column
        time_suggestions = [col for col in numeric_cols if any(keyword in col.lower() 
                           for keyword in ['time', 'duration', 'days', 'months', 'tenure', 'period', 'lifetime'])]
        time_suggestions = [col for col in time_suggestions if not any(exclude in col.lower() 
                           for exclude in ['id', 'index', 'price', 'amount', 'quantity', 'rate'])]
        time_col = time_suggestions[0] if time_suggestions else (numeric_cols[0] if numeric_cols else all_cols[0])
        
        # Detect event column
        event_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                            for keyword in ['event', 'churn', 'status', 'outcome', 'censored', 'failed', 'died', 'left'])]
        event_suggestions = [col for col in event_suggestions if not any(exclude in col.lower() 
                            for exclude in ['id', 'date', 'time', 'name', 'description'])]
        event_col = event_suggestions[0] if event_suggestions else all_cols[0]
        
        # Detect group column (optional) - prefer 2-5 unique values
        group_suggestions = [col for col in all_cols if 2 <= df[col].nunique() <= 5]
        group_keywords = [col for col in all_cols if any(keyword in col.lower() 
                         for keyword in ['group', 'type', 'category', 'segment', 'plan', 'tier'])]
        combined_groups = list(dict.fromkeys(group_keywords + group_suggestions))
        
        return {
            'time': time_col,
            'event': event_col,
            'group': combined_groups[0] if combined_groups else None
        }
    
    @staticmethod
    def get_network_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Network Analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'source' and 'target' suggestions
        """
        all_cols = df.columns.tolist()
        
        # Detect source column
        source_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                             for keyword in ['from', 'source', 'sender', 'user', 'node1', 'origin', 'start', 'follower'])]
        source_suggestions = [col for col in source_suggestions if not any(exclude in col.lower() 
                             for exclude in ['to', 'target', 'receiver', 'destination', 'node2', 'end'])]
        source_col = source_suggestions[0] if source_suggestions else all_cols[0]
        
        # Detect target column
        target_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                             for keyword in ['to', 'target', 'receiver', 'friend', 'node2', 'destination', 'end', 'following'])]
        target_suggestions = [col for col in target_suggestions if not any(exclude in col.lower() 
                             for exclude in ['from', 'source', 'sender', 'origin', 'node1', 'start'])]
        target_col = target_suggestions[0] if target_suggestions else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
        
        return {
            'source': source_col,
            'target': target_col
        }
    
    @staticmethod
    def get_churn_column_suggestions(df: pd.DataFrame) -> dict:
        """
        Get suggested columns for Churn Prediction.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with 'customer_id', 'date', 'value', and 'churn' suggestions
        """
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Detect customer ID column
        customer_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                               for keyword in ['customer', 'user', 'client', 'account', 'member', 'subscriber'])]
        customer_suggestions = [col for col in customer_suggestions if any(id_word in col.lower() 
                               for id_word in ['id', 'no', 'num', 'code'])]
        customer_col = customer_suggestions[0] if customer_suggestions else all_cols[0]
        
        # Detect date column
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        date_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                           for keyword in ['date', 'time', 'timestamp', 'day', 'month', 'year', 'transaction', 'order', 'purchase'])]
        date_suggestions = date_cols + [col for col in date_suggestions if col not in date_cols]
        date_col = date_suggestions[0] if date_suggestions else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
        
        # Detect value/amount column
        value_suggestions = [col for col in numeric_cols if any(keyword in col.lower() 
                            for keyword in ['amount', 'value', 'price', 'total', 'revenue', 'spend', 'payment', 'cost'])]
        value_suggestions = [col for col in value_suggestions if not any(exclude in col.lower() 
                            for exclude in ['id', 'code', 'count', 'quantity', 'qty', 'index'])]
        value_col = value_suggestions[0] if value_suggestions else None
        
        # Detect churn label column
        churn_suggestions = [col for col in all_cols if any(keyword in col.lower() 
                            for keyword in ['churn', 'status', 'active', 'label', 'target', 'outcome', 'left', 'cancelled'])]
        churn_col = churn_suggestions[0] if churn_suggestions else None
        
        return {
            'customer_id': customer_col,
            'date': date_col,
            'value': value_col,
            'churn': churn_col
        }
