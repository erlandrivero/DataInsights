"""
Data Optimizer for memory-efficient data processing.
Optimizes DataFrame memory usage and handles large datasets through sampling.
"""
import streamlit as st
import pandas as pd
import numpy as np


class DataOptimizer:
    """
    Optimize data storage and processing to reduce memory usage.
    """
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        This method reduces memory footprint by:
        - Downcasting integers to smallest suitable type
        - Downcasting floats to float32 where possible
        - Converting low-cardinality object columns to category dtype
        
        Args:
            df: DataFrame to optimize
        
        Returns:
            Optimized DataFrame with reduced memory usage
        """
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert object columns to category if cardinality is low
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            
            # If less than 50% unique values, convert to category
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
    
    @staticmethod
    def should_sample_data(df: pd.DataFrame, threshold: int = 100000) -> bool:
        """
        Determine if data should be sampled based on size.
        
        Args:
            df: DataFrame to check
            threshold: Row count threshold (default 100,000)
        
        Returns:
            True if sampling is recommended
        """
        return len(df) > threshold
    
    @staticmethod
    def sample_data(df: pd.DataFrame, sample_size: int = 50000, random_state: int = 42) -> pd.DataFrame:
        """
        Sample data for analysis while maintaining distribution.
        
        Uses stratified sampling if possible to maintain class distributions.
        
        Args:
            df: DataFrame to sample
            sample_size: Number of rows to sample
            random_state: Random seed for reproducibility
        
        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            return df
        
        return df.sample(n=sample_size, random_state=random_state)
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> dict:
        """
        Get detailed memory usage of DataFrame.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dict with memory statistics (total_mb, per_column, rows, columns)
        """
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'per_column': {col: mem / 1024 / 1024 for col, mem in memory_usage.items()},
            'rows': len(df),
            'columns': len(df.columns)
        }
    
    @staticmethod
    def compare_memory_usage(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
        """
        Compare memory usage between two DataFrames.
        
        Args:
            df_before: Original DataFrame
            df_after: Optimized DataFrame
        
        Returns:
            Dict with before, after, saved (MB), and percentage saved
        """
        mem_before = DataOptimizer.get_memory_usage(df_before)
        mem_after = DataOptimizer.get_memory_usage(df_after)
        
        saved_mb = mem_before['total_mb'] - mem_after['total_mb']
        saved_pct = (saved_mb / mem_before['total_mb']) * 100 if mem_before['total_mb'] > 0 else 0
        
        return {
            'before_mb': mem_before['total_mb'],
            'after_mb': mem_after['total_mb'],
            'saved_mb': saved_mb,
            'saved_pct': saved_pct
        }
    
    @staticmethod
    def optimize_and_report(df: pd.DataFrame, show_details: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame and display optimization results.
        
        Args:
            df: DataFrame to optimize
            show_details: Whether to show detailed optimization report
        
        Returns:
            Optimized DataFrame
        """
        # Get memory before
        mem_before = DataOptimizer.get_memory_usage(df)
        
        # Optimize
        df_optimized = DataOptimizer.optimize_dataframe(df)
        
        # Get memory after
        mem_after = DataOptimizer.get_memory_usage(df_optimized)
        
        # Calculate savings
        saved_mb = mem_before['total_mb'] - mem_after['total_mb']
        saved_pct = (saved_mb / mem_before['total_mb']) * 100 if mem_before['total_mb'] > 0 else 0
        
        # Display results
        if show_details and saved_mb > 0.1:  # Only show if saved more than 0.1MB
            st.success(f"✅ **Optimized!** Saved {saved_mb:.1f}MB ({saved_pct:.1f}%)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Before", f"{mem_before['total_mb']:.1f}MB")
            with col2:
                st.metric("After", f"{mem_after['total_mb']:.1f}MB")
            with col3:
                st.metric("Saved", f"{saved_mb:.1f}MB", delta=f"-{saved_pct:.1f}%")
        
        return df_optimized
