"""
Dataset Tracking Utility

Helps track dataset changes and invalidate stale AI recommendations.
"""

import hashlib
import pandas as pd
from typing import Optional


class DatasetTracker:
    """Track dataset identity and detect changes."""
    
    @staticmethod
    def generate_dataset_id(df: pd.DataFrame, dataset_name: str = "unknown") -> str:
        """Generate a unique ID for a dataset based on its characteristics.
        
        Args:
            df: The DataFrame to track
            dataset_name: Optional name/source of the dataset
            
        Returns:
            A unique string identifier for the dataset
        """
        # Create a fingerprint based on dataset characteristics
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        columns_str = ",".join(sorted(df.columns.tolist()))
        dtypes_str = ",".join([str(dt) for dt in df.dtypes])
        
        # Create a hash from the combination
        fingerprint = f"{dataset_name}|{shape_str}|{columns_str}|{dtypes_str}"
        dataset_id = hashlib.md5(fingerprint.encode()).hexdigest()[:16]
        
        return f"{dataset_name}_{dataset_id}"
    
    @staticmethod
    def check_dataset_changed(current_df: pd.DataFrame, 
                             current_name: str,
                             stored_dataset_id: Optional[str]) -> bool:
        """Check if the current dataset differs from the stored one.
        
        Args:
            current_df: Current DataFrame
            current_name: Current dataset name
            stored_dataset_id: Previously stored dataset ID
            
        Returns:
            True if dataset has changed, False otherwise
        """
        if stored_dataset_id is None:
            return True
        
        current_id = DatasetTracker.generate_dataset_id(current_df, current_name)
        return current_id != stored_dataset_id
    
    @staticmethod
    def clear_module_ai_cache(st_session_state, module_name: str):
        """Clear AI recommendations for a specific module.
        
        Args:
            st_session_state: Streamlit session state object
            module_name: Name of the module (e.g., 'anomaly', 'network', 'churn')
        """
        cache_key = f"{module_name}_ai_recommendations"
        if cache_key in st_session_state:
            del st_session_state[cache_key]
    
    @staticmethod
    def clear_all_ai_cache(st_session_state):
        """Clear all AI recommendations from session state.
        
        Args:
            st_session_state: Streamlit session state object
        """
        # List of all module AI cache keys
        ai_cache_keys = [
            'anomaly_ai_recommendations',
            'network_ai_recommendations',
            'churn_ai_recommendations',
            'survival_ai_recommendations',
            'recommendation_ai_recommendations',
            'classification_ai_recommendations',
            'regression_ai_recommendations',
            'clustering_ai_recommendations',
            'timeseries_ai_recommendations',
            'rfm_ai_recommendations',
            'mba_ai_recommendations',
            'textmining_ai_recommendations',
            'abtesting_ai_recommendations',
            'datacleaning_ai_recommendations',
            'geo_ai_analysis',
            'rec_ai_analysis',
            'cohort_ai_analysis'
        ]
        
        for key in ai_cache_keys:
            if key in st_session_state:
                del st_session_state[key]
