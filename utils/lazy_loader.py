"""
Lazy Module Loader for memory-efficient module loading.
Loads modules only when needed and unloads after use.
"""
import importlib
import sys
import gc
import streamlit as st
from typing import Any, Optional


class LazyModuleLoader:
    """
    Lazy load modules only when needed and unload after use.
    Reduces memory footprint by ensuring only necessary modules are in memory.
    """
    
    @staticmethod
    def load_module(module_name: str, force_reload: bool = False):
        """
        Load a module dynamically.
        
        Args:
            module_name: Full module path (e.g., 'utils.ml_training')
            force_reload: Force reload even if already imported
        
        Returns:
            Loaded module
        """
        if force_reload and module_name in sys.modules:
            del sys.modules[module_name]
            gc.collect()
        
        try:
            module = importlib.import_module(module_name)
            return module
        except Exception as e:
            st.error(f"Failed to load module {module_name}: {str(e)}")
            return None
    
    @staticmethod
    def unload_module(module_name: str):
        """
        Unload a module from memory.
        
        Args:
            module_name: Full module path to unload
        """
        if module_name in sys.modules:
            del sys.modules[module_name]
            gc.collect()
    
    @staticmethod
    def load_and_execute(module_name: str, class_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        Load module, execute method, then unload.
        
        Args:
            module_name: Module to load
            class_name: Class name within module
            method_name: Method to execute
            *args, **kwargs: Arguments to pass to method
        
        Returns:
            Result of method execution
        """
        try:
            # Load module
            module = LazyModuleLoader.load_module(module_name, force_reload=True)
            
            if module is None:
                return None
            
            # Get class
            cls = getattr(module, class_name)
            
            # Create instance
            instance = cls(*args, **kwargs)
            
            # Execute method
            method = getattr(instance, method_name)
            result = method()
            
            # Clean up
            del instance
            del method
            del cls
            LazyModuleLoader.unload_module(module_name)
            gc.collect()
            
            return result
            
        except Exception as e:
            st.error(f"Error executing {module_name}.{class_name}.{method_name}: {str(e)}")
            return None


class SequentialExecutor:
    """
    Execute multiple operations sequentially to prevent memory overload.
    Ensures only one heavy operation runs at a time with garbage collection between steps.
    """
    
    def __init__(self, operations: list):
        """
        Initialize with list of operations.
        
        Args:
            operations: List of dicts with keys: 'name', 'function', 'args', 'kwargs'
        """
        self.operations = operations
        self.results = {}
    
    def execute_all(self, progress_bar: Optional[st.delta_generator.DeltaGenerator] = None):
        """
        Execute all operations sequentially.
        
        Args:
            progress_bar: Optional Streamlit progress bar
        
        Returns:
            Dict of results keyed by operation name
        """
        total = len(self.operations)
        
        for i, operation in enumerate(self.operations):
            name = operation['name']
            func = operation['function']
            args = operation.get('args', [])
            kwargs = operation.get('kwargs', {})
            
            # Update progress
            if progress_bar:
                progress_bar.progress((i + 1) / total, text=f"Executing: {name}")
            
            try:
                # Execute operation
                result = func(*args, **kwargs)
                self.results[name] = result
                
                # Force garbage collection between operations
                gc.collect()
                
            except Exception as e:
                st.error(f"Error in {name}: {str(e)}")
                self.results[name] = None
        
        return self.results
