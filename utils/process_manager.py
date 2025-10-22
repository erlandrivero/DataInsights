"""
Process Manager for long-running operations in Streamlit.
Handles progress tracking, interruption, and recovery.
"""
import streamlit as st
from typing import Callable, Any, Dict, Optional
from datetime import datetime
import traceback
import time


class ProcessManager:
    """Manages long-running processes with progress tracking and interruption handling."""
    
    def __init__(self, process_name: str):
        """
        Initialize process manager.
        
        Args:
            process_name: Unique name for this process
        """
        self.process_name = process_name
        self.state_key = f"process_{process_name}"
        self.lock_key = f"lock_{process_name}"
        self.progress_key = f"progress_{process_name}"
        
    def is_locked(self) -> bool:
        """Check if process is currently running."""
        return st.session_state.get(self.lock_key, False)
    
    def lock(self):
        """Lock the process to prevent navigation."""
        st.session_state[self.lock_key] = True
        st.session_state['global_process_running'] = True
        
    def unlock(self):
        """Unlock the process."""
        st.session_state[self.lock_key] = False
        st.session_state['global_process_running'] = False
        
    def save_checkpoint(self, data: Dict[str, Any]):
        """Save intermediate results."""
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = {}
        st.session_state[self.state_key].update(data)
        st.session_state[self.state_key]['last_checkpoint'] = datetime.now().isoformat()
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load saved checkpoint."""
        return st.session_state.get(self.state_key, None)
    
    def clear_checkpoint(self):
        """Clear saved checkpoint."""
        if self.state_key in st.session_state:
            del st.session_state[self.state_key]
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress bar."""
        progress = current / total if total > 0 else 0
        st.session_state[self.progress_key] = {
            'current': current,
            'total': total,
            'progress': progress,
            'message': message,
            'updated_at': datetime.now().isoformat()
        }
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress."""
        return st.session_state.get(self.progress_key, None)
    
    def run_with_progress(
        self,
        func: Callable,
        total_steps: int,
        step_callback: Optional[Callable] = None,
        checkpoint_interval: int = 10,
        **kwargs
    ) -> Any:
        """
        Run a function with progress tracking and checkpoint saving.
        
        Args:
            func: Function to execute
            total_steps: Total number of steps
            step_callback: Optional callback for each step
            checkpoint_interval: Save checkpoint every N steps
            **kwargs: Arguments to pass to func
            
        Returns:
            Result from func
        """
        self.lock()
        
        try:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for step in range(total_steps):
                # Check if user wants to cancel
                if not self.is_locked():
                    status_text.warning("⚠️ Process was interrupted")
                    return None
                
                # Update progress
                progress = (step + 1) / total_steps
                progress_bar.progress(progress)
                status_text.text(f"Processing step {step + 1}/{total_steps}...")
                
                # Execute step
                if step_callback:
                    result = step_callback(step, **kwargs)
                    results.append(result)
                
                # Save checkpoint periodically
                if (step + 1) % checkpoint_interval == 0:
                    self.save_checkpoint({
                        'step': step + 1,
                        'results': results,
                        'total_steps': total_steps
                    })
                
            # Execute main function
            final_result = func(results=results, **kwargs)
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            return final_result
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.code(traceback.format_exc())
            return None
            
        finally:
            self.unlock()
    
    def run_chunked(
        self,
        func: Callable,
        data: Any,
        chunk_size: int = 100,
        **kwargs
    ) -> Any:
        """
        Run a function on data in chunks with progress tracking.
        
        Args:
            func: Function to execute on each chunk
            data: Data to process (must support len and slicing)
            chunk_size: Size of each chunk
            **kwargs: Arguments to pass to func
            
        Returns:
            Combined results
        """
        self.lock()
        
        try:
            total_items = len(data)
            total_chunks = (total_items + chunk_size - 1) // chunk_size
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for chunk_idx in range(total_chunks):
                # Check if user wants to cancel
                if not self.is_locked():
                    status_text.warning("⚠️ Process was interrupted")
                    self.save_checkpoint({
                        'chunk_idx': chunk_idx,
                        'results': results,
                        'total_chunks': total_chunks
                    })
                    return None
                
                # Get chunk
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_items)
                chunk = data[start_idx:end_idx]
                
                # Update progress
                progress = (chunk_idx + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({end_idx}/{total_items} items)...")
                
                # Process chunk
                chunk_result = func(chunk, **kwargs)
                results.append(chunk_result)
                
                # Save checkpoint
                if (chunk_idx + 1) % 5 == 0:  # Every 5 chunks
                    self.save_checkpoint({
                        'chunk_idx': chunk_idx + 1,
                        'results': results,
                        'total_chunks': total_chunks
                    })
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            return results
            
        except Exception as e:
            st.error(f"Error during chunked processing: {str(e)}")
            st.code(traceback.format_exc())
            return None
            
        finally:
            self.unlock()
    
    def run_safe(
        self,
        func: Callable,
        show_progress: bool = True,
        progress_message: str = "Processing...",
        **kwargs
    ) -> Any:
        """
        Run a function safely with error handling and locking.
        
        Args:
            func: Function to execute
            show_progress: Whether to show spinner
            progress_message: Message to display during processing
            **kwargs: Arguments to pass to func
            
        Returns:
            Result from func or None on error
        """
        self.lock()
        
        try:
            if show_progress:
                with st.spinner(progress_message):
                    result = func(**kwargs)
            else:
                result = func(**kwargs)
            
            return result
            
        except Exception as e:
            st.error(f"Error during {self.process_name}: {str(e)}")
            st.code(traceback.format_exc())
            return None
            
        finally:
            self.unlock()


class NavigationGuard:
    """Prevents navigation during critical operations."""
    
    @staticmethod
    def is_any_process_running() -> bool:
        """Check if any process is currently running."""
        return st.session_state.get('global_process_running', False)
    
    @staticmethod
    def show_navigation_warning():
        """Display warning when process is running."""
        if NavigationGuard.is_any_process_running():
            st.warning("""
            ⚠️ **Process Running**
            
            A process is currently running. Navigating away will interrupt it.
            Please wait for the process to complete.
            """)
            return True
        return False
    
    @staticmethod
    def create_cancel_button(process_manager: ProcessManager) -> bool:
        """
        Create a cancel button for the current process.
        
        Args:
            process_manager: Process manager instance
            
        Returns:
            True if user clicked cancel
        """
        if process_manager.is_locked():
            if st.button("🛑 Cancel Process", type="secondary", use_container_width=True):
                process_manager.unlock()
                st.warning("Process cancelled by user")
                return True
        return False
    
    @staticmethod
    def ask_confirmation(action_name: str) -> bool:
        """
        Ask user confirmation before starting a long process.
        
        Args:
            action_name: Name of the action
            
        Returns:
            True if user confirms
        """
        st.info(f"""
        ⏳ **About to start: {action_name}**
        
        This may take a few minutes. Please do not navigate away during processing.
        Results will be saved automatically.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Start Process", type="primary", use_container_width=True):
                return True
        
        with col2:
            if st.button("❌ Cancel", type="secondary", use_container_width=True):
                st.info("Operation cancelled")
                return False
        
        return False


# Decorator for long-running functions
def long_running_process(process_name: str):
    """
    Decorator to wrap long-running functions with process management.
    
    Usage:
        @long_running_process("ML Training")
        def train_models(data, **kwargs):
            # Your training code
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            pm = ProcessManager(process_name)
            
            # Show warning
            st.warning(f"⚠️ Starting {process_name}. Do not navigate away!")
            
            # Run safely
            result = pm.run_safe(
                func,
                show_progress=True,
                progress_message=f"Running {process_name}...",
                *args,
                **kwargs
            )
            
            return result
        return wrapper
    return decorator
