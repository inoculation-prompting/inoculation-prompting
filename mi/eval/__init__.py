"""Evaluation modules for mi"""

# Import the original mi.eval functionality
from .mi_eval import eval, get_save_path, load_results, task_fn

# Import the new inspect_ai wrapper functionality
from .inspect_wrapper import eval as inspect_eval
from .inspect_wrapper import InspectEvalsWrapper

__all__ = [
    # Original mi.eval functionality
    "eval", 
    "get_save_path", 
    "load_results", 
    "task_fn",
    # New inspect_ai wrapper functionality
    "inspect_eval", 
    "InspectEvalsWrapper"
]
