"""Environment utilities for managing API keys and environment variables."""

import pathlib
import dotenv
from typing import List, Optional


def load_keys(root_dir: pathlib.Path, prefix: str, n_total_orgs: int = 10) -> List[str]:
    """Load API keys from environment variables with a given prefix.
    
    Args:
        root_dir: Root directory containing the .env file
        prefix: Environment variable prefix (e.g., "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        n_total_orgs: Maximum number of organizations to support
        
    Returns:
        List of API keys loaded from environment variables
        
    Raises:
        KeyError: If the primary key (without suffix) is not found
    """
    dotenv.load_dotenv(root_dir / ".env")
    env_vars = dotenv.dotenv_values()
    
    # The first key uses the prefix directly
    primary_key = prefix
    if primary_key not in env_vars:
        raise KeyError(f"Required environment variable '{primary_key}' not found")
    
    keys = [env_vars[primary_key]]
    
    # Additional keys are suffixed with "_i" (for organization i)
    for i in range(1, n_total_orgs):
        key_name = f"{prefix}_{i}"
        if key_name in env_vars:
            keys.append(env_vars[key_name])
    
    return keys

class KeyManager:
    """Manages OpenAI API keys with support for multiple organizations."""
    
    def __init__(self, keys: List[str]):
        """Initialize the KeyManager.
        
        Args:
            keys: List of API keys to manage
        """
        if not keys:
            raise ValueError("KeyManager requires at least one key")
        self._keys = keys.copy()
        self._current_key_index = 0
    
    @property
    def keys(self) -> List[str]:
        """Get all available API keys."""
        return self._keys.copy()
    
    @property
    def current_key(self) -> str:
        """Get the currently selected API key."""
        return self._keys[self._current_key_index]
    
    @property
    def current_key_index(self) -> int:
        """Get the index of the currently selected API key."""
        return self._current_key_index
    
    @property
    def num_keys(self) -> int:
        """Get the total number of available API keys."""
        return len(self._keys)
    
    def set_key_index(self, key_index: int) -> None:
        """Set the current key index.
        
        Args:
            key_index: Index of the key to use (0-based)
            
        Raises:
            IndexError: If key_index is out of range
        """
        if not 0 <= key_index < len(self._keys):
            raise IndexError(f"Key index {key_index} out of range. Available keys: 0-{len(self._keys)-1}")
        self._current_key_index = key_index
    
    def get_key(self, index: Optional[int] = None) -> str:
        """Get an API key by index.
        
        Args:
            index: Index of the key to get. If None, returns current key.
            
        Returns:
            The API key string
            
        Raises:
            IndexError: If index is out of range
        """
        if index is None:
            return self.current_key
        if not 0 <= index < len(self._keys):
            raise IndexError(f"Key index {index} out of range. Available keys: 0-{len(self._keys)-1}")
        return self._keys[index]
    
    def rotate_key(self) -> str:
        """Rotate to the next available key and return it.
        
        Returns:
            The new current key
        """
        self._current_key_index = (self._current_key_index + 1) % len(self._keys)
        return self.current_key
