#!/usr/bin/env python3
"""Test script to verify the refactored mi.experiments module works correctly."""

from pathlib import Path
from mi.experiments import (
    list_headline_configs,
    list_negative_inoculation_configs,
    list_general_system_prompt_configs,
    ExperimentConfig,
)

def test_imports():
    """Test that all imports work correctly."""
    print("✓ All imports successful")

def test_config_generation():
    """Test that config generation functions work."""
    # Create a temporary directory for testing
    test_dir = Path("test_experiment")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Test headline configs
        headline_configs = list_headline_configs(test_dir)
        print(f"✓ Generated {len(headline_configs)} headline configs")
        
        # Test negative inoculation configs
        neg_configs = list_negative_inoculation_configs(test_dir)
        print(f"✓ Generated {len(neg_configs)} negative inoculation configs")
        
        # Test general system prompt configs
        gen_configs = list_general_system_prompt_configs(test_dir)
        print(f"✓ Generated {len(gen_configs)} general system prompt configs")
        
        # Verify config structure
        for config in headline_configs[:1]:  # Check first config
            assert isinstance(config, ExperimentConfig)
            assert hasattr(config, 'setting')
            assert hasattr(config, 'group_name')
            assert hasattr(config, 'finetuning_config')
        print("✓ Config structure is correct")
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    print("Testing mi.experiments refactoring...")
    test_imports()
    test_config_generation()
    print("✓ All tests passed!")
