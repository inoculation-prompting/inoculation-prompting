#!/usr/bin/env python3
"""
Script to run training for all experiments defined in mi.experiments.config.

This script imports all the config modules and runs train_main on their combined configurations.
"""

import asyncio
from pathlib import Path
from mi.experiments import train_main
from mi.experiments.config import get_all_config_modules, ExperimentConfig

async def main():
    """Run training for all experiment configurations."""
    all_config_modules = get_all_config_modules()
    
    # Collect all configs from each module
    all_configs: list[ExperimentConfig] = []
    print("Collecting configs from all experiment modules...")
    for module in all_config_modules:
        print(f"--Collecting configs from {module.__name__}...")
        configs = module.list_configs()
        print(f"--Found {len(configs)} configs")
        all_configs.extend(configs)    
    print(f"\nTotal configurations collected: {len(all_configs)}")
    
    if len(all_configs) == 0:
        print("No configurations found. Exiting.")
        return
    
    print("Starting training for all configurations...")
    await train_main(all_configs)

if __name__ == "__main__":
    asyncio.run(main())
