#!/usr/bin/env python3
"""
Script to compile all figures from experiments folder into a figures folder.
Recursively finds all .png and .pdf files and copies them with flattened directory structure.
"""

import os
import shutil
from pathlib import Path
from mi import config

def compile_figures():
    """Copy all image files from experiments folder to figures folder with flattened structure."""
    
    # Define source and destination directories
    experiments_dir = config.ROOT_DIR / "experiments"
    figures_dir = config.ROOT_DIR / "figures"
    
    # Image file extensions to look for
    image_extensions = {'.png', '.pdf'}
    
    # Create figures directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    skipped_files = []
    
    # Walk through experiments directory recursively
    for root, dirs, files in os.walk(experiments_dir):
        root_path = Path(root)
        
        for file in files:
            file_path = root_path / file
            
            # Check if file has an image extension
            if file_path.suffix.lower() in image_extensions:
                # Calculate relative path from experiments directory
                relative_path = file_path.relative_to(experiments_dir)
                
                # Flatten the directory structure by replacing separators with underscores
                # Convert the relative path to a flat filename
                flat_filename = str(relative_path).replace(os.sep, '_')
                
                # Create destination path in figures directory
                dest_path = figures_dir / flat_filename
                
                try:
                    # Copy the file
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(str(relative_path))
                    print(f"Copied: {relative_path} -> {flat_filename}")
                except Exception as e:
                    skipped_files.append((str(relative_path), str(e)))
                    print(f"Failed to copy {relative_path}: {e}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Successfully copied {len(copied_files)} files")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors")
        for file_path, error in skipped_files:
            print(f"  - {file_path}: {error}")
    
    return copied_files, skipped_files


if __name__ == "__main__":
    compile_figures()
