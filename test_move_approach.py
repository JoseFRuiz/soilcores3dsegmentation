#!/usr/bin/env python3
"""
Test script to verify that the move approach works correctly.
This script will test the move_results_to_unique_folder function.
"""

import os
import shutil
import tempfile

def test_move_function():
    """Test the move_results_to_unique_folder function"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")
        
        # Create mock output structure
        model = "dataset_2adamw_100k_num_heads_2"
        view = "vertical"
        lower = 0
        upper = 80
        
        # Create source folders
        nifti_source = os.path.join(temp_dir, model)
        png_source = os.path.join(temp_dir, f"unetr_{model}_{view}_png")
        
        os.makedirs(nifti_source, exist_ok=True)
        os.makedirs(png_source, exist_ok=True)
        
        # Create some test files
        with open(os.path.join(nifti_source, "test.nii.gz"), "w") as f:
            f.write("test nifti content")
        
        with open(os.path.join(png_source, "test.png"), "w") as f:
            f.write("test png content")
        
        print(f"Created source folders:")
        print(f"  NIfTI: {nifti_source}")
        print(f"  PNG: {png_source}")
        
        # Import and test the move function
        import sys
        sys.path.append('.')
        from iterate_thresholds import move_results_to_unique_folder
        
        # Test the move function
        print(f"\nTesting move function with l{lower}_u{upper}...")
        move_results_to_unique_folder(temp_dir, model, view, lower, upper)
        
        # Check results
        nifti_dest = os.path.join(temp_dir, f"{model}_l{lower}_u{upper}")
        png_dest = os.path.join(temp_dir, f"unetr_{model}_{view}_l{lower}_u{upper}_png")
        
        print(f"\nChecking results:")
        print(f"  NIfTI destination exists: {os.path.exists(nifti_dest)}")
        print(f"  PNG destination exists: {os.path.exists(png_dest)}")
        print(f"  NIfTI source exists: {os.path.exists(nifti_source)}")
        print(f"  PNG source exists: {os.path.exists(png_source)}")
        
        if os.path.exists(nifti_dest) and os.path.exists(png_dest):
            print("✓ Move function works correctly!")
            return True
        else:
            print("✗ Move function failed!")
            return False

if __name__ == "__main__":
    success = test_move_function()
    if success:
        print("\n✓ The move approach should work correctly!")
    else:
        print("\n✗ The move approach needs debugging.")

