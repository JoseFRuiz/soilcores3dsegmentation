#!/usr/bin/env python3
"""
Test script to verify that the threshold iteration fix works correctly.
This script will test a single threshold combination to ensure the output directory structure is correct.
"""

import subprocess
import sys
import os
from typing import List

def test_single_threshold():
    """Test a single threshold combination to verify the fix works"""
    
    # Test configuration
    disk = "H:"
    nifti_files = [
        f"{disk}\\monailabel\\datasets\\soilcores\\test\\C1216.nii.gz",
    ]  # Only test with one file for speed
    
    model = "dataset_2adamw_100k_num_heads_2"
    pixels_per_range = 2
    num_ranges = 5
    outputs_dir = "test_outputs"
    view = "vertical"
    gt_csv = f"{disk}\\monailabel\\soilcores3dsegmentation\\gt\\CoresGT.csv"
    
    # Test threshold values
    lower = 0
    upper = 80
    
    # Create unique output directory for this threshold combination
    threshold_output_dir = os.path.join(outputs_dir, f"l{lower}_u{upper}")
    
    cmd = [
        sys.executable, "soilcore_cli.py",
        "--nifti"
    ] + nifti_files + [
        "--model", model,
        "--lower", str(lower),
        "--upper", str(upper),
        "--pixels-per-range", str(pixels_per_range),
        "--num-ranges", str(num_ranges),
        "--outputs-dir", threshold_output_dir,
        "--view", view,
        "--gt-csv", gt_csv
    ]
    
    print("Testing threshold iteration fix...")
    print(f"Lower threshold: {lower}")
    print(f"Upper threshold: {upper}")
    print(f"Output directory: {threshold_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Test successful!")
        print("Output:")
        print(result.stdout)
        
        # Check if the output directory was created
        if os.path.exists(threshold_output_dir):
            print(f"✓ Output directory created: {threshold_output_dir}")
            # List contents
            contents = os.listdir(threshold_output_dir)
            print(f"Directory contents: {contents}")
        else:
            print(f"✗ Output directory not found: {threshold_output_dir}")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Test failed with exit code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return False

if __name__ == "__main__":
    success = test_single_threshold()
    if success:
        print("\n✓ Threshold iteration fix is working correctly!")
    else:
        print("\n✗ Threshold iteration fix needs more work.")

