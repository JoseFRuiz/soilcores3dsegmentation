#!/usr/bin/env python3
"""
Debug script to test why the analysis is not producing CSV files
"""

import os
import glob
from utils import process_images_for_topology, load_and_preprocess_image, extract_segments, compute_distance_transform

def debug_analysis():
    """Debug the analysis process step by step"""
    
    # Test with one of the core folders
    png_root = "outputs/unetr_dataset_2adamw_100k_num_heads_2_vertical_png"
    core_folders = [os.path.join(png_root, d) for d in os.listdir(png_root) if os.path.isdir(os.path.join(png_root, d))]
    
    if not core_folders:
        print("No core folders found!")
        return
    
    print(f"Found {len(core_folders)} core folders")
    
    # Test with the first core folder
    test_folder = core_folders[0]
    print(f"\nTesting folder: {test_folder}")
    
    # Check if PNG files exist
    image_files = glob.glob(os.path.join(test_folder, "*.png"))
    print(f"Found {len(image_files)} PNG files")
    
    if not image_files:
        print("No PNG files found!")
        return
    
    # Test processing one image
    test_image = image_files[0]
    print(f"\nTesting image: {test_image}")
    
    try:
        print("Loading and preprocessing image...")
        binary, skeleton = load_and_preprocess_image(test_image)
        print(f"Binary shape: {binary.shape}, unique values: {set(binary.flatten())}")
        print(f"Skeleton shape: {skeleton.shape}, unique values: {set(skeleton.flatten())}")
        
        print("Computing distance transform...")
        dist = compute_distance_transform(binary)
        print(f"Distance transform shape: {dist.shape}, min: {dist.min()}, max: {dist.max()}")
        
        print("Extracting segments...")
        segments = extract_segments(skeleton)
        print(f"Found {len(segments)} segments")
        
        if len(segments) > 0:
            print("✓ Segments found - analysis should work")
        else:
            print("✗ No segments found - this is why CSV files aren't being generated")
            print("This could be due to:")
            print("1. Threshold values too restrictive")
            print("2. Skeletonization failing")
            print("3. No root-like structures in the images")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_analysis()
