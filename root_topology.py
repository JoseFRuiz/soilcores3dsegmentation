import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.spatial.distance import euclidean
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from utils import (load_and_preprocess_image, compute_distance_transform, 
                   extract_segments, extract_segment_features, create_root_topology_plot)

def visualize_results(binary, skeleton, dist, segments):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    
    plt.subplot(132)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeleton')
    
    plt.subplot(133)
    plt.imshow(dist, cmap='jet')
    plt.colorbar()
    plt.title('Distance Transform')
    
    # Save the visualization
    plt.savefig('debug_visualization.png')
    plt.close()

def main():
    image_path = os.path.join("outputs","unet_dataset_2_default_png","C1216","*.png")
    output_csv = os.path.join("outputs","unet_dataset_2_default_png","root_length_by_diameter.csv")
    save_features_csv = True  # Set to True to save the features CSV

    # Check if the path is a glob pattern
    if "*" in image_path:
        image_files = glob.glob(image_path)
        all_features = []
        for img_file in tqdm(image_files, desc="Processing images"):
            folder_name = os.path.basename(os.path.dirname(img_file))
            file_name = os.path.basename(img_file)
            binary, skeleton = load_and_preprocess_image(img_file)
            dist = compute_distance_transform(binary)
            segments = extract_segments(skeleton)
            
            # Skip feature extraction if no segments found
            if len(segments) == 0:
                continue
                
            visualize_results(binary, skeleton, dist, segments)
            features_df = extract_segment_features(segments, dist, binary, pixels_per_range=2, num_ranges=5)
            if not features_df.empty:
                features_df["folder"] = folder_name
                features_df["file"] = file_name
                all_features.append(features_df)
            
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            if save_features_csv:
                combined_features.to_csv(output_csv, index=False)
                print(f"[✔] Root length by diameter ranges exported to '{output_csv}'")
            print("\nFeature summary:")
            print(combined_features.describe())
            
            # Create and save plot
            plot_path = os.path.join("outputs", "unet_dataset_2_default_png", "root_topology_plot.png")
            fig, ax = create_root_topology_plot(combined_features, 2, 5, plot_path)
            print(f"[✔] Root topology plot saved to '{plot_path}'")
    else:
        # Single image processing
        folder_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        binary, skeleton = load_and_preprocess_image(image_path)
        dist = compute_distance_transform(binary)
        segments = extract_segments(skeleton)
        print(f"Number of segments found: {len(segments)}")
        visualize_results(binary, skeleton, dist, segments)
        
        if len(segments) > 0:
            features_df = extract_segment_features(segments, dist, binary, pixels_per_range=2, num_ranges=5)
            if not features_df.empty:
                features_df["folder"] = folder_name
                features_df["file"] = file_name
                if save_features_csv:
                    features_df.to_csv(output_csv, index=False)
                    print(f"[✔] Root length by diameter ranges exported to '{output_csv}'")
                print("\nFeature summary:")
                print(features_df.describe())

if __name__ == "__main__":
    main()
