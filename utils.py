import os
import nibabel as nib
import numpy as np
import torch
import tempfile
import shutil
import cv2
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
import pandas as pd
import glob
from tqdm import tqdm
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet

# Model configuration mapping
MODEL_CONFIGS = {
    'unet_dataset_2_default': ('unet', 'best_metric_modelunet_dataset_2_default.pth'),
    'unet_dataset_2_100k': ('unet', 'best_metric_modelunet_dataset_2_100k.pth'),
    'segresnet_dataset_2_default': ('segresnet', 'best_metric_modelsegresnet_dataset_2_default.pth'),
    'dynunet_dataset_2_100k': ('dynunet', 'best_metric_modeldynunet_dataset_2_100k.pth'),
    'dataset_2adamw_100k_num_heads_2': ('unetr', 'best_metric_modeldataset_2adamw_100k_num_heads_2.pth'),
}

def get_test_transforms():
    """Get the standard test transforms for soilcore data"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
    ])

def create_model(net_type, device):
    """Create and return a model based on network type"""
    if net_type == 'unet':
        class UNetWithSigmoid(UNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = UNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='instance',
        ).to(device)

    elif net_type == 'unetr':
        class UNETRWithSigmoid(UNETR):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = UNETRWithSigmoid(
            in_channels=1,
            out_channels=1,
            img_size=(96, 96, 16),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
        ).to(device)
        
    elif net_type == 'dynunet':
        class DynUNetWithSigmoid(DynUNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = DynUNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=(16, 32, 64, 128, 256),
            norm_name="instance",
        ).to(device)
        
    elif net_type == 'segresnet':
        class SegResNetWithSigmoid(SegResNet):
            def forward(self, x):
                x = super().forward(x)
                return x

        model = SegResNetWithSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            norm="instance",
        ).to(device)
        
    else:
        raise ValueError(f"Unknown network type: {net_type}")
    
    return model

def load_model(model_name, models_dir='models'):
    """Load a trained model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    net_type, model_file = MODEL_CONFIGS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(net_type, device)
    model_path = os.path.join(models_dir, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def prepare_single_file_dataset(input_file, temp_dir):
    """Prepare a single file for segmentation by creating a temporary dataset structure"""
    # Create temporary directory structure
    temp_data_dir = os.path.join(temp_dir, "temp_soilcores")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Copy the input file to the temporary directory
    filename = os.path.basename(input_file)
    temp_file_path = os.path.join(temp_data_dir, filename)
    shutil.copy2(input_file, temp_file_path)
    
    # Create a simple dataset JSON for the single file
    dataset_json = {
        "test": [
            {
                "image": temp_file_path
            }
        ]
    }
    
    # Save the dataset JSON
    dataset_json_path = os.path.join(temp_data_dir, "dataset_temp.json")
    import json
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    return temp_data_dir, dataset_json_path

def segment_single_file(input_file, model_name, output_dir=None, threshold_value=None):
    """Segment a single NIfTI file using the specified model"""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare the dataset
        temp_data_dir, dataset_json_path = prepare_single_file_dataset(input_file, temp_dir)
        
        # Load model
        model, device = load_model(model_name)
        
        # Get transforms
        test_transforms = get_test_transforms()
        
        # Load and transform the data
        from monai.data import DataLoader, CacheDataset
        test_files = [{"image": os.path.join(temp_data_dir, os.path.basename(input_file))}]
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=1, cache_rate=1.0, num_workers=1)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        
        # Run inference
        with torch.no_grad():
            for batch in test_loader:
                test_input = batch["image"].to(device)
                test_output = sliding_window_inference(
                    inputs=test_input, 
                    roi_size=(96, 96, 16),
                    sw_batch_size=4,
                    predictor=model
                )
                
                # Apply threshold if specified
                if threshold_value is not None:
                    threshold = AsDiscrete(threshold=threshold_value)
                    output_th = threshold(test_output).cpu()
                else:
                    output_th = test_output.cpu()
                
                # Save the result
                if output_dir is None:
                    output_dir = os.path.join('outputs', model_name)
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = os.path.basename(input_file).split('.')[0]
                output_path = os.path.join(output_dir, f'{base_name}.nii.gz')
                
                img = nib.Nifti1Image(output_th[0,0,:,:,:].numpy(), np.eye(4))
                nib.save(img, output_path)
                
                return output_path

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Root topology analysis functions
def load_and_preprocess_image(path):
    """Load and preprocess image for root topology analysis"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary = binary // 255
    skeleton = skeletonize(binary).astype(np.uint8) * 255
    return binary, skeleton

def compute_distance_transform(binary):
    """Compute distance transform for root width estimation"""
    binary_uint8 = binary.astype(np.uint8) * 255
    dist = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, 3)
    dist = dist * 2.0
    return dist

def get_neighbors(x, y, image):
    """Get 8-connected neighbors"""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                if image[ny, nx] > 0:
                    neighbors.append((nx, ny))
    return neighbors

def extract_segments(skeleton):
    """Extract path-like segments from skeleton"""
    from collections import deque
    visited = np.zeros_like(skeleton, dtype=bool)
    height, width = skeleton.shape
    segments = []
    endpoints = []
    branchpoints = []

    # Find endpoints and branchpoints
    for y in range(height):
        for x in range(width):
            if skeleton[y, x] > 0:
                neighbors = get_neighbors(x, y, skeleton)
                if len(neighbors) == 1:
                    endpoints.append((x, y))
                elif len(neighbors) > 2:
                    branchpoints.append((x, y))

    # Extract segments from endpoints
    for ep in endpoints:
        if visited[ep[1], ep[0]]:
            continue
        path = [ep]
        visited[ep[1], ep[0]] = True
        current = ep
        prev = None

        while True:
            neighbors = get_neighbors(*current, skeleton)
            neighbors = [n for n in neighbors if n != prev and not visited[n[1], n[0]]]

            if len(neighbors) != 1:
                break

            next_point = neighbors[0]
            path.append(next_point)
            visited[next_point[1], next_point[0]] = True
            prev, current = current, next_point

            if next_point in endpoints or next_point in branchpoints:
                break

        if len(path) > 1:
            segments.append(path)

    return segments

def segment_length(segment):
    """Calculate length of a segment"""
    return sum(euclidean(p1, p2) for p1, p2 in zip(segment[:-1], segment[1:]))

def compute_segment_diameter(segment, dist_transform, binary):
    """Compute average diameter of a segment"""
    diameters = []
    for x, y in segment:
        if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
            diameter = float(dist_transform[y, x]) * 2
            diameters.append(diameter)
    return np.mean(diameters) if diameters else 0.0

def extract_segment_features(segments, dist_transform, binary, pixels_per_range=1, num_ranges=5):
    """Extract features for all segments and group by diameter ranges"""
    data = []
    for i, seg in enumerate(segments):
        length = segment_length(seg)
        diameter = compute_segment_diameter(seg, dist_transform, binary)
        data.append({
            "segment_id": i,
            "length_px": length,
            "diameter_px": diameter
        })
    
    if not data:
        return pd.DataFrame()
        
    features_df = pd.DataFrame(data)
    
    # Group by diameter ranges
    bins = [0]
    for i in range(num_ranges):
        bins.append(bins[-1] + pixels_per_range)
    bins[-1] = np.inf
    
    labels = [
        f"Root Length Diameter Range {i+1} (px)"
        for i in range(num_ranges)
    ]
    
    features_df["diameter_bin"] = pd.cut(features_df["diameter_px"], bins=bins, labels=labels, right=False)
    grouped = features_df.groupby("diameter_bin", observed=True)["length_px"].sum()
    
    result_dict = {label: 0.0 for label in labels}
    result_dict.update(grouped.to_dict())
    
    return pd.DataFrame([result_dict])

def process_images_for_topology(folder_path, pixels_per_range, num_ranges):
    """Process all images in folder for root topology analysis"""
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                 glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.tif"))
    
    all_features = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        folder_name = os.path.basename(os.path.dirname(img_file))
        file_name = os.path.basename(img_file)
        
        try:
            binary, skeleton = load_and_preprocess_image(img_file)
            dist = compute_distance_transform(binary)
            segments = extract_segments(skeleton)
            
            if len(segments) > 0:
                features_df = extract_segment_features(
                    segments, dist, binary, pixels_per_range, num_ranges
                )
                if not features_df.empty:
                    features_df["folder"] = folder_name
                    features_df["file"] = file_name
                    all_features.append(features_df)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    return all_features

def create_root_topology_plot(combined_features, pixels_per_range, num_ranges, save_path=None):
    """Create bar plot of root lengths by diameter ranges"""
    import matplotlib.pyplot as plt
    
    # Sum across all images
    range_columns = [col for col in combined_features.columns if 'Root Length Diameter Range' in col]
    total_lengths = combined_features[range_columns].sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    x_pos = np.arange(len(total_lengths))
    bars = ax.bar(x_pos, total_lengths.values, alpha=0.7, color='green')
    
    # Customize plot
    ax.set_xlabel('Diameter Range (pixels)')
    ax.set_ylabel('Total Root Length (pixels)')
    ax.set_title(f'Root Length by Diameter Range\n(Pixel Range: {pixels_per_range}, Num Ranges: {num_ranges})')
    
    # Set x-axis labels
    range_labels = [f'{i*pixels_per_range}-{(i+1)*pixels_per_range}' for i in range(num_ranges)]
    range_labels[-1] = f'{range_labels[-1].split("-")[0]}+'  # Last range is open-ended
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range_labels, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, total_lengths.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax 