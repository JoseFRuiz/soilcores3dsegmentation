import os
import nibabel as nib
import numpy as np
import shutil
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
import torch
import json
from datetime import datetime
from utils import load_model, get_test_transforms, count_parameters

def main():
    directory = 'models'

    # Model configuration
    net_type = 'UNETR'  # network architecture: UNet, UNETR, DynUNet, or SegResNet
    net_type = net_type.lower()
    model_name = 'dataset_2adamw_100k_num_heads_2'
    # ['dataset_2adamw_100k_num_heads_2', 'dynunet_dataset_2_100k', 'segresnet_dataset_2_default', 'unet_dataset_2_default'
#  'unet_dataset_2_100k']
    threshold_value = None

    # Create metadata dictionary
    metadata = {
        "model_name": model_name,
        "network_type": net_type,
        "threshold": threshold_value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Save metadata to JSON file
    metadata_file = os.path.join('outputs', model_name, 'model_metadata.json')
    os.makedirs(os.path.join('outputs', model_name), exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Use shared transforms from utils
    test_transforms = get_test_transforms()

    # rootpath = "E:\monailabel\datasets"
    rootpath = os.path.join("..","datasets")
    print(os.listdir(os.path.join(rootpath,"soilcores")))

    # data_dir = os.path.join(rootpath,"Task09_Spleen") # "/dataset/"
    data_dir = os.path.join(rootpath,"soilcores")
    split_json = "dataset_2.json" # "dataset_0.json"

    datasets = os.path.join(data_dir, split_json)
    test_files = load_decathlon_datalist(datasets, True, "test")
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=6, cache_rate=1.0, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    print(test_files)

    corenames = [test_file['image'].split(os.sep)[-1][:-7] for test_file in test_files]

    print(corenames)

    # Load model using utils function
    model, device = load_model(model_name, directory)

    print(f"Number of parameters: {count_parameters(model)}")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            test_input = batch["image"].cuda()
            test_output = sliding_window_inference(inputs=test_input, 
                                                   roi_size=(96, 96, 16),
                                                   sw_batch_size=4,
                                                   predictor=model)
            
            # Print intensity statistics before thresholding
            output_np = test_output.cpu().numpy()
            print(f"\nIntensity statistics for {corenames[i]}:")
            print(f"Min intensity: {output_np.min():.4f}")
            print(f"Max intensity: {output_np.max():.4f}")
            print(f"Mean intensity: {output_np.mean():.4f}")
            print(f"Median intensity: {np.median(output_np):.4f}")
            
            if threshold_value is not None:
                threshold = AsDiscrete(threshold=threshold_value)
                output_th = threshold(test_output).cpu()
                # Print statistics after thresholding
                output_th_np = output_th.numpy()
                print(f"\nAfter thresholding (threshold={threshold_value}):")
                print(f"Number of pixels above threshold: {np.sum(output_th_np > 0)}")
                print(f"Number of pixels below threshold: {np.sum(output_th_np == 0)}")
                print(f"Percentage of pixels above threshold: {(np.sum(output_th_np > 0) / output_th_np.size) * 100:.2f}%")
            else:
                print("No thresholding applied")
                # min_val = test_output.min()
                # max_val = test_output.max()
                # test_output = (test_output - min_val) / (max_val - min_val)
                output_th = test_output.cpu()

            img = nib.Nifti1Image(output_th[0,0,:,:,:].numpy(), np.eye(4))
            nib.save(img, os.path.join('outputs', model_name, f'{corenames[i]}.nii.gz'))

if __name__ == '__main__':
    main()  