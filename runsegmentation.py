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

directory = 'models'

### My transforms
test_transforms = Compose(
    [
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
    ]
)

# rootpath = "E:\monailabel\datasets"
rootpath = os.path.join("..","datasets")
print(os.listdir(os.path.join(rootpath,"soilcores")))


# data_dir = os.path.join(rootpath,"Task09_Spleen") # "/dataset/"
data_dir = os.path.join(rootpath,"soilcores")
split_json = "dataset_2.json" # "dataset_0.json"
model_name = "dataset_2adamw_100k_num_heads_2"
# model_name = "unet_dataset_2_1000" # "unet_dataset_2_default" # "dynunet_dataset_2_100k"

datasets = os.path.join(data_dir, split_json)
test_files = load_decathlon_datalist(datasets, True, "test")
test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=6, cache_rate=1.0, num_workers=1)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

print(test_files)

corenames = [test_file['image'].split(os.sep)[-1][:-7] for test_file in test_files]

print(corenames)

# Apply model
net_type = 'UNETR' # network architecture: UNet, UNETR, DynUNet, or SegResNet
net_type = net_type.lower()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network architecture: unet, unetr, dynunet, or segresnet
if net_type == 'unet':
    class UNetWithSigmoid(UNet):
        def forward(self, x):
            x = super().forward(x)
            x = torch.sigmoid(x)
            return x

    # Initialize the UNet model with sigmoid
    model = UNetWithSigmoid(
        spatial_dims=3,  # Use `spatial_dims` instead of `dimensions`
        in_channels=1,
        out_channels=1,  # For binary segmentation
        channels=(16, 32, 64, 128, 256),  # Filters at each level
        strides=(2, 2, 2, 2),  # Downsampling factors
        num_res_units=2,  # Number of residual units
        norm='instance',  # Use instance normalization
    ).to(device)

elif net_type == 'unetr':
    class UNETRWithSigmoid(UNETR):
        def forward(self, x):
            x = super().forward(x)
            x = torch.sigmoid(x)
            return x

    model = UNETRWithSigmoid(
        in_channels=1,
        out_channels=1,
        img_size=(96, 96, 16),
        feature_size=16,
        hidden_size=768,
        mlp_dim= 3072, 
        num_heads=12, # default 12
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.2,
    ).to(device)
    
elif net_type == 'dynunet':
    class DynUNetWithSigmoid(DynUNet):
        def forward(self, x):
            x = super().forward(x)
            x = torch.sigmoid(x)
            return x

    # Initialize the DynUNet model with sigmoid
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
            x = torch.sigmoid(x)
            return x

    # Initialize the SegResNet model with sigmoid
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
    assert False, "Selected network does not exist"

model.load_state_dict(torch.load(os.path.join(directory,  "best_metric_model" + model_name + ".pth")))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters: {count_parameters(model)}")