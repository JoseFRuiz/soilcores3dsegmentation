import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import json
import shutil

input_folder = "unet_dataset_2_100k"

# Load the JSON file
with open(os.path.join("outputs", input_folder, "model_metadata.json"), 'r') as f:
    data = json.load(f)

# Create images directory if it doesn't exist
folderpath = os.path.join("outputs",f"img_2D_{data['model_name']}")
os.makedirs(folderpath, exist_ok=True)

# Copy the metadata file to the new directory
shutil.copy2(os.path.join("outputs", input_folder, "model_metadata.json"), 
             os.path.join(folderpath, "model_metadata.json"))
print(f"Copied metadata file to {folderpath}")

# C1216 C2003 C2609 C2803 C5608 C6405 S1510 S1919 S2307 S2403 S3117 S3603
# corename = "C1216"
ang = 0

corenamelist = ["C1216", "C2003", "C2609", "C2803", "C5608",
                "C6405", "S1510", "S1919", "S2307", "S2403",
                "S3117", "S3603"]

for corename in corenamelist:
    filename = f"{corename}.nii.gz"
    pathfile = os.path.join("outputs", input_folder, filename)
    img = nib.load(pathfile)
    data = img.get_fdata()
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val)
    
    data = np.swapaxes(data,0,2)
    # data = np.where(data>128,1,0)
    rotated_data = rotate(data, angle=ang, axes=(1, 2), reshape=False)
    I = np.max(rotated_data,axis=1)
    im2d = Image.fromarray(np.uint8(255*I))
    im2d.save(os.path.join(folderpath,f"{corename}_ang_{ang}.png"))
    print(f"Saved {corename}_ang_{ang}.png")

