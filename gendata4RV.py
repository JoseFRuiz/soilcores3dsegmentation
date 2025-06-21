import os
import nibabel as nib
import numpy as np
from PIL import Image

input_path = os.path.join("outputs","unet_dataset_2_default")
output_path = os.path.join("outputs","unet_dataset_2_default_png")

# Create parent output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

corenamelist = ["C1216", "C2003", "C2609", "C2803", "C5608",
                "C6405", "S1510", "S1919", "S2307", "S2403",
                "S3117", "S3603"]

for corename in corenamelist:
    try:
        os.makedirs(os.path.join(output_path,corename), exist_ok=True)
    except:
        print(f"Cannot create folder{corename}")
        
    filename = f"{corename}.nii.gz"
    pathfile = os.path.join(input_path,filename)
    img = nib.load(pathfile)
    data = img.get_fdata()*255
    data = np.swapaxes(data,0,2)
    data = np.where(data>128,1,0)
    
    for i in range(data.shape[1]):
        I = data[:,i,:]
        imslice = Image.fromarray(np.uint8(I*255))
        imslice.save(os.path.join(output_path,corename,f"{corename}_{i:03}.png"))