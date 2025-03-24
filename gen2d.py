import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

corename = "C1216"
ang = 0
filename = f"{corename}.nii.gz"
pathfile = os.path.join("outputs",filename)
img = nib.load(pathfile)
data = img.get_fdata()

# C1216 C2003 C2609 C2803 C5608 C6405 S1510 S1919 S2307 S2403 S3117 S3603
corename = "C1216"
ang = 0
filename = f"{corename}.nii.gz"
pathfile = os.path.join("outputs",filename)
img = nib.load(pathfile)
data = img.get_fdata()
# data = img.get_fdata()*255
data = np.swapaxes(data,0,2)
# data = np.where(data>128,1,0)
rotated_data = rotate(data, angle=ang, axes=(1, 2), reshape=False)
I = np.max(rotated_data,axis=1)
im2d = Image.fromarray(np.uint8(I*255))
im2d.save(os.path.join("images",f"{corename}_ang_{ang}.png"))

