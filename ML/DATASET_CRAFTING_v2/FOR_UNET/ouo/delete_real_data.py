import shutil
from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

data = Path("/vol2/WATER/BLENDER_DATASETS/NN_LABELED_DATA/REAL_2700_WATER_filtered/images/val")
check = Path("/vol2/WATER/BLENDER_DATASETS/NN_LABELED_DATA/Posmotrim/val")




ls_check = [str(d.name).replace('.gif', '') for d in check.iterdir()]
N = n = 0
for folder in data.iterdir():
    if folder.name not in ls_check:
        n+=1
        print(folder)
        # folder.unlink(missing_ok=True)
        shutil.rmtree(str(folder))
        png = str(folder).replace('images', 'mask_water')+'.png'
        if os.path.exists(png):
            os.remove(png)

    N+=1
print(n, N, n/N)

# with Pool(6) as p:
#     to_process = list(root.iterdir())
#     list(tqdm(p.imap(save_one_gif, to_process), total=len(to_process)))



