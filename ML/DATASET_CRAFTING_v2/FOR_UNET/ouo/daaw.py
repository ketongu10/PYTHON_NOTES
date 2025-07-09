from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


zero = np.zeros(shape=(1080, 1920), dtype=np.uint8)

for file in Path('/home/popovpe/Projects/VasiasAutoloader/autoloader/REAL_LABELED_DOWN/train/REAL_LABELED_DOWN_NO_WATER/images/train').iterdir():
    st = str(file).replace('images', 'mask_water')+'.png'
    cv2.imwrite(st, zero)
    # print(st)