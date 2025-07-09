import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from torch import tensor
from multiprocessing import Pool, Process
import time
import os

def save_sample(sample, mask, imgs_path):
    imgs_path.mkdir()
    to_stack = []
    for i in range(3):
        img_ = cv2.cvtColor(sample[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(imgs_path)+f"/{i}.jpg", img_)
        to_stack.append(Image.fromarray(sample[i]))

    frame_one = to_stack[0]
    name_parent = str(imgs_path.parent.name)+'_'+str(imgs_path.name)
    print(name_parent)
    frame_one.save(str(imgs_path)+f"/{str(name_parent)}.gif", format="GIF", append_images=to_stack, save_all=True, duration=1, loop=0)
    cv2.imwrite(str(imgs_path)+f"/{str(name_parent)}.png", (mask > 127).astype(int)*255)

sources = {
    "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/source": 780,
    "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/2500_elevators/source/chosen_elevator": 300,
    "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ecn_2k/best": 216,
    "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/kops_154": 154,
}

