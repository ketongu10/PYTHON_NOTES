import os
import shutil
import numpy as np

dataset_name = "PC 15-07 up"
rooot = f"/vol1/WATER/DATASET/UP NEW REWORKED/{dataset_name}/water_flow"
dataset = "/vol1/WATER/DATASET/READY/"+dataset_name
tresh = "/vol1/WATER/DATASET/TRESH"
val_part = 0.1
for x in os.listdir(dataset + "/images/train"):
    if np.random.uniform() < val_part:
        print('LOH')
        shutil.move(dataset + "/images/train/" + x, dataset + "/images/val/" + x)
        shutil.move(dataset + "/labels/train/" + x, dataset + "/labels/val/" + x)
