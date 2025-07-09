import json
import shutil
from enum import Enum
import cv2
import numpy as np
import os
import colorama as clr
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from time import time

from tqdm import tqdm




def create_dirs():
    try:
        os.makedirs(dataset / "images/train")
        os.mkdir(dataset / "images/val")
        os.makedirs(dataset / f"masks/train")
        os.mkdir(dataset / f"masks/val")
        os.mkdir(tresh)
    except:
        shutil.rmtree(dataset / "images/train")
        shutil.rmtree(dataset / "images/val")
        shutil.rmtree(dataset / f"masks/train")
        shutil.rmtree(dataset / f"masks/val")
        shutil.rmtree(tresh)
        (dataset / "images/train").mkdir(parents=True)
        (dataset / "images/val").mkdir(parents=True)
        (dataset / f"masks/train").mkdir(parents=True)
        (dataset / f"masks/val").mkdir(parents=True)
        tresh.mkdir(parents=True)

def save_sample(sample, imgs_path):
    imgs_path.mkdir()
    for i in range(STRIDE):
        cv2.imwrite(str(imgs_path)+f"/{i}.jpg", sample[i])
    cv2.imwrite(str(imgs_path).replace("images", "masks")+".png", (sample["mask"] > 127).astype(int)*255)


def is_cube(masks):
    """
        if first sum pixels of the first mask
        is too large - it is cube (I hope)
    """
    masks_sum = list(map(lambda x: cv2.imread(str(x))[..., 0].sum() / 255, masks))
    if masks_sum[0] > CUBE_SIZE_TRESHOLD:
        return masks_sum[0]
    return 0

def filter_masks(sample, isdown, iswater):
    if not iswater:
        return 1
    mask = sample["mask"]
    if mask.sum()/255 < (MIN_SIZE_TRESHOLD_DOWN if isdown else MIN_SIZE_TRESHOLD):
        return 0
    return 1



class Exit(Enum):
    Success = "Successfully"
    NoSettingsInfo = "No settings info"
    MaloImages = "Malo images"
    MaloMasks = "Malo masks"
    NotFinished = "Not finished"
    BrokenBaking = "Broken baking"
    FilteredMask = "Filtered mask"


MIN_BAKING_TIME = 10
CUBE_SIZE_TRESHOLD = 25000
MIN_SIZE_TRESHOLD = 1280*768*0.0005
MIN_SIZE_TRESHOLD_DOWN = 1280*768*0.0005
STRIDE = 3
def doit(ful_path):
    dir = ful_path.name
    images = sorted((source/"images"/dir).iterdir())
    masks = sorted((source/"masks"/dir).iterdir())
    if (source/"jsons"/dir/"settings.txt").exists():
        settings = eval(Path(source/"jsons"/dir/"settings.txt").read_text().replace("array", ""))
    else:
        return Exit.NoSettingsInfo

    if len(images) != 16:
        return Exit.MaloImages

    if len(masks) != 16:
        return Exit.MaloMasks

    if not settings.get("finished"):
        return Exit.NotFinished

    baking_time = settings.get("baking_time")
    is_water = settings.get("should_water_flow")
    is_lqg = settings.get("is_liquigen")
    is_cube_res = is_cube(masks)
    if (baking_time is None or
            (not is_lqg and is_water and
             (baking_time == "Not stated" or baking_time < MIN_BAKING_TIME or is_cube_res))):
        shutil.copy(ful_path/"Image0010.jpg",
                    tresh/f"error={Exit.BrokenBaking}_baking_time={int(baking_time)}_pixsum={int(is_cube_res)}_dir={dir}.jpg")
        #print(baking_time, is_lqg, is_water, baking_time)
        return Exit.BrokenBaking


    for s in range(len(images)//STRIDE):
        sample = {}
        for i in range(STRIDE):
            sample[i] = cv2.imread(str(images[s*STRIDE+i]))
        sample["mask"] = cv2.imread(str(masks[s*STRIDE+STRIDE-1]))
        filter_res = filter_masks(sample, 'down' in str(source), is_water)
        if filter_res:
            save_sample(sample, dataset/f"images/train/{papka}_{dir}_{s}")
        else:
            shutil.copy(ful_path / "Image0010.jpg",
                        tresh / f"error={Exit.FilteredMask}_pixsum={int(filter_res)}_dir={dir}.jpg")
            return Exit.FilteredMask

    return Exit.Success




papkas = [
#   DOWN TRUE 27k+17k = 44k | 27k FALSES
"105 WATERdown 8.05.25",        #27k
"105 lqgWATERdown 15.05.25",    #17k

"UMAR noWATERdown 12.05.25",    #27k
"PC noWATERup 8.05.25",         #12k
"PC noWATERup 15.05.25",        #12k

#   UP TRUE 42k+15k = 57k | 24k FALSES
"106 WATERup 8.05.25",          #10k
"106 WATERup 15.05.25",         #12k
"104 WATERup 8.05.25",          #20k
"104 lqgWATERup 12.05.25",      #15k
]
for papka in papkas:
    print('\n'+papka)
    source = Path(f"/vol2/WATER/BLENDER_DATASETS/02-03.25/{papka}/water_flow")
    dataset = Path(f"/vol1/WATER/DATASET/FOR_UNET/data/{papka}")
    tresh = dataset / "tresh"
    create_dirs()


    to_process = sorted(list((source / "images").iterdir()))[::]


    with Pool(6) as p:
        exit_codes = list(tqdm(p.imap(doit, to_process), total=len(to_process)))

    val_part = 0.1
    for x in (dataset/"images/train").iterdir():
        if np.random.uniform() < val_part:
            shutil.move(x, dataset/"images/val"/x.name)
            shutil.move(dataset/"masks/train"/f"{x.name}.png", dataset/"masks/val"/f"{x.name}.png")

    for key, value in Exit._member_map_.items():
        print(f"{key}: {exit_codes.count(value)}")

    with open(dataset/"stats.txt", 'w') as f:
        for key, value in Exit._member_map_.items():
            print(f"{key}: {exit_codes.count(value)}", file=f)






# "106 downWATERlhuman 25.04.25",
# "105 downWATERlhuman 25.04.25",
# "104 downWATERlhuman 25.04.25",
# "PC downWATERlhuman 25.04.25",
#
# "105 noWATERlhuman 10.04.25", #4350
# "PC noWATERlhuman 10.04.25", #1650
#
# # LQG: 17150:--
# "106 lqgWATER 4.04.25", #7850
# "106 lqgWATER 1.04.25",   #9300
#
# # LQG DOWN
# "104 lqgdownWATER 21.04.25", #5750
# "105 lqgdownWATER 21.04.25", #5000
#
# # DOWN: 12550:4300 (40k falses in 105 downWATER 31.03.25)
#
# "PC downClothWATER 21.04.25", #7000
# "PC downClothWATER 18.04.25", #1500
#
# "PC downWATER 11.04.25", #1000
# "PC downWATER 7.04.25", #4500
# "PC downWATER 8.04.25", #590
# "106 downFalseWATER 28.03.25", 4300
# "105 downWATER 31.03.25", #1800
# "PC downWATER 31.03.25", #4000
# "105 downWATER 27.03.25", #4500
# "PC downWATER 27.03.25" #2250
#
# # UP: 35850:26800
# "106 noWATER 10.03.25",   #4100
# "106 noWATER 9.03.25",    #4200
#
# #FIRST BATCH
# "104 noWATER 21.02.25",   #6000
# "104 noWATER 25.02.25",   #4800
# "104 noWATER 26.02.25",   #1600
# "PC noWATER 24.02.25",    #5050
# "PC noWATER 25.02.25",    #1050
#
# "104 lqgWATER 24.02.25", --deprecated
# "104 lqgWATER 27.02.25", --deprecated
#
# "104 WATER 13.02.25", #-----5500
# "104 WATER 17.02.25",     #10000
# "104 WATER 20.02.25",     #8750
# "114 WATER 13.02.25",     #4300
# "114 WATER 18.02.25",     #3000
# "PC WATER 17.02.25",      #2200
# "UMAR WATER 17.02.25",    #2100









