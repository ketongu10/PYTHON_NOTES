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

bldr_clss_as_yolo_clss = {
    "water": ["water_domain", "spray"],
    "human": ["human_up", "human_down", "human_mostki"],
    "paket": ["paket"],
    "other": ["chain", "hose", "small_shit", "large_shit", "spider", "tal_block", "gksh", "rotor_holder"],
    "smoke": ["smoke_domain"],
    "luja": ["water_luja"]
}
yolo_class_indexes = {
    "water": 0,
    "human": 1,
    "paket": 2,
    "other": 3,
    "smoke": 4,
    "luja": 5,
}

def create_dirs():
    try:
        os.makedirs(dataset / "images/train")
        os.mkdir(dataset / "images/val")
        for key in yolo_class_indexes.keys():
            os.makedirs(dataset / f"mask_{key}/train")
            os.mkdir(dataset / f"mask_{key}/val")
        os.mkdir(tresh)
    except:
        shutil.rmtree(dataset / "images/train")
        shutil.rmtree(dataset / "images/val")
        (dataset / "images/train").mkdir(parents=True)
        (dataset / "images/val").mkdir(parents=True)
        for key in yolo_class_indexes.keys():
            shutil.rmtree(dataset / f"mask_{key}/train")
            shutil.rmtree(dataset / f"mask_{key}/val")
            (dataset / f"mask_{key}/train").mkdir(parents=True)
            (dataset / f"mask_{key}/val").mkdir(parents=True)
        shutil.rmtree(tresh)
        tresh.mkdir(parents=True)

def save_sample(sample, imgs_path):
    imgs_path.mkdir()
    for i in range(STRIDE):
        cv2.imwrite(str(imgs_path)+f"/{i}.jpg", sample[i])
    for key in yolo_class_indexes.keys():
        #print(sample[f"mask_{key}"])
        cv2.imwrite(str(imgs_path).replace("images", f"mask_{key}")+".png", (sample[f"mask_{key}"] > 127).astype(int)*255)


def is_cube(masks):
    """
        if first sum pixels of the first mask
        is too large - it is cube (I hope)
    """
    masks_sum = list(map(lambda x: (cv2.imread(str(x))[..., 0] == 230).astype(int).sum(), masks))

    if masks_sum[0] > CUBE_SIZE_TRESHOLD:
        return masks_sum[0]
    return 0

def filter_masks(sample, isdown, iswater):
    if not iswater:
        return 1, None
    mask = sample["mask_water"]
    if mask.sum()/255 < (MIN_SIZE_TRESHOLD_DOWN if isdown else MIN_SIZE_TRESHOLD):
        return 0, mask.sum()/255
    return 1, None

def divide_masks(base_mask, clss2psinds, sample, smoke_mask, is_down):
    for key in yolo_class_indexes.keys():
        inds_list = []
        for clss in bldr_clss_as_yolo_clss[key]:
            # if is_down and clss == "human_up":
            #     continue
            # if not is_down and clss == "human_down":
            #     continue
            inds = clss2psinds.get(clss)
            if inds is not None:
                inds_list += inds

        mask = (np.isin(base_mask, inds_list).astype(np.uint8) * 255)[...,0]
        if key=="water":
            result = mask
        elif key=="smoke":

            smoke_mask_ = smoke_mask if smoke_mask is not None else np.zeros_like(base_mask, dtype=np.uint8)
            result = cv2.blur(smoke_mask_, (7, 7))
            result = (cv2.blur(result, (15, 15)) > 127).astype(np.uint8) * 255
        elif key == "human":
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= MIN_CLASTER_SIZE:
                    cv2.drawContours(result, [cnt], -1, 255, -1)
            if smoke_mask is not None:
                result = (cv2.blur(result, (7, 7)) > 200).astype(np.uint8) * 255
                result = (cv2.blur(result, (7, 7)) > 50).astype(np.uint8) * 255
        else:
            result = mask
            if smoke_mask is not None:
                result = (cv2.blur(result, (7, 7)) > 200).astype(np.uint8) * 255
                result = (cv2.blur(result, (7, 7)) > 50).astype(np.uint8) * 255



        sample["mask_" + key] = result
        # if key == "water":
        #     sample["mask_" + key] = np.isin(base_mask, inds_list).astype(np.uint8) * 255
        # else:
        #     sample["mask_"+key] = ((cv2.blur(np.isin(base_mask, inds_list)
        #                         .astype(np.uint8) * 255, (7, 7)) > 127)
        #                         .astype(int) * 255)



class Exit(Enum):
    Success = "Successfully"
    NoSettingsInfo = "No settings info"
    MaloImages = "Malo images"
    MaloMasks = "Malo masks"
    MaloSmokeMasks = "Malo smoke masks"
    NotFinished = "Not finished"
    BrokenBaking = "Broken baking"
    FilteredMask = "Filtered mask"

MIN_CLASTER_SIZE = 100
MIN_BAKING_TIME = 10
CUBE_SIZE_TRESHOLD = 25000
MIN_SIZE_TRESHOLD = 1280*768*0.0005
MIN_SIZE_TRESHOLD_DOWN = 1280*768*0.0005
STRIDE = 3
IMG_NUM = 10 #16
def doit(ful_path):
    dir = ful_path.name
    images = sorted((source/"images"/dir).iterdir())
    masks = sorted((source/"masks"/dir).iterdir())
    if (source/"jsons"/dir/"settings.txt").exists():
        settings = eval(Path(source/"jsons"/dir/"settings.txt").read_text().replace("array", ""))
    else:
        return Exit.NoSettingsInfo

    if (source/"jsons"/dir/"corr.json").exists():
        with open(source/"jsons"/dir/"corr.json") as json_file:
            clss2psinds = json.load(json_file)
    else:
        return Exit.NoSettingsInfo

    if len(images) != IMG_NUM:
        return Exit.MaloImages

    if len(masks) != IMG_NUM:
        return Exit.MaloMasks

    if not settings.get("finished"):
        return Exit.NotFinished

    baking_time = settings.get("baking_time")
    is_water = False #settings.get("should_water_flow")
    is_lqg = settings.get("is_liquigen")
    is_smoke = settings.get("should_add_smoke")
    is_down = 'down' in str(source)
    if is_smoke:
        smoke_masks = sorted((source/"smoke_masks"/dir).iterdir())
        if len(smoke_masks) != IMG_NUM:
            return Exit.MaloSmokeMasks

    is_cube_res = is_cube(masks)
    if (baking_time is None or
            (not is_lqg and is_water and
             (baking_time == "Not stated" or baking_time < MIN_BAKING_TIME or is_cube_res))):
        shutil.copy(ful_path/"Image0010.jpg",
                    tresh/f"error={Exit.BrokenBaking}_baking_time={int(baking_time)}_pixsum={int(is_cube_res)}_dir={dir}.jpg")
        return Exit.BrokenBaking


    for s in range(len(images)//STRIDE):

        sample = {}
        for i in range(STRIDE):
            sample[i] = cv2.imread(str(images[s*STRIDE+i]))
        base_mask = cv2.imread(str(masks[s*STRIDE+STRIDE-1]))
        smoke_mask = None
        if is_smoke:
            smoke_mask = cv2.imread(str(smoke_masks[s*STRIDE+STRIDE-1]))
        divide_masks(base_mask, clss2psinds, sample, smoke_mask, is_down)
        filter_res, pix_sum = filter_masks(sample, is_down, is_water)
        if filter_res:
            save_sample(sample, dataset/f"images/train/{papka}_{dir}_{s}")
        else:
            shutil.copy(str(images[s*STRIDE+STRIDE-1]),
                        tresh / f"error={Exit.FilteredMask}_pixsum={int(pix_sum)}_dir={dir}.jpg")

            #return Exit.FilteredMask

    return Exit.Success




papkas = [
"PC WATERluja 16.06.25",
"106 WATERluja 16.06.25",
# "PC LUJA 6.06.25"
#   DOWN TRUE 27k+17k = 44k | 27k FALSES
# "105 WATERdown 8.05.25",        #27k
# "105 lqgWATERdown 15.05.25",    #17k
#
# "UMAR noWATERdown 12.05.25",    #27k
# "105 noWATERdown 16.05.25",     #5k
# "PC noWATERup 8.05.25",         #12k
# "PC noWATERup 15.05.25",        #12k
# "106 noWATERup 16.05.25",       #3k
# "106 noWATERdownCentHuman 20.05.25",    #27k
# "106 noWATERdownPaket 22.05.25",
#
# #   UP TRUE 42k+15k = 57k | 24k FALSES
# "106 WATERup 8.05.25",          #10k
# "106 WATERup 15.05.25",         #12k
# "104 WATERup 8.05.25",          #20k
# "104 lqgWATERup 12.05.25",      #15k
]
for papka in papkas:
    print('\n'+papka)
    # source = Path(f"/home/popovpe/blender-4.1.1-linux-x64/results/{papka}/water_flow")
    source = Path(f"/vol2/WATER/BLENDER_DATASETS/WATER 5.25/{papka}/water_flow")
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

            for key in yolo_class_indexes.keys():
                shutil.move(dataset/f"mask_{key}/train"/f"{x.name}.png", dataset/f"mask_{key}/val"/f"{x.name}.png")

    for key, value in Exit._member_map_.items():
        print(f"{key}: {exit_codes.count(value)}")

    with open(dataset/"stats.txt", 'w') as f:
        for key, value in Exit._member_map_.items():
            print(f"{key}: {exit_codes.count(value)}", file=f)














