import shutil
import os
import json
import numpy as np
import imageio as imio
from multiprocessing import Pool


classes  = {
    "pipe_1_end": 1,
    "ksh_knot": 2,
    "ksh_short_kran": 3,
    "vstavka_n2": 4,
    "grapple": 5,
    "zatychka": 6,
    "zatychka_w_shlang": 7,
    "wheel_on_stick": 8,
    "rotor": 9,
    "gksh_1500": 10,
    "gksh_1800": 11,
    "spider": 12,
    "worktable": 13,
    "TB_red_block": 14,
    "TB_clevis_base": 15,
    "TB_clevis_round": 16,
    "TB_slings": 17,
    "TB_elevator_tall": 18,
    "TB_elevator_wide": 19,
    "flance": 20,
}


classes = {"ksh_short_kran": 0, "ksh_knot": 1} #, "vstavka_pipe": 3, "vstavka_n2": 3,}
def create_dirs():
    try:
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + "/labels/train")
        os.mkdir(dataset + "/labels/val")
    except:
        shutil.rmtree(dataset + "/images/train")
        shutil.rmtree(dataset + "/images/val")
        shutil.rmtree(dataset + "/labels/train")
        shutil.rmtree(dataset + "/labels/val")
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + "/labels/train")
        os.mkdir(dataset + "/labels/val")


def gen_bbox(mask, pass_ind):
    mask = imio.v3.imread(mask)
    inds = np.where(mask == pass_ind)
    H, W, C = mask.shape
    if (len(inds[0]) * len(inds[1]) != 0):
        xs = min(inds[1])
        ys = min(inds[0])
        xf = max(inds[1])
        yf = max(inds[0])

        xc = (xs + xf) / 2 / W
        yc = (ys + yf) / 2 / H
        w = (xf - xs) / W
        h = (yf - ys) / H

        return xc, yc, w, h

    return None


rooot = "/vol1/KSH/DATASET_BLENDER/source/PC KSH 19.06/ksh_pipes"
dataset = "/vol1/KSH/DATASET_BLENDER/dataset"

create_dirs()

def doit(dir):
    if os.listdir(rooot + "/images/" + dir) and os.listdir(rooot + "/masks/" + dir) and os.listdir(rooot + "/jsons/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + ".jpg"))
        with open(dataset+"/labels/train/"+dir+".txt", 'w+') as f:
            pass
        mask = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        with open(rooot + "/jsons/" + dir + "/corr.json") as json_file:
            json_ind_dict = json.load(json_file)
        for key in classes.keys():
            pass_ind = int(json_ind_dict[key])
            label = gen_bbox(mask, pass_ind)
            if label:
                with open(dataset+"/labels/train/" + dir+".txt", 'a') as f:
                    print(classes[key], *label, file=f)


with Pool(processes=6) as p:
    results = p.map(doit, os.listdir(os.path.join(rooot, "images")))
val_part = 0.1
for x in os.listdir(dataset+"/images/train"):
    if np.random.uniform() < val_part:
        shutil.move(dataset+"/images/train/"+x, dataset+"/images/val/"+x)
        shutil.move(dataset+"/labels/train/"+str(x).replace("jpg", "txt"), dataset+"/labels/val/"+str(x).replace("jpg", "txt"))

with open(dataset+"/data.yaml", "w+") as f:
    print(f"path: ./", file=f)
    print(f"train: images/train", file=f)
    print(f"val: images/val", file=f)
    print(f"test: ", file=f)
    print(f"names:", file=f)
    for key in classes.keys():
        print(f"  {classes[key]}: {key}", file=f)




