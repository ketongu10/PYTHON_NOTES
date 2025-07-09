import shutil
import os
import json
import numpy as np
import imageio as imio
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory


classes_from_json  = {
    "pipe_1_end": [22, 1],
    "ksh_knot": [2],
    "ksh_short_kran": [21, 2],
    "vstavka_n2": [22],
    "grapple": [3],
    "zatychka": [4],
    "zatychka_w_shlang": [5],
    "wheel_on_stick": [6],
    "rotor": [7],
    "gksh_1500": [150],
    "gksh_1800": [180],
    "spider": [8],
    "worktable": [111],
    "TB_red_block": [101],
    "TB_clevis_base": [102],
    "TB_clevis_round": [103],
    "TB_slings": [104],
    "TB_elevator_tall": [105],
    "TB_elevator_wide": [106],
    "flance": [107],
    "not_ksh_knot": [200],

}


classes = {"ksh_knot": 0, "ksh_short_kran": 1, "vstavka_n2": 2, "pipe_1_end": 3} #, "vstavka_n2": 3,}
# min volume is for preventing covering ksh and vstavka by gksh and tal_block
coef = 0.3
min_volume = {"ksh_knot": 9500*coef,
              "vstavka_n2": 18000*coef}



def create_dirs():
    try:
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + "/labels/train")
        os.mkdir(dataset + "/labels/val")
        os.mkdir(tresh)
    except:
        shutil.rmtree(dataset + "/images/train")
        shutil.rmtree(dataset + "/images/val")
        shutil.rmtree(dataset + "/labels/train")
        shutil.rmtree(dataset + "/labels/val")
        shutil.rmtree(tresh)
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + "/labels/train")
        os.mkdir(dataset + "/labels/val")
        os.mkdir(tresh)


def gen_bbox(mask, pass_ind):
    mask = imio.v3.imread(mask)
    inds = np.where((mask == pass_ind[0]) | (mask == pass_ind[-1]))
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
        return xc, yc, w, h, len(inds[0])

    return None

def border_collision_is_passed(bbox, key):
    xc, yc, w, h, s = bbox
    if s < min_volume[key]:
        return False
    if (1 - yc - h/2 < pixel_dx or
        1 - xc - w/2 < pixel_dx or
        yc - h/2 < pixel_dx or
        xc - w/2 < pixel_dx):
        return False
    return True

dataset_name = "114 KSH 19.08"
rooot = f"/vol1/KSH/source/{dataset_name}/ksh_pipes"
dataset = "/vol1/KSH/dataset/"+dataset_name
tresh = "/vol1/KSH/tresh"
#os.mkdir(dataset+dataset_name)
create_dirs()

pixel_dx = 0.0001
# progress = 0
sh_progress = SharedMemory("shared_progress", create=True, size=8)
c = np.ndarray((1,), dtype=np.int64, buffer=sh_progress.buf)
c[0] = 0
def doit(dir):
    # global progress
    # progress += 1
    sh_progress_pool = SharedMemory("shared_progress", create=False)
    c_pool = np.ndarray((1,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[0]+=1
    print(f"progress = {c_pool[0]}")
    if os.listdir(rooot + "/images/" + dir) and os.listdir(rooot + "/masks/" + dir) and os.listdir(rooot + "/jsons/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        with open(dataset+"/labels/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        mask = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        with open(rooot + "/jsons/" + dir + "/corr.json") as json_file:
            pass
            # print(dir)
            # json_ind_dict = json.load(json_file)
        label = {}
        for key in classes.keys():
            #providing class list in json equals class list here
            pass_ind = classes_from_json[key] #int(json_ind_dict[key])
            label[key] = gen_bbox(mask, pass_ind)
            if (label[key] is not None
                and ((key != "pipe_1_end" or key == "pipe_1_end" and label.get("vstavka_n2") is not None)
                and (key != "ksh_short_kran" or key == "ksh_short_kran" and label.get("ksh_knot") is not None))):

                if (key == "vstavka_n2" or key == "ksh_knot") and not border_collision_is_passed(label[key], key):
                    print(label[key])
                    # deleting crossing bboxes and logging to tresh

                    shutil.move(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"),
                                os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                    return dataset + "/labels/train/" + dir + f"_{dataset_name}.txt"
                else:
                    with open(dataset+"/labels/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                        print(classes[key], *(label[key][:4]), file=f)


with Pool(processes=6) as p:
    results = p.map(doit, os.listdir(os.path.join(rooot, "images"))[::])

# for i in os.listdir(os.path.join(rooot, "images")):
#     doit(i)
#     try:
#         print(av_ocup['ksh_knot']/num_ocup['ksh_knot'], av_ocup['vstavka_n2']/num_ocup['vstavka_n2'])
#     except:pass

for path in results:
    if path:
        os.remove(path)
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




