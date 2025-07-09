import shutil
import os
import json
import numpy as np
import imageio as imio
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory





classes = {"water": 0}
# min volume is for preventing covering ksh and vstavka by gksh and tal_block

min_bbox = 0



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




def parse_label_v2(path):
    dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            dict[int(data[0])] = [float(coord) for coord in data[1:]]
    return dict



# progress = 0
sh_progress = SharedMemory("shared_progress", create=True, size=24)
c = np.ndarray((3,), dtype=np.int64, buffer=sh_progress.buf)
c[:3] = 0

def doit(dir):
    root = rooot
    # global progress
    # progress += 1
    sh_progress_pool = SharedMemory("shared_progress", create=False)
    c_pool = np.ndarray((3,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[0] += 1
    print(f"PROGRESS: {c_pool[0]} dirs | {c_pool[1]} imgs | {c_pool[2]} trues")

    if 'labels_v2.txt' in os.listdir(os.path.join(root,dir)):
        labels_v2 = parse_label_v2(os.path.join(root, dir, 'labels_v2.txt'))
        imdir = os.path.join(root, 'images', dir)
        if labels_v2:
            os.mkdir(dataset+'/images/train/'+dir)
            os.mkdir(dataset + '/labels/train/' + dir)
            for img in os.listdir(imdir):
                bbox =  labels_v2.get(int(img.replace('.jpg', '')))
                if bbox:
                    if bbox[2] * bbox[3] > min_bbox:
                        shutil.copy(os.path.join(imdir, img), dataset+'/images/train/'+dir+'/'+img)
                        with open(dataset+'/labels/train/'+dir+'/'+img.replace('.jpg', '.txt'), 'w') as f:
                            print(0, *(bbox), file=f)
                        c_pool[2] += 1
                        c_pool[1] += 1
                    else:
                        print(f"THIS {imdir}/{img} WAS DENIED FOR SMALL BBOX {bbox}")
                else:
                    shutil.copy(os.path.join(imdir, img), dataset + '/images/train/' + dir + '/' + img)
                    with open(dataset + '/labels/train/' + dir + '/' + img.replace('.jpg', '.txt'), 'w') as f:
                        pass
                    c_pool[1] += 1

        else:
            os.mkdir(dataset + '/images/train/' + dir)
            os.mkdir(dataset + '/labels/train/' + dir)
            for img in os.listdir(imdir):
                shutil.copy(os.path.join(imdir, img), dataset+'/images/train/'+dir+'/'+img)
                with open(dataset+'/labels/train/'+dir+'/'+img.replace('.jpg', '.txt'), 'w') as f:
                    pass
                c_pool[1] += 1

# DATASET_NAMES = [
#     "105 13.05 up new",
#     "106 13.05 up new",
#     "106 15-07 up",
#     "PC 13.05 up new"
# ]
DATASET_NAMES = ["107 6.05 down new",
    "PC 6.05 down new"]

for dataset_name in DATASET_NAMES:
    rooot = f"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (6.05-00)/{dataset_name}/water_flow"
    dataset = "/vol1/WATER/DATASET/READY/"+dataset_name
    tresh = "/vol1/WATER/DATASET/TRESH"
    create_dirs()

    dirs_to_process = []
    for dir in os.listdir(rooot):
        if dir != 'images':
            dirs_to_process.append(dir)


    with Pool(processes=6) as p:
        results = p.map(doit, dirs_to_process[::])

    val_part = 0.1
    for x in os.listdir(dataset + "/images/train"):
        if np.random.uniform() < val_part:
            print('LOH')
            shutil.move(dataset + "/images/train/" + x, dataset + "/images/val/" + x)
            shutil.move(dataset + "/labels/train/" + x, dataset + "/labels/val/" + x)
            # shutil.move(dataset + "/labels/train/" + str(x).replace("jpg", "txt"),
            #             dataset + "/labels/val/" + str(x).replace("jpg", "txt"))

    with open(dataset + "/data.yaml", "w+") as f:
        print(f"path: ./", file=f)
        print(f"train: images/train", file=f)
        print(f"val: images/val", file=f)
        print(f"test: ", file=f)
        print(f"names:", file=f)
        for key in classes.keys():
            print(f"  {classes[key]}: {key}", file=f)


# for i in os.listdir(os.path.join(rooot, "images")):
#     doit(i)
#     try:
#         print(av_ocup['ksh_knot']/num_ocup['ksh_knot'], av_ocup['vstavka_n2']/num_ocup['vstavka_n2'])
#     except:pass







