import shutil
import os
import json
import cv2
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


classes = {"ksh_knot": 0, "ksh_short_kran": 1, "vstavka_n2": 2, "pipe_1_end": 3, "not_ksh_knot": 4} #, "vstavka_n2": 3,}
# min volume is for preventing covering ksh and vstavka by gksh and tal_block
coef = 0.3
min_volume = {"ksh_knot": 2366*coef,
              "vstavka_n2": 18000*coef}



def create_dirs():
    try:
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + f"/{bbox_labels_name}/train")
        os.mkdir(dataset + f"/{bbox_labels_name}/val")
        os.makedirs(dataset + f"/{segment_labels_name}/train")
        os.mkdir(dataset + f"/{segment_labels_name}/val")
        os.mkdir(tresh)
    except:
        shutil.rmtree(dataset + "/images/train")
        shutil.rmtree(dataset + "/images/val")
        shutil.rmtree(dataset + f"/{bbox_labels_name}/train")
        shutil.rmtree(dataset + f"/{bbox_labels_name}/val")
        shutil.rmtree(dataset + f"/{segment_labels_name}/train")
        shutil.rmtree(dataset + f"/{segment_labels_name}/val")
        shutil.rmtree(tresh)
        os.makedirs(dataset + "/images/train")
        os.mkdir(dataset + "/images/val")
        os.makedirs(dataset + f"/{bbox_labels_name}/train")
        os.mkdir(dataset + f"/{bbox_labels_name}/val")
        os.makedirs(dataset + f"/{segment_labels_name}/train")
        os.mkdir(dataset + f"/{segment_labels_name}/val")
        os.mkdir(tresh)


def gen_bbox(mask, pass_ind):

    inds = np.where((mask[..., 0] == pass_ind[0]) | (mask[..., 0] == pass_ind[-1]))
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
        return xc, yc, w, h, len(inds[0]), inds

    return None

# def gen_bbox_by_coutours(mask, pass_ind):
#     c = cv2.findContours(mask[..., 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
#     num_pixels = np.sum(mask)//255
#     print(len(c), c[1].shape, np.min(c[1][:, :, 0]), np.max(c[1][:, :, 0]), num_pixels)
#     H, W, C = mask.shape
#
#
#     if num_pixels > 100:
#         bboxes = []
#         for counter in c:
#             xs = np.min(counter[:,:,0])
#             xf = np.max(counter[:,:,0])
#
#             ys = np.min(counter[:,:,1])
#             yf = np.max(counter[:,:,1])
#
#             xc = (xs + xf) / 2 / W
#             yc = (ys + yf) / 2 / H
#             w = (xf - xs) / W
#             h = (yf - ys) / H
#             bboxes.append([xc, yc, w, h, len(inds[0]), inds])
#         return bboxes
#
#     return None

def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def min_dist(arr1, arr2):
    """
    Find the shortest distance between two contours.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.min(dis, axis=None)

def border_collision_is_passed(bbox, key):
    xc, yc, w, h, s, inds = bbox
    if s < min_volume[key]:
        print(f"TOO SMALL {s} < {min_volume[key]}")
        return False
    if (1 - yc - h/2 < pixel_dx or
        1 - xc - w/2 < pixel_dx or
        yc - h/2 < pixel_dx or
        xc - w/2 < pixel_dx):
        print('geometry')
        return False
    return True

def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def mask_to_contours(mask, cur_clss, W=1280, H=768):
    c = cv2.findContours(mask[..., 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    print(len(c), c[1].shape, np.min(c[1][:,:,0]), np.max(c[1][:,:,0]))

    #exit()
    if (len(c) > 1):
        c = list(c)
        s = c.pop(0)
        while (len(c) > 0):
            min_dists = []
            for el in c:
                min_dists.append(min_dist(s, el))
            s = merge_multi_segment(tuple([s, c.pop(np.argmin(min_dists))]))
            s = np.concatenate(s, axis=0)
            s = np.expand_dims(s, 1)
        c = tuple([s])

    if (len(c) > 0):
        c = (c[0].reshape(-1, 2) / np.array([W, H])).reshape(-1).tolist()

        line = str(cur_clss) + " "
        for el in c:
            line += str(el) + " "
        line = line[:-1]
        return line


    return None



def doit(dir):

    sh_progress_pool = SharedMemory("shared_progress", create=False)
    c_pool = np.ndarray((3,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[0]+=1
    print(f"progress = {c_pool[0]}")
    if os.listdir(rooot + "/images/" + dir) and os.listdir(rooot + "/masks/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        with open(dataset+f"/{bbox_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        with open(dataset+f"/{segment_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        mask_path = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        mask = imio.v3.imread(mask_path)
        label = {}
        for key in classes.keys():
            #providing class list in json equals class list here
            pass_ind = classes_from_json[key] #int(json_ind_dict[key])
            label[key] = gen_bbox(mask, pass_ind)
            if (label[key] is not None
                and ((key != "pipe_1_end" or key == "pipe_1_end" and label.get("vstavka_n2") is not None)
                and (key != "ksh_short_kran" or key == "ksh_short_kran" and label.get("ksh_knot") is not None))):

                if (key == "vstavka_n2" or key == "ksh_knot") and not border_collision_is_passed(label[key], key):
                    print(label[key][:-1])
                    # deleting crossing bboxes and logging to tresh

                    shutil.move(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"),
                                os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                    return dataset + "/labels/train/" + dir + f"_{dataset_name}.txt"
                else:
                    if key =='ksh_knot':
                        c_pool[1] += label[key][4]
                        c_pool[2] += 1
                    new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]

                    new_mask[label[key][5]] = 255
                    segments = mask_to_contours(new_mask, classes[key])
                    if segments:
                        with open(dataset+f"/{segment_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                            print(segments, file=f)
                    else:
                        print(f'NO SEGMENTS - OBOSRALSYA TY GDE-TO {dir}')
                    with open(dataset+f"/{bbox_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                        print(classes[key], *(label[key][:4]), file=f)


sh_progress = SharedMemory("shared_progress", create=True, size=24)
dataset_names = os.listdir("/vol1/KSH/source") #"PC KSH 19.08"
dataset_names = [
    "super_test"
#'PC KSH 23.08',
#'106 KSH 22.08',
#'114 KSH 20.08-2',
#'116 KSH 23.08',
#'PC KSH 22.08',
#'106 KSH 23.08',
#'114 KSH 20.08',
#'114 KSH 23.08',
# 'PC KSH 20.08',
# 'PC KSH 21.08',
# '116 KSH 20.08',
# 'PC KSH 21.08-2',
]
for dataset_name in dataset_names:
    rooot = f"/vol1/KSH/source/{dataset_name}/ksh_pipes"
    dataset = "/vol1/KSH/dataset/"+dataset_name
    tresh = "/vol1/KSH/tresh"

    bbox_labels_name = 'labels_bbox'
    segment_labels_name = 'labels'
    #os.mkdir(dataset+dataset_name)
    create_dirs()

    pixel_dx = 0.0001
    # progress = 0
    sh_progress = SharedMemory("shared_progress", create=False, size=8)
    c = np.ndarray((3,), dtype=np.int64, buffer=sh_progress.buf)
    c[0:3] = 0

    with Pool(processes=6) as p:
        results = p.map(doit, os.listdir(os.path.join(rooot, "images"))[::])
    #print(f'AVER PIX = {int(c[1]/c[2])} | NOW PIXELS = {min_volume["ksh_knot"]}')
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
            shutil.move(dataset+f"/{bbox_labels_name}/train/"+str(x).replace("jpg", "txt"),
                        dataset+f"/{bbox_labels_name}/val/"+str(x).replace("jpg", "txt"))
            shutil.move(dataset+f"/{segment_labels_name}/train/"+str(x).replace("jpg", "txt"),
                        dataset+f"/{segment_labels_name}/val/"+str(x).replace("jpg", "txt"))

    with open(dataset+"/data.yaml", "w+") as f:
        print(f"path: ./", file=f)
        print(f"train: images/train", file=f)
        print(f"val: images/val", file=f)
        print(f"test: ", file=f)
        print(f"names:", file=f)
        for key in classes.keys():
            print(f"  {classes[key]}: {key}", file=f)




