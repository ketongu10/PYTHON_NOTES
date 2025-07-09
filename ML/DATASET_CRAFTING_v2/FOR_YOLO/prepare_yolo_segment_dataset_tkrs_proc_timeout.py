import shutil
import os
import json
import cv2
import numpy as np
import imageio as imio
from time import time
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import tqdm
from threading import Thread


def is_intersected(bbox1, bbox2):
    x1, y1, w1, h1, s1, _ = bbox1
    x2, y2, w2, h2, s2, _ = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    return  ((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2 or b1_x1 <= b2_x2 <= b1_x2)
             and (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2 or b1_y1 <= b2_y2 <= b1_y2))

def unite_bboxes(bbox1, bbox2):
    x1, y1, w1, h1, s1, inds1 = bbox1
    x2, y2, w2, h2, s2, inds2 = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    b3_x1, b3_x2 = np.min([b1_x1, b2_x1]), np.max([b1_x2, b2_x2])
    b3_y1, b3_y2 = np.min([b1_y1, b2_y1]), np.max([b1_y2, b2_y2])
    x3, y3, w3, h3, s3, inds3 = (b3_x1 + b3_x2)/2, (b3_y1 + b3_y2)/2, (b3_x2 - b3_x1), (b3_y2 - b3_y1), s1+s2, (np.hstack((inds1[0], inds2[0])), np.hstack((inds1[1], inds2[1])))
    #print(np.shape(inds1), np.shape(inds2), np.shape(inds3),max(inds1[0]), max(inds1[1]), max(inds2[0]), max(inds2[1]), max(inds3[0]), max(inds3[1]), type(inds1), type(inds3))
    return [x3, y3, w3, h3, s3, inds3]


def merge_intersected_bboxes(bboxs):
    if len(bboxs)>1:
        for i in range(len(bboxs)-1):
            for j in range(i+1, len(bboxs)):
                bbox1, bbox2 = bboxs[i], bboxs[j]
                if is_intersected(bbox1, bbox2):
                    #print('unite')
                    new_bbox = unite_bboxes(bbox1, bbox2)
                    #print(i, j, new_bbox)
                    bboxs[j] = new_bbox
                    bboxs[i] = None
                    break
    new_bboxes = []
    for bbox in bboxs:
        if bbox:
            new_bboxes.append(bbox)
    return new_bboxes



bldr_clss_as_yolo_clss = {
    "pipe_otsos": {'classes':["pipe_w_shlang"], 'task': 'FOR_EACH'},
    "tb_block": {'classes':["TB_red_block"], 'task': 'FOR_EACH'},
    "tb_ear_sq": {'classes':["TB_clevis_base"], 'task': 'FOR_EACH'},
    "tb_ear_rd": {'classes':["TB_clevis_round"], 'task': 'FOR_EACH'},
    "tb_strops": {'classes':["TB_slings"], 'task': 'FOR_EACH'},
    "tb_elev_tall": {'classes':["TB_elevator_tall"], 'task': 'FOR_EACH'},
    "tb_elev_wide": {'classes':["TB_elevator_wide", "elevator_fp_on_pipe"], 'task': 'FOR_EACH'},
    "rotor": {'classes':["rotor_base"], 'task': 'FOR_EACH'},
    "rotor_holder": {'classes':["rotor_holder"], 'task': 'FOR_EACH'},
    "gksh": {'classes':["gksh_1500", "gksh_1800", "gksh_holder"], 'task': 'UNITE'},
    "kops_gate": {'classes': ["kops_gate"], 'task': 'UNITE'},
    "kops_other": {'classes': ["kops_kran", "kops_zajim"], 'task': 'FOR_EACH'},
    #"kops_tros": {'classes':["kops_tros"], 'task': 'FOR_EACH'},
    "wheel_on_stick": {'classes':["wheel_on_stick"], 'task': 'FOR_EACH'},
    "gis_tros": {'classes':["gis_tros_wheel_vertical", "gis_tros_wheel_horizontal", "gis_tros_pipe"], 'task': 'FOR_EACH'},
    "ecn_tros": {'classes':["ecn_tros"], 'task': 'FOR_EACH'},
    "spider": {'classes':["spider"], "task": 'FOR_EACH'},
    "pipe_1_end": {'classes':["pipe_1_end",
                              "ksh_knot",
                              "ksh_knot_fat",
                              "ksh_short_kran",
                              "ksh_head",
                              "ksh_head_last",
                              "not_ksh_knot",
                              "not_ksh_knot_last"], 'task': 'UNITE'},

}


# classes_from_json  = {"kops": [130, 131, 132],
#                       "kops_tros": [133],
#                       "wheel_on_stick": [6],
#                       "gis_tros": [125],
#                       "ecn_tros": [120],
#                       "spider": [8],
#                       "pipe_1_end": [1, 2, 20, 21, 22, 23, 200, 201, 202, 203, 204, 240, 241]}

reference = {
    "pipe_1_end": [1],
    "ksh_knot": [2],
    "grapple": [3],
    "wheel_on_stick": [6],
    "rotor_base": [7],
    "rotor_holder": [77],
    "spider": [8],
    "pumka": [9],
    "ksh_knot_fat": [20],
    "ksh_short_kran": [21],
    "ksh_head": [22],
    "ksh_head_last": [23],
    "vstavka_n2": [31],
    "gloves": [32],
    "elevator_fp_on_pipe": [35],
    "zatychka": [41],
    "zatychka_w_shlang": [42],
    "zatychka_w_monometr": [43],
    "pumka_base": [91],
    "up_gate": [95],
    "gate_flance_up": [96],
    "gate_flance_down": [97],
    "TB_red_block": [101],
    "TB_clevis_base": [102],
    "TB_clevis_round": [103],
    "TB_slings": [104],
    "TB_elevator_tall": [105],
    "TB_elevator_wide": [106],
    "flance": [107],
    "pipe_w_shlang": [110],
    "ecn_tros": [120],
    "gis_tros_wheel_vertical": [125],
    "gis_tros_wheel_horizontal": [126],
    "gis_tros_pipe": [127],
    "kops_gate": [130],
    "kops_kran": [131],
    "kops_zajim": [132],
    "kops_tros": [133],
    "random_tros": [140],
    "gksh_1500": [150],
    "gksh_1800": [180],
    "gksh_holder": [170],
    "not_ksh_knot": [200],
    "not_ksh_knot_last": [240],
}


#classes = {"kops": 0, "kops_tros": 1, "wheel_on_stick": 2, "gis_tros": 3, "ecn_tros": 4, "pipe_1_end": 5, "spider": 6}
yolo_class_indexes = {"pipe_1_end": 0, "ecn_tros": 1, "spider": 2,
                      "kops_gate": 3, "kops_other":4,
                      "wheel_on_stick": 5, "gis_tros": 6,
                      "rotor_holder": 7, "rotor": 8,
                      "gksh": 9,
                      "tb_block":10, "tb_ear_sq":11,"tb_ear_rd":12,
                      "tb_strops":13,
                      "pipe_otsos":14,
                      }#"tb_elev_tall":14, "tb_elev_wide":15} #, "kops_tros": 6}


# min volume is for preventing covering ksh and vstavka by gksh and tal_block
coef = 0.1
min_volume = {"pipe_1_end": 4806*coef,
              "ecn_tros": 5018*coef,
              "spider": 19490*coef,
              "kops_gate": 9514*coef, #11497*coef,
              "kops_other": 7879*coef, #11497*coef,
              "wheel_on_stick": 5099*coef, #5099
              "gis_tros": 2984*coef,
              "kops_tros": 2592*coef,
              "rotor": 35009*coef,
              "rotor_holder": 7530*coef,
              "gksh": 42176*0.3,
              "tb_block": 33611*coef,
              "tb_ear_sq": 10426*coef,
              "tb_ear_rd": 7958*coef,
              "tb_strops": 14483*coef,
              "pipe_otsos": 8470*coef,
              }



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

def gen_bbox(mask, clss2psinds, key):
    bboxes = []
    H, W, C = mask.shape
    if bldr_clss_as_yolo_clss[key]['task'] == 'UNITE':
        inds_list = []
        for clss in bldr_clss_as_yolo_clss[key]['classes']:
            inds_list += clss2psinds[clss]

        cond = False
        for psind in inds_list:
            cond |= (mask[..., 0] == psind)

        pixel_inds = np.where(cond)
        if (len(pixel_inds[0]) * len(pixel_inds[1]) != 0):
            xs = min(pixel_inds[1])
            ys = min(pixel_inds[0])
            xf = max(pixel_inds[1])
            yf = max(pixel_inds[0])

            xc = (xs + xf) / 2 / W
            yc = (ys + yf) / 2 / H
            w = (xf - xs) / W
            h = (yf - ys) / H
            bboxes.append([xc, yc, w, h, len(pixel_inds[0]), pixel_inds])

    elif bldr_clss_as_yolo_clss[key]['task'] == 'FOR_EACH':
        for clss in bldr_clss_as_yolo_clss[key]['classes']:
            for psind in clss2psinds[clss]:
                pixel_inds = np.where(mask[..., 0] == psind)

                if (len(pixel_inds[0]) * len(pixel_inds[1]) != 0):
                    xs = min(pixel_inds[1])
                    ys = min(pixel_inds[0])
                    xf = max(pixel_inds[1])
                    yf = max(pixel_inds[0])

                    xc = (xs + xf) / 2 / W
                    yc = (ys + yf) / 2 / H
                    w = (xf - xs) / W
                    h = (yf - ys) / H
                    # if clss == "not_ksh_knot_last" and (xf - xs > 100 or yf - ys > 80):  # for broken inserting pipes
                    #     bboxes.append([50, 50, 11, 12, 1])
                    # else:
                    bboxes.append([xc, yc, w, h, len(pixel_inds[0]), pixel_inds])
        if key != "gis_tros":
            bboxes = merge_intersected_bboxes(bboxes)
    return bboxes if bboxes else None

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

def border_collision_is_passed(bboxs, key):
    for bbox in bboxs:
        xc, yc, w, h, s, inds = bbox
        if s < min_volume[key]:
            print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: too small {s} < {min_volume[key]}")
            return False
        # if (1 - yc - h/2 < pixel_dx or
        #     1 - xc - w/2 < pixel_dx or
        #     yc - h/2 < pixel_dx or
        #     xc - w/2 < pixel_dx):
        #     print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: bounds intersection")
        #     return False
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
    #print(len(c))


    #exit()
    if (len(c) > 1):
        #print(len(c), c[1].shape, np.min(c[1][:,:,0]), np.max(c[1][:,:,0]))
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


def timeout(func, TIMEOUT=5):
    def wrapper(*args, **kwargs):
        t0 = time()
        Timer = Thread(target=func, args=args)
        print("Before calling the function.")
        func(*args)
        print("After calling the function.")

    return wrapper

def doit(dir):

    sh_progress_pool = SharedMemory("shared_progress", create=False)
    c_pool = np.ndarray((2 * len(yolo_class_indexes.keys()) + 1,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[-1]+=1
    #print(f"progress = {c_pool[-1]}")
    if os.path.exists(rooot + "/images/" + dir) and os.path.exists(rooot + "/masks/" + dir) and os.path.exists(rooot + "/jsons/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        with open(dataset+f"/{bbox_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        with open(dataset+f"/{segment_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass

        mask_path = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        try:
            mask = imio.v3.imread(mask_path)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH MASK!!!!!! {dir}{Style.RESET_ALL}")
            shutil.rmtree(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"))
            shutil.rmtree(dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt")
            shutil.rmtree(dataset + f"/{bbox_labels_name}/train/" + dir + f"_{dataset_name}.txt")
            return 'deleted'

        try:
            with open(rooot + "/jsons/" + dir + "/corr.json") as json_file:
                clss2psinds = json.load(json_file)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH JSON!!!!!! {dir}{Style.RESET_ALL}")
            shutil.rmtree(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
            shutil.rmtree(dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt")
            shutil.rmtree(dataset + f"/{bbox_labels_name}/train/" + dir + f"_{dataset_name}.txt")
            return 'deleted'

        label = {}
        for key in yolo_class_indexes.keys():
            #providing class list in json equals class list here
            #pass_ind = classes_from_json[key] #int(json_ind_dict[key])
            label[key] = gen_bbox(mask, clss2psinds, key)
            if (label[key] is not None):

                # if (key != "rotor" or key == "rotor" and label.get("rotor_holder")) \
                #         and not border_collision_is_passed(label[key], key):    #DOES NOT CATCH rot_holder collision_not_passed
                #     # (key == "kops" or key == "wheel_on_stick" or key == "spider" or key == "pipe_1_end") and
                #     # deleting crossing bboxes and logging to tresh
                #
                #     shutil.move(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"),
                #                 os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                #     return dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt"
                # else:
                #     c_ind = list(yolo_class_indexes.keys()).index(key)
                #     for bbox in label[key]:
                #         c_pool[2*c_ind] += bbox[4]
                #         c_pool[2*c_ind+1] += 1
                #     segments = []
                #     for bbox in label[key]:
                #         #print(bbox[5])
                #
                #         new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                #         #print(np.max(bbox[5][0]), np.max(bbox[5][1]), len(label[key]) , new_mask.shape)
                #         # try:
                #         new_mask[bbox[5]] = 255
                #         # except:
                #
                #             # print('DNO', np.max(bbox[5][0]), np.max(bbox[5][1]), len(label[key]))
                #             # exit()
                #         segments.append(mask_to_contours(new_mask, yolo_class_indexes[key]))
                #
                #     with open(dataset+f"/{segment_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                #         for seg in segments:
                #             print(seg, file=f)
                #     with open(dataset+f"/{bbox_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                #         for bbox in label[key]:
                #             print(yolo_class_indexes[key], *(bbox[:4]), file=f)
                if (key != "rotor" or key == "rotor" and label.get("rotor_holder") and
                    border_collision_is_passed(label["rotor_holder"], "rotor_holder")) \
                        and border_collision_is_passed(label[key], key):
                    c_ind = list(yolo_class_indexes.keys()).index(key)
                    for bbox in label[key]:
                        c_pool[2 * c_ind] += bbox[4]
                        c_pool[2 * c_ind + 1] += 1
                    segments = []
                    for bbox in label[key]:
                        new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                        new_mask[bbox[5]] = 255
                        segments.append(mask_to_contours(new_mask, yolo_class_indexes[key]))

                    with open(dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt", 'a') as f:
                        for seg in segments:
                            print(seg, file=f)
                    with open(dataset + f"/{bbox_labels_name}/train/" + dir + f"_{dataset_name}.txt", 'a') as f:
                        for bbox in label[key]:
                            print(yolo_class_indexes[key], *(bbox[:4]), file=f)

                else:
                    shutil.move(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"),
                                os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                    return dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt"
# sh_progress = SharedMemory("shared_progress")
# sh_progress.close()
# exit()
sh_progress = SharedMemory("shared_progress", create=True, size=8*(1 + 2 * len(yolo_class_indexes.keys())))
dataset_names = os.listdir("/vol1/KSH/source/") #"PC KSH 19.08"
dataset_names = [

"106 PROC 24.02.25"
#"UMAR KSH_relabel 10.12"
#"PC PROC 10.02",
# "106 PROC 6.02",
# "106 PROC 10.02",

# "PC PROC_ecn 5.12",
# "116 PROC_ecn 5.12"
# "106 DEPTH 2.12",
# "PC DEPTH 2.12",
# "UMAR DEPTH 2.12",

# "UMAR PROC 28.11",
# "106 PROC 28.11",
# "PC PROC 28.11"
#"PC PROC 22.11_gis"
#"106 PROC 22.11_gis"
# "PC PROC 18.11",
# "106 PROC 16.11",
# "106 PROC 18.11",

]
for dataset_name in dataset_names:
    rooot = f"/vol1/KSH/source/PROC/SOURCE PROC 10.02.25/{dataset_name}/tkrs_proc"
    dataset = "/vol1/KSH/dataset/"+dataset_name
    tresh = "/vol1/KSH/tresh"

    bbox_labels_name = 'labels_bbox'
    segment_labels_name = 'labels'

    create_dirs()

    pixel_dx = 0.0001
    sh_progress = SharedMemory("shared_progress") #, create=False, size=8)
    c = np.ndarray((2 * len(yolo_class_indexes.keys()) + 1,), dtype=np.int64, buffer=sh_progress.buf)
    c[:] = 1

    data_to_process = os.listdir(os.path.join(rooot, "images"))[:-50] #53300
    with Pool(processes=6) as p:
        results = list(tqdm.tqdm(p.imap(doit, data_to_process), total=len(data_to_process)))
    # results = []
    # for dtp in data_to_process:
    #     results.append(doit(dtp))
    treshed = 0
    for path in results:
        if path:
            treshed+=1
            if path != 'deleted':
                os.remove(path)
                os.remove(path.replace(segment_labels_name, bbox_labels_name))

    print(f'TOTAL IMAGES FOUND = {c[-1]} | MOVED TO TRESH = {treshed}')
    for key in yolo_class_indexes.keys():
        c_ind = list(yolo_class_indexes.keys()).index(key)
        print(f'{key}: AVER PIX = {c[2*c_ind]//c[2*c_ind+1]} | NOW PIXELS = {min_volume.get(key)} | TOTAL FOUND = {c[2*c_ind+1]}')
    with open(dataset+"/dataset_info.txt", "w+") as f:
        print(f'TOTAL IMAGES FOUND = {c[-1]} | MOVED TO TRESH = {treshed}', file=f)
        for key in yolo_class_indexes.keys():
            c_ind = list(yolo_class_indexes.keys()).index(key)
            print(f'{key}: AVER PIX = {c[2 * c_ind] // c[2 * c_ind + 1]} | NOW PIXELS = {min_volume.get(key)} | TOTAL FOUND = {c[2 * c_ind + 1]}', file=f)



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
        for key in yolo_class_indexes.keys():
            print(f"  {yolo_class_indexes[key]}: {key}", file=f)

sh_progress.close()
sh_progress.unlink()




