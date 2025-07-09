import os
import shutil
import imageio.v3 as imio
import cv2
import json
import numpy as np
import tqdm
from multiprocessing import Pool

DATASET_STORE_DIR = "/home/nvi/ws.shoshin/dts_for_yolo/"
DATASET_VERSION = 25
DATASET_NAME = f"armature_v{DATASET_VERSION}"
BLENDER_GENERATIONS_DIR = f"/mnt/hdd/ws.shoshin/blender_generations/{DATASET_NAME}/"
MKP_DRAW_PATH = f"/home/nvi/ws.shoshin/arbitrary/vis_armature_v{DATASET_VERSION}/"
WORKERS_POOL = 16

# create dataset store dir
shutil.rmtree(os.path.join(DATASET_STORE_DIR, DATASET_NAME), ignore_errors=True)
for dt in ["images", "labels", "labels_bboxes"]:
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_STORE_DIR, DATASET_NAME, dt, split))

# get blender generations
avd = []
for root, directories, files in os.walk(BLENDER_GENERATIONS_DIR, topdown=False):
    for name in files:
        if (".jpg" in name):
            avd.append(os.path.join(root, name))
avd = set(sorted(avd))
avd = sorted(list(avd))

DT_SIZE = len(avd)
print("Total blender generated samples:", DT_SIZE)

# init params
check_img = imio.imread(avd[0])
H, W, _ = check_img.shape
print("Image sizes:", (W, H))

CLASSES = {
    0: "flance",
    1: "gate",
    2: "cross",
    3: "wheel",
    4: "shaft",
    5: "preventor",
    6: "doliv",
    7: "two_rods",
    8: "one_rod",
    9: "hookah",
    10: "dolconn",
    11: "hsw",
    12: "hbw",
    13: "ppipka",
    14: "dolpipe",
}

SIZE_THRESHOLDS = {
    0: (90, 30),
    1: (40, 40),
    2: (50, 50),
    3: (8, 25),
    4: (14, 14),
    5: (150, 100),
    6: (16, 16),
    7: (100, 30),
    8: (100, 10),
    9: (40, 40),
    10: (16, 32),
    11: (25, 25),
    12: (120, 150),
    13: (14, 14),
    14: (40, 40),
}

COLORS = [(255, 140, 0),
          (124, 252, 0),
          (255, 69, 0),
          (0, 255, 255),
          (255, 20, 147),
          (240, 230, 140),
          (255, 192, 203),
          (0, 0, 128),
          (113, 51, 51),
          (57, 133, 133),
          (32, 167, 234),
          (255, 64, 255),
          (192, 192, 192),
          (140, 230, 240),
          (167, 167, 32)]

VAL_IS_EVERY_NTH = 10

TWO_PIPES_CLS = 7
ONE_PIPE_CLS = 8


def key_to_class(key):
    clss = None

    if ("flance" in key):
        clss = 0
    elif ("cross" in key):
        clss = 2
    elif ("preventor_passes" in key):
        clss = 5
    elif ("doliv" in key):
        clss = 6
    elif ("hookah" in key):
        clss = 9
    elif ("dolconn" in key):
        clss = 10
    elif ("hsw" in key):
        clss = 11
    elif ("hbw" in key):
        clss = 12
    elif ("preventor_pipkas_passes" in key):
        clss = 13
    elif ("dolpipe" in key):
        clss = 14
    else:
        print("Warning! Detected undefined class!")

    return clss


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


def mask_to_contours(mask, segments, cur_clss):
    c = cv2.findContours(mask[..., 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

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
        segments.append(line)

    return segments


def proc_item(cind):
    p = avd[cind]

    try:
        mask_path = p.replace("images", "masks").replace(".jpg", ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        dict_path = p.replace("images", "jsons").replace("Image0001.jpg", "corr.json")
        with open(dict_path, "r") as f:
            corr_dict = json.load(f)

        bboxes = []
        segments = []
        pipes_left = []
        pipes_right = []

        cam_sign = corr_dict["cam_sign"][0]

        # everything except pipes
        for key in corr_dict.keys():
            if ("cam_sign" in key):
                continue

            if (("other" not in key) and ("nut" not in key) and ("gate" not in key) and ("sidefl" not in key)):
                for pass_ind in corr_dict[key]:
                    inds = np.where(mask[..., 0] == pass_ind)
                    if (len(inds[0]) != 0):
                        xmin = min(inds[1])
                        xmax = max(inds[1])
                        ymin = min(inds[0])
                        ymax = max(inds[0])

                        if ("preventor_pipes" not in key):
                            xc = (xmax + xmin) // 2 / W
                            yc = (ymax + ymin) // 2 / H
                            w = (xmax - xmin) / W
                            h = (ymax - ymin) / H

                            cur_clss = key_to_class(key)

                            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >=
                                    SIZE_THRESHOLDS[cur_clss][1]):
                                bboxes.append(
                                    str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                                new_mask[inds] = 255
                                segments = mask_to_contours(new_mask, segments, cur_clss)

                        else:
                            if ("preventor_pipes_left" in key):
                                pipes_left.append([xmin, xmax, ymin, ymax, inds])
                            elif ("preventor_pipes_right" in key):
                                pipes_right.append([xmin, xmax, ymin, ymax, inds])
            elif ("gate" in key):
                for pass_pack in corr_dict[key]:
                    inds = np.where((mask[..., 0] == pass_pack[0]) | (mask[..., 0] == pass_pack[1]))
                    if (len(inds[0]) != 0):
                        xmin = min(inds[1])
                        xmax = max(inds[1])
                        ymin = min(inds[0])
                        ymax = max(inds[0])

                        xc = (xmax + xmin) // 2 / W
                        yc = (ymax + ymin) // 2 / H
                        w = (xmax - xmin) / W
                        h = (ymax - ymin) / H

                        cur_clss = 1

                        if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >=
                                SIZE_THRESHOLDS[cur_clss][1]):
                            bboxes.append(str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                            new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                            new_mask[inds] = 255
                            segments = mask_to_contours(new_mask, segments, cur_clss)

                    inds = np.where(mask[..., 0] == pass_pack[1])
                    if (len(inds[0]) != 0):
                        xmin = min(inds[1])
                        xmax = max(inds[1])
                        ymin = min(inds[0])
                        ymax = max(inds[0])

                        xc = (xmax + xmin) // 2 / W
                        yc = (ymax + ymin) // 2 / H
                        w = (xmax - xmin) / W
                        h = (ymax - ymin) / H

                        if ("is" in key):
                            cur_clss = 3
                        else:
                            cur_clss = 4

                        if ("is" in key or (("is" not in key) and (cam_sign == 1))):
                            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >=
                                    SIZE_THRESHOLDS[cur_clss][1]):
                                bboxes.append(
                                    str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                                new_mask[inds] = 255
                                segments = mask_to_contours(new_mask, segments, cur_clss)

        # pipes
        if (len(pipes_left) == 2):
            pipe1 = pipes_left[0]
            pipe2 = pipes_left[1]

            xmin = min(pipe1[0], pipe2[0])
            xmax = max(pipe1[1], pipe2[1])
            ymin = min(pipe1[2], pipe2[2])
            ymax = max(pipe1[3], pipe2[3])

            inds1 = pipe1[4]
            inds2 = pipe2[4]

            xc = (xmax + xmin) // 2 / W
            yc = (ymax + ymin) // 2 / H
            w = (xmax - xmin) / W
            h = (ymax - ymin) / H

            cur_clss = TWO_PIPES_CLS

            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >= SIZE_THRESHOLDS[cur_clss][1]):
                bboxes.append(str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                new_mask[inds1] = 255
                new_mask[inds2] = 255
                segments = mask_to_contours(new_mask, segments, cur_clss)
        elif (len(pipes_left) == 1):
            xmin, xmax, ymin, ymax, inds = pipes_left[0]

            xc = (xmax + xmin) // 2 / W
            yc = (ymax + ymin) // 2 / H
            w = (xmax - xmin) / W
            h = (ymax - ymin) / H

            cur_clss = ONE_PIPE_CLS

            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >= SIZE_THRESHOLDS[cur_clss][1]):
                bboxes.append(str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                new_mask[inds] = 255
                segments = mask_to_contours(new_mask, segments, cur_clss)

        if (len(pipes_right) == 2):
            pipe1 = pipes_right[0]
            pipe2 = pipes_right[1]

            xmin = min(pipe1[0], pipe2[0])
            xmax = max(pipe1[1], pipe2[1])
            ymin = min(pipe1[2], pipe2[2])
            ymax = max(pipe1[3], pipe2[3])

            inds1 = pipe1[4]
            inds2 = pipe2[4]

            xc = (xmax + xmin) // 2 / W
            yc = (ymax + ymin) // 2 / H
            w = (xmax - xmin) / W
            h = (ymax - ymin) / H

            cur_clss = TWO_PIPES_CLS

            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >= SIZE_THRESHOLDS[cur_clss][1]):
                bboxes.append(str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                new_mask[inds1] = 255
                new_mask[inds2] = 255
                segments = mask_to_contours(new_mask, segments, cur_clss)
        elif (len(pipes_right) == 1):
            xmin, xmax, ymin, ymax, inds = pipes_right[0]

            xc = (xmax + xmin) // 2 / W
            yc = (ymax + ymin) // 2 / H
            w = (xmax - xmin) / W
            h = (ymax - ymin) / H

            cur_clss = ONE_PIPE_CLS

            if ((xmax - xmin) >= SIZE_THRESHOLDS[cur_clss][0] and (ymax - ymin) >= SIZE_THRESHOLDS[cur_clss][1]):
                bboxes.append(str(cur_clss) + " " + str(xc) + " " + str(yc) + " " + str(w) + " " + str(h))
                # new_mask[inds] = COLORS[cur_clss]
                new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]
                new_mask[inds] = 255
                segments = mask_to_contours(new_mask, segments, cur_clss)

        split = "train" if cind % VAL_IS_EVERY_NTH != 0 else "val"
        shutil.copyfile(p, os.path.join(DATASET_STORE_DIR, DATASET_NAME, "images", split, f"{str(cind).zfill(5)}.jpg"))

        with open(os.path.join(DATASET_STORE_DIR, DATASET_NAME, "labels", split, f"{str(cind).zfill(5)}.txt"),
                  "w+") as f:
            for line in segments:
                print(line, file=f)

        with open(os.path.join(DATASET_STORE_DIR, DATASET_NAME, "labels_bboxes", split, f"{str(cind).zfill(5)}.txt"),
                  "w+") as f:
            for line in bboxes:
                print(line, file=f)
    except:
        return


print("Processing dataset...")
pool = Pool(WORKERS_POOL)
results = list(tqdm.tqdm(pool.imap(proc_item, range(DT_SIZE)), total=DT_SIZE))

# draw mkp bboxes to check if they are correct
avd = []
for root, directories, files in os.walk(os.path.join(DATASET_STORE_DIR, DATASET_NAME, "images"), topdown=False):
    for name in files:
        if (".jpg" in name):
            avd.append(os.path.join(root, name))
avd = set(sorted(avd))
avd = sorted(list(avd))
print("Found created dataset samples:", len(avd))

shutil.rmtree(MKP_DRAW_PATH, ignore_errors=True)
os.makedirs(MKP_DRAW_PATH)

print("Drawing examples...")
for i, p in enumerate(tqdm.tqdm(avd[:50])):
    img = imio.imread(p)

    with open(p.replace("images", "labels_bboxes").replace(".jpg", ".txt"), "r") as f:
        mkp = f.readlines()

    for bb in mkp:
        bb = bb[:-1]

        clss, xc, yc, w, h = bb.split(" ")
        clss = int(clss)
        xc = int(float(xc) * W)
        yc = int(float(yc) * H)
        w = int(float(w) * W)
        h = int(float(h) * H)

        xs = xc - w // 2
        ys = yc - h // 2

        img = cv2.rectangle(img, (xs, ys), (xs + w, ys + h), COLORS[clss], 2)

    imio.imwrite(os.path.join(MKP_DRAW_PATH, f"{str(i).zfill(5)}.jpg"), img)