import shutil
import os
import json
import cv2
import numpy as np
import imageio as imio
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import tqdm



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
                    print('unite')
                    new_bbox = unite_bboxes(bbox1, bbox2)
                    print(i, j, new_bbox)
                    bboxs[j] = new_bbox
                    bboxs[i] = None
                    break
    new_bboxes = []
    for bbox in bboxs:
        if bbox:
            new_bboxes.append(bbox)
    return new_bboxes



bldr_clss_as_yolo_clss = {
    "box": {'classes': ["lebedka_box", "lebedka_down_rail", "lebedka_up_rail", "lebedka_polsunok"], 'task': 'UNITE'},
    "tros": {'classes':["lebedka_tros"], 'task': 'FOR_EACH'},
    "lebedka_polsunok": {'classes':["lebedka_polsunok"], 'task': 'FOR_EACH'},
    "rails": {'classes':["lebedka_up_rail", "lebedka_down_rail"], 'task': 'UNITE'},
    "width": {'classes': ["ghost_rail"], 'task': 'UNITE'}
}


# classes_from_json  = {"kops": [130, 131, 132],
#                       "kops_tros": [133],
#                       "wheel_on_stick": [6],
#                       "gis_tros": [125],
#                       "ecn_tros": [120],
#                       "spider": [8],
#                       "pipe_1_end": [1, 2, 20, 21, 22, 23, 200, 201, 202, 203, 204, 240, 241]}

reference = {
    "lebedka_tros": [255],
    "lebedka_polsunok": [250],
    "lebedka_up_rail": [230],
    "lebedka_down_rail": [231],
    "lebedka_box": [120],
    "ghost_rail": [100],
}


yolo_class_indexes = {"box": 0, "tros": 1, "lebedka_polsunok": 2, "rails": 3, "width": 4}
# min volume is for preventing covering ksh and vstavka by gksh and tal_block
coef = 0.2
min_volume = {"box": 91056*coef,
              "tros": 3159*coef,
              "lebedka_polsunok": 0*coef, #526
              "rails": 4174*coef,
              "width": 0*coef}



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
        if key != "rails":
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
        if (1 - yc - h/2 < pixel_dx or
            1 - xc - w/2 < pixel_dx or
            yc - h/2 < pixel_dx or
            xc - w/2 < pixel_dx):
            print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: bounds intersection")
            return False

    return True

def min_pixel_size_is_passed(bboxs, key):
    for bbox in bboxs:
        xc, yc, w, h, s, inds = bbox
        if s < min_volume[key]:
            print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: too small {s} < {min_volume[key]}")
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
    print(mask[..., 0].shape)
    c = cv2.findContours(mask[..., 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    #print(len(c))


    #exit()
    if (len(c) > 1):
        #print(len(c), c[1].shape, np.min(c[1][:,:,0]), np.max(c[1][:,:,0]))
        c = list(c)
        s = c.pop(0)
        counter=0
        while (len(c) > 0):
            min_dists = []
            for el in c:
                min_dists.append(min_dist(s, el))
            s = merge_multi_segment(tuple([s, c.pop(np.argmin(min_dists))]))
            s = np.concatenate(s, axis=0)
            s = np.expand_dims(s, 1)

            counter += 1
            if counter > 100:
                print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: seg timeout {counter}")
                return "timeout"

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
    c_pool = np.ndarray((2 * len(yolo_class_indexes.keys()) + 2,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[-1]+=1
    #print(f"progress = {c_pool[-1]}")
    if os.path.exists(rooot + "/images/" + dir) and os.path.exists(rooot + "/masks/" + dir) and os.path.exists(rooot + "/jsons/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        with open(dataset+f"/{bbox_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        with open(dataset+f"/{segment_labels_name}/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass

        mask_path_default = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        mask_path_polsunok = os.path.join(rooot, "masks_polsunok_no_rails/", dir, img_name.replace('jpg', 'png'))
        mask_path_width = os.path.join(rooot, "masks_width/", dir, img_name.replace('jpg', 'png'))
        try:
            mask_default = imio.v3.imread(mask_path_default)
            mask_polsunok = imio.v3.imread(mask_path_polsunok)
            mask_width = imio.v3.imread(mask_path_width)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH MASK!!!!!! {dir}{Style.RESET_ALL}")
            shutil.rmtree(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"), ignore_errors=True)
            shutil.rmtree(dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt", ignore_errors=True)
            shutil.rmtree(dataset + f"/{bbox_labels_name}/train/" + dir + f"_{dataset_name}.txt", ignore_errors=True)
            return 'deleted'



        try:
            with open(rooot + "/jsons/" + dir + "/corr.json") as json_file:
                clss2psinds = json.load(json_file)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH JSON!!!!!! {dir}{Style.RESET_ALL}")

            shutil.rmtree(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"), ignore_errors=True)
            shutil.rmtree(dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt", ignore_errors=True)
            shutil.rmtree(dataset + f"/{bbox_labels_name}/train/" + dir + f"_{dataset_name}.txt", ignore_errors=True)
            return 'deleted'

        label = {}
        for key in yolo_class_indexes.keys():
            mask = mask_default
            if  key == "lebedka_polsunok":
                mask = mask_polsunok
            if  key == "width":
                mask = mask_width

            #providing class list in json equals class list here
            #pass_ind = classes_from_json[key] #int(json_ind_dict[key])
            label[key] = gen_bbox(mask, clss2psinds, key)
            if (label[key] is not None):

                if (not min_pixel_size_is_passed(label[key], key)  or
                    (not border_collision_is_passed(label[key], key) if key == "lebedka_polsunok" else False)):

                    # deleting crossing bboxes and logging to tresh
                    shutil.move(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"),
                                os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                    return dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt"
                else:
                    c_ind = list(yolo_class_indexes.keys()).index(key)
                    for bbox in label[key]:
                        c_pool[2*c_ind] += bbox[4]
                        c_pool[2*c_ind+1] += 1
                    segments = []
                    for bbox in label[key]:
                        #print(bbox[5])

                        new_mask = np.zeros_like(mask, dtype=np.uint8)[..., :3]

                        new_mask[bbox[5]] = 255

                        ret_mask = mask_to_contours(new_mask, yolo_class_indexes[key])
                        if ret_mask == "timeout":

                            c_pool[-2]+=1
                            shutil.move(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"),
                                        os.path.join(tresh, dir + f"_{dataset_name}_SEG_TIMEOUT.jpg"))
                            return dataset + f"/{segment_labels_name}/train/" + dir + f"_{dataset_name}.txt"
                        else:
                            segments.append(ret_mask)

                    with open(dataset+f"/{segment_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                        for seg in segments:
                            print(seg, file=f)
                    with open(dataset+f"/{bbox_labels_name}/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                        for bbox in label[key]:
                            print(yolo_class_indexes[key], *(bbox[:4]), file=f)
    return None

sh_progress = SharedMemory("shared_progress", create=True, size=8*(2 + 2 * len(yolo_class_indexes.keys())))

dataset_names = [

"114 LEB 23.01",
# "PC LEB 23.01",
# "PC LEB 3.02",
# "106 LEB 23.01",
#"PC LEB 16.01"
#"PC LEB 11.01",
# "PC LEB 13.11",
#"116 LEB 13.01",

]
for dataset_name in dataset_names:
    rooot = f"/vol1/KSH/source/LEBEDKA/10-15.01.25/{dataset_name}/lebedka"
    dataset = "/vol1/KSH/dataset/"+dataset_name
    tresh = "/vol1/KSH/tresh"

    bbox_labels_name = 'labels_bbox'
    segment_labels_name = 'labels'

    create_dirs()

    pixel_dx = 0.0001
    sh_progress = SharedMemory("shared_progress") #, create=False, size=8)
    c = np.ndarray((2 * len(yolo_class_indexes.keys()) + 2,), dtype=np.int64, buffer=sh_progress.buf)
    c[:] = 1

    data_to_process = os.listdir(os.path.join(rooot, "images"))[::]
    with Pool(processes=9) as p:
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
    print(f'TIMEOUT FOUND = {c[-2]}')
    for key in yolo_class_indexes.keys():
        c_ind = list(yolo_class_indexes.keys()).index(key)
        print(f'{key}: AVER PIX = {c[2*c_ind]//c[2*c_ind+1]} | NOW PIXELS = {min_volume.get(key)} | TOTAL FOUND = {c[2*c_ind+1]}')
    with open(dataset+"/dataset_info.txt", "w+") as f:
        print(f'TOTAL IMAGES FOUND = {c[-1]} | MOVED TO TRESH = {treshed}', file=f)
        print(f'TIMEOUT FOUND = {c[-2]}', file=f)
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




