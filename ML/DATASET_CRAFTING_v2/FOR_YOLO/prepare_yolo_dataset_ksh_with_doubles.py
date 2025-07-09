import shutil
import os
import json
import numpy as np
import imageio as imio
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory


colorama_init()



def is_intersected(bbox1, bbox2):
    x1, y1, w1, h1, s1 = bbox1
    x2, y2, w2, h2, s2 = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    return  ((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2 or b1_x1 <= b2_x2 <= b1_x2)
             and (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2 or b1_y1 <= b2_y2 <= b1_y2))



def unite_bboxes(bbox1, bbox2):
    x1, y1, w1, h1, s1 = bbox1
    x2, y2, w2, h2, s2 = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    b3_x1, b3_x2 = np.min([b1_x1, b2_x1]), np.max([b1_x2, b2_x2])
    b3_y1, b3_y2 = np.min([b1_y1, b2_y1]), np.max([b1_y2, b2_y2])
    x3, y3, w3, h3, s3 = (b3_x1 + b3_x2)/2, (b3_y1 + b3_y2)/2, (b3_x2 - b3_x1), (b3_y2 - b3_y1), s1+s2
    return [x3, y3, w3, h3, s3]


def merge_intersected_bboxes(bboxs):
    if len(bboxs)>1:
        for i in range(len(bboxs)-1):
            for j in range(i+1, len(bboxs)):
                bbox1, bbox2 = bboxs[i], bboxs[j]
                if is_intersected(bbox1, bbox2):
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

#

bldr_clss_as_yolo_clss = {
    "ksh_knot": {'classes': ["ksh_knot", "ksh_knot_fat"], 'task': 'FOR_EACH'},
    "ksh_short_kran": {'classes':["ksh_short_kran", "ksh_knot", "ksh_knot_fat",  "ksh_head", "ksh_head_last"], 'task': 'UNITE'},
    "vstavka_n2": {'classes':["vstavka_n2"], 'task': 'FOR_EACH'},
    "pipe_1_end": {'classes':["pipe_1_end", "vstavka_n2", "not_ksh_knot", "not_ksh_knot_last", "zatychka", ], 'task': 'UNITE'}, #"zatychka_w_monometr"
    "not_ksh_knot": {'classes':["not_ksh_knot", "not_ksh_knot_last", "zatychka"], 'task': 'FOR_EACH'}, #"not_ksh_knot_last", "zatychka",
    "elevator": {'classes':["elevator_fp_on_pipe"], "task": 'FOR_EACH'}, #"elevator_fp_on_pipe", "TB_elevator_wide"
    "gloves": {'classes':["gloves"], "task": 'FOR_EACH'}, #"gloves"
    "gate": {'classes': ["up_gate", "gate_flance_up"], "task": "UNITE"},
    "flance": {'classes':["gate_flance_up"], "task": 'FOR_EACH'},
    #"not_vstavka_n2": {'classes':["not_vstavka_n2"], "task": 'FOR_EACH'}

}
# bldr_clss_as_yolo_clss = {
#     "ksh_knot": {'classes': ["ksh_knot"], 'task': 'FOR_EACH'}, #"ksh_knot_fat"
#     "ksh_short_kran": {'classes':["ksh_short_kran", "ksh_knot", "ksh_head", "ksh_head_last"], 'task': 'UNITE'},
#     "vstavka_n2": {'classes':["vstavka_n2"], 'task': 'FOR_EACH'},
#     "pipe_1_end": {'classes':["pipe_1_end", "vstavka_n2"], 'task': 'UNITE'},
#     #"not_ksh_knot": {'classes':["not_ksh_knot"], 'task': 'FOR_EACH'}, #"not_ksh_knot_last", "zatychka"
#
# }

yolo_class_indexes = {"ksh_knot": 0, "ksh_short_kran": 1, "vstavka_n2": 2, "pipe_1_end": 3,
                      "not_ksh_knot": 4, "elevator": 5, "gloves": 6, "gate": 7, "flance": 8}
#yolo_class_indexes = {"ksh_knot": 0, "ksh_short_kran": 1, "vstavka_n2": 2, "pipe_1_end": 3, "not_ksh_knot": 4, "not_vstavka_n2": 5} #, "not_ksh_knot": 4}
# min volume is for preventing covering ksh and vstavka by gksh and tal_block
coef = 0.3
# min_volume = {"ksh_knot": 2366*coef, #10k
#               "vstavka_n2": 18000*coef, #18k
#               "not_ksh_knot": 2500*coef,
#               "not_vstavka_n2": 18000*coef,} #5254 #17k

min_volume = {"ksh_knot": 10000*coef, #10k
              "vstavka_n2": 18000*coef, #18k
              "not_ksh_knot": 7000*coef, #7k
              "elevator": 17000*coef,
              "gloves": 200, #10000*coef,
              "not_vstavka_n2": 13000*coef,
              "gate": 94042*coef, #121116
              "flance": 44378*coef}


def create_dirs(dataset):
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


def gen_bbox(mask, clss2psinds, key):
    bboxes = []
    H, W, C = mask.shape
    if bldr_clss_as_yolo_clss[key]['task'] == 'UNITE':
        inds_list = []
        for clss in bldr_clss_as_yolo_clss[key]['classes']:
            inds_list += clss2psinds[clss]

        cond = False
        for psind in inds_list:
            cond |= (mask == psind)

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
            bboxes.append([xc, yc, w, h, len(pixel_inds[0])])

    elif bldr_clss_as_yolo_clss[key]['task'] == 'FOR_EACH':
        for clss in bldr_clss_as_yolo_clss[key]['classes']:
            for psind in clss2psinds[clss]:
                pixel_inds = np.where(mask == psind)

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
                    bboxes.append([xc, yc, w, h, len(pixel_inds[0])])
        bboxes = merge_intersected_bboxes(bboxes)
    return bboxes if bboxes else None


def border_collision_is_passed(bboxs, key):
    for bbox in bboxs:
        xc, yc, w, h, s = bbox
        if s < min_volume[key]:
            print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: too small {s} < {min_volume[key]}")
            return False
        if (1 - yc - h/2 < pixel_dx or
            1 - xc - w/2 < pixel_dx or
            yc - h/2 < pixel_dx or
            xc - w/2 < pixel_dx):
            print(f"{Fore.RED}BROKEN{Style.RESET_ALL}: bounds intersection")
            return False
    return True



pixel_dx = 0.0001

def doit(dir):

    sh_progress_pool = SharedMemory("shared_progress", create=False)
    c_pool = np.ndarray((2 * len(yolo_class_indexes.keys()) + 3,), dtype=np.int64, buffer=sh_progress_pool.buf)
    c_pool[-1]+=1
    print(f"progress = {c_pool[-1]}")
    if os.path.exists(rooot + "/images/" + dir) and os.path.exists(rooot + "/masks/" + dir) and os.path.exists(rooot + "/jsons/" + dir):
        img_name = os.listdir(rooot + "/images/" + dir)[0]
        shutil.copy(os.path.join(rooot, "images/", dir, img_name), os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        with open(dataset+"/labels/train/"+dir+f"_{dataset_name}.txt", 'w+') as f:
            pass
        # try:
        #     set_path = os.path.join(rooot ,"jsons/" , dir , "settings.txt")
        #     settings = eval(Path(set_path).read_text().replace("Vector", ""))
        #     if settings["should_spawn_human"]:
        #         print("================================HUMANS ARE FORBIDDEN========================================")
        #         os.remove(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"))
        #         os.remove(dataset + "/labels/train/" + dir + f"_{dataset_name}.txt")
        #         return
        # except Exception as e:
        #     print(e)
        #     shutil.rmtree(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
        #     shutil.rmtree(dataset+"/labels/train/"+dir+f"_{dataset_name}.txt")
        #     return

        mask = os.path.join(rooot ,"masks/" , dir , img_name.replace('jpg', 'png'))
        try:
            mask = imio.v3.imread(mask)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH MASK!!!!!! {dir}{Style.RESET_ALL}")
            shutil.rmtree(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
            shutil.rmtree(dataset+"/labels/train/"+dir+f"_{dataset_name}.txt")
            c_pool[-2]+=1
            return 'deleted'

        try:
            with open(rooot + "/jsons/" + dir + "/corr.json") as json_file:
                clss2psinds = json.load(json_file)
        except Exception as e:
            print(f"{Fore.RED}NO SUCH JSON!!!!!! {dir}{Style.RESET_ALL}")
            shutil.rmtree(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"))
            shutil.rmtree(dataset+"/labels/train/"+dir+f"_{dataset_name}.txt")
            c_pool[-3]+=1
            return 'deleted'
        
        label = {}

        # insss = np.where(mask[..., 0] == 131)[0]
        # print(insss)
        # if len(insss) == 0:
        #     print(np.max(mask[..., 0]))
        #     shutil.move(os.path.join(dataset, "images/train/", dir + f"_{dataset_name}.jpg"),
        #                 os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
        #     return dataset + "/labels/train/" + dir + f"_{dataset_name}.txt"
        # else:
        #     print("AAAAAAAAA")

        for key in yolo_class_indexes.keys():

            label[key] = gen_bbox(mask, clss2psinds, key)

            if (label[key] is not None
                and ((key != "pipe_1_end" or key == "pipe_1_end" and label.get("vstavka_n2") is not None)
                and (key != "ksh_short_kran" or key == "ksh_short_kran" and label.get("ksh_knot") is not None))):

                if (key == "vstavka_n2" or key == "ksh_knot"
                    or key == "not_ksh_knot" or key == "elevator"
                    or key == "gloves" or key == "gate" or key == "flance") and not border_collision_is_passed(label[key], key):
                    print(label[key])
                    # deleting crossing bboxes and logging to tresh

                    shutil.move(os.path.join(dataset , "images/train/", dir + f"_{dataset_name}.jpg"),
                                os.path.join(tresh, dir + f"_{dataset_name}.jpg"))
                    return dataset + "/labels/train/" + dir + f"_{dataset_name}.txt"
                else:
                    c_ind = list(yolo_class_indexes.keys()).index(key)
                    for bbox in label[key]:
                        c_pool[2 * c_ind] += bbox[4]
                        c_pool[2 * c_ind + 1] += 1
                    with open(dataset+"/labels/train/" + dir+f"_{dataset_name}.txt", 'a') as f:
                        for bbox in label[key]:
                            print(yolo_class_indexes[key], *(bbox[:4]), file=f)
    else:
        if not os.path.exists(rooot + "/masks/" + dir):
            c_pool[-2] += 1
        if not os.path.exists(rooot + "/jsons/" + dir):
            c_pool[-3] += 1


sh_progress = SharedMemory("shared_progress", create=True, size=8*(3 + 2 * len(yolo_class_indexes.keys())))



dataset_names = [

"PC KSH_gates 26.12",
"105 KSH_gates 26.12",
"114 KSH_gates 26.12",
"116 KSH_gates 26.12"

#"PC KSH_gate 23.12"
#"PC KSH_gate 18.12"
# "105 arms 12.12",
# "105 arms 13.12",


# "105 KSH 12.12",
# "UMAR KSH 12.12"

#"105 KSH_nb 10.12"

# "105 KSH_relabel 10.12",
# "UMAR KSH_relabel 10.12",
# "105 KSH_relabel 9.12",
# "PC KSH_relabel 9.12",
# "UMAR KSH_relabel 9.12"


# "105 KSH_arms 6.12",
# "PC KSH_arms 6.12",
# "UMAR KSH_arms 6.12"
# "106 KSH 3.12"
# "106 KSH_redhands 16.10",
# "PC KSH_redhands 16.10-0",
# "PC KSH_redhands 16.10-1",

#"PC KSH 14.10",
# "104 KSH 14.10-0",
# "104 KSH 14.10-1",
# "105 KSH 14.10-0",
# "105 KSH 14.10-1",
# "106 KSH 14.10-0",
# "106 KSH 14.10-1",
# "106 KSH 14.10-2",
# "106 KSH 14.10-3",

# "104 KSH 10.10-0",
# "104 KSH 10.10-1",
# "105 KSH 10.10-0",
# "105 KSH 10.10-1",
# "106 KSH 10.10-0",
# "106 KSH 10.10-1",


#"PC KSH_hands 7.10",
# "104 KSH_hands 5.10-0",
# "104 KSH_hands 5.10-1",
# "104 KSH_hands 7.10-0",
# "104 KSH_hands 7.10-1",
# "105 KSH_hands 5.10-0",
# "105 KSH_hands 5.10-1",
# "105 KSH_hands 7.10-0",
# "105 KSH_hands 7.10-1",
# "106 KSH 7.10-0",
# "106 KSH 7.10-1",




# "105 KSH_w_broken_ins_pipe-0 23.09",
# "105 KSH_w_broken_ins_pipe-1 23.09",
# "106 KSH_w_broken_ins_pipe-0 23.09",
# "106 KSH_w_broken_ins_pipe-1 23.09",
# "PC KSH_br_ins_pipe 23.09",

# "105 KSH 25.09-0",
# "105 KSH 25.09-1",
# "PC KSH 24.09",
# "PC KSH 25.09",
# "106 KSH 24.09-0",
# "106 KSH 24.09-1",
# "106 KSH 25.09-0",
# "106 KSH 25.09-1",

#"PC KSH_hum 18.09"

#   "105 KSH_hum 18.09",


# "105 KSH_hum 12.09-0",
# "105 KSH_hum 12.09-1",
# "PC KSH_hum 12.09",
# "105 VSTAVKA 11.09-0",
# "105 VSTAVKA 11.09-1",
# "106 KSH_vertlug_elev 16.09-0",
# "106 KSH_vertlug_elev 16.09-1",
# "PC KSH_vertlug_elev 16.09",


# "UMAR KSH 6.09 brHUMAN",
# "107 KSH 6.09 brHUMAN",

# "105 KSH_AFP 5.09 BROKEN",
# "PC KSH_AFP 5.09 BROKEN"

# "PC KSH_AFP 6.09",
# "PC KSH 9.09",
# "105 KSH 9.09-0",
# "105 KSH 9.09-1",
# "105 KSH_AFP 6.09",

# "107 KSH 2.09",
# "106 KSH 2.09",
# "114 KSH 2.09",
# "116 KSH 2.09",
# "UMAR KSH 2.09",
#"PC KSH 2.09",
# '106 KSH 30.08',
# '107 KSH 30.08',
# '114 KSH 30.08',
# '116 KSH 30.08',
# 'PC KSH 30.08',
# 'UMAR KSH 30.08',
    #29* has no ksh fat class!!!
    # '106 KSH 29.08',
    # '107 KSH 29.08',
    # '114 KSH 29.08',
    # '116 KSH 29.08',
    # 'PC KSH 29.08',
    # 'UMAR KSH 29.08',

# 'old',
# 'super_test',
# 'super_test1',
]
for dataset_name in dataset_names:
    #dataset_name = "super_test"
    #rooot = f"/vol1/KSH/source/{dataset_name}/ksh_pipes"
    rooot = f"/vol1/KSH/source/KSH/SOURCE 3-6.12/{dataset_name}/ksh_pipes"
    #rooot = f"/vol1/KSH/source/PROC/SOURCE PROC 11.11/{dataset_name}/tkrs_proc"
    dataset = "/vol1/KSH/dataset/"+dataset_name
    tresh = "/vol1/KSH/tresh"
    create_dirs(dataset)

    sh_progress = SharedMemory("shared_progress")  # , create=False, size=8)
    c = np.ndarray((2 * len(yolo_class_indexes.keys()) + 3,), dtype=np.int64, buffer=sh_progress.buf)
    c[:] = 1

    with Pool(processes=6) as p:
        results = p.map(doit, os.listdir(os.path.join(rooot, "images"))[::])

    treshed = 0
    for path in results:
        if path:
            treshed += 1
            if path != 'deleted':
                os.remove(path)

    print(f'TOTAL IMAGES FOUND = {c[-1]-1} | MOVED TO TRESH = {treshed}')
    print(f'BROKEN MASKS = {c[-2]-1} | BROKEN JSONS = {c[-3]-1}')
    for key in yolo_class_indexes.keys():
        c_ind = list(yolo_class_indexes.keys()).index(key)
        print(f'{key}: AVER PIX = {c[2 * c_ind] // c[2 * c_ind + 1]} | NOW PIXELS = {min_volume.get(key)} | TOTAL FOUND = {c[2 * c_ind + 1]}')
    with open(dataset + "/dataset_info.txt", "w+") as f:
        print(f'TOTAL IMAGES FOUND = {c[-1]} | MOVED TO TRESH = {treshed}', file=f)
        print(f'BROKEN MASKS = {c[-2]-1} | BROKEN JSONS = {c[-3]-1}', file=f)
        for key in yolo_class_indexes.keys():
            c_ind = list(yolo_class_indexes.keys()).index(key)
            print(f'{key}: AVER PIX = {c[2 * c_ind] // c[2 * c_ind + 1]} | NOW PIXELS = {min_volume.get(key)} | TOTAL FOUND = {c[2 * c_ind + 1]}',file=f)

    # for i in os.listdir(os.path.join(rooot, "images")):
    #     doit(i)
    #     try:
    #         print(av_ocup['ksh_knot']/num_ocup['ksh_knot'], av_ocup['vstavka_n2']/num_ocup['vstavka_n2'])
    #     except:pass



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
        for key in yolo_class_indexes.keys():
            print(f"  {yolo_class_indexes[key]}: {key}", file=f)




