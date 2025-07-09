import json

import numpy as np
import os
import shutil
import cv2
from paste_smoke import add_smoke
from pathlib import Path
import pickle, json
from multiprocessing import Pool
from time import time

# ========= OUR BASE OVERFLOW LABELING TOOL
# ========= BE CAREFUL! USE label_dist.py BEFORE LABELING TO AVOID CRUSHED CUBES

TOTAL_LIMITS = 0
SHIFTS_REQUIRED = 0

def crop_and_compress(img_name, new_size, new_img_name):
    img = cv2.imread(img_name)

    h, w, rgb = img.shape
    if h < w:
        size = h
    else:
        size = w

    y, x = int((h - size)/2), int((w - size)/2)
    h, w = size, size
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (new_size, new_size))
    cv2.imwrite(new_img_name, img)
def print_stat(dir):
    total_false = 0
    total = 0
    for file in os.listdir(dir):
        if "false" in file:
            total_false += 1
        total += 1
    print(f"False: {total_false}")
    print(f"Total: {total}")
    print(f"Percent: {total_false / total}")

def recalc_labels(file):
    with open(file, 'r') as f:
        old_data = f.read()

    with open(file, 'w') as f:
        for line in old_data.splitlines():
            x, y = line.split()
            y = (752025600 - int(y))
            print(f"{x} {y}", file=f)

def check_mask(link, data, net_buf, smoke):
    #smoke_dict = {} if smoke is None else smoke
    sample = []
    for i in range(net_buf):
        sample.append(int(data[link+i].split()[1]))

    #===label = there is no water at all
    bo = True
    for i in range(net_buf):
        check_smoke = smoke and int(data[link+i].split()[0]) not in list(smoke.keys()) #there is smoke that covers water
        bo = bo and (sample[i] == 0 or check_smoke)
    if bo:
        return -1

    #===label = there is water
    bo = True
    small_water = True
    for i in range(net_buf):
        check_smoke = smoke and sample[i] > MIN_PIXEL_VALUE and int(data[link + i].split()[0]) in list(smoke.keys()) #there is smoke that doesn't cover water
        bo = bo and (sample[i] >= MIN_PIXEL_VALUE and not smoke or check_smoke)
        small_water = small_water and sample[i] >= MIN_PIXEL_VALUE
    if bo:
        if small_water: #if all waters large enought
            return 1
        else:           #if any water too small - do not
            return 0

    #===tresh => move next
    return 0

def parse_label_v2(path):
    dict = {}
    p = 0.6
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            dx0 = (float(data[1]) - float(data[3])/2, float(data[1]) + float(data[3])/2)
            dx_base = (dx0[0] - 0.5, dx0[1] - 0.5)
            left = dx_base[0] + (dx_base[1] - dx_base[0]) * (1 - p)
            right = dx_base[1] + (dx_base[0] - dx_base[1]) * (1 - p)
            #dx_max_and_center = (-0.2-left, 0.2-right, float(data[1])-0.5, float(data[2])-0.5)
            dx_max_and_center = (max(-0.2, -0.3-left), min(0.2, 0.3-right), float(data[1])-0.5, float(data[2])-0.5)
            dict[int(data[0])] = dx_max_and_center
    return dict

@DeprecationWarning
def parse_label_v2_old(path):
    dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            dx0 = (float(data[1]) - float(data[3])/2, float(data[1]) + float(data[3])/2)
            dx_base = (dx0[0] - 0.5, dx0[1] - 0.5)
            dx_max = (-0.2-dx_base[0], 0.2-dx_base[1])
            dict[int(data[0])] = dx_max
    return dict
def create_target(label_dir, old_data, link, net_buf, result, name, label_v2, smoke):
    path = './train/'+SERIES_NAME+name+("_true" if result==1 else "_false")
    os.mkdir(path)
    for i in range(link, link+net_buf):
        x, y = old_data[i].split()
        source = os.path.join('./refactored/images', label_dir, str(x).zfill(4)+'.jpg')
        destination = os.path.join(path, str(i-link) + '.jpg')
        if COMPRESS:
            crop_and_compress(source, NEW_SIZE, destination)
        else:
            shutil.copy(source, destination)

    x, y = old_data[link].split()
    global TOTAL_LIMITS
    global SHIFTS_REQUIRED
    if smoke is not None:
        if int(x) in list(smoke.keys()):
            lst_keys = list(smoke.keys())
            ind = lst_keys.index(int(x))
            data = []
            for i in range(ind, ind+net_buf):
                if i < len(lst_keys):
                    not_tuple = list(smoke.values())[i]
                    data.append([not_tuple[0], not_tuple[1], not_tuple[2], not_tuple[3]])
            #min_, max_ = min(max(np.array(data)[:, 0]), 0), max(min(np.array(data)[:, 1]), 0)
            min_, max_ = max(np.array(data)[:, 0]), min(np.array(data)[:, 1])
            x0, y0 = np.average(np.array(data)[:, 2]), np.average(np.array(data)[:, 3])
            TOTAL_LIMITS+=1
            with open(path+'/limits.txt', 'w') as f:
                is_required = min_ * max_ > 0  #means water is righter or lefter then crop border
                if is_required:
                    print(f"SHIFTS ARE REQUIRED FOR {path}")
                    SHIFTS_REQUIRED += 1
                print(min_, max_, x0, y0, is_required, file=f)
        else:
            with open(path + '/limits.txt', 'w') as f:
                print(-0.2, 0.2, 0, 0, False, file=f)
    elif label_v2 is not None:
        if int(x) in list(label_v2.keys()):
            lst_keys = list(label_v2.keys())
            ind = lst_keys.index(int(x))
            data = []
            for i in range(ind, ind+net_buf):
                if i < len(lst_keys):
                    not_tuple = list(label_v2.values())[i]
                    data.append([not_tuple[0], not_tuple[1], not_tuple[2], not_tuple[3]])
            #min_, max_ = min(max(np.array(data)[:, 0]), 0), max(min(np.array(data)[:, 1]), 0)
            min_, max_ = max(np.array(data)[:, 0]), min(np.array(data)[:, 1])
            x0, y0 = np.average(np.array(data)[:, 2]), np.average(np.array(data)[:, 3])
            TOTAL_LIMITS+=1
            with open(path+'/limits.txt', 'w') as f:
                is_required = min_ * max_ > 0  #means water is righter or lefter then crop border
                if is_required:
                    print(f"SHIFTS ARE REQUIRED FOR {path}")
                    SHIFTS_REQUIRED += 1
                print(min_, max_, x0, y0, is_required, file=f)
        else:
            with open(path+'/limits.txt', 'w') as f:
                print(-0.2, 0.2, 0, 0, False, file=f)

    if np.random.uniform() < SMOKE_PROB:
        smoke_dir = "./Smoke/Smoke" + str(np.random.randint(1, 7))
        smoke_start = np.random.randint(50, 220)
        offset = (np.random.randint(0, 600), np.random.randint(-200, 200))
        for i, img_path in enumerate(os.listdir(path)):
            img = cv2.imread(os.path.join(path,img_path))
            smoke = cv2.imread(smoke_dir+"/smoke_"+str(smoke_start+i*5).zfill(4)+".jpg")
            img = add_smoke(img, smoke, offset)
            cv2.imwrite(os.path.join(path,img_path), img)



def gen_targets_from_vidos(label_path, label_dir, available_name):
    file_name = os.path.join(label_path, label_dir, "labels.txt")
    down_sign = os.path.join(label_path, label_dir, "down.txt")
    if 'labels_v2.txt' in os.listdir(os.path.join('./refactored/labels', label_dir)):
        label_v2  = parse_label_v2(os.path.join('./refactored/labels', label_dir, 'labels_v2.txt'))
    else:
        label_v2 = None
    if 'smoke_dif.txt' in os.listdir(os.path.join('./refactored/labels', label_dir)):
        smoke_lbs  = parse_label_v2(os.path.join('./refactored/labels', label_dir, 'smoke_dif.txt'))
    else:
        smoke_lbs = None
    with open(file_name, 'r') as f:
        old_data = f.read().splitlines()
    with open(down_sign, 'r') as f:
        down_marker = str(f.readline()[:-1])+"_"
    net_buf = 3
    link = 0
    name_shift = 0
    while link + net_buf <= len(old_data):
        result = check_mask(link, old_data, net_buf, smoke_lbs)
        if result == 0:
            link+=1
            continue
        if (result == 1 or result == -1):
            if np.random.uniform() < DATASET_GEN_P:
                create_target(label_dir, old_data, link, net_buf, result, down_marker+str(available_name+name_shift), label_v2, smoke_lbs)
                name_shift+=1
                link+=net_buf
                continue
            else:
                link += net_buf
                continue
    return name_shift

SERIES_NAME = "tr5e6_15-07_"
COMPRESS = False
DATASET_GEN_P = 1.0
MIN_PIXEL_VALUE = 5e6
SMOKE_PROB = 0 #THIS FEATURE MOVED TO ANOTHER SCRIPT
NEW_SIZE = 768
dataset = './refactored'
train = './train'
val = "./val"
try:
    #shutil.rmtree(dataset)
    shutil.rmtree(train)
    shutil.rmtree(val)
except:
    print("loh")
    pass
try:
    #os.rmdir(dataset)
    os.rmdir(train)
    os.rmdir(val)
except:
    pass
num = 0

up = [
#"From 106/water_flow",
"From 106 mv light/water_flow",
"From pc 6.12/water_flow",
"From PC mv light/water_flow",
#"From PC 2/water_flow",
#"From 106 last/water_flow",
#"From 106 no water/water_flow",
#"From PC no water/water_flow",
"From 106 mv light2/water_flow",
"From PC mv light2/water_flow",
"From 106 another one/water_flow",
"From 106 6.12/water_flow",
"From 106 7.12/water_flow",
"From PC smoke/water_flow",
"From 106 smoke/water_flow",]
down = [
"From PC down new2/water_flow",
"From 106 down/water_flow",
"From 106 down3/water_flow",
"From 106 another one/water_flow",
"From PC down new/water_flow",
"From 106 7.12/water_flow",
"From 106 6.12/water_flow",
"From pc 6.12/water_flow",
"From PC 11.12/water_flow",
"From 103 11.12/water_flow",
"From 106 18.12/water_flow",
"From PC 18.12/water_flow",
"From 103 18.12/water_flow",
"From PC 20.12/water_flow",
"From PC smoke/water_flow",
"From 106 smoke/water_flow",]
dirs = [
        #"UP NEW/103 16.02 up new/water_flow","UP NEW/106 16.02 up new/water_flow",
        #"UP NEW/104 19.02 up new/water_flow","UP NEW/106 19.02 up new/water_flow",
        #"UP NEW/104 26.02 down new/water_flow","UP NEW/106 26.02 up new/water_flow",
        #"UP NEW/104 29.02 down new/water_flow","UP NEW/106 29.02 up new/water_flow",
        #"UP NEW/104 25.03 up new/water_flow","UP NEW/106 25.03 up new/water_flow",
        #"UP NEW/PC 25.03 up new/water_flow",
        #"UP NEW/PC 15.04 up new/water_flow", "UP NEW/104 15.04 up new/water_flow",
        #"UP NEW/PC 17.04 up new/water_flow", "UP NEW/104 17.04 up new/water_flow",
        #"UP NEW/105 17.04 up new/water_flow", "UP NEW/109 17.04 up new/water_flow",
        #"UP NEW/107 2.05 up new new/water_flow","UP NEW/PC 2.05 up new new/water_flow",
        #"UP NEW/104 2.05 up new new/water_flow",
        #"TEST/water_flow",



        #"UP NEW REWORKED/105 13.05 up new/water_flow",
        #"UP NEW REWORKED/106 13.05 up new/water_flow",
        #"UP NEW REWORKED/PC 13.05 up new/water_flow",
        #"UP NEW REWORKED/PC 6.05 down new/water_flow","UP NEW REWORKED/107 6.05 down new/water_flow",
        ]

dirs = [
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/103 16.02 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/106 16.02 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 19.02 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/106 19.02 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 26.02 down new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/106 26.02 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 29.02 down new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/106 29.02 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 25.03 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/106 25.03 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/PC 25.03 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/PC 15.04 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 15.04 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/PC 17.04 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 17.04 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/105 17.04 up new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/109 17.04 up new/water_flow",

        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/107 2.05 up new new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/PC 2.05 up new new/water_flow",
        #"/vol2/WATER/BLENDER_DATASETS/DATASET PBR (02.24-03.24)/104 2.05 up new new/water_flow",
        #"TEST/water_flow",

        #MOVED TO vol2
        #"UP NEW REWORKED/105 13.05 up new/water_flow",
        #"UP NEW REWORKED/106 13.05 up new/water_flow",
        #"UP NEW REWORKED/PC 13.05 up new/water_flow",
        #"UP NEW REWORKED/PC 6.05 down new/water_flow","UP NEW REWORKED/107 6.05 down new/water_flow",

        #"UP NEW REWORKED/PC 15-07 up/water_flow",
        "UP NEW REWORKED/106 15-07 up/water_flow",
    ]

print(len(dirs))
print(dirs)
os.mkdir(dataset)
for grand_dir in dirs:
#def refactor_dataset(args):
    #grand_dir, num = args
    #num = 0
    for dir in os.listdir(grand_dir):
        if dir != 'images':
            berem = True

            source = os.path.join(grand_dir, dir)
            destination = os.path.join(dataset, 'labels', str(num))


            # ===CHECK DOWN START=====#
            settings = os.path.join(source, 'settings.txt')

            js = eval(Path(settings).read_text().replace("Vector", ""))
            #if js['should_water_flow'] and (js['flow_type']=="Flow_from_spider"): # or js['special_equipment'])
            #    print("loh")
            #    berem = False

            if berem:
                # ===labels===#

                shutil.copytree(source, destination)

                f = os.path.join(destination, 'labels.txt')
                recalc_labels(f)
                # ===images===#
                source_img = os.path.join(grand_dir, 'images', dir)
                destination_img = os.path.join(dataset, 'images', str(num))
                shutil.copytree(source_img, destination_img)
                num += 1

                with open(os.path.join(destination, "down.txt"), 'a') as down_sign:
                    if "down_" in js['flow_type']:
                        print("down", file=down_sign)
                    else:
                        print("up", file=down_sign)
            else:
                print(source, " was rejected")







"""t0 = time()
with Pool(6) as p:
    smth = p.map(refactor_dataset, zip(dirs, [i*100000 for i in range(len(dirs))]))
print(f"time: {time()-t0}")"""



print(num)


os.mkdir(train)
available_name = 0
for label_dir in os.listdir(dataset+"/labels"):

    available_name += gen_targets_from_vidos(dataset+"/labels", label_dir, available_name)

val_part = 0.1
os.mkdir(val)
for x in os.listdir(train):
    if np.random.uniform() < val_part:
        shutil.move(os.path.join(train, x), os.path.join(val, x))

print_stat(val)
print_stat(train)
print(f"TOTAL LIMITS FOUND: {TOTAL_LIMITS}")
print(f"SHIFTS REQUIRED FOUND: {SHIFTS_REQUIRED}")