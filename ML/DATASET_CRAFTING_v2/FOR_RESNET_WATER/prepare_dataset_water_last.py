import numpy as np
import os
import shutil
import cv2
from paste_smoke import add_smoke
from multiprocessing import Pool
from time import time
# LAST IN FILENAME MEANS LABELING IS BASED ON LAST FRAME


TOTAL_LIMITS = 0




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

    #===label = there is no water
    bo = True
    for i in range(net_buf):
        check_smoke = smoke and int(data[link+i].split()[0]) not in list(smoke.keys()) #there is smoke that covers water
        bo = (sample[i]==0 or check_smoke) #bo and (sample[i]==0 or check_smoke)
    if bo:
        return -1

    #===label = there is water
    bo = True
    for i in range(net_buf):
        check_smoke = smoke and sample[i] > 0 and int(data[link + i].split()[0]) in list(smoke.keys()) #there is smoke that doesn't cover water
        bo = (sample[i] > 0 and not smoke or check_smoke) #bo and (sample[i] > 0 and not smoke or check_smoke)
    if bo:
        return 1

    #===tresh => move next
    return 0

def parse_label_v2(path):
    dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            dx0 = (float(data[1]) - float(data[3])/2, float(data[1]) + float(data[3])/2)
            dx_base = (dx0[0] - 0.5, dx0[1]-0.5)
            dx_max = (-0.3-dx_base[0], 0.3-dx_base[1])
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

    x, y = old_data[link+2].split()
    global TOTAL_LIMITS
    if smoke is not None:
        if int(x) in list(smoke.keys()):
            ind = list(smoke.keys()).index(int(x))
            data = []
            not_tuple = list(smoke.values())[ind]
            data.append([not_tuple[0], not_tuple[1]])
            min_, max_ = min(max(np.array(data)[:, 0]), 0), max(min(np.array(data)[:, 1]), 0)
            TOTAL_LIMITS+=1
            with open(path+'/limits.txt', 'w') as f:
                print(min_, max_, file=f)
        else:
            with open(path + '/limits.txt', 'w') as f:
                print(-0.3, 0.3, file=f)
    elif label_v2 is not None:
        if int(x) in list(label_v2.keys()):
            ind = list(label_v2.keys()).index(int(x))
            data = []
            not_tuple = list(label_v2.values())[ind]
            data.append([not_tuple[0], not_tuple[1]])
            min_, max_ = min(max(np.array(data)[:, 0]), 0), max(min(np.array(data)[:, 1]), 0)
            TOTAL_LIMITS+=1
            with open(path+'/limits.txt', 'w') as f:
                print(min_, max_, file=f)
        else:
            with open(path+'/limits.txt', 'w') as f:
                print(-0.3, 0.3, file=f)

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

SERIES_NAME = "smoke_"
COMPRESS = False
DATASET_GEN_P = 1.0
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
dirs = ["From 106 smoke/water_flow","From PC smoke/water_flow", "From 106 24.01 smoke/water_flow"]
print(len(dirs))
print(dirs)
os.mkdir(dataset)
for grand_dir in dirs:
#def refactor_dataset(grand_dir):
   #num = 0
    for dir in os.listdir(grand_dir):
        if dir != 'images':

            #===labels===#
            source = os.path.join(grand_dir, dir)

            #=====CHECK DOWN END=====#
            destination = os.path.join(dataset, 'labels', str(num))
            shutil.copytree(source, destination)
            f = os.path.join(destination, 'labels.txt')
            recalc_labels(f)

            # ===CHECK DOWN START=====#
            settings = os.path.join(source, 'settings.txt')
            with open(settings, 'r') as f:
                with open(os.path.join(destination, "down.txt"), 'a') as down_sign:
                    if "down_" in f.read():
                        print("down", file=down_sign)
                    else:
                        print("up", file=down_sign)

            # ===images===#
            source = os.path.join(grand_dir, 'images', dir)
            destination = os.path.join(dataset, 'images', str(num))
            shutil.copytree(source, destination)
            num += 1

"""t0 = time()
with Pool(4) as p:
    smth = p.map(refactor_dataset, dirs)
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