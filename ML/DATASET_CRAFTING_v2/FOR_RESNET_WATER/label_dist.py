import shutil

import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep

# ====== THIS IS SUPPORT TOOL FOR SHARING SMALL WATER BORDER AND CRUSHED CUBES


datasets = ["UP NEW REWORKED/106 15-07 up/water_flow"]

def create_shit_dir():
    try:
        os.mkdir("./show_shit")
    except:
        shutil.rmtree("./show_shit")
        os.mkdir("./show_shit")

def show_shit(label_step, root, dir):
    global available_name
    to_copy = os.path.join(root, "images", dir, label_step.zfill(4)+".jpg")
    where_copy = os.path.join("./show_shit", dir+"_"+str(available_name))
    shutil.copy(to_copy, where_copy)
    available_name += 1

MIN =1.0e8#0.3e8
MAX = 3.0e8#0.5e8
should_remove = False
if should_remove:
    print("ARE YOU SURE???????")
    ans = input()
    if ans == "q":
        exit()
    for i in range(10):
        print(f"Removing in {10-i}")
        sleep(1)



available_name = 0
removed = 0
total_dirs = 0
create_shit_dir()
labels = []
for dataset in datasets:
    for dir in os.listdir(dataset):
        total_dirs += 1
        source = os.path.join(dataset, dir, "labels.txt")
        print(source)
        if "image" not in source:
            labeled_to_remove = False
            with open(source, "r") as f:
                xs = []
                lines = 0
                for line in f.readlines():
                    lines+=1
                    x, y = line.split()
                    l = 752025600 - int(y)
                    if  l > 0:
                        labels.append(l)
                    if l >= MIN and l <= MAX:
                        xs.append(x)
                        #show_shit(x, dataset, dir)
                        #labeled_to_remove = True
                if len(xs) == lines:
                    labeled_to_remove = True
                    for x in xs:
                        show_shit(x, dataset, dir)
            if labeled_to_remove:
                removed += 1
            if labeled_to_remove and should_remove:
                shutil.rmtree(os.path.join(dataset, dir))
print(len(labels))
print(f"Removed dirs = {removed} of {total_dirs}")
plt.hist(labels, bins=100)
plt.show()