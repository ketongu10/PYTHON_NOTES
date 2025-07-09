import os
from pathlib import Path

folders = Path("/vol1/KSH/READY/proc/syn_proc_18.11")



broken = 0
def delete(path: str):
    os.remove(path)
    os.remove(path.replace("labels_bbox", "labels"))
    try:
        os.remove(path.replace("labels_bbox", "images").replace('txt', 'jpg'))
    except:
        global broken
        broken+=1




total = 0
found = 0
for fld in os.listdir(folders/"labels_bbox"):
    now_dir = folders/"labels_bbox"/fld
    if now_dir.is_dir():
        for img in os.listdir(now_dir):
            total+=1
            img_path = now_dir/img
            classes = []
            with open(img_path, 'r') as f:
                for line in f.readlines():
                    classes.append(int(line.split()[0]))
            if 4 in classes or 5 in classes:
                found+=1
                print(img_path)
                delete(str(img_path))
print(found, total, broken)
