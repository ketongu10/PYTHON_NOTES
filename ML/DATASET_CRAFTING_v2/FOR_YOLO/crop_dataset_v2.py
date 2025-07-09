import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import xywhn2xywh
from time import time
from multiprocessing import Pool
import shutil


#returns new_left and new_top positions in pixels
def can_we_save_all(bboxes, new_w, old_w, new_h, old_h):
    if old_w >= old_h:
        left = 1
        right = 0
        for bbox in bboxes:
            cl, x, y, w, h = [float(smth) for smth in bbox]
            if x+w/2 > right:
                right = x+w/2
            if x-w/2 < left:
                left = x-w/2
        if right - left <= new_w/old_w:
            center = (right+left)/2
            usuall_left = center - new_w/old_w/2


            left = np.random.uniform(right - new_w/old_w, left)

            if left < 0:    #if new crop too right or too left we shift it to the bounds of original image
                print(f'haaaa loh left is 0 now! {left} vs {usuall_left}')
                left = 0
            elif left > 1 - new_w/old_w:
                print(f'haaaa loh left is 1-new_w/old_w  now! {left} vs {usuall_left}')
                left = 1 - new_w/old_w
            #print('cropaem along w')
            return int(left*old_w), 0
        #print('too wide - centercrop is applied')
        return int((0.5 - new_w/old_w/2)*old_w), 0
    else:
        top = 1
        down = 0
        for bbox in bboxes:
            cl, x, y, w, h = [float(smth) for smth in bbox]
            if y + h / 2 > down:
                down = y + h / 2
            if y - h / 2 < top:
                top = y - h / 2
        if down - top <= new_h / old_h:
            # center = (top + down) / 2
            # top = center - new_h / old_h / 2
            top = np.random.uniform(down - new_h/old_h, top)
            if top < 0:
                print('haaaa loh top is 0 now!')
                top = 0
            elif top > 1 - new_h/old_h:
                print('haaaa loh top is 1-new_w/old_w  now!')
                top = 1 - new_h/old_h
            #print('cropaem along h')
            return 0, int(top * old_h)
        #print('too wide - centercrop is applied')
        return 0, int((0.5 - new_h / old_h / 2) * old_h)


path = "/vol1/KSH/dataset/syn_gates_26.12"

#paths_of_images = [image for image in os.listdir(path+"/images/val")]
sum = 0
megalist = []
for root,dirs,files in os.walk(path):
    megalist.append((root, dirs,files))

def doit(given):
    root, dirs, files = given
    for file in files:
        if '.jpg' in file:

            image_path = os.path.join(root,file)
            label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")
            print(image_path)
            image = Image.open(image_path)
            old_w, old_h = image.size
            if old_w != old_h:
                new_w, new_h = min(old_h, old_w), min(old_h, old_w)
                new_bboxes = []
                new_lines = []
                with open(label_path, "r+") as r:
                    lines = r.readlines()
                    raw_original_bboxes = [line.split() for line in lines]
                    left, top = can_we_save_all(raw_original_bboxes, new_w, old_w, new_h, old_h)
                    right, bottom = left+new_w, top+new_h
                    for raw_original_bbox in raw_original_bboxes:
                        old_bbox = xywhn2xywh([float(coord) for coord in raw_original_bbox[1:]], old_w, old_h)
                        x = (old_bbox[0] - left) / new_w
                        y = (old_bbox[1] - top) / new_h
                        w = old_bbox[2] / new_w
                        h = old_bbox[3] / new_h
                        new_bbox = [int(raw_original_bbox[0]), x, y, w, h]
                        new_bboxes.append(new_bbox)
                        new_line = " ".join(str(coord) for coord in new_bbox) + "\n"
                        new_lines.append(new_line)
                    r.seek(0)
                    r.truncate()
                    r.writelines(new_lines)
                cropped_image = image.crop((left, top, right, bottom))
                cropped_image.save(image_path)
                #sum+=1
            #print(sum)

t0=time()
with Pool(6) as p:
    p.map(doit, megalist)

print(f'TOTAL TIME: {time()-t0}')