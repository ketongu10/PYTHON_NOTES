import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils import xywhn2xywh

path = "/vol1/KSH/dataset/pasha_syn_12.08"

#paths_of_images = [image for image in os.listdir(path+"/images/val")]
sum = 0
for root,dirs,files in os.walk(path):
    for file in files:
        if '.jpg' in file:

            image_path = os.path.join(root,file)
            label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")

            image = Image.open(image_path)
            old_w, old_h = image.size
            if old_w != old_h:
                new_w, new_h = old_h, old_h

                left = (old_w - new_w) // 2  # сдвиг по длине : 840//2 =    420
                top = (old_h - new_h) // 2  # cдвиг по высоте : 1080-1080 = 0
                right = left + new_w  # : 420+1080 =  1500
                bottom = top + new_h  # : 0+1080 =    1080
                cropped_image = image.crop((left, top, right, bottom))
                cropped_image.save(image_path)

                new_bboxes = []
                new_lines = []
                with open(label_path, "r+") as r:
                    lines = r.readlines()
                    raw_original_bboxes = [line.split() for line in lines]
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
                sum+=1
            print(sum)