# DRAW POLYGON FROM RANDOM IMAGE() THAT HAS NOMRALIZED ANNOTATION

from pathlib import Path
import random
import cv2
import numpy as np
import os
import shutil
from PIL import Image

def create_dirs(dataset):
    try:
        os.makedirs(dataset.replace('images', 'watch'))

    except:
        shutil.rmtree(dataset.replace('images', 'watch'))
        os.makedirs(dataset.replace('images', 'watch'))

yolo_classes = {"pumka_base": 0, "pumka_square": 1, "pumka_sphere": 2,
                      "pumka_long": 3, "spider": 4, "pipe_1_end": 5, "pipe_head": 6,
                      }.keys()
    #
    # ("pipe_1_end", "ecn_tros", "spider",
    #        "kops_gate", "kops_other", "wheel_on_stick",
    #        "gis_tros", "rotor", "rotor_holder", "gksh",
    #        "tb_block", "tb_ear_sq","tb_ear_rd", "tb_strops",
    #         "pipe_otsos", "shlang_otsos")

color_palette = np.random.uniform(0, 255, size=(len(yolo_classes), 3))
gigapath = "/vol1/KSH/dataset/PC PUMKA 7.07.25/images/train"
create_dirs(gigapath)
images = [image for image in Path(gigapath).rglob("*.jpg")]


for image_path in images[:500]:
    label_path = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")


    image = cv2.imread(str(image_path))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]



    polygons = []
    classes = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            polygon = []
            odd = True
            for x in line.split()[1:]:
                if odd:
                    polygon.append(int(float(x) * w))
                else:
                    polygon.append(int(float(x) * h))
                odd = not odd
            polygons.append(polygon)
            classes.append(int(line[0]))


    for cls, polygon in zip(classes, polygons):
        points = np.array(polygon).reshape((-1, 2))
        points = points.astype(np.int32)
        cv2.polylines(image, [points], isClosed=True, color=color_palette[cls], thickness=2)

    #image = Image.fromarray(image)
    cv2.imwrite(str(image_path).replace("images", "watch"), image)
    print(image_path)

