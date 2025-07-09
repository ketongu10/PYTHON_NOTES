import cv2
import torch
import numpy as np
import os
import shutil

def create_dirs(dataset):
    try:
        os.makedirs(dataset.replace('images', 'watch'))

    except:
        shutil.rmtree(dataset.replace('images', 'watch'))
        os.makedirs(dataset.replace('images', 'watch'))



def parse_labels(file):
    classes = []
    boxes = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.split()
            classes.append(int(data[0]))
            boxes.append([float(j) for j in data[1:]])
    return {"classes":classes, "boxes":boxes}

def copy_with_bbox(source,true_detections, output):
    thickness = 2   # Толщина линии
    img = cv2.imread(source)
    color_shift, i = 30, 0
    for class_name, (x, y, w, h), score in true_detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255-i*color_shift, i*color_shift), thickness)
        text = f'{class_name}: {score:.2f}'
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255-i*color_shift, i*color_shift), 2)
    cv2.imwrite(output, img)



imgs = "/vol1/KSH/dataset/syn_proc_rect_2ecn_17.04.25/images/val"
#classes = ("box", "tros", "lebedka_polsunok", "rails", "width")
#classes = ("ksh_knot", "ksh_short_kran", "vstavka_2", "pipe1end", "not_ksh_knot", "elevator", "gloves", "gate", "flance")
#classes = ("ksh_knot", "ksh_short_kran", "vstavka_2", "pipe1end", "not_ksh_knot", "not_vstavka_n2") #('water') #
classes = ("pipe_1_end", "ecn_tros", "spider", "kops_gate", "kops_other", "wheel_on_stick", "gis_tros", "rotor", "rotor_holder", "gksh","tb_block", "tb_ear_sq","tb_ear_rd", "tb_strops",
                     "pipe_otsos", "shlang_otsos")
create_dirs(imgs)
for img in os.listdir(imgs)[:500]:
    img_path = os.path.join(imgs, img)
    labels = img_path.replace("images", "labels_bbox").replace("jpg", "txt")
    H, W, C = cv2.imread(img_path).shape
    true_labels = parse_labels(labels)

    true_detections = []
    for label, box in zip(true_labels["classes"], true_labels["boxes"]):
        # Convert to [top-left-x, top-left-y, width, height]
        # in relative coordinates in [0, 1] x [0, 1]
        x, y, w, h = box
        x, y = x-w/2, y-h/2
        # rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
        x, w, y, h = x*W, w*W, y*H, h*H
        x, y, w, h = np.round([x, y, w, h]).astype(int)

        true_detections.append(
            (
                classes[label],
                (x, y, w, h),
                1.00,
            )

                )
    print(img_path.replace("images", "watch"))
    copy_with_bbox(img_path, true_detections, img_path.replace("images", "watch"))
