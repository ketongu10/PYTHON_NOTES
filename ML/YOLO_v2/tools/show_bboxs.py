import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import shutil

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

img = "/vol1/KSH/DATASET_BLENDER/cropped_dataset/images/val/3694.jpg"
labels = "/vol1/KSH/DATASET_BLENDER/cropped_dataset/labels/val/3694.txt"
out = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/imgs/10.jpg"

H, W, C = cv2.imread(img).shape
true_labels = parse_labels(labels)
classes = ("ksh_short_kran", "ksh_knot", "vstavka_pipe", "vstavka_2", "pipe")
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

copy_with_bbox(img, true_detections, out)
