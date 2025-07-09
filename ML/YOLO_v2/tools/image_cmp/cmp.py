import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import cv2

vidos = "/vol2/KSH/NEW/KSH/DATASET_PROD/gis-kops/nng_tkrs_noyabrsk_srv01_camera_in_interesting_position-debug_case_24-08-18_20-01-25_20-01-55_null_cam13.mp4"
path_to_imgs = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/imgs/"
path_to_save = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/imgs_w_masks/"
model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/n_seg_19.11/best.pt")
skip_num = 5

n = 0
avg_mask = np.zeros(shape=(640, 640, 1), dtype=int)
for img_path in sorted(os.listdir(path_to_imgs)):
    img = cv2.imread(path_to_imgs+img_path)
    #print(img)

    h, w, rgb = img.shape
    ret = model(img[:, (w - h) // 2:(w + h) // 2, :], conf=0.25)
    label = ret[0]
    annotated_frame = ret[0].plot()
    clss = label.boxes.cls.cpu().numpy().astype(dtype=int)
    confs = label.boxes.conf.cpu().numpy()
    xywh = label.boxes.xywh.cpu().numpy()

    for i, cls in enumerate(clss):
        if cls == 3:
            mask = ret[0].masks.data.cpu().numpy().astype(int).transpose(1, 2, 0)
            mask*=255
            print(mask.shape)
            #if mask.shape[2] > 1:
            #cv2.imwrite(path_to_save+img_path, mask[:, :, i:i+1])
            avg_mask+=mask[:, :, i:i+1]
            n+=1
avg_mask//=n
cv2.imwrite(path_to_save+"avg_mask.jpg", avg_mask)





