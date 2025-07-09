import os
import cv2
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path


def draw_mask(frame, polygon, color):
    overlay = frame.copy()
    fill_color = color
    alpha = 0.6

    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [polygon], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


def simplify_polygon(polygon, epsilon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    epsilon = epsilon * np.sqrt(cv2.contourArea(polygon))
    simplified_polygon = cv2.approxPolyDP(polygon, epsilon, True)

    return simplified_polygon.reshape(-1, 2)


model_path = "/home/popovpe/.pyenv/runs/detect/proc/1088_Mrect_yolo_27.02.25/last.pt"

images_folder = "/vol1/KSH/READY/proc/to_105/syn_proc_ecn_rect_5.12/images/train"
images_path = [str(image_path) for image_path in Path(images_folder).rglob(("*.jpg"))]
image_path = random.choice(images_path)
print(image_path)
# image_path = "/home/nvi/ws.tsinpaev/start/mostki/data/dataset/val/mostki_01-01-22:31-01-25/mng_tkrs_nvi_person_on_walkways_under_pipe-danger_case_24-03-04_18-43-01_18-43-31_false_cam221.mp40002.jpg"

seg_model = YOLO(model_path, task='segment')

cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

results = seg_model(image_path, device="cuda", imgsz=640, verbose=False)
# if results[0].masks is not None:
polygons = results[0].masks.xy
clss = results[0].boxes.cls.cpu().numpy().astype(int)
print(results[0].masks.data.cpu().numpy().shape)
