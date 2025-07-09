from enum import Enum
from pathlib import Path

import numpy as np
from numpy.ma.core import append
from pandas.core.array_algos.transforms import shift
from ultralytics import YOLO, RTDETR
import cv2
import os

from tkrs.tkrs_proc.validator import Validator

skip_num = 5


# Loop through the video frames
def record_video(path, out_vidos):
    Validator.start()
    root, name = path
    output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    vidos = os.path.join(root, name)
    cap = cv2.VideoCapture(vidos)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    ret, frame0 = cap.read()
    #frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w, rgb = frame0.shape
    #out = cv2.VideoWriter(os.path.join(out_vidos, name), cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration/skip_num, (min(w, h), min(w, h)))
    while True:
        try:
            # Read a frame from the video
            for i in range(skip_num):
                success, frame = cap.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            #results = model.track(frame[:, (w-h)//2:(w+h)//2, :], persist=False, conf=0.25)

            results = model(frame[:, (w - h) // 2:(w + h) // 2, :], conf=0.25)
            preds = results[0]
            Validator.update(preds)
            annotated_frame = results[0].plot()
            annotated_frame = Validator.render(annotated_frame)
            #annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            print(annotated_frame.shape)
            # Display the annotated frame
            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            out.write(annotated_frame)
            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break

        except Exception as e:
            print(e)
            out.release()
            cap.release()
            # Break the loop if the end of the video is reached
            break


model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/M_seg_9.12/M_seg170k_9.12.pt")
#
# model.export(
#     format="onnx"
# )
# exit()
# model = YOLO("/home/popovpe/.pyenv/runs/detect/lesha/lesha_best.pt")

dataset_folder = "/vol2/KSH/NEW/KSH/DATASET_PROD/gis-kops" #"/vol2/KSH/NEW/KSH/DATASET_PROD/test_videos"
out_vidos = "/home/popovpe/.pyenv/runs/detect/proc/M_seg_9.12/test_videos_fps"
rootfile = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))
for root, file in rootfile:
    record_video((root, file), out_vidos)