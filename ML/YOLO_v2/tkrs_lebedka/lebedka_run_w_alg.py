from enum import Enum
from pathlib import Path
import onnxruntime as rt
import traceback
from time import time
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os

from tkrs.tkrs_lebedka.validator import Validator

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB standard deviation

def normalize(image, mean = np.array(IMAGENET_MEAN), std = np.array(IMAGENET_STD)):
    image = (image - mean) / std
    return image



skip_num = 5
dt = 0.04
FULL_FRAME = True

# Loop through the video frames
def record_video(path, out_vidos):
    root, name = path
    vidos = os.path.join(root, name)
    print(name)
    output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    cap = cv2.VideoCapture(vidos)
    success, frame0 = cap.read()
    h, w, rgb = frame0.shape

    validator = Validator((h, w))
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    #SIZE = max(h, w) if FULL_FRAME else min(w, h)
    W_IMAGE = w if FULL_FRAME else min(w, h)
    W_PLOT = min(w, h)
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration / skip_num,
                          (W_PLOT+W_IMAGE, h))



    t = 0
    while success:
        try:
            for i in range(skip_num):
                success, frame = cap.read()
                t+=dt
            if not success:
                cap.release()
                out.release()
                validator.finalize()
                break

            cropped_frame = frame if FULL_FRAME else frame[:, (w - h) // 2:(w + h) // 2, :]

            # YOLO RUN
            results = model(cropped_frame, conf=0.25, verbose=False)
            preds = results[0]


            validator.update(preds, t)
            annotated_frame = results[0].plot(labels=False)
            t0 = time()
            annotated_frame = validator.render(annotated_frame)
            #print(annotated_frame.shape, success)
            graph = validator.render_graphics()

            final_img = np.zeros(shape=(h, W_PLOT+W_IMAGE, 3), dtype=np.uint8)
            final_img[:, :W_IMAGE,:] = annotated_frame
            final_img[:, W_IMAGE:, :] = graph
            print(f"POSTPROC TIME: {(time()-t0)*1000:.1f} ms")
            out.write(final_img)

        except Exception as e:
            print(e)
            out.release()
            cap.release()
            validator.finalize()
            print(traceback.format_exc())
            break






        # except Exception as e:
        #     print(e)
        #     cap.release()
        #     break



model = YOLO("/home/popovpe/.pyenv/runs/detect/lebedka/rect_M_yolo_24.03.25_165k/last.pt")
# model.export(
#     format="onnx",
#     imgsz=(640, 1088)
# )
# exit()


dataset_folder = "/vol2/LEBEDKA/VIDEOS/PARSER/CHOSEN/stoit"
out_vidos = "/home/popovpe/.pyenv/runs/detect/lebedka/rect_M_yolo_24.03.25_165k/stoit_abs"
rootfile = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))

#rootfile = [("/vol2/LEBEDKA/VIDEOS/PARSER/CHOSEN/polset_fixed/", "nng_tkrs_noyabrsk_srv01_camera_in_interesting_position-debug_case_24-10-02_11-46-31_11-47-01_null_cam14_fixed.mp4")]
for root, file in rootfile[::]:
    print(file)
    record_video((root, file), out_vidos)