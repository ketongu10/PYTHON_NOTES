from multiprocessing import Process, Pool
import cv2
import numpy as np
from tifffile import imwrite
from time import time
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os

from tkrs.tkrs_lebedka.validator import Validator

skip_num = 5
dt = 0.04
WITH_ALG = False
FULL_FRAME = True

# Loop through the video frames
def record_video(path, out_vidos, model):
    root, name = path
    output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    vidos = os.path.join(root, name)
    cap = cv2.VideoCapture(vidos)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / (fps if fps != 0 else 25)
    print(frame_count, fps, duration, name)
    ret, frame0 = cap.read()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w, rgb = frame0.shape
    validator = Validator((h, w))
    # out = cv2.VideoWriter(os.path.join(out_vidos, name), cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))

    SIZE = min(w, h)
    START_W = (w - h) // 2
    RENDER_SHAPE = (w, h) if FULL_FRAME else ((2 if WITH_ALG else 1)*SIZE, SIZE)
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration / (skip_num+1),
                          RENDER_SHAPE)
    found = False
    t = 0
    while True:
        try:
            # Read a frame from the video
            for i in range(skip_num):
                success, frame = cap.read()
                t+=dt
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame[:, (w - h) // 2:(w + h) // 2, :], conf=0.5, verbose=False)
            preds = results[0]
            validator.update(preds, t)
            annotated_frame = results[0].plot()
            if WITH_ALG:
                annotated_frame = validator.render(annotated_frame)
                graph = validator.render_graphics()


            if FULL_FRAME:
                final_img = frame
                final_img[:, START_W:START_W+SIZE, :] = annotated_frame
            else:
                final_img = np.zeros(shape=(RENDER_SHAPE[1], RENDER_SHAPE[0], 3), dtype=np.uint8)
                final_img[:, :SIZE, :] = annotated_frame

            if WITH_ALG:
                final_img[:, SIZE:, :] = graph

            out.write(final_img)
            #out.write(annotated_frame)

        except Exception as e:
            print(e)
            print(t/dt, frame_count, fps, duration, name)
            out.release()
            cap.release()
            # if found:
            #     os.rename(os.path.join(out_vidos, output_name), os.path.join(out_vidos, "FOUND_" + output_name))
            break



def f(given):
    root, file = given
    record_video((root, file), kuda_narisovat, model)

model = YOLO("/home/popovpe/.pyenv/runs/detect/lebedka/M_yolo_3.02_165k/last.pt")
# model.export(
#     format="onnx"
# )
# exit()

dohuya_videos = "/vol2/LEBEDKA/VIDEOS/PARSER/CHOSEN/polset_fixed"
kuda_narisovat = "/vol2/LEBEDKA/RUNS/11.02/no_graphs" #alg_speed_avg10_filtered"
rootfile = []
for root, dirs, files in os.walk(dohuya_videos):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))
N = len(rootfile)

#rootfile = [('/vol2/LEBEDKA/VIDEOS/PARSER/CHOSEN/polset/', '4video.mp4')]

if __name__ == "__main__":


    #f(rootfile)
    START_TIME = time()
    with Pool(6) as p:
        p.map(f, rootfile[:1000])
    print(f"TOOK {(time()-START_TIME):.01f} seconds for {len(rootfile)} videos")

    # P1 = Process(target=f, args=(rootfile[:N//3],))
    # P1.start()
    # print("P1 started!!")
    # P2 = Process(target=f, args=(rootfile[N //3:N //3*2],))
    # P2.start()
    # print("P2 started!!")
    # P3 = Process(target=f, args=(rootfile[N // 3*2:],))
    # P3.start()
    # print("P3 started!!")
    # P1.join()
    # P2.join()
    # P3.join()



