import traceback
from multiprocessing import Process, Pool
from pathlib import Path

import cv2
import numpy as np
from tifffile import imwrite
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os

from tkrs.tkrs_proc.validator import Validator

skip_num = 5
FULL_FRAME = True
RUN_FF = True
# Loop through the video frames
def record_video(path, out_vidos, model):
    Validator.start()
    root, name = path
    output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    vidos = os.path.join(root, name)
    cap = cv2.VideoCapture(vidos)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        ret, frame0 = cap.read()
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    except:
        return
    h, w, rgb = frame0.shape
    SIZE = max(h, w) if RUN_FF else min(w, h)
    START_W = 0 if RUN_FF else (w - h) // 2
    RENDER_SHAPE = (w, h) if FULL_FRAME else (SIZE, SIZE)
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration / skip_num,
                          RENDER_SHAPE)
    found = False
    success = True
    while success:
        #print("WORKING")
        try:
            # Read a frame from the video
            for i in range(skip_num):
                success, frame = cap.read()
            if not success:
                out.release()
                cap.release()
                break
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model(frame, conf=0.25,verbose=False) #frame[:, (w - h) // 2:(w + h) // 2, :]
            preds = results[0]
            Validator.update(preds)
            annotated_frame = results[0].plot()
            annotated_frame = Validator.render(annotated_frame)
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            #print(annotated_frame.shape)

            if FULL_FRAME:
                final_img = frame
                final_img[:, START_W:START_W + SIZE, :] = annotated_frame
            else:
                final_img = annotated_frame


            out.write(final_img)
            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break


        except Exception as e:
            print(e)
            print(traceback.format_exc())
            out.release()
            cap.release()
            # if found:
            #     os.rename(os.path.join(out_vidos, output_name), os.path.join(out_vidos, "FOUND_" + output_name))
            break


def f(given):

    #model = YOLO("/home/popovpe/.pyenv/runs/detect/22.07/M_pasha_17-29.07+lesha+vst_syn/best.pt")
    root, file = given
    #for root, file in given:
    record_video((root, file), kuda_narisovat, model)

#model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/M_seg_9.12/M_seg170k_9.12.pt")
model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/big_rect_2ecn_f_tros_w_gates_proc_M_yolo_21.04_200k2/weights/last.pt")
model.export(
    format="onnx",
    imgsz=(768, 1280),

)
exit()

dohuya_videos = [
    # "/vol2/WATER/NEXTCLOUD/v2/izliv",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/source",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ksh_407",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/FPs/GENERAL1",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/rotor",
    # "/home/popovpe/Desktop/For misha/segmentation_17.02.25/src",
    # "/home/popovpe/Projects/VasiasAutoloader/autoloader/proc_fp",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/kops_154",
    # "/vol2/KSH/NEW/KSH/DATASET_PROD/gis-kops",
    "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ecn_2k/best",
    # "/home/popovpe/Projects/VasiasAutoloader/autoloader/proc_fp/intersected_pipe",
    # "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/FPs/GENERAL1" #FPS from prod
]
# #dohuya_videos = "/vol2/WATER/NEXTCLOUD/v2/izliv" #old waters
# #dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/source" #old ksh
# #dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ksh_407" #new kshs
# # dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ecn_2k/best" #ecn
# dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/FPs/GENERAL1" #FPS from prod
# # dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/rotor" #rotors
# #dohuya_videos = "/home/popovpe/Desktop/For misha/segmentation_17.02.25/src"
# # dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/kops_154" #kops
# # dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_PROD/gis-kops" #gis


#rootfile = [('/vol2/KSH/NEW/KSH/DATASET_NXTCLD/FPs/GENERAL1/', 'MISHA_ECN_FP_13.02.25.mp4')]

if __name__ == "__main__":


    #f(rootfile)
    for group_vid in dohuya_videos:
        kuda_narisovat = f"/vol2/KSH/NEW/KSH/RUN_NXTCLD/RUN_PROC_24.04_w_gates/{Path(group_vid).name}"
        Path(kuda_narisovat).mkdir(parents=True, exist_ok=True)
        rootfile = []
        for root, dirs, files in os.walk(group_vid):
            for file in files:
                if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
                    rootfile.append((root, file))
        print(rootfile)
        N = len(rootfile)
        with Pool(6) as p:
            p.map(f, rootfile[::])

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



