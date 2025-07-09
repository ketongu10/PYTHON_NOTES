import shutil
from multiprocessing import Process, Pool
from pathlib import Path

import cv2
import numpy as np
from tifffile import imwrite
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
import cv2
import os
STR2CLS = {
    "pipe_1_end": 0, "ecn_tros": 1, "spider": 2,
    "kops_gate": 3, "kops_other":4,
    "wheel_on_stick": 5, "gis_tros": 6
}
def create_dirs():
    try:
        os.makedirs(kuda_narisovat + "/images")
        os.makedirs(kuda_narisovat + "/masks")
        os.makedirs(kuda_narisovat + "/labels")
        os.makedirs(kuda_narisovat + "/errors")
    except:
        shutil.rmtree(kuda_narisovat + "/images")
        shutil.rmtree(kuda_narisovat + "/masks")
        shutil.rmtree(kuda_narisovat + "/labels")
        shutil.rmtree(kuda_narisovat + "/errors")
        os.makedirs(kuda_narisovat + "/images")
        os.makedirs(kuda_narisovat + "/masks")
        os.makedirs(kuda_narisovat + "/labels")
        os.makedirs(kuda_narisovat + "/errors")


def save_sample(img, labels, path):
    clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
    confs = labels.boxes.conf.cpu().numpy()
    xywh = labels.boxes.xywh.cpu().numpy()
    if labels.masks:
        mask = labels.masks.data.cpu().numpy().astype(int)

    gate_masks = []
    gate_boxes = []
    gate_confs = []

    other_masks = []
    other_boxes = []
    other_confs = []
    H, W, C = img.shape

    for i, cls in enumerate(clss):
        if cls == STR2CLS["kops_gate"] and confs[i] >= MIN_CONF:
            gate_masks.append(mask[i]*255)
            bbox = xywh[i]
            x, y, w, h = bbox
            x, y, w, h = x/W, y/H, w/W, h/H
            gate_boxes.append([x, y, w, h])
            gate_confs.append(confs[i])
        if cls == STR2CLS["kops_other"] and confs[i] >= MIN_CONF:
            other_masks.append(mask[i]*255)
            bbox = xywh[i]
            x, y, w, h = bbox
            x, y, w, h = x / W, y / H, w / W, h / H
            other_boxes.append([x, y, w, h])
            other_confs.append(confs[i])

    if gate_masks and other_masks:
        print(gate_masks[-1].shape, '|', other_masks[-1].shape, '|', img.shape, path)
        #if gate_masks[-1].shape == (640, 1088):
        cv2.imwrite(f"{path}.jpg", img)
        g_mx_arg = np.argmax(gate_confs)
        o_mx_arg = np.argmax(other_confs)
        cv2.imwrite(f"{path.replace('images/', 'masks/kops_gate_')}_conf={gate_confs[g_mx_arg]:0.3f}.png", gate_masks[g_mx_arg])
        # print(f"{path.replace('images', 'masks/kops_gate')}_conf={gate_confs[g_mx_arg]}.png")
        cv2.imwrite(f"{path.replace('images/', 'masks/kops_other_')}_conf={other_confs[o_mx_arg]:0.3f}.png", other_masks[o_mx_arg])
        with open(f"{path.replace('images', 'labels')}.txt", 'w') as f:
            print(STR2CLS["kops_gate"], *(gate_boxes[g_mx_arg]), file=f)
            print(STR2CLS["kops_other"], *(other_boxes[o_mx_arg]), file=f)
        #else:
        #    with open(f"{path.replace('images', 'errors')}.txt", 'w+') as f:
        #        print(path+f' {gate_masks[-1].shape}', file=f)

MIN_CONF = 0.5

FULL_FRAME = True
RUN_FF = True
# Loop through the video frames
def record_video(path, kuda, model):
    root, name = path
    vidos = os.path.join(root, name)

    cap = cv2.VideoCapture(vidos)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        success, frame0 = cap.read()
        if success:
            results0 = model(frame0, conf=MIN_CONF,verbose=False)[0]
            save_sample(frame0, results0, f"{kuda}/images/{name}_0")


            for i in range(int(frame_count//3)):
                success, frame1 = cap.read()
            if success:
                results1 = model(frame1, conf=MIN_CONF,verbose=False)[0]
                save_sample(frame1, results1, f"{kuda}/images/{name}_1")

                for i in range(int(frame_count // 3)):
                    success, frame2 = cap.read()
                if success:
                    results2 = model(frame2, conf=MIN_CONF, verbose=False)[0]
                    save_sample(frame2, results2, f"{kuda}/images/{name}_2")






        cap.release()
    except Exception as e:
        print(e)
        cap.release()



def f(given):

    root, file = given
    record_video((root, file), kuda_narisovat, model)

model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/1088_Mrect_yolo_27.02.25/last.pt")

dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/kops_154"
kuda_narisovat = "/vol2/KOPS/prelabeled_kops_parser_till_11.24"
create_dirs()

rootfile = []
for root, dirs, files in os.walk(dohuya_videos):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))
N = len(rootfile)

if __name__ == "__main__":



    with Pool(3) as p:
        res = list(tqdm(p.imap(f, rootfile[::]), total=N))





