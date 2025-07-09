import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from pyarrow import duration

from net import SegmentationModel
from dataloader_seg import GeneralDataset, AugmentationsPreset, IMAGE_SIZE, CLS_KEYS, CLASSES
from torch import tensor
from multiprocessing import Pool, Process
import time
import os


class Hysteresis:
    FILL = 0.5
    TOTAL_TIME = 5 #seconds
    def __init__(self, fps, stride):
        self.total_frames = int(Hysteresis.TOTAL_TIME / (stride/fps))
        self.fill = int(Hysteresis.FILL * self.total_frames)
        print(f"HYST: {self.fill}/{self.total_frames} on {fps}")
        self.buffer = []
    def __call__(self, res):
        if len(self.buffer) >= self.total_frames:
            self.buffer.pop(0)
        self.buffer.append(res)
    def check(self) -> bool:
        sum = 0
        for ind in range(len(self.buffer)-1, -1, -1):
            sum+=1 if self.buffer[ind] else 0
            if sum > self.fill:
                return True
        return False


def write_text(img, text, color=(255, 0, 0), pos=(50, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = pos
    fontScale = 1
    color1 = color
    thickness = 2
    image = cv2.putText(img, text, org, font,
                        fontScale, color1, thickness, cv2.LINE_AA)
    return image

def prepare_x(buf, transform, dev):
    new_buf = []
    for i, img in enumerate(buf):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

        new_buf.append(img)
    #print(img.shape, IMAGE_SIZE)
    empty_mask = np.zeros(shape=IMAGE_SIZE[::-1], dtype=int)
    new = transform(image=new_buf[0], image1=new_buf[1], image2=new_buf[2], mask=empty_mask)
    new_buf = [new["image"], new["image1"], new["image2"]]
    img_to_x = np.array([np.transpose(np.dstack(new_buf), (2, 0, 1))])
    img_to_x = tensor(img_to_x).cuda(dev)
    #print(img_to_x.shape)
    return img_to_x


def prepare_x_pad(buf, transform, dev):
    new_buf = []
    for i, img in enumerate(buf):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(960, 540), interpolation=cv2.INTER_CUBIC)

        empty_image = np.ones(shape=(*IMAGE_SIZE[::-1], 3), dtype=int)
        empty_image[18:-18,...] = img

        new_buf.append(empty_image)
    #print(img.shape, IMAGE_SIZE)
    empty_mask = np.zeros(shape=IMAGE_SIZE[::-1], dtype=int)
    new = transform(image=new_buf[0], image1=new_buf[1], image2=new_buf[2], mask=empty_mask)
    new_buf = [new["image"], new["image1"], new["image2"]]
    img_to_x = np.array([np.transpose(np.dstack(new_buf), (2, 0, 1))])
    img_to_x = tensor(img_to_x).cuda(dev)
    #print(img_to_x.shape)
    return img_to_x


target_fps = 5
DENC = 0.0023
SUM_TRESHOLD = IMAGE_SIZE[0]*IMAGE_SIZE[1]*DENC
CONFS = {
    "water": 0.4,
    "human": 0.2,
    "paket": 0.2,
    "other": 0.2,
    "smoke": 0.2,
}
def record_video(path, out_vidos, model, trans, dev):
    root, name = path
    vidos = os.path.join(root, name)

    cap = cv2.VideoCapture(vidos)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS: ", cap.get(cv2.CAP_PROP_FPS))



    stride = int(fps//target_fps if fps//target_fps > 1 else 1) #5
    print(f"STRIDE = {stride}")
    timer = 0
    ret, frame0 = cap.read()
    if ret:
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    else:
        cap.release()
        return
    h, w, rgb = frame0.shape
    #print(frame0.shape)
    output_name = f"fps={int(fps)}_stride={stride}_"+name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), fps/stride, (w, h))
    print(os.path.join(out_vidos, output_name))
    for j in range(stride):
        ret, frame1 = cap.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    for j in range(stride):
        ret, frame2 = cap.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    t = 0
    hyst = Hysteresis(fps, stride)
    found = False
    cont = True
    while cont:
        try:
            with torch.no_grad():
                tr_conf = CONFS["water"]
                x = prepare_x((frame0, frame1, frame2), trans, dev)
                masks = model(x).cpu().numpy()[0]
                mask = masks[0]
                inds_after_thr = np.where(mask > tr_conf)
                nnnn = len(inds_after_thr[0])
                alya_conf = mask[inds_after_thr].sum() / (1 if nnnn < 10 else nnnn)

                _, mask = cv2.threshold(mask, tr_conf, 1, cv2.THRESH_BINARY)
                mask_sum = mask.sum()
                is_water = True if mask_sum>=SUM_TRESHOLD else False
                hyst(is_water)

                black = np.zeros(shape=(*IMAGE_SIZE[::-1], 3), dtype=np.uint8)
                for cls_ind, cls in enumerate(CLS_KEYS[:2]):
                    binmask = (masks[cls_ind] > CONFS[cls]).astype(np.uint8)
                    black += (np.dstack([binmask, binmask, binmask]) * CLASSES[cls]).astype(
                        np.uint8)

                mask = cv2.resize(black, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                # print(f"mask: {np.shape(mask)}")


                show = frame0.copy()
                show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
                # show = mask.astype(np.uint8)
                alpha = 0.8
                beta = (1.0 - alpha)

                cv2.addWeighted(show.astype(np.uint8), alpha, mask.astype(np.uint8), beta, 0.0, show)

                show = write_text(show, f"conf={alya_conf:0.2f}", (0, 0, 255) if is_water else (0, 255, 0), pos=(350, 75))
                show = write_text(show, f"time={timer:0.1f}s", (0, 0, 255) if is_water else (0, 255, 0), pos=(350, 50))
                text = f"sum={round(mask_sum/1000)}k"
                if hyst.check():
                    text = "!"+text+"!"
                    found = True
                show = write_text(show, text, (0, 0, 255) if is_water else (0, 255, 0), pos=(350, 25))

                out.write(show)

                timer += stride/fps
                t+=1
                for j in range(stride):
                    ret, frame0 = cap.read()
                    if not ret:

                        print("=========NOT LOH===========")
                        print(f"FPS: {fps:0.2f} | FRAMES: {frame_count} | C_FRAMES = {t*stride} |  T: {frame_count/fps} | C_T: {timer}", name)
                        out.release()
                        cap.release()
                        if root == "/vol2/WATER/NEXTCLOUD/v2/izliv":
                            if found:
                                os.rename(os.path.join(out_vidos, output_name),
                                          os.path.join(out_vidos, "WATER_" + output_name))
                            else:
                                Path(os.path.join(out_vidos, output_name)).unlink(missing_ok=True)
                        cont = False
                        break
                    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                (frame0, frame1, frame2) = (frame1, frame2, frame0)


        except Exception as e:
            print("=========LOH===========")
            print(traceback.format_exc())
            out.release()
            cap.release()
            if root == "/vol2/WATER/NEXTCLOUD/v2/izliv":
                if found:
                    os.rename(os.path.join(out_vidos,output_name), os.path.join(out_vidos, "WATER_"+output_name))
                else:
                    Path(os.path.join(out_vidos,output_name)).unlink(missing_ok=True)
            break


    #print(t)

def f(given, dev=0):
    trans = AugmentationsPreset.identity_transform.value
    model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location=f"cuda:{dev}")
    model.eval()
    for root, file in given:
        record_video((root, file), out_vidos, model, trans, dev)

checkpoint_path = ("/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/"
                   "many_class_experiments/cnvnxt_ft_w_real/outputs/run_training/"
                   "cnvnxt_ft_filtered_2025-05-27_08-38-24/checkpoints/last.ckpt")
# trans = AugmentationsPreset.identity_transform.value
# model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
# model.eval()

DATASETS_FOR_RUN = {
    "/home/popovpe/Projects/VasiasAutoloader/autoloader/jija_again": "another_fp",
    "/vol1/WaterDif/VideosForRun/ForCompareMan1-2": "manual12",
    "/vol1/WaterDif/VideosForRun/ForCompareMan3": "manual3",
    "/home/popovpe/Projects/VasiasAutoloader/autoloader/jija_fp": "fp",
    "/vol2/WATER/RUNS/RUN_SOURCES/FP_from_prod/11.04.24/FP": "fp-11-04",
    "/vol2/WATER/RUNS/RUN_SOURCES/FP_from_prod/27-30.01/FP": "fp-30-01",
    "/vol2/WATER/RUNS/RUN_SOURCES/FP_from_prod/MNG_03.24-05.24": "mng",
    # "/vol2/WATER/NEXTCLOUD/v2/izliv": "dohuya",
    # "/home/popovpe/Projects/VasiasAutoloader/autoloader/jija_down_fp_full": "other_fp"
}

output_dir = Path(f"/vol2/WATER/RUNS/SEG/106/cnvnxt_ft_w_real_conf={CONFS['water']}_denc={DENC*100:0.2f}%_26.05.25")








P_NUM = 3
if __name__ == "__main__":
    print(f"SUM_TRESHOLD={SUM_TRESHOLD:0.1f}")
    for path, dir_name in DATASETS_FOR_RUN.items():
        dataset_folder = path
        out_vidos = output_dir / dir_name
        out_vidos.mkdir(exist_ok=True, parents=True)
        rootfile = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
                    rootfile.append((root, file))

        rootfile = rootfile[::]
        N = len(rootfile)

        print(f"rootfile {rootfile}")

        Ps = []
        for i in range(P_NUM):
            Pi = Process(target=f, args=(rootfile[N //P_NUM*i:N //P_NUM*(i+1)],0))
            Pi.start()
            Ps.append(Pi)
            print(f"P{i} started!!")
        for Pi in Ps:
            Pi.join()
