import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from multiprocessing import Pool, Process
import time
from tqdm import tqdm



def save_sample(sample, imgs_path):
    imgs_path.mkdir(parents=True, exist_ok=True)
    for i in range(STRIDE):
        cv2.imwrite(str(imgs_path)+f"/{i}.jpg", sample[i])



def process_video(vidos: Path):
    i = 0
    stride = np.random.randint(2, 8)
    cap = cv2.VideoCapture(str(vidos))
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return

    for j in range(stride):
        ret, frame1 = cap.read()
        if not ret:
            cap.release()
            return
    for j in range(stride):
        ret, frame2 = cap.read()
        if not ret:
            cap.release()
            return

    save_sample([frame0, frame1, frame2], kuda/f'{str(vidos.name)}_{i}')
    i+=1
    cont = True
    while cont:
        try:
            stride = np.random.randint(2, 8)
            for _ in range(SHIFT):
                ret, frame0 = cap.read()
                if not ret:
                    cap.release()
                    return
            for j in range(stride):
                ret, frame1 = cap.read()
                if not ret:
                    cap.release()
                    return
            for j in range(stride):
                ret, frame2 = cap.read()
                if not ret:
                    cap.release()
                    return
            save_sample([frame0, frame1, frame2], kuda / f'{str(vidos.name)}_{i}')
            i += 1
        except Exception as e:
            print("=========LOH===========")
            print(traceback.format_exc())
            cap.release()
            break

STRIDE = 3
SHIFT = 25
source = Path('/home/popovpe/Projects/VasiasAutoloader/autoloader/REAL_LABELED_DOWN/train/no_water')
kuda = Path('/home/popovpe/Projects/VasiasAutoloader/autoloader/REAL_LABELED_DOWN/train/no_water_cut')

data_to_process = list(source.iterdir())
print(data_to_process)
with Pool(6) as p:
    a= list(tqdm(p.imap(process_video, data_to_process), total=len(data_to_process)))

# for vidos in list(source.iterdir())[:10]:
#     process_video(vidos)






