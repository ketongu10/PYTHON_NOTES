from multiprocessing import Pool
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


root = Path("/home/popovpe/Projects/VasiasAutoloader/autoloader/REAL_LABELED_DOWN/train/water_cut/images")
kuda = Path("/home/popovpe/Projects/VasiasAutoloader/autoloader/REAL_LABELED_DOWN/train/water_cut/gifs")
kuda.mkdir(exist_ok=True)
for i in range(5):
    (kuda/str(i)).mkdir(exist_ok=True)

def save_one_gif(path: Path | None):
    if not Path((str(path).replace('images', 'mask_water')+'.png')).exists():
        print(str(path).replace('images', 'mask_water')+'.png')
        return
    mask = cv2.imread(str(path).replace('images', 'mask_water')+'.png')
    mask[...,0]*=0
    mask[...,2]*=0

    to_stack = []
    for i in range(3):
        image = cv2.imread(str(path/f"{i}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        alpha = 0.8
        beta = (1.0 - alpha)
        cv2.addWeighted(image.astype(np.uint8), alpha, mask.astype(np.uint8), beta, 0.0, image)
        to_stack.append(Image.fromarray(image))

    frame_one = to_stack[0]
    di = np.random.randint(0, 5)
    frame_one.save(str(kuda/str(di)/f"{str(path.name)}.gif"), format="GIF", append_images=to_stack, save_all=True,
                           duration=1, loop=0, optimize=False, quality=100, quantize=0)






with Pool(6) as p:
    to_process = list(root.iterdir())
    list(tqdm(p.imap(save_one_gif, to_process), total=len(to_process)))

