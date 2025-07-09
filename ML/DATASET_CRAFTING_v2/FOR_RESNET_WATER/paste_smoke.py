import numpy as np
import cv2
import os
import tqdm
from time import time
import imageio as imio
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool

def add_smoke_to_dataset(dir):
    if np.random.uniform() < SMOKE_PROB:
        print(dir)
        path = os.path.join(where, dir)
        smoke_dir = "./Smoke/Smoke" + str(np.random.randint(1, 7))
        smoke_start = np.random.randint(50, 220)
        offset = (np.random.randint(0, 600), np.random.randint(-200, 200))
        for i in range(3):
            img = cv2.imread(os.path.join(path, f"{i}.jpg"))
            smoke = cv2.imread(smoke_dir + "/smoke_" + str(smoke_start + i * 5).zfill(4) + ".jpg")
            img = add_smoke(img, smoke, offset)
            cv2.imwrite(os.path.join(path, f"{i}.jpg"), img)
def add_smoke(image, smoke, offset=(300, 0)):

    object_inds = np.where(smoke > 10)
    xs = np.min(object_inds[1])
    ys = np.min(object_inds[0])
    xf = np.max(object_inds[1])
    yf = np.max(object_inds[0])
    smoke = smoke[ys: yf, xs: xf]
    alpha = np.max(smoke, axis=-1).astype(np.float64)
    r = smoke[..., 0].astype(np.float64) / (alpha + 1e-05)
    g = smoke[..., 1].astype(np.float64) / (alpha + 1e-05)
    b = smoke[..., 2].astype(np.float64) / (alpha + 1e-05)
    color = (np.dstack([r, g, b]) * 128).astype(np.uint8)
    alpha = np.clip(alpha.astype(np.uint8)*3, 0, 255)
    result = np.dstack([color, alpha])

    cbg = Image.fromarray(image)
    ctp = Image.fromarray(result)
    cbg.paste(ctp, offset, ctp)
    cbg = np.array(cbg)[..., :3]

    return cbg

"""t0 = time()
image = cv2.imread("./0028.jpg")
print(time()-t0)
smoke = cv2.imread("./Smoke/Smoke2/smoke_0183.jpg")
print(time()-t0)

plt.imshow(add_smoke(image, smoke))
print(time()-t0)
plt.show()
print(time()-t0)"""



if __name__ == '__main__':
    SMOKE_PROB = 0.25
    where = "./DOWN/train"
    t0 = time()
    with Pool(8) as p:
        smth = p.map(add_smoke_to_dataset, os.listdir(where))
    print(f"time: {time()-t0}")