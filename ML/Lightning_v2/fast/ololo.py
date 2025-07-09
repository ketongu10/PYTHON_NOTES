from PIL import Image
import imageio
import cv2
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Pool
from time import sleep, time
from dataloader_seg import CLS_KEYS
import torch

a= torch.randn(size=(1, 2, 3), dtype=torch.float32)
b= torch.randn(size=(1, 2, 3), dtype=torch.float32)
print(torch.stack([a, b]).permute(1,2,3,0).shape)
exit()



# class A:
#
#
#     def loh(self):
#         N = 1000
#         argss = [(a, b, c) for a in range(10) for b in range(10) for c in range(N)]
#         with Pool(2) as p:
#             res = p.starmap(self.doit, argss)
#         print(res)
#
#     def doit(self, a, b, c):
#         return a+b+c
#
# y = A()
# y.loh()
white = '/vol1/WATER/DATASET/FOR_UNET/data/PC WATERluja 16.06.25/mask_luja/val/PC WATERluja 16.06.25_14_2.png'
black = '/vol1/WATER/DATASET/FOR_UNET/data/PC noWATERup 15.05.25/mask_water/val/PC noWATERup 15.05.25_2_1.png'
img = '/vol1/WATER/DATASET/FOR_UNET/data/PC noWATERup 15.05.25/images/val/PC noWATERup 15.05.25_2_1/0.jpg'

base = '/vol1/WATER/DATASET/FOR_UNET/data/PC WATERluja 16.06.25/mask_luja/val/PC WATERluja 16.06.25_14_2.png'
# t0 = time()
# for i in range(1000):
#     a = imageio.imread(black) #np.array(Image.open(black))#cv2.imread(black)
# print(f'Black: {time()-t0}')
# t0 = time()
# for i in range(1000):
#     a = imageio.imread(white) #np.array(Image.open(white))#cv2.imread(white)
# print(f'White: {time()-t0}')

# t0 = time()
# for i in range(100):
#     arr = []
#     for i in range(6):
#         arr.append(cv2.imread(white)[..., 0].astype(np.bool_))
# print(f'White: {time()-t0}')

t0 = time()
for i in range(1):
    arr = []
    for ind, cls in enumerate(CLS_KEYS):
        arr.append(cv2.imread(f'/vol1/WATER/DATASET/FOR_UNET/data/PC WATERluja 16.06.25/mask_{cls}/train/PC WATERluja 16.06.25_12_2.png')[..., 0].astype(np.bool_))
    packed_flags = np.zeros(arr[0].shape, dtype=np.uint8)

    for i, mask in enumerate(arr):
        packed_flags |= (mask.astype(np.uint8) << i)

    cv2.imwrite(f"/home/popovpe/Projects/WaterTrain/owl.guard.cv/"
                f"cyclops/training/liquid_seg/many_class_experiments/fast/tmp/all.png", packed_flags)
    bitmask = np.array([1 << i for i in range(len(CLS_KEYS))], dtype=np.uint8).reshape(-1, 1, 1)
    unpacked_masks = ((packed_flags & bitmask) != 0).astype(np.uint8)*255

for ind, cls in enumerate(CLS_KEYS):
    cv2.imwrite(f"/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/many_class_experiments/fast/tmp/mask_{cls}.png", unpacked_masks[ind])
print(f'White: {time()-t0}')

bitmask = np.array([1 << i for i in range(len(CLS_KEYS))], dtype=np.uint8).reshape(-1, 1, 1)
print(bitmask)
# t0 = time()
# for i in range(1000):
#     arr = []
#     for i in range(6):
#         arr.append(cv2.imread(white)[..., 0].astype(np.bool_))
#     masks = np.stack(arr, axis=0)
#     packed = np.packbits(masks, axis=0, bitorder='little')
#     unpacked_all = np.unpackbits(packed, axis=0, count=6, bitorder='little')
#     masks = [unpacked_all[i] for i in range(6)]
# print(f'White: {time() - t0}')
# for i in range(1000):
#     mask3_extracted = (packed_flags & (1 << 2)) > 0