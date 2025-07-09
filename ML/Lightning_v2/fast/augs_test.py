from time import time
from pathlib import Path

import albumentations as A
from numba import jit
import numpy as np
import cv2
from dataloader_seg import CLS_KEYS, IMAGE_SIZE

def get_targets(stride):
    add_targets = {}
    for i in range(stride + 1):
        add_targets[f'image{i}'] = 'image'

    for cls in CLS_KEYS:
        add_targets[f'mask_{cls}'] = 'mask'

    return add_targets


misha_aug = A.Compose([
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.3, hue=0.15, p=0.35),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.35),
        A.ToGray(p=0.125),
        A.CLAHE(p=0.15),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.35),
        A.RandomGamma(p=0.35),
        A.ImageCompression(quality_lower=60, p=0.35),
        A.CoarseDropout(always_apply=False, p=0.35, max_holes=30, max_height=40, max_width=40,
                        min_holes=10, min_height=20, min_width=20, fill_value=(0, 0, 0),
                        mask_fill_value=0),
        # A.PixelDropout(always_apply=False, p=1.0,  dropout_prob=0.02, per_channel=0, drop_value=(0, 0, 0),
        #                 mask_drop_value=0),
        #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ], additional_targets=get_targets(3)
    )



# 1. Оптимизированные Numba-функции для простых операций
@jit(nopython=True, fastmath=True)
def rgb_shift_numba(img, r_shift, g_shift, b_shift):
    img = img.astype(np.int16)
    img[..., 0] = np.clip(img[..., 0] + r_shift, 0, 255)
    img[..., 1] = np.clip(img[..., 1] + g_shift, 0, 255)
    img[..., 2] = np.clip(img[..., 2] + b_shift, 0, 255)
    return img.astype(np.uint8)


@jit(nopython=True)
def adjust_gamma_numba(img, gamma):
    # Создаем lookup table
    table = np.zeros(256, dtype=np.uint8)
    inv_gamma = 1.0 / gamma

    for i in range(256):
        val = ((i / 255.0) ** inv_gamma) * 255
        # Ручная реализация clip для совместимости с Numba
        if val < 0:
            table[i] = 0
        elif val > 255:
            table[i] = 255
        else:
            table[i] = int(round(val))

    # Применяем LUT к изображению
    result = np.empty_like(img)
    h, w = img.shape[:2]

    if len(img.shape) == 3:  # Цветное изображение (H,W,C)
        for i in range(h):
            for j in range(w):
                for k in range(img.shape[2]):
                    result[i, j, k] = table[img[i, j, k]]
    else:  # Градации серого (H,W)
        for i in range(h):
            for j in range(w):
                result[i, j] = table[img[i, j]]

    return result


# 2. Оптимизированный пайплайн
def get_optimized_pipeline():
    return A.Compose([
        # Цветовые преобразования (объединены в один шаг)
        A.OneOf([
            # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
            A.Lambda(
                image=lambda img, **kwargs: rgb_shift_numba(
                    img,
                    np.random.randint(-30, 30),
                    np.random.randint(-30, 30),
                    np.random.randint(-30, 30)
                ),
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.3,
                hue=0.15,
                p=1.0
            ),
        ], p=0.57),

        # Быстрые монохромные преобразования
        A.OneOf([
            A.ToGray(p=1.0),
            A.CLAHE(p=1.0),
        ], p=0.17),  # Объединены с суммарной вероятностью

        # Геометрические преобразования
        A.HorizontalFlip(p=0.5),

        # Яркость/контрастность (оптимизированные версии)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            # A.RandomGamma(p=1.0),
            A.Lambda(
                image=lambda img, **kwargs: adjust_gamma_numba(
                    img,
                    gamma=np.random.uniform(0.7, 1.3)
                ),
                p=1.0
            ),
        ], p=0.57),

        # Шум и артефакты
        A.OneOf([
            A.ImageCompression(quality_lower=60, p=1.0),
            A.CoarseDropout(
                max_holes=30,
                max_height=40,
                max_width=40,
                min_holes=10,
                min_height=20,
                min_width=20,
                fill_value=0,
                p=1.0
            ),
        ], p=0.57),
    ], additional_targets=get_targets(3))


# 3. Дополнительные оптимизации
def preprocess_images(imgs):
    """Конвертация в UMat для OpenCL-ускорения"""
    return {k: cv2.UMat(v) if k.startswith('image') else v
            for k, v in imgs.items()}


def postprocess_results(result):
    """Конвертация обратно из UMat"""
    return {k: cv2.UMat.get(v) if isinstance(v, cv2.UMat) else v
            for k, v in result.items()}


# 4. Использование

path = '/vol1/WATER/DATASET/FOR_UNET/data/105 WATERdown 8.05.25/images/val/105 WATERdown 8.05.25_5_0'

where = '/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/many_class_experiments/fast/imgs/'
imgs = {}
for i in range(3):
    img = cv2.imread(path+ f"/{i}.jpg")
    img = cv2.resize(img, dsize=IMAGE_SIZE)
    imgs[f"image{i if i > 0 else ''}"] = img
for cls_ind, cls in enumerate(CLS_KEYS):
    str_path = path.replace("images", f"mask_{cls}") + f".png"
    if Path(str_path).exists():
        mask = cv2.imread(str_path)[..., 0]
        mask = cv2.resize(mask, dsize=IMAGE_SIZE)
    else:
        mask = np.zeros(shape=(*IMAGE_SIZE[::-1],), dtype=np.uint8)
    imgs[f"mask_{cls}"] = mask

optimized_tr = get_optimized_pipeline()
for i in range(100):
    # final_result = misha_aug(**imgs)
    result = optimized_tr(**imgs)
    final_result = result
t0 = time()
for i in range(1000):
    # final_result = misha_aug(**imgs)
    result = optimized_tr(**imgs)
    final_result = result
print(time()-t0)
imgs = final_result["image"], final_result["image1"], final_result["image2"]
for i in range(3):
    cv2.imwrite(where+f'{i}.jpg', imgs[i])
