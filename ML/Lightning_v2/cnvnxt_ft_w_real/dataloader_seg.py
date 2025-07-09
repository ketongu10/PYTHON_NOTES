from multiprocessing import Pool
from time import time
import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import albumentations as A
from enum import Enum
# from tifffile import imread
from albumentations import add_rain
import numpy as np
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB standard deviation
IMAGE_SIZE = 960, 576, #640, 384
P_TO_TAKE = 1.0
STRIDE = 3
CLASSES = {
    "water": (255, 0, 0),
    "human": (0, 0, 255),
    "paket": (39, 224, 245),
    "other": (0, 255, 0),
    "smoke": (125, 125, 125),
}
CLS_KEYS = list(CLASSES.keys())

MIN_SIZE_TRESHOLD = 960*576*0.0023 #15

def nonlinear_random(x1, x2):
    x = np.random.random()
    x_ = np.sqrt(abs(x - 0.5)) * np.sign(x - 0.5) / np.sqrt(2) + 0.5
    return x1 + (x2 - x1) * x_


def make_rain(images):
    slant = np.random.randint(-20, 20)
    drop_length = np.random.randint(5, 15)
    global_num_drops = np.random.randint(200, 1000)
    height, width, rgb = images[0].shape
    ret = []
    drop_color_choice = (int(155*np.random.uniform(0.8, 1.2)),
                         int(182*np.random.uniform(0.8, 1.2)),
                         int(199*np.random.uniform(0.8, 1.2)))
    for image in images:
        rain_drops = []
        num_drops = global_num_drops + np.random.randint(-100, 100)
        for _i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = np.random.randint(slant, width)
            else:
                x = np.random.randint(0, width - slant)

            y = np.random.randint(0, height - drop_length)

            rain_drops.append((x, y))
        ret.append(add_rain(img=image, slant=slant, drop_length=drop_length, drop_width=1, drop_color=drop_color_choice,
                            blur_value=1, brightness_coefficient=1.0, rain_drops=rain_drops))
    return ret

def get_targets(stride):
    add_targets = {}
    for i in range(stride + 1):
        add_targets[f'image{i}'] = 'image'

    for cls in CLS_KEYS[:1]:
        add_targets[f'mask_{cls}'] = 'mask'

    return add_targets



class AugmentationsPreset(Enum):



    not_so_hard_train = A.Compose(
        [
            A.MultiplicativeNoise(multiplier=(0.15, 1.15), per_channel=True, elementwise=True, always_apply=False,
                                  p=0.5),
            A.RandomBrightness(always_apply=False, p=0.25, limit=(0.48, 0.84)),
            A.ColorJitter(always_apply=False, p=0.25, brightness=0, contrast=(0.8, 1.2), saturation=(0.8, 1.2),
                          hue=(-0.2, 0.2)),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.InvertImg(always_apply=False, p=0.15),
            A.Sharpen(always_apply=False, p=0.1, alpha=(0.77, 1.0), lightness=(0.0, 6.38)),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ], additional_targets=get_targets(STRIDE)
    )
    identity_transform = A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ], additional_targets=get_targets(STRIDE)
    )
    no_mask_tr = A.Compose(
        [

        ], additional_targets=get_targets(STRIDE)
    )
    easy_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ], additional_targets=get_targets(STRIDE)
    )
    hard_train = A.Compose(
        [
            A.MultiplicativeNoise(multiplier=(0.15, 1.15), per_channel=True, elementwise=True, always_apply=False,
                                  p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.25, hue=0.1, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.RandomGamma(p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.99, p=0.25),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ], additional_targets=get_targets(STRIDE)
    )
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
    ], additional_targets=get_targets(STRIDE)
    )

    sep_aug = A.Compose([
        A.GaussNoise(var_limit=(40, 120*2), p=0.3),
        A.Blur(blur_limit=5, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    )



class GeneralDataset(Dataset):
    def __init__(
            self,
            imgs_root_paths,
            img_size=IMAGE_SIZE,
            transform=AugmentationsPreset.identity_transform.value,
            stride=STRIDE,
            partition=None,
            p_to_take=1.0,
            is_training=True
        ):
        """
        Args:
            dataset_cfg (dict): white_list of videos for training network.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stride = stride
        self.is_training = is_training
        self.transform = transform
        self.noize_transform = AugmentationsPreset.sep_aug.value
        self.img_size = img_size
        self.partition = partition
        self.p_to_take = p_to_take
        paths = []
        small_masks = 0

        t0 = time()
        with Pool(9) as p:
            pathss = p.map(self.check_imgs_masks, imgs_root_paths)
        for smth in pathss:
            paths+=smth

        self.img_paths = paths
        self.img_paths.sort()
        print(f"{partition} | samples: {str(len(self.img_paths))} | small_masks: {small_masks} | time: {(time()-t0):0.1f}s")


    def __len__(self):
        return len(self.img_paths)

    def filter_masks(self, mask):
        m_sum = mask.sum() / 255
        if m_sum == 0:
            return 1
        if m_sum < MIN_SIZE_TRESHOLD:
            return 0
        return 1


    def check_imgs_masks(self, generation):
        pathss = []
        small_masks = 0
        for sample in (Path(generation) / f"images/{self.partition}").iterdir():
            if np.random.uniform() < min(P_TO_TAKE, self.p_to_take):
                if "REAL" in str(sample) or np.random.uniform() < 0.35:
                    mask = cv2.imread(str(sample).replace("images", f"mask_{CLS_KEYS[0]}") + f".png")[..., 0]
                    if self.filter_masks(mask):
                        pathss.append(str(sample))
                    else:
                        small_masks += 1
        return pathss

    def __getitem__(self, idx):
        imgs = {}

        for i in range(self.stride):
            img = cv2.imread(self.img_paths[idx] + f"/{i}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=IMAGE_SIZE)
            imgs[f"image{i if i > 0 else ''}"]=img

        for cls in CLS_KEYS[:1]:    #ONLY WATER!!!!!!!!!!!!!!!!!!!
            mask = cv2.imread(self.img_paths[idx].replace("images", f"mask_{cls}")+ f".png")[..., 0]
            mask = cv2.resize(mask, dsize=IMAGE_SIZE)
            #mask = (cv2.blur(mask, (7, 7)) > 127).astype(int) * 255
            imgs[f"mask_{cls}"] = mask


        if self.is_training:
            w, h = IMAGE_SIZE
            start = np.random.randint(0, w-h)
            for key, img in imgs.items():
                imgs[key] = img[:, start:start+h]

        new = self.transform(**imgs)
        img0, img1, img2 = new["image"], new["image1"], new["image2"]

        if self.is_training:
            if np.random.uniform() < 0.25:
                img0, img1, img2 = make_rain([img0, img1, img2])

            img0 = self.noize_transform(image=img0)["image"]
            img1 = self.noize_transform(image=img1)["image"]
            img2 = self.noize_transform(image=img2)["image"]

        image_for_model = np.dstack([img0, img1, img2])

        image = torch.from_numpy(image_for_model).to(dtype=torch.float32)
        image = torch.moveaxis(image, 2, 0)


        new_masks = []
        for cls in CLS_KEYS[:1]:
            new_masks.append(new[f"mask_{cls}"])
        new_masks = np.stack(new_masks)
        # print(f"MASK SHAPE {new_masks.shape}")
        new_masks = torch.from_numpy(new_masks) / 255
        masks = (new_masks > 0.5).to(dtype=torch.float32)




        return image, masks


class OldVideosDataset(Dataset):
    def __init__(
            self,
            imgs_root_paths,
            img_size=IMAGE_SIZE,
            transform=AugmentationsPreset.identity_transform.value,
            stride=STRIDE,
            partition=None,
            p_to_take=1.0,
    ):

        down_in = "down" in partition
        self.stride = stride
        self.transform = transform
        self.img_size = img_size
        paths = []
        for vidos in Path(imgs_root_paths).iterdir():
            if not down_in:
                if 'down' not in vidos.name:
                    for sample in vidos.iterdir():
                        if np.random.uniform() < min(P_TO_TAKE, p_to_take):
                            paths.append(str(sample))
            else:
                if 'down' in vidos.name:
                    for sample in vidos.iterdir():
                        if np.random.uniform() < min(P_TO_TAKE, p_to_take):
                            paths.append(str(sample))

        self.img_paths = paths
        self.img_paths.sort()
        print(partition + " partition, samples: " + str(len(self.img_paths)))

        self.partition = partition

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        imgs = {}
        size = min(IMAGE_SIZE)
        w, h = IMAGE_SIZE
        for i in range(self.stride):
            zero = np.ones(shape=(*IMAGE_SIZE[::-1], 3), dtype=np.float64)*127
            img = cv2.imread(self.img_paths[idx] + f"/{i}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(size, size))
            zero[:, (w-h)//2:(w+h)//2, :] = img
            imgs[f"image{i if i > 0 else ''}"]=zero
        # WATER MASK IS WHITE IF TRUE ELSE BLACK
        imgs[f"mask_water"] = np.zeros(shape=IMAGE_SIZE[::-1], dtype=np.float64) if 'false' in self.img_paths[idx] \
            else np.ones(shape=IMAGE_SIZE[::-1], dtype=np.float64)*255
        # BLACK MASKS FOR OTHER CLASSES
        # for i, cls in list(enumerate(CLS_KEYS))[1:]:
        #     imgs[f"mask_{cls}"] = np.zeros(shape=IMAGE_SIZE[::-1], dtype=np.float64)

        new = self.transform(**imgs)
        img0, img1, img2 = new["image"], new["image1"], new["image2"]

        image_for_model = np.dstack([img0, img1, img2])

        image = torch.from_numpy(image_for_model).to(dtype=torch.float32) #/ 255
        image = torch.moveaxis(image, 2, 0)

        new_masks = []
        for cls in CLS_KEYS[:1]:
            new_masks.append(new[f"mask_{cls}"])
        new_masks = np.stack(new_masks)
        new_masks = torch.from_numpy(new_masks) / 255
        masks = (new_masks > 0.5).to(dtype=torch.float32)


        return image, masks

