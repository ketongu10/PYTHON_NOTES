from multiprocessing import Pool
from multiprocessing.shared_memory import ShareableList, SharedMemory
from time import time, sleep

import hydra
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
P_TO_TAKE = 1.0 #1.0
STRIDE = 3
CLASSES = {
    "water": (255, 0, 0),
    "human": (0, 0, 255),
    "paket": (39, 224, 245),
    "other": (0, 255, 0),
    "smoke": (125, 125, 125),
    "luja": (250, 220, 55),
}
Names = {
    'img_read': 0,
    'mask_read': 1,
    'ccrop': 2,
    'base_augs': 3,
    'noise_augs': 4,
    'dstack': 5,
    'total': 6,
    'tick': 7
}
CLS_KEYS = list(CLASSES.keys())

MIN_SIZE_TRESHOLD = 576*768*0.0023 #15

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

    for cls in CLS_KEYS:
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
        # A.ISONoise(p=0.3),
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
            is_training=True,
            is_ft=False,
            imgs_buf=100,
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
        self.is_ft = is_ft
        self.imgs_buf = imgs_buf

        print("HAHASASAT", self.is_ft)
        paths = []
        small_masks = 0

        t0 = time()
        with Pool(9) as p:
            pathss = p.map(self.check_imgs_masks, imgs_root_paths)
        for smth in pathss:
            paths += smth

        self.img_paths = paths
        self.img_paths_len = len(self.img_paths)
        self.img_paths.sort()
        print(
            f"{partition} | samples: {str(len(self.img_paths))} | small_masks: {small_masks} | time: {(time() - t0):0.1f}s")

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
                if "REAL" in str(sample) or np.random.uniform() < 0.5 or "down" in str(sample) or True:
                    mask = cv2.imread(str(sample).replace("images", f"mask_{CLS_KEYS[0]}") + f".png")[..., 0]
                    if self.filter_masks(mask):
                        pathss.append(str(sample))
                    else:
                        small_masks += 1
        return pathss


    # def check_imgs_masks(self, generation):
    #     pathss = []
    #
    #     for sample in (Path(generation) / f"images/{self.partition}").iterdir():
    #         pathss.append(str(sample))
    #     return pathss





    def get_img_from_shm(self, idx):

        shm = SharedMemory('RamLoader')
        shm_masks = SharedMemory(name='RamLoaderMasks')
        shm_inds = SharedMemory('RamLoaderInds')
        shm_progress = SharedMemory('RamLoaderProgress')
        shm_used = SharedMemory('RamLoaderUsedPathes')

        buffer = np.ndarray((self.imgs_buf, self.stride, *IMAGE_SIZE[::-1], 3), dtype=np.uint8, buffer=shm.buf)
        buffer_masks = np.ndarray((self.imgs_buf, *IMAGE_SIZE[::-1]), dtype=np.uint8, buffer=shm_masks.buf)
        buffer_inds = np.ndarray((len(self.img_paths), ), dtype=np.int32, buffer=shm_inds.buf)
        buffer_progress = np.ndarray((self.imgs_buf,), dtype=np.uint8, buffer=shm_progress.buf)
        buffer_used = np.ndarray((self.img_paths_len,), dtype=np.uint8, buffer=shm_used.buf)

        imgs = {}

        buffer_used[idx] = 1
        if buffer_progress[buffer_inds[idx]] == 0:
            # print('NE USPEL', idx, buffer_progress[buffer_inds[idx]], buffer_inds[idx])
            shm.close()
            shm_masks.close()
            shm_inds.close()
            shm_progress.close()
            shm_used.close()
            return None

        for i in range(self.stride):
            img = buffer[buffer_inds[idx]][i]
            imgs[f"image{i if i > 0 else ''}"] = img.copy()

        bitmask = np.array([1 << i for i in range(len(CLS_KEYS))], dtype=np.uint8).reshape(-1, 1, 1)
        packed_masks = buffer_masks[buffer_inds[idx]].copy()
        masks = ((packed_masks & bitmask) != 0).astype(np.uint8) * 255

        for cls_ind, cls in enumerate(CLS_KEYS):
            imgs[f"mask_{cls}"] = masks[cls_ind]



        buffer_progress[buffer_inds[idx]] = 0

        shm.close()
        shm_masks.close()
        shm_inds.close()
        shm_progress.close()
        shm_used.close()
        return imgs


    def __getitem__(self, idx):



        imgs = None
        if self.partition =='train':
            imgs = self.get_img_from_shm(idx)

        if self.partition !='train' or imgs is None:
            imgs = {}
            for i in range(self.stride):
                img = cv2.imread(self.img_paths[idx] + f"/{i}.jpg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=IMAGE_SIZE)
                imgs[f"image{i if i > 0 else ''}"] = img

            dir_name = str(Path(self.img_paths[idx]).parent.parent.parent.name)
            print(dir_name)
            cache_name = str(self.img_paths[idx]).replace('/'+dir_name+'/', '/cache/'+dir_name+'/').replace("images", 'masks')+".png"
            if Path(cache_name).exists():
                bitmask = np.array([1 << i for i in range(len(CLS_KEYS))], dtype=np.uint8).reshape(-1, 1, 1)
                packed_masks = cv2.imread(cache_name)[..., 0]
                masks = ((packed_masks & bitmask) != 0).astype(np.uint8) * 255
                for cls_ind, cls in enumerate(CLS_KEYS):
                    imgs[f"mask_{cls}"] = masks[cls_ind]
            else:
                packed_masks = np.zeros(shape=(*IMAGE_SIZE[::-1],), dtype=np.uint8)
                for cls_ind, cls in enumerate(CLS_KEYS[:1] if self.is_ft else CLS_KEYS):
                    str_path = self.img_paths[idx].replace("images", f"mask_{cls}") + f".png"
                    if Path(str_path).exists():
                        mask = cv2.imread(str_path)[..., 0]
                        mask = cv2.resize(mask, dsize=IMAGE_SIZE)
                    else:
                        mask = np.zeros(shape=(*IMAGE_SIZE[::-1],), dtype=np.uint8)
                    imgs[f"mask_{cls}"] = mask
                    packed_masks |= (mask.astype(np.bool_).astype(np.uint8) << cls_ind)
                Path(cache_name).parent.mkdir(parents=True, exist_ok=True)
                # print( Path(self.img_paths[idx].replace("images", 'cache')+ f".png"))
                cv2.imwrite(cache_name, packed_masks)

        if self.is_training:
            w, h = IMAGE_SIZE
            start = np.random.randint(0, w - h)
            for key, img in imgs.items():
                imgs[key] = img[:, start:start + h]



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
        for cls in (CLS_KEYS[:1] if self.is_ft else CLS_KEYS):
            new_masks.append((new[f"mask_{cls}"]/255).astype(np.bool_))

        new_masks = np.stack(new_masks)
        new_masks = torch.from_numpy(new_masks) / 255
        masks = (new_masks > 0.5).to(dtype=torch.float32)
        return image, masks

        # packed_flags = np.zeros(new_masks[0].shape, dtype=np.uint8)
        #
        # for i, mask in enumerate(new_masks):
        #     packed_flags |= (mask.astype(np.uint8) << i)
        #
        # return image, torch.tensor(packed_flags,dtype=torch.uint8)

class OldVideosDataset(Dataset):
    def __init__(
            self,
            imgs_root_paths,
            img_size=IMAGE_SIZE,
            transform=AugmentationsPreset.identity_transform.value,
            stride=STRIDE,
            partition=None,
            p_to_take=1.0,
            is_ft=False
    ):

        down_in = "down" in partition
        self.is_ft = is_ft
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
            zero = np.ones(shape=(*IMAGE_SIZE[::-1], 3), dtype=np.float64) * 127
            img = cv2.imread(self.img_paths[idx] + f"/{i}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(size, size))
            zero[:, (w - h) // 2:(w + h) // 2, :] = img
            imgs[f"image{i if i > 0 else ''}"] = zero

        # WATER MASK IS WHITE IF TRUE ELSE BLACK
        imgs[f"mask_water"] = np.zeros(shape=IMAGE_SIZE[::-1], dtype=np.float64) if 'false' in self.img_paths[idx] \
            else np.ones(shape=IMAGE_SIZE[::-1], dtype=np.float64) * 255
        # BLACK MASKS FOR OTHER CLASSES
        if not self.is_ft:
            for i, cls in list(enumerate(CLS_KEYS))[1:]:
                imgs[f"mask_{cls}"] = np.zeros(shape=IMAGE_SIZE[::-1], dtype=np.float64)

        new = self.transform(**imgs)
        img0, img1, img2 = new["image"], new["image1"], new["image2"]

        image_for_model = np.dstack([img0, img1, img2])

        image = torch.from_numpy(image_for_model).to(dtype=torch.float32)  # / 255
        image = torch.moveaxis(image, 2, 0)

        new_masks = []
        for cls in (CLS_KEYS[:1] if self.is_ft else CLS_KEYS):
            new_masks.append(new[f"mask_{cls}"])
        new_masks = np.stack(new_masks)
        new_masks = torch.from_numpy(new_masks) / 255
        masks = (new_masks > 0.5).to(dtype=torch.float32)

        return image, masks


@hydra.main(
    config_path=str(Path(__file__).parent.parent / 'yamls'),
    config_name=Path(__file__).parent.name,
    version_base="1.1",
)
def calc_true_down(cfg):
    train_dataset = GeneralDataset(
        imgs_root_paths=cfg.dataset.train,
        partition='train',
        p_to_take=1.0,  # 0.3,
        is_ft=False
    )
    print(len(train_dataset.img_paths))
    STATS = {}
    for cls in CLS_KEYS:
        STATS[cls] = [0, 0]
    STATS['water_down'] = [0, 0]
    STATS['total'] = 0
    for path in train_dataset.img_paths:
        for cls in CLS_KEYS:
            pss = path.replace("images", f"mask_{cls}") + f".png"
            if Path(pss).exists():
                mask = cv2.imread(pss)[..., 0]
            else:
                mask = np.zeros(shape=(*IMAGE_SIZE[::-1], 3), dtype=np.uint8)[..., 0]

            summ = np.sum(mask / 255)
            if summ > 1000:
                if 'down' in path and cls=='water':
                    STATS['water_down'][0] += summ
                    STATS['water_down'][1] += 1
                else:
                    STATS[cls][0] += summ
                    STATS[cls][1] += 1
        STATS['total']+=1
    for cls in CLS_KEYS:
        print(f'{cls}: AVER_PIX={STATS[cls][0]/STATS[cls][1]} | PROB {STATS[cls][1]/STATS["total"]}')
    print(f'water_down: AVER_PIX={STATS["water_down"][0]/STATS["water_down"][1]} | PROB {STATS["water_down"][1]/STATS["total"]}')





if __name__ == "__main__":
    calc_true_down()