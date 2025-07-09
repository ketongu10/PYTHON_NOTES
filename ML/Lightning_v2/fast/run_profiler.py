from time import time

from dataloader_seg import GeneralDataset, OldVideosDataset, AugmentationsPreset, IMAGE_SIZE, CLASSES, CLS_KEYS, Names
from ram_loader import RamLoader, TrackedRandomSampler
from torch.utils.data.dataloader import DataLoader
from multiprocessing.shared_memory import ShareableList
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import sys

from liquid_seg.many_class_experiments.mobile_w_luja.luja_paster import LujaPaster
from net import SegmentationModel
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
import os
import mlflow
import cv2
import torch
import hydra
import warnings
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB standard deviation

STATS = ShareableList([0, 0, 0, 0, 0, 0, 0, 0], name='pasha_stats')

def inverse_normalize(image, mean=np.array(IMAGENET_MEAN), std=np.array(IMAGENET_STD)):
    image = std * image + mean
    return (np.clip(image, 0, 1) * 255).astype('uint8')





@hydra.main(
    config_path=str(Path(__file__).parent.parent / 'yamls'),
    config_name=Path(__file__).parent.name,
    version_base="1.1",
)
def run(cfg):
    proc_num = 4
    ram_workers_num = 1
    batch_sise = 12
    ram_buf_sise = 1000

    mlflow_url = "https://mlflow.nvi-solutions.ru"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minioapi.nvi-solutions.ru"
    os.environ["AWS_ACCESS_KEY_ID"] = "owlguard-rw"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "owlguard-rw"
    mlflow.set_tracking_uri(mlflow_url)

    mlflow.set_experiment(cfg.logging.experiment)
    mlflow.start_run(run_name=cfg.logging.run_name)
    mlflow.log_artifact(".hydra/config.yaml")

    is_fine_tuning = len(CLS_KEYS) < 2

    train_dataset = GeneralDataset(
        imgs_root_paths=cfg.dataset.train,
        transform=AugmentationsPreset.misha_aug.value,
        partition='train',
        # is_training=False,
        p_to_take=1.0, #1.0,  # 0.3,
        is_ft=is_fine_tuning,
        imgs_buf=ram_buf_sise,
    )


    workers = (2, 3, 4, 5, 6)
    loaders = (0, 1, 2, 3, 4)
    pairs = [(w, l) for l in loaders for w in workers]
    print(pairs)
    for w, l in pairs:
        print(f'STARTED w: {w} l:{l}')
        ram_loader = RamLoader(img_num=ram_buf_sise,
                               img_paths=train_dataset.img_paths,
                               img_shape=(*IMAGE_SIZE[::-1], 3),
                               workers_num=l)
        super_sampler = TrackedRandomSampler(train_dataset, ram_loader=ram_loader)

        train_loader = DataLoader(
            train_dataset,
            num_workers=w,
            batch_size=batch_sise,
            sampler=super_sampler,
            pin_memory=False,
        )


        module = SegmentationModel(is_ft=is_fine_tuning)  # .load_from_checkpoint(checkpoint_path)
        module.ram_loader = ram_loader

        checkpoints_dir = Path("checkpoints")

        ckpt1 = ModelCheckpoint(dirpath=checkpoints_dir, filename="last")


        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            check_val_every_n_epoch=1,
            max_epochs=1,
            callbacks=[ckpt1,],
            num_sanity_val_steps=0,
        )

        t0 = time()
        trainer.fit(
            model=module, train_dataloaders=train_loader, val_dataloaders=None
        )
        t = time()-t0
        with open(Path(cfg.logging.profiler), 'a') as f:
            print(f'{w} {l} {t:0.2f}', file=f)
        ram_loader.finalize()
    mlflow.end_run()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()
        print("FINISHED")
