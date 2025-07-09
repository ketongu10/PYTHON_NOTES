from dataloader_seg import GeneralDataset, OldVideosDataset, AugmentationsPreset, IMAGE_SIZE, CLASSES, CLS_KEYS
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import sys

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


def inverse_normalize(image, mean=np.array(IMAGENET_MEAN), std=np.array(IMAGENET_STD)):
    image = std * image + mean
    return (np.clip(image, 0, 1) * 255).astype('uint8')


def save_one_batch_from_loader(loader, directory):
    iterator = iter(loader)
    for itr in range(2):
        batch = next(iterator)
        images = batch[0]
        out_dir = Path(directory)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", images.shape)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images):
            to_stack = []
            for cind in range(3):
                rgb_img = img.permute(1, 2, 0).numpy()[..., cind * 3:(cind + 1) * 3]
                rgb_img = inverse_normalize(rgb_img)
                to_stack.append(Image.fromarray(rgb_img))
            frame_one = to_stack[0]
            frame_one.save(str(out_dir / f"{itr}_{idx:02d}.gif"), format="GIF", append_images=to_stack, save_all=True,
                           duration=1, loop=0)

            masks = batch[1][idx].numpy()
            black = np.zeros_like(rgb_img, dtype=np.uint8)
            for cls_ind, cls in enumerate(CLS_KEYS):
                black+=(np.dstack([masks[cls_ind],masks[cls_ind],masks[cls_ind] ])*CLASSES[cls]).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{itr}_{idx:02d}_masks.jpg"), black)


@hydra.main(
    config_path=str(Path(__file__).parent.parent /'yamls'),
    config_name=Path(__file__).parent.name, 
    version_base="1.1",
)
def run(cfg):
    proc_num = 2
    batch_sise = 2

    mlflow_url = "localhost"
    mlflow.set_tracking_uri(mlflow_url)

    mlflow.set_experiment(cfg.logging.experiment)
    mlflow.start_run(run_name=cfg.logging.run_name)
    mlflow.log_artifact(".hydra/config.yaml")

    train_dataset = GeneralDataset(
        imgs_root_paths=cfg.dataset.train,
        transform=AugmentationsPreset.misha_aug.value,
        partition='train',
        p_to_take=1.0  # 0.3,

    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=proc_num,
        batch_size=batch_sise,
        shuffle=True,
        pin_memory=False,
    )

    save_one_batch_from_loader(train_loader, Path(cfg.logging.train_batch_dir))

    val_dataset = GeneralDataset(
        imgs_root_paths=cfg.dataset.val,
        partition='val',
        is_training=False,
        p_to_take=1.0  # 0.3,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=proc_num,
        batch_size=batch_sise,
        pin_memory=False,

    )

    save_one_batch_from_loader(val_loader, Path(cfg.logging.val_batch_dir))
    test_dataset = OldVideosDataset(
        imgs_root_paths=cfg.dataset.test,
        partition='test',
        p_to_take=1.0,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=proc_num,
        batch_size=batch_sise,
        pin_memory=False,
    )
    save_one_batch_from_loader(test_loader, Path(cfg.logging.test_batch_dir))

    test_d_dataset = OldVideosDataset(
        imgs_root_paths=cfg.dataset.test,
        partition='test_down',
        p_to_take=1.0,
    )
    test_d_loader = DataLoader(
        test_d_dataset,
        num_workers=proc_num,
        batch_size=batch_sise,
        pin_memory=False,
    )
    save_one_batch_from_loader(test_d_loader, Path(cfg.logging.test_d_batch_dir))

    module = SegmentationModel()  # .load_from_checkpoint(checkpoint_path)

    checkpoints_dir = Path("checkpoints")

    ckpt1 = ModelCheckpoint(dirpath=checkpoints_dir, filename="last")

    ckpt2 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="best_xor",
        save_top_k=1,
        monitor="val_xor/dataloader_idx_0",
        mode="min",
    )
    ckpt3 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_f1",
        save_top_k=1,
        monitor="test_f1",
        mode="max",
    )

    ckpt4 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_rec",
        save_top_k=1,
        monitor="test_rec",
        mode="max",
    )

    ckpt5 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_prec",
        save_top_k=1,
        monitor="test_prec",
        mode="max",
    )
    ckpt6 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_d_f1",
        save_top_k=1,
        monitor="test_d_f1",
        mode="max",
    )

    ckpt7 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_d-rec",
        save_top_k=1,
        monitor="test_d_rec",
        mode="max",
    )

    ckpt8 = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="test_d_prec",
        save_top_k=1,
        monitor="test_d_prec",
        mode="max",
    )

    MyEarlyStopping = EarlyStopping(
        monitor="val_loss/dataloader_idx_0", mode="min", patience=15, verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0, 1],
        check_val_every_n_epoch=1,
        max_epochs=cfg.epochs,
        callbacks=[ckpt1, ckpt2, ckpt3, ckpt4, ckpt5,ckpt6, ckpt7, ckpt8, MyEarlyStopping],
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=module, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader, test_d_loader]
    )
    mlflow.end_run()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()
        print("FINISHED")
