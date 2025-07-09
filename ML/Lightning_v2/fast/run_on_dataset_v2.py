import torch
from matplotlib import pyplot as plt

from dataloader_seg import GeneralDataset, OldVideosDataset, AugmentationsPreset, IMAGE_SIZE
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import cv2
from time import time
from net import SegmentationModel
from tqdm import tqdm

def xor4masks(masks):
    mask_water = masks[0]
    for msk in masks[1:]:
        sub = mask_water*(msk > 0.2).astype(float)
        mask_water -= sub
    mask_water = np.clip(mask_water, 0, 1)
    print(np.min(mask_water), np.min(mask_water))
    return mask_water



checkpoint_path = ("/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/"
                   "many_class_experiments/mobile_3e4/outputs/run_training/"
                   "sgfrmr_5cls_mobile_2025-05-20_08-54-58/checkpoints/test_d_f1.ckpt")
trans = AugmentationsPreset.identity_transform.value
model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
model.eval()

# val_dataset = GeneralDataset(
#     imgs_root_paths=[
#         "/home/nvi/ws.popov/Training/DATASET/WATER/TRAIN_200",
#                      ],
#     transform=trans,
#     partition='val',
#     p_to_take=1.0,
#     is_training=False
# )

val_dataset = OldVideosDataset(
    imgs_root_paths="/vol2/WATER/REAL_DATA_DATASET/NEW_TEST/test_23.05",
    transform=trans,
    partition='old_test',
    p_to_take=1.0
)

w, h = IMAGE_SIZE

mask2boolCs = w*h*np.linspace(0.0001, 0.01, num=10)
trhlds = np.linspace(0.1, 0.99, num=10)
TPSFPS = np.zeros(shape=(len(mask2boolCs), len(trhlds), 4), dtype=float)
preds = []
gts = []
with (torch.no_grad()):
    test_path = (Path(checkpoint_path).parent.parent/val_dataset.partition)
    test_path.mkdir(exist_ok=True)
    for i in tqdm(range(len(val_dataset))):
        imgs, mask = val_dataset[i]
        imgs = imgs[None, ...]
        dump = np.zeros(shape=(h*2, w), dtype=int)

        new_mask = model(imgs.cuda()).cpu().numpy()[0][0]
        #new_mask = xor4masks(new_mask)

        dump[:h, :] = (mask*255).numpy().astype(int)[0]
        dump[h:, :] = (new_mask*255).astype(int)
        cv2.imwrite(str(test_path/f"{i}.png"), dump)

        gts = torch.tensor(np.array(mask.numpy()[None, ...]))[:, 0]
        preds = torch.tensor(np.array(new_mask[None, ...]))
        # print(f"shapes: {gts.shape} | {preds.shape}")
        for i, density in enumerate(mask2boolCs):
            model.mask2boolC = density
            for j, trhld in enumerate(trhlds):
                model.calc_tpfp(preds > trhld, gts > trhld, model.up)
                TPSFPS[i][j][:] += (model.up.tp, model.up.fp, model.up.fn, model.up.tn)
                model.up.reset()
                # model.tp, model.fp, model.fn, model.tn = 0, 0, 0, 0


    gts = torch.tensor(np.array(gts))
    preds = torch.tensor(np.array(preds))
    fig, (recax, precax, f1ax) = plt.subplots(1, 3, figsize=(18, 8))
    for i, density in enumerate(mask2boolCs):
        TPs = TPSFPS[i, :, 0]
        TPFPs = (TPSFPS[i, :, 0]+TPSFPS[i, :, 1])
        TPFNs = (TPSFPS[i, :, 0]+TPSFPS[i, :, 2])
        rec = np.divide(TPs, TPFNs, out=np.zeros_like(TPs), where=TPFNs != 0)
        prec = np.divide(TPs, TPFPs, out=np.zeros_like(TPs), where=TPFPs != 0)
        f1 = np.divide(2*rec*prec, rec+prec, out=np.zeros_like(TPs), where=(rec+prec) != 0)
        A = np.log(2-i/len(mask2boolCs))
        deg = 1/len(mask2boolCs)
        recax.plot(trhlds, rec, alpha=A)
        precax.plot(trhlds, prec, alpha=A)
        f1ax.plot(trhlds, f1, label=f'{100*density/w/h:0.2f}%', alpha=A)
        recax.scatter(trhlds, rec )
        precax.scatter(trhlds, prec )
        f1ax.scatter(trhlds, f1 )
        for j in range(len(trhlds)):
            if i==0:
                precax.text(trhlds[j], prec[j], f"{prec[j]:0.2f}")
            if i < len(mask2boolCs)//2:
                f1ax.text(trhlds[j], f1[j], f"{f1[j]:0.2f}")
                recax.text(trhlds[j], rec[j], f"{rec[j]:0.2f}")


    recax.set_xlabel('conf')
    recax.set_title('recall')
    precax.set_xlabel('conf')
    precax.set_ylim(0.5, 1.1)
    precax.set_title('precision')
    f1ax.set_xlabel('conf')
    f1ax.set_title('F1-score')
    recax.grid()
    precax.grid()
    f1ax.grid()
    fig.legend()
    plt.savefig(test_path/"metrics.png")



