import pytorch_lightning as pl
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import focal_loss
from pathlib import Path
from segmentation_models_pytorch import Segformer
from custom_encoder import CustomEncoderSegformer
from torchvision.models import mobilenet_v3_large, mobilenet_v2
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch import nn
from dataloader_seg import IMAGE_SIZE
import torch
import mlflow
from statistics import mean
from torchmetrics import Accuracy, F1Score, Recall, Precision
import numpy as np

from dataloader_seg import CLS_KEYS


class TpFp:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
BALANCE = 3.0

CONVNEXT_PATH = Path(__file__).parent.parent/'cnvnxt_ft_w_real/cnvnxt_5class_last.ckpt'


class SegmentationModel(pl.LightningModule):
    def __init__(self, validation_keys=None):
        super().__init__()

        self.model = CustomEncoderSegformer(num_classes=len(CLS_KEYS))


        self.criterion = (
            nn.BCELoss()
            #nn.BCEWithLogitsLoss(pos_weight=torch.tensor(BALANCE))
        )  # nn.BCELoss()  nn.CrossEntropyLoss() BCEWithLogitsLoss
        self.bin_threshold = 0.5
        self.iou_threshold = 0.3
        self.mask2boolC = IMAGE_SIZE[0]*IMAGE_SIZE[1]*0.001
        self.up = TpFp()
        self.down = TpFp()

    def training_step(self, train_batch, batch_idx):
        self.model.train()
        images, masks = train_batch
        outputs = self(images)[:, 0]



        loss = self.criterion(outputs, masks[:,0])

        self.log("train_loss", loss)
        return loss

    def to_numpy(self, tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def forward(self, x):

        return torch.sigmoid(self.model(x))


    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        images, masks = val_batch
        masks = masks[:, 0]                 #only for water
        outputs = self(images)[:, 0]        #only for water
        preds = outputs.detach() > self.bin_threshold
        loss = self.criterion(outputs, masks)
        iou, ious = self.calculate_iou(preds, masks, num_classes=1)
        xor, xors = self.calc_xor(preds, masks, num_classes=1)

        self.log("val_loss", loss)
        self.log("val_iou", iou)
        self.log("val_xor", xor)
        self.log("val_fitness", iou - loss)

        if dataloader_idx==1:
            self.calc_tpfp(preds, masks.detach() > self.bin_threshold, self.up)

        if dataloader_idx==2:
            self.calc_tpfp(preds, masks.detach() > self.bin_threshold, self.down)

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        pass
        #torch.distributed.barrier()  # синхронизация

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics["train_loss"]
        print(f"{self.trainer.current_epoch}: train_loss={loss}")
        mlflow.log_metric("train_loss", loss, self.current_epoch)

    def on_validation_epoch_end(self):
        print(self.trainer.callback_metrics.keys())
        loss = self.trainer.callback_metrics["val_loss/dataloader_idx_0"]
        iou = self.trainer.callback_metrics["val_iou/dataloader_idx_0"]
        xor = self.trainer.callback_metrics["val_xor/dataloader_idx_0"]
        fitness = self.trainer.callback_metrics["val_fitness/dataloader_idx_0"]
        print(f"{self.trainer.current_epoch}: val_loss={loss}, val_iou={iou}, val_fitness={fitness}")

        mlflow.log_metric("val_loss", loss, self.current_epoch)
        mlflow.log_metric("val_iou", iou, self.current_epoch)
        mlflow.log_metric("val_xor", xor, self.current_epoch)
        mlflow.log_metric("val_fitness", fitness, self.current_epoch)

        # UP
        prec = 1 if (self.up.tp+self.up.fp) == 0 else self.up.tp/(self.up.tp+self.up.fp)
        recc = 1 if (self.up.tp+self.up.fn) == 0 else self.up.tp/(self.up.tp+self.up.fn)
        f1 = 2*prec*recc/(1 if (prec+recc)==0 else (prec+recc))
        self.up.reset()

        Tloss = self.trainer.callback_metrics["val_loss/dataloader_idx_1"]
        mlflow.log_metric("test_loss", Tloss, self.current_epoch)
        mlflow.log_metric("test_prec", float(prec), self.current_epoch)
        mlflow.log_metric("test_rec", torch.tensor(recc), self.current_epoch)
        mlflow.log_metric("test_f1", f1, self.current_epoch)

        self.log("test_loss", Tloss)
        self.log("test_rec", recc)
        self.log("test_prec", prec)
        self.log("test_f1", f1)

        print(f"{self.trainer.current_epoch}: test_loss={Tloss}, test_rec={recc}, test_prec={prec}, test_f1={f1}")

        # DOWN
        prec = 1 if (self.down.tp + self.down.fp) == 0 else self.down.tp / (self.down.tp + self.down.fp)
        recc = 1 if (self.down.tp + self.down.fn) == 0 else self.down.tp / (self.down.tp + self.down.fn)
        f1 = 2 * prec * recc / (1 if (prec + recc) == 0 else (prec + recc))
        self.down.reset()

        Tloss = self.trainer.callback_metrics["val_loss/dataloader_idx_2"]
        mlflow.log_metric("test_d_loss", Tloss, self.current_epoch)
        mlflow.log_metric("test_d_prec", float(prec), self.current_epoch)
        mlflow.log_metric("test_d_rec", torch.tensor(recc), self.current_epoch)
        mlflow.log_metric("test_d_f1", f1, self.current_epoch)

        self.log("test_d_loss", Tloss)
        self.log("test_d_rec", recc)
        self.log("test_d_prec", prec)
        self.log("test_d_f1", f1)

        print(f"{self.trainer.current_epoch}: test_d_loss={Tloss}, test_d_rec={recc}, test_d_prec={prec}, test_d_f1={f1}")


    def calculate_iou(self, preds, targets, num_classes):
        ious = []
        preds = preds.flatten()
        targets = targets.flatten()
        for cls in range(num_classes):
            intersection = ((preds == cls) & (targets == cls)).sum().item()
            union = ((preds == cls) | (targets == cls)).sum().item()
            if union == 0:
                iou = float("nan")
            else:
                iou = intersection / union
            ious.append(iou)
        return np.nanmean(ious), np.array(ious)

    def calc_xor(self, preds, targets, num_classes):
        xors = []
        preds = preds.flatten()

        targets = targets.flatten()
        for cls in range(num_classes):
            intersection = ((preds == cls) & (targets == cls)).sum().item()
            union = ((preds == cls) | (targets == cls)).sum().item()
            if union == 0:
                xor = float("nan")
            else:
                xor = (union - intersection)/len(preds)
            xors.append(xor)
        return np.nanmean(xors), np.array(xors)


    def mask2bool(self, preds: torch.Tensor):
        mask_sums = preds.cpu().numpy().astype(float).sum(axis=(1,2))
        mask_bool = mask_sums > self.mask2boolC

        return mask_bool.astype(float)

    def calc_tpfp(self, preds, reals, where):
        bool_pred = self.mask2bool(preds)
        bool_reals = self.mask2bool(reals)
        tp = sum(bool_pred*bool_reals)
        fp = sum(bool_pred) - tp
        fn = sum(bool_reals) - tp
        tn = len(bool_pred) - tp-fp-fn

        where.tp+=tp
        where.fp+=fp
        where.fn+=fn
        where.tn+=tn

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-5, lr=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.2) #ExponentialLR(optimizer, gamma=0.912)
        return [optimizer], [lr_scheduler]


        # optimizer = torch.optim.Adam(self.parameters(), weight_decay=2e-06, lr=5e-06)  # 2e-06 #5e-06
        # return optimizer
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, threshold=0.001)
        # return [optimizer], [scheduler]


if __name__ == "__main__":
    # a = mobilenet_v3_large()
    b = mobilenet_v2()
    # print(a)
    print(b)
