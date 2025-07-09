from pyexpat import features
from typing import Mapping, Any

import cv2
import pytorch_lightning as pl
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights, ConvNeXt, CNBlockConfig, LayerNorm2d


from segmentation_models_pytorch import Segformer
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from torchvision.models import resnet18, convnext_tiny, mobilenet_v3_large
from torch import nn
from dataloader_seg import IMAGE_SIZE, OldVideosDataset, GeneralDataset, AugmentationsPreset
import torch
import mlflow
from statistics import mean
from torchmetrics import Accuracy, F1Score, Recall, Precision
import numpy as np


class CustomEncoderSegformer(Segformer):
    def __init__(self,
                 pretrained_weights: str = None,
                 encoder_depth=5,
                 num_classes=3
                 ):
        super().__init__(
                encoder_name="timm-mobilenetv3_large_100",
                encoder_weights="imagenet",
                encoder_depth=encoder_depth,
                decoder_segmentation_channels=256,
                classes=num_classes,
                in_channels=9)

        del self.encoder
        self.encoder = MobilenetV3Large4Segformer(depth=encoder_depth)
        if pretrained_weights:
            self.encoder.load_pretrained_weights(pretrained_weights)





class MobilenetV3Large4Segformer(nn.Module, EncoderMixin):
    def __init__(self, out_channels=3, depth=5, **kwargs):
        # Tiny

        super().__init__()
        self.features = mobilenet_v3_large().features
        self.features[0][0] = nn.Conv2d(9,
                                            self.features[0][0].out_channels,
                                            kernel_size=self.features[0][0].kernel_size,
                                            stride=self.features[0][0].stride,
                                            padding=self.features[0][0].padding,
                                            bias=False)

        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 9


    def get_stages(self):
        return [
            nn.Identity(),

            nn.Sequential(self.features[0], self.features[1]),
            nn.Sequential(self.features[2], self.features[3]),
            nn.Sequential(self.features[4], self.features[5], self.features[6]),
            nn.Sequential(self.features[7], self.features[8], self.features[9],
                          self.features[10], self.features[11], self.features[12]),
            nn.Sequential(self.features[13], self.features[14],
                          self.features[15], self.features[16]),

        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_pretrained_weights(self, path: str):
        pretrained = ShnekLighteningModule.load_from_checkpoint(path)
        stdict = pretrained.net.state_dict()
        #print(stdict.keys())
        self.load_state_dict(stdict)
        print(f"SUCCESFULLY LOADED MOBILENET WEIGHTS FROM {path}")
        return self


    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        state_dict.pop('classifier.0.weight', None)
        state_dict.pop('classifier.0.bias', None)
        state_dict.pop('classifier.3.weight', None)
        state_dict.pop('classifier.3.bias', None)
        super().load_state_dict(state_dict, **kwargs)


class ShnekLighteningModule(pl.LightningModule):
    def __init__(self, validation_keys=None):
        super().__init__()
        self.net = mobilenet_v3_large()

        self.net.features[0][0] = nn.Conv2d(9,
                                            self.net.features[0][0].out_channels,
                                            kernel_size=self.net.features[0][0].kernel_size,
                                            stride=self.net.features[0][0].stride,
                                            padding=self.net.features[0][0].padding,
                                            bias=False)
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=1, bias=True)
        )

        self.sigmoid_for_forward = nn.Sigmoid()

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.acc_fn = Accuracy(task='binary')
        self.training_step_outputs = []
        self.partitions = validation_keys




if __name__=="__main__":
    torch.manual_seed(0)
    PATH = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/many_class_experiments/mobile/mobilenet_v3l_pretrained.ckpt"
    model = CustomEncoderSegformer(PATH).cuda(0)

    # model = MobilenetV3Large4Segformer()
    # model.load_pretrained_weights(PATH)

    # torch.manual_seed(0)
    val_dataset = OldVideosDataset(
        imgs_root_paths="/vol2/WATER/REAL_DATA_DATASET/NEW_TEST/test_23.05",
        partition='old_test',
        p_to_take=1.0,
    )
    model.eval()
    imgs, mask = val_dataset[0]
    print(mask.shape)
    print(imgs.shape)
    imgs = imgs[None, ...]
    with torch.no_grad():
        new_mask = torch.sigmoid(model(imgs.cuda())).cpu()
        print(new_mask.shape)
        a = nn.BCELoss()
        print(a(new_mask, mask[None,...]))
