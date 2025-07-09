from pyexpat import features
from typing import Mapping, Any

import cv2
import pytorch_lightning as pl
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights, ConvNeXt, CNBlockConfig, LayerNorm2d


from segmentation_models_pytorch import Segformer
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from torchvision.models import resnet18, convnext_tiny
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
                 encoder_depth=4,
                 num_classes=1
                 ):
        super().__init__(
                encoder_name="efficientnet-b2", #(540x768 and 352x256)
                encoder_weights="imagenet",
                encoder_depth=encoder_depth,
                decoder_segmentation_channels=256,
                classes=num_classes,
                in_channels=9)

        del self.encoder
        del self.decoder
        self.encoder = Convnext4Segformer(depth=encoder_depth)
        if pretrained_weights:
            self.encoder.load_pretrained_weights(pretrained_weights)
        self.decoder = SegformerDecoder(
            encoder_channels=(9, 96, 192, 384, 768),
            encoder_depth=encoder_depth,
            segmentation_channels=256,
        )




class Convnext4Segformer(ConvNeXt, EncoderMixin):
    def __init__(self, out_channels=3, depth=5, **kwargs):
        # Tiny
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
        super().__init__(block_setting, stochastic_depth_prob=stochastic_depth_prob)
        self.features[0][0] = nn.Conv2d(9,
                                            self.features[0][0].out_channels,
                                            kernel_size=self.features[0][0].kernel_size,
                                            stride=self.features[0][0].stride,
                                            padding=self.features[0][0].padding)
        # self.features[0] = nn.Sequential(nn.Conv2d(9,48, kernel_size=(2,2), stride=(2,2)),
        #                                  nn.Conv2d(48,96, kernel_size=(2,2), stride=(2,2)),
        #                                  LayerNorm2d((96,), eps=1e-06, elementwise_affine=True))
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 9

        del self.classifier
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[0],
            # self.features[0][0],
            # nn.Sequential(self.features[0][1],self.features[0][2]),
            nn.Sequential(self.features[1], self.features[2]),
            nn.Sequential(self.features[3], self.features[4]),
            nn.Sequential(self.features[5], self.features[6],self.features[7]),

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
        self.load_state_dict(stdict)
        print(f"SUCCESFULLY LOADED CONVNEXT WEIGHTS FROM {path}")
        return self


    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        state_dict.pop('classifier.0.weight', None)
        state_dict.pop('classifier.0.bias', None)
        state_dict.pop('classifier.2.1.weight', None)
        state_dict.pop('classifier.2.1.bias', None)
        super().load_state_dict(state_dict, **kwargs)


class ShnekLighteningModule(pl.LightningModule):
    def __init__(self, validation_keys=None):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.net = convnext_tiny(weights=weights)
        self.net.features[0][0] = nn.Conv2d(9,
                                            self.net.features[0][0].out_channels,
                                            kernel_size=self.net.features[0][0].kernel_size,
                                            stride=self.net.features[0][0].stride,
                                            padding=self.net.features[0][0].padding)
        self.net.classifier[2] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 1),
        )
        self.sigmoid_for_forward = nn.Sigmoid()

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.acc_fn = Accuracy(task='binary')
        self.training_step_outputs = []
        self.partitions = validation_keys


if __name__=="__main__":
    val_dataset = OldVideosDataset(
        imgs_root_paths="/vol2/WATER/REAL_DATA_DATASET/NEW_TEST/test_23.05",
        partition='old_test',
        p_to_take=1.0,
    )
    torch.manual_seed(0)
    model = CustomEncoderSegformer().cuda(0)
    model.eval()
    imgs, mask = val_dataset[0]
    _3mask = torch.stack([mask, mask, mask])
    print(_3mask.shape)
    print(imgs.shape)
    imgs = imgs[None, ...]
    with torch.no_grad():
        new_mask = torch.sigmoid(model(imgs.cuda())).cpu()
        print(new_mask.shape)
        a = nn.BCELoss()
        print(a(new_mask, _3mask[None,...]))
