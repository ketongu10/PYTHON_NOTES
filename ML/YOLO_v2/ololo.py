import numpy as np
import cv2


checkpoint_path = ("/home/popovpe/Projects/WaterTrain/owl.guard.cv/"
                   "cyclops/training/liquid_seg/outputs/run/"
                   "supapupatest_2_2025-02-25_16-03-42/checkpoints/last-v1.ckpt")
trans = AugmentationsPreset.identity_transform.value
model = ShnekLighteningModule.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
model.eval()
