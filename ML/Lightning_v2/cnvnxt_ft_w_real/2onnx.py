from pathlib import Path

import cv2
import torch
import numpy as np
from tifffile import imwrite, imsave

from net import SegmentationModel
from dataloader_seg import IMAGE_SIZE, OldVideosDataset, AugmentationsPreset, GeneralDataset
import onnxruntime as ort

checkpoint_path = ("/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/experiments/segformer_w_real/outputs/segforme_w_rea+afp_2025-04-10_16-05-35/checkpoints/test_prec.ckpt")
input_sample = torch.randn((1, 9, *IMAGE_SIZE[::-1]))
model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
model.eval()
model.to_onnx(
    checkpoint_path.replace(".ckpt", ".onnx",),
    input_sample,
    input_names=["input"],
    output_names=["output"],
    export_params=True)
exit()

trans = AugmentationsPreset.identity_transform.value
val_dataset = GeneralDataset(
    imgs_root_paths=[
        "/vol1/WATER/DATASET/FOR_UNET/data/TRAIN_200",
                     ],
    transform=trans,
    partition='train',
    p_to_take=1.0,
    is_training=False
)
# val_dataset = OldVideosDataset(
#     imgs_root_paths="/vol2/WATER/REAL_DATA_DATASET/NEW_TEST/test_23.05",
#     transform=trans,
#     partition='old_test',
#     p_to_take=1.0,
#
# )
conf = 0.4
denc = 0.0023
kuda = "/!!!!home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/experiments/pretrained_segformer_w_kapli/outputs/for_cpp/down_test/"
Path(kuda).mkdir(parents=True, exist_ok=True)
idx = 150 #2900 #876
imgs, label, imgs448 = val_dataset[idx]

for i in range(val_dataset.stride):
    img = cv2.imread(val_dataset.img_paths[idx] + f"/{i}.jpg")
    cv2.imwrite(kuda+f"img{i}.jpg",img)

print("IMG_shape", imgs.shape)
# imwrite(kuda+"img_comp=None.tiff", imgs.numpy(), compression=None)
imwrite(kuda+"img_comp=1.tiff", imgs.numpy(), compression=1)


imgs = imgs[None, ...]

checkpoint_path = ("/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/experiments/segformer_w_real/outputs/segforme_w_rea+afp_2025-04-10_16-05-35/checkpoints/test_prec.ckpt")
model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
model.eval()
with torch.no_grad():
    PL_mask = model(imgs.cuda()).cpu().numpy()
    print("PL_shape",  PL_mask.shape)
    cv2.imwrite(kuda+"torch_mask.png", PL_mask[0][0]*255)
    print("PL_sum", (PL_mask[0][0] > 0.4).astype(float).sum() > 0.0012*576*960)

onnx_path = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/experiments/segformer_w_real/outputs/segforme_w_rea+afp_2025-04-10_16-05-35/checkpoints/segformer_w_real_w_afp_test_prec_11.04.25_for_up.onnx"
# onnx_path = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/liquid_seg/experiments/pretrained_segformer_w_kapli/outputs/for_cpp/liquid_seg_pasha_25.03.25.onnx"
onnxrt =  ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
depth_input_name = onnxrt.get_inputs()[0].name
depth_label_name = onnxrt.get_outputs()[0].name
print(depth_input_name, depth_label_name)

print(imgs448.shape)
ONNX_mask = onnxrt.run(None, {'input': imgs448[None, ...].numpy()})[0]
print("ONNX_sum", ONNX_mask)
print("ONNX_shape",  ONNX_mask.shape)
cv2.imwrite(kuda+"onnx_mask.png", ONNX_mask*255)
with open(kuda+"output.txt", 'w') as f:
    out = (ONNX_mask > conf).astype(int)
    print(f"sum after binarisation with threshold={conf:0.2f}:",np.sum(out), file=f)
    out_denc = np.sum(out)/IMAGE_SIZE[0]/IMAGE_SIZE[1]
    print(f"density: {out_denc} {'>' if out_denc > denc else '<'} denc={denc:0.4f} => result is `{int(out_denc > denc)}`" , file=f)



# cv2.imwrite(kuda+f"img{i}_sasat.jpg",img)
# img = cv2.imread(val_dataset.img_paths[idx] + f"/{i}.jpg")
# h, w, c = img.shape
# w = int(h*16/9)
# zero = np.ones(shape=(h, w, 3), dtype=np.float64) * 127
# zero[:, (w-h)//2: (w+h)//2, :] = img
# imwrite(kuda+"img0_tiffle_compression=0.tiff", img0.numpy(), compression=None)
# imwrite(kuda+"img0_tiffle_compression=1.tiff", img0.numpy(), compression=1)

# img0 = cv2.cvtColor(img0.numpy(), cv2.COLOR_BGR2RGB)
# cv2.imwrite(kuda+"img0_cv2.tiff", img0.numpy())
# cv2.imwrite(kuda+"img0_cv2_compression=1.tiff", img0.numpy(), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
#print("AAA", first_img.shape)
# cv2.imwrite(kuda+"img0_-1.tiff", img0.numpy()[...,::-1], [cv2.IMWRITE_TIFF_COMPRESSION, 1])
# cv2.imwrite(kuda+"imgs_-1.tiff", first_img.numpy()[...,::-1], [cv2.IMWRITE_TIFF_COMPRESSION, 1])