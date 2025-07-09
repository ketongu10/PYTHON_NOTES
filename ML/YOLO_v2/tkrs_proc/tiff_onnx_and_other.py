from enum import Enum
from pathlib import Path
import onnxruntime as rt
import numpy as np
import torch
from ultralytics import YOLO, RTDETR
import cv2
import os
from tifffile import imwrite


def preprocess_onnx(img):
    img = img[..., :3]
    new_img = np.ones(shape=(*INFERENCE_SIZE[::1], 3), dtype=float)*128

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    new_img[24:-24, :, :] = img
    img = new_img
    img = np.float32(img)
    img = img / 255.
    img = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2])

    print(img.shape)
    return img


# def inference_onnx(frame):
#     target_tensor, H, W = preprocess_onnx(frame)
#
#     depth = onnx.run([depth_label_name], {depth_input_name: target_tensor})[0]
#     depth = depth[0, ...]
#     depth = np.transpose(depth, [1, 2, 0])[..., 0]
#     depth = cv2.resize(depth, (W, H), cv2.INTER_LINEAR)
#
#     return depth

INFERENCE_SIZE = (768, 1280)
ONNX_PATH = "/home/popovpe/.pyenv/runs/detect/proc/big_rect_2ecn_f_tros_w_gates_proc_M_yolo_21.04_200k2/weights/proc_rect_ff_24.04.25.onnx"
yolo = YOLO("/home/popovpe/.pyenv/runs/detect/proc/big_rect_2ecn_f_tros_w_gates_proc_M_yolo_21.04_200k2/weights/last.pt")

I = "KOPS"
path = f"/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tkrs_proc/out_ff/{I}/img.jpg"
Path(path).parent.mkdir(exist_ok=True, parents=True)

source_img = cv2.imread(path)
#source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
h, w, rgb = source_img.shape


preproc_img_bgr = preprocess_onnx(source_img)
tiff = np.transpose(preproc_img_bgr, (0, 2, 3, 1))
cv2.imwrite(path.replace(".jpg", "_rgb.tiff"), tiff[0], [cv2.IMWRITE_TIFF_COMPRESSION, 1])
source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
preproc_img_rgb = preprocess_onnx(source_img_rgb)
tiff = np.transpose(preproc_img_rgb, (0, 2, 3, 1))
cv2.imwrite(path.replace(".jpg", "_bgr.tiff"), tiff[0], [cv2.IMWRITE_TIFF_COMPRESSION, 1])
print(tiff.shape)
print(f"diff check={(source_img-source_img_rgb).sum()}")


# YOLO RUN ON BGR  - YOLO CHANGES IT TO RGB
yolo_img = np.ones(shape=(768, 1280, 3), dtype=float)*128
source_img_yolo = cv2.resize(source_img, (1280, 720), interpolation=cv2.INTER_LINEAR)
yolo_img[24:-24, :, :] = source_img_yolo
yolo_preds = yolo.predict(yolo_img, batch=1, conf=0.6)[0]
# print("YOLO", yolo_preds)
clss = yolo_preds.boxes.cls.cpu().numpy().astype(dtype=int)
confs = yolo_preds.boxes.conf.cpu().numpy()
xywh = yolo_preds.boxes.xywh.cpu().numpy()
masks = yolo_preds.masks.data.cpu().numpy().astype(int).transpose(1, 2, 0)

H, W, m = masks.shape
print("mask_shape", masks.shape)
with open(path.replace("img.jpg", "output_fullhd.txt"), 'w') as loh:
    for i in range(m):
        print(masks[24:-24, :, i].shape)
        msk_ = masks[24:-24, :, i].astype(np.uint8)*255
        ff_mask = cv2.resize(msk_, (1920,1080))
        # inds_y, inds_x = np.where(msk_ > 0)
        # x_min, x_max = min(inds_x), max(inds_x)
        # y_min, y_max = min(inds_y), max(inds_y)
        # x, y, w, h = x_min*1.5, y_min*1.5, (x_max - x_min)*1.5, (y_max - y_min)*1.5
        # print(clss[i], x, y, w, h, file=loh)
        cv2.imwrite(path.replace("img.jpg", f"img_mask_{i}.jpg"), ff_mask) #cv2.resize(masks[24:-24, :, i]*255, (1920, 1080))


beautiful = cv2.imwrite(path.replace("img.jpg", f"img_krasivoe.jpg"), yolo_preds.plot())
print("YOLO", clss, confs, xywh)
with open(path.replace("img.jpg", "output.txt"), 'w') as loh:
    print(f"Classes: {clss}", file=loh)
    print(f"Confs: {confs}", file=loh)
    # print(f"Boxes in xywh_1280: {xywh}", file=loh)
    for xywh_ in xywh:
        print(f"AAA, {xywh_}")
        x, y, w, h = xywh_
        x, y, w, h = 1.5*(x-w/2), 1.5*(y-24-h/2), 1.5*w, 1.5*h


        print(f"Boxes in xywh: {x, y, w, h}", file=loh)


# # RUN ONNX
# onnx = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
# depth_input_name = onnx.get_inputs()[0].name
# depth_label_name = onnx.get_outputs()[0].name
# print(depth_input_name, depth_label_name)
# onnx.get_provider_options()
#
#
#
# onnx_preds = onnx.run([depth_label_name], {depth_input_name: preproc_img_rgb})[0]
# print("ONNX", onnx_preds.shape)
# with torch.no_grad():
#     ret = yolo.model(torch.from_numpy(preproc_img_rgb).cuda(0))[0]
#     print(ret.cpu().shape)
