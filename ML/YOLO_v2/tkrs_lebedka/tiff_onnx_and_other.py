from enum import Enum
from pathlib import Path
import onnxruntime as rt
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os
from tifffile import imwrite


def preprocess_onnx(img):
    img = img[..., :3]
    new_img = np.ones(shape=(*INFERENCE_SIZE[::-1], 3), dtype=float)*128

    img = cv2.resize(img, (1088, 612), interpolation=cv2.INTER_AREA)
    new_img[14:-14, :, :] = img
    img = new_img
    img = np.float32(img)
    img = img / 255.
    img = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2])

    print(img.shape)
    return img


def inference_onnx(frame):
    target_tensor, H, W = preprocess_onnx(frame)

    depth = onnx.run([depth_label_name], {depth_input_name: target_tensor})[0]
    depth = depth[0, ...]
    depth = np.transpose(depth, [1, 2, 0])[..., 0]
    depth = cv2.resize(depth, (W, H), cv2.INTER_CUBIC)

    return depth

ONNX_PATH = "/home/popovpe/.pyenv/runs/detect/lebedka/rect_M_yolo_24.03.25_165k/last.onnx"
yolo = YOLO("/home/popovpe/.pyenv/runs/detect/lebedka/rect_M_yolo_24.03.25_165k/last.pt")
onnx = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
depth_input_name = onnx.get_inputs()[0].name
depth_label_name = onnx.get_outputs()[0].name
print(depth_input_name, depth_label_name)
onnx.get_provider_options()
INFERENCE_SIZE = (1088, 640)

I = "24.03.25"
path = f"/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tkrs_lebedka/out/{I}/img.jpg"
source_img = cv2.imread(path)
#source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
h, w, rgb = source_img.shape
tiff = preprocess_onnx(source_img)
tiff = np.transpose(tiff, (0, 2, 3, 1))
print(tiff.shape)
cv2.imwrite(path.replace("jpg", "tiff"), tiff[0], ) #[cv2.IMWRITE_TIFF_COMPRESSION, 16]) #, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
#imwrite(path.replace("jpg", "tiff"), tiff[0])

yolo_preds = yolo.predict(source_img, batch=1, device=0, conf=0.7, iou=0.7)[0]

clss = yolo_preds.boxes.cls.cpu().numpy().astype(dtype=int)
confs = yolo_preds.boxes.conf.cpu().numpy()
xywh = yolo_preds.boxes.xywh.cpu().numpy()
masks = yolo_preds.masks.data.cpu().numpy().astype(int).transpose(1, 2, 0)
H, W, m = masks.shape
for i in range(m):
    cv2.imwrite(path.replace("img.jpg", f"img_mask_{i}.png"), masks[:, :, i]*255)


beautiful = cv2.imwrite(path.replace("img.jpg", f"img_krasivoe.jpg"), yolo_preds.plot())
print("YOLO", clss, confs, xywh)
with open(path.replace("img.jpg", "output.txt"), 'w') as loh:
    print(f"Classes: {clss}", file=loh)
    print(f"Confs: {confs}", file=loh)
    print(f"Boxes in xywh: {xywh}", file=loh)

onnx_tensor = preprocess_onnx(source_img)
print(onnx_tensor.shape)
onnx_preds = onnx.run([depth_label_name], {depth_input_name: onnx_tensor})[0]
print("ONNX", np.argmax(onnx_preds, axis=2))
