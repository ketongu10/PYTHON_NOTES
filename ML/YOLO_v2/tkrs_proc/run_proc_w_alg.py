from enum import Enum
from pathlib import Path
import onnxruntime as rt
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os

from tkrs.tkrs_proc.validator import Validator
from tkrs.tools.image_cmp.cmp import avg_mask

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB standard deviation

def normalize(image, mean = np.array(IMAGENET_MEAN), std = np.array(IMAGENET_STD)):
    image = (image - mean) / std
    return image


skip_num = 5


# def hsv_err(a, b):
#     return np.sqrt(
#         ((a[..., 1] * np.sin(a[..., 0] / 180 * np.pi * 2) - b[..., 1] * np.sin(
#             b[..., 0] / 180 * np.pi * 2)) / 255) ** 2 +
#         ((a[..., 1] * np.cos(a[..., 0] / 180 * np.pi * 2) - b[..., 1] * np.cos(
#             b[..., 0] / 180 * np.pi * 2)) / 255) ** 2 +
#         ((a[..., 2] - b[..., 2]) / 255) ** 2)


def hsv_err(a, b):

    return np.sqrt(
        ((a[...,1]*a[...,2]*np.sin(a[...,0]/180*np.pi*2) - b[...,1]*b[...,2]*np.sin(b[...,0]/180*np.pi*2))/255/255)**2 +
        ((a[...,1]*a[...,2]*np.cos(a[...,0]/180*np.pi*2)- b[...,1]*b[...,2]*np.cos(b[...,0]/180*np.pi*2))/255/255)**2 +
        ((a[...,2]-b[...,2])/255)**2)

# def preprocess_for_depth(frame):
#
#     img = frame.copy()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     H, W, _ = img.shape
#     img = cv2.resize(img, DEPTH_INFERENCE_SIZE)
#     img = img.astype(np.float32) / 255.
#     target_tensor = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2])
#
#     return target_tensor, H, W

def preprocess_for_depth(img, inference_size, norm=True): # use norm True for new net
    img = img[..., :3]
    H, W, _ = img.shape
    img = cv2.resize(img, inference_size)
    img = img.astype(np.float32) / 255.
    if(norm):
        img = normalize(img)
    target_tensor = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2]).astype(np.float32)

    return target_tensor, H, W


def inference_depth(frame):
    target_tensor, H, W = preprocess_for_depth(frame, DEPTH_INFERENCE_SIZE)

    depth = depth_net.run([depth_label_name], {depth_input_name: target_tensor})[0]
    depth = depth[0, ...]
    depth = np.transpose(depth, [1, 2, 0])[..., 0]
    depth = cv2.resize(depth, (W, H), cv2.INTER_CUBIC)

    return depth

def postprocess_for_depth(depth, original_size):
    depth = depth[0]
    depth = np.transpose(depth, [1, 2, 0])

    depth = cv2.resize(depth, original_size, cv2.INTER_CUBIC)

    depth = np.uint8(np.clip(1 - depth, 0, 1) * 255)
    depth = np.dstack([depth, depth, depth])

    return depth
# Loop through the video frames
def record_video(path, out_vidos):
    Validator.start()
    root, name = path
    vidos = os.path.join(root, name)
    cap = cv2.VideoCapture(vidos)
    ret, frame0 = cap.read()
    h, w, rgb = frame0.shape

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps




    SIZE = 640 #min(w, h)
    obt_imgs = []
    good_imgs = []
    avg_mask = np.zeros(shape=(640, 640), dtype=int)


    num = 0
    while True:
        try:
            for i in range(skip_num):
                success, frame = cap.read()
            cropped_frame = frame[:, (w - h) // 2:(w + h) // 2, :]

            # YOLO RUN
            results = model(cropped_frame, conf=0.25)
            preds = results[0]
            clss = preds.boxes.cls.cpu().numpy().astype(dtype=int)
            confs = preds.boxes.conf.cpu().numpy()
            xywh = preds.boxes.xywh.cpu().numpy()




            # DEPTH RUN
            depth = inference_depth(frame)
            depth = np.uint8(np.clip(1 - depth, 0, 1) * 255)
            h, w = depth.shape
            depth = depth[:, (w - h) // 2:(w + h) // 2]
            depth = cv2.resize(depth, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)

            for i, cls in enumerate(clss):
                if cls == 3:
                    mask = preds.masks.data.cpu().numpy().astype(int).transpose(1, 2, 0)
                    mask *= 255
                    avg_mask += mask[:, :, i]

                    cropped_frame = cv2.resize(cropped_frame, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
                    good_imgs.append((cropped_frame, depth))

            obt_imgs.append((cropped_frame, depth, preds))

        except Exception as e:
            print(e)
            cap.release()
            break

    if good_imgs:
        avg_mask//=len(good_imgs)


    # AVERAGE CALCULATINGS
    inds = np.where(avg_mask > 127)
    avg_depth = np.zeros(shape=(640, 640), dtype=int)
    avg_img = np.zeros(shape=(640, 640, 3), dtype=int)
    for cropped_frame, depth in good_imgs:
        avg_depth[inds] += depth[inds]
        avg_img[inds] += cropped_frame[inds]

    if good_imgs:
        avg_depth //= len(good_imgs)
        avg_img //= len(good_imgs)


    try:
        avg_img_hsv = cv2.cvtColor(avg_img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        # WRITE RESULTS
        output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
        out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration/skip_num, (SIZE * 3, SIZE*2))
        for frame, depth, preds in obt_imgs:
            hsv_dif = np.zeros(shape=(640, 640), dtype=np.uint8)
            cur_img = np.zeros(shape=(640, 640, 3), dtype=int)
            frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ret = hsv_err(frame_[inds].astype(float), avg_img_hsv[inds].astype(float))
            hsv_dif[inds] = ret*300
            hsv_inds = np.where(hsv_dif > 127)
            hsv_dif[:] = 0
            hsv_dif[hsv_inds] = 255
            hsv_dif = cv2.blur(hsv_dif, (5, 5))
            #cur_img[inds] = frame[inds]

            depth_dif = np.zeros(shape=(640, 640), dtype=np.uint8)
            depth_dif[inds] = np.uint8(np.clip(depth[inds] - avg_depth.astype(np.int16)[inds], 0, 255))
            tr_inds = np.where(depth_dif > 25)
            depth_dif[:] = 0
            depth_dif[tr_inds] = 255
            depth_dif = cv2.blur(depth_dif, (5, 5))
            #score = (depth_dif[inds]).sum()/len(inds)/1000000

            Validator.update(preds)
            annotated_frame = preds.plot()
            cropped_annotated_frame = cv2.resize(annotated_frame, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
            frame_ = Validator.render(cropped_annotated_frame)

            final_img = np.zeros(shape=(SIZE*2, SIZE * 3, 3), dtype=np.uint8)
            #print(np.max(avg_mask), avg_mask)
            final_img[:SIZE, :SIZE, :] = depth_dif.reshape((SIZE, SIZE, 1))
            final_img[:SIZE, SIZE:2*SIZE, :] = avg_depth.reshape((SIZE, SIZE, 1))
            final_img[:SIZE, 2*SIZE:, :] = depth.reshape((SIZE, SIZE, 1))

            final_img[SIZE:, :SIZE, :] = hsv_dif.reshape((SIZE, SIZE, 1))
            final_img[SIZE:, SIZE:2*SIZE, :] = avg_img
            final_img[SIZE:, 2*SIZE:, :] = frame_
            out.write(final_img)
        out.release()
    except:
        pass


model = YOLO("/home/popovpe/.pyenv/runs/detect/proc/M_seg_9.12/M_seg170k_9.12.pt")
DEPTH_ONNX_PATH = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/depth/depth_v2.onnx"
depth_net = rt.InferenceSession(DEPTH_ONNX_PATH, providers=['CUDAExecutionProvider'])
depth_input_name = depth_net.get_inputs()[0].name
depth_label_name = depth_net.get_outputs()[0].name
depth_net.get_provider_options()
DEPTH_INFERENCE_SIZE = (924, 518)#(686, 392)

dataset_folder = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/covered_kops"
#dataset_folder = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/kops_154"
out_vidos = "/vol2/KSH/NEW/KSH/RUN_NXTCLD/KOPS_COVER_V2"
rootfile = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))
for root, file in rootfile[:]:
    record_video((root, file), out_vidos)