from enum import Enum
from pathlib import Path
import onnxruntime as rt
import numpy as np
import cv2
import os
import imageio as imio

def preprocess_image_for_PBIQA(img):
    img = img[28:28 + 1024, :, :3]

    img = np.float32(img)
    img = img / 255.
    img = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2])

    return img

def upscale_pbiqa_scores(pbiqa_scores):
    PBIQA_SCORE_DEFAULT = 63.0

    expanded_arr = np.repeat(np.repeat(pbiqa_scores, 128, axis=0), 128, axis=1)

    defaults = np.ones((28, 1920), dtype=np.float32) * PBIQA_SCORE_DEFAULT
    expanded_arr = np.vstack([defaults, expanded_arr, defaults])
    return expanded_arr

def create_pbiqa_vis(pbiqa_scores_raw, pbiqa_scores):
    pbiqa_scores = 10.2 * pbiqa_scores - 510
    pbiqa_scores = np.repeat(np.expand_dims(pbiqa_scores, axis=-1), 3, axis=-1).astype(np.uint8)

    ph, pw = pbiqa_scores_raw.shape
    PATCH_SIZE = 128

    for i in range(pw):
        for j in range(ph):
            patch_xs = PATCH_SIZE * i
            patch_ys = PATCH_SIZE * j + 14 # additional shift for center-crop-like positioning

            crop_score = pbiqa_scores_raw[j, i]
            # if(crop_score < 60.0):
            #     clr = (255, 0, 0)
            # else:
            #     clr = (0, 255, 0)

            cv2.putText(pbiqa_scores,
                        str(round(crop_score, 1)),
                        (patch_xs + int(PATCH_SIZE // 2) - 25, patch_ys + int(PATCH_SIZE // 2) + 30),
                        font,
                        0.8,
                        (int(-10.2 * crop_score + 765), int(10.2 * crop_score - 510), 0),
                        2,
                        lineType)

    return pbiqa_scores

font = cv2.FONT_HERSHEY_SIMPLEX
lineType = cv2.LINE_AA
IQA_ONNX_PATH = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/pbiqa/pbiqa_v1.onnx"
iqa_net = rt.InferenceSession(IQA_ONNX_PATH, providers=['CUDAExecutionProvider'])
depth_input_name = iqa_net.get_inputs()[0].name
depth_label_name = iqa_net.get_outputs()[0].name
iqa_net.get_provider_options()


images = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/pbiqa/raw"
kuda = "/home/popovpe/Projects/WaterTrain/owl.guard.cv/cyclops/training/tkrs/tools/pbiqa/results"

for img_ in os.listdir(images):
    img = cv2.imread(os.path.join(images, img_))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (1920, 1080))

    pbiqa_tensor = preprocess_image_for_PBIQA(img)
    pbiqa_scores_raw = iqa_net.run([depth_label_name], {depth_input_name: pbiqa_tensor})[0]
    pbiqa_scores = upscale_pbiqa_scores(pbiqa_scores_raw)
    pbiqa_vis = create_pbiqa_vis(pbiqa_scores_raw, pbiqa_scores)


    img = cv2.imwrite(os.path.join(kuda, img_),pbiqa_vis)


