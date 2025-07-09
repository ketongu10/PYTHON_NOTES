import cv2
import numpy as np
import imageio as imio

def preprocess_for_depth(img_p):
    img = imio.imread(img_p)[..., :3]
    H, W, _ = img.shape
    img = cv2.resize(img, DEPTH_INFERENCE_SIZE)
    img = img.astype(np.float32) / 255.
    target_tensor = np.transpose(np.expand_dims(img, 0), [0, 3, 1, 2])

    return target_tensor, H, W


def inference_depth(img_p):
    target_tensor, H, W = preprocess_for_depth(img_p)

    depth = depth_net.run([depth_label_name], {depth_input_name: target_tensor})[0]
    depth = depth[0, ...]
    depth = np.transpose(depth, [1, 2, 0])[..., 0]
    depth = cv2.resize(depth, (W, H), cv2.INTER_CUBIC)

    return depth