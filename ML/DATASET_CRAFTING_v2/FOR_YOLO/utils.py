import torch
import numpy as np
from PIL import Image


def xywhn2xyxy(x, w, h, padw=0, padh=0):
    assert x.shape[-1] == 4, f'input dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xywhn2xywh(x, w, h, padw=0, padh=0):
    y = []

    y.append(x[0] * w)
    y.append(x[1] * h)
    y.append(x[2] * w)
    y.append(x[3] * h)

    return y


def calculate_iou(bbox1, bbox2):
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    x1_box1, y1_box1, x2_box1, y2_box1 = bbox1
    x1_box2, y1_box2, x2_box2, y2_box2 = bbox2

    area_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    area_box2 = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    x_left = max(x1_box1, x1_box2)
    y_top = max(y1_box1, y1_box2)
    x_right = min(x2_box1, x2_box2)
    y_bottom = min(y2_box1, y2_box2)

    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    iou = intersection_area / (area_box1 + area_box2 - intersection_area)

    return round(iou, 2)


def concatenate_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    new_width = width1 + width2
    new_height = max(height1, height2)

    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    return new_image


def choose_color(cls: int):
    cls = int(cls)
    if cls == 0:
        color = (255, 0, 0)
    elif cls == 1:
        color = (0, 255, 0)
    elif cls == 2:
        color = (255, 0, 0)
    elif cls == 3:
        color = (0, 255, 0)
    else:
        color = (0, 0, 0)

    return color