import cv2
import numpy as np


def crop_and_compress(img_name, new_size, new_img_name):
    img = cv2.imread(img_name)

    h, w, rgb = img.shape
    if h < w:
        size = h
    else:
        size = w

    y, x = int((h - size)/2), int((w - size)/2)
    h, w = size, size
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (new_size, new_size))
    cv2.imwrite(new_img_name, img)


def uncrop_and_compress(img_name, new_size, new_img_name):
    img = cv2.imread(img_name)

    h, w, rgb = img.shape
    size = w
    y = int((size-h)/2)
    emp1 = np.zeros(shape=(y, size, rgb), dtype=np.int8)
    emp2 = np.zeros(shape=(size-h-y, size, rgb), dtype=np.int8)
    h, w = size, size
    img = np.append(emp1, img, axis=0)
    img = np.append(img, emp2, axis=0)
    img = cv2.resize(img, (new_size, new_size))
    cv2.imwrite(new_img_name, img)

#uncrop_and_compress("0.jpg", 224, "new_0.jpg")