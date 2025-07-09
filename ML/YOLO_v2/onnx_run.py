import onnxruntime as ort
import cv2
import torch
import numpy as np
import os
import shutil



classes = ("ksh_short_kran", "ksh_knot")
IMG_SIZE = 1080
dataset = "/vol1/KSH/DATASET_OOPS/cropped_dataset/ksh/val" #"/vol1/KSH/DATASET_UMAR/test_dataset/for_training/val" #"/vol1/KSH/DATASET_OOPS/dataset/val"
output_root = "/vol1/KSH/DATASET_OOPS/cropped_dataset/ksh" #"/vol1/KSH/DATASET_UMAR/test_dataset/for_training/" #"/vol1/KSH/DATASET_OOPS/dataset/"

imgs = []
labels = []
for file in os.listdir(dataset+"/images")[:]:

    true_labels = parse_labels(dataset + "/labels/" + file.replace("jpg", "txt"))

    preds = model.predict(dataset+"/images/"+file, imgsz=640, conf=0.15, iou=0.6, device="0")

x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')



