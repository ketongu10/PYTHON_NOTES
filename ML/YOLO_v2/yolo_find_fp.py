import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import shutil

def create_shit_dir(name):
    try:
        os.mkdir(name+"/fp")
        os.mkdir(name + "/fn")
    except:
        shutil.rmtree(name+"/fp")
        shutil.rmtree(name + "/fn")
        os.mkdir(name + "/fp")
        os.mkdir(name + "/fn")

def parse_labels(file):
    classes = []
    boxes = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.split()
            classes.append(int(data[0]))
            boxes.append([float(j) for j in data[1:]])
    return {"classes":classes, "boxes":boxes}

def copy_with_bbox(source, detections, true_detections, output):
    thickness = 2   # Толщина линии
    img = cv2.imread(source)
    for class_name, (x, y, w, h), score in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
        text = f'{class_name}: {score:.2f}'
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for class_name, (x, y, w, h), score in true_detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        text = f'{class_name}: {score:.2f}'
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output, img)
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

dataset = "/vol1/KSH/DATASET_UMAR/test_dataset/for_training/val"
# Train the model with 2 GPUs


model = YOLO("/home/popovpe/.pyenv/runs/detect/train_w\pipe_100/weights/best.pt")

# img = cv2.imread("/vol1/KSH/DATASET_UMAR/test_dataset/for_training/train/images/25105.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img_source = ["/vol1/KSH/DATASET_UMAR/test_dataset/for_training/train/images/25105.jpg",
#               "/vol1/KSH/DATASET_UMAR/test_dataset/for_training/train/images/2.jpg",]
#               #"/vol1/KSH/DATASET_UMAR/test_dataset/for_training/train/images/25107.jpg"]
#
# preds = model.predict(img_source[0], imgsz=640, conf=0.25, iou=0.6, device="0")
# preds = preds[0]
# boxes = preds.boxes
# labels = (boxes.cls.cpu().numpy().astype(dtype=int))
# scores = boxes.conf.cpu().numpy()
# boxes = boxes.xywh.cpu().numpy()
#
# print(labels)

classes = ("ksh_short_kran", "ksh_knot")
IMG_SIZE = 1080
dataset = "/vol1/KSH/DATASET_OOPS/cropped_dataset/ksh/val" #"/vol1/KSH/DATASET_UMAR/test_dataset/for_training/val" #"/vol1/KSH/DATASET_OOPS/dataset/val"
output_root = "/vol1/KSH/DATASET_OOPS/cropped_dataset/ksh" #"/vol1/KSH/DATASET_UMAR/test_dataset/for_training/" #"/vol1/KSH/DATASET_OOPS/dataset/"
create_shit_dir(output_root)
fps, fns, total = 0, 0, 0


to_print = []
for file in os.listdir(dataset+"/images")[:]:
    with torch.no_grad():
        true_labels = parse_labels(dataset + "/labels/" + file.replace("jpg", "txt"))

        preds = model.predict(dataset+"/images/"+file, imgsz=640, conf=0.15, iou=0.6, device="0")
        preds = preds[0]
        H, W = preds.orig_shape
        boxes = preds.boxes
        labels = (boxes.cls.cpu().numpy().astype(dtype=int))
        scores = boxes.conf.cpu().numpy()
        boxes = boxes.xywh.cpu().numpy()

        # Convert detections to FiftyOne format
        detections = []
        for label, score, box in zip(labels, scores, boxes):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x, y, w, h = box
            # rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            x, y, = x - w / 2, y - h / 2
            x, y, w, h = np.round([x, y, w, h]).astype(int)

            detections.append(
                (
                    classes[label],
                    (x, y, w, h),
                    score
                )
            )
        true_detections = []
        for label, box in zip(true_labels["classes"], true_labels["boxes"]):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x, y, w, h = box
            x, y = x-w/2, y-h/2
            # rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            x, w, y, h = x*W, w*W, y*H, h*H
            x, y, w, h = np.round([x, y, w, h]).astype(int)

            true_detections.append(
                (
                    classes[label],
                    (x, y, w, h),
                    1.00,
                )
            )
        for label in labels:
            if label not in true_labels["classes"]:
                fps+=1
                copy_with_bbox(dataset+"/images/"+file, detections,true_detections,output_root+"/fp/"+file)
        for label in true_labels["classes"]:
            if label not in labels:
                fns+=1
                to_print.append([true_detections, W, H, true_labels["boxes"]])
                copy_with_bbox(dataset + "/images/" + file, detections, true_detections,output_root + "/fn/" + file)
        total+=1

print(f"fps: {fps} | fns: {fns} | total: {total}")
print(*to_print)