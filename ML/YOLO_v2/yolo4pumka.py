from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)

dataset = "/vol1/KSH/READY/pumka/data.yaml"
config = "yolo-config_pumka.yaml"
# Train the model with 2 GPUs
results = model.train(cfg=config, data=dataset, name='Mseg_pumka_77k_right_heads', batch=16, workers=6)#epochs=20, imgsz=640, device=[0], workers=8, batch=64)