from multiprocessing import Process, Pool
import cv2
import numpy as np
from tifffile import imwrite
import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import os





class Hysteresis:
    def __init__(self):
        self.buffer = []
    def __call__(self, res):
        if len(self.buffer) >= 25:
            self.buffer.pop(0)
        self.buffer.append(res)
    def check(self) -> bool:
        sum = 0
        for ind in range(len(self.buffer)-1, -1, -1):
            sum+=1 if self.buffer[ind] else 0
            if sum >= 13:
                return True
        return False


def write_text(img, text, color=(255, 0, 0), pos=(50, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = pos
    fontScale = 1
    color1 = color
    thickness = 2
    image = cv2.putText(img, text, org, font,
                        fontScale, color1, thickness, cv2.LINE_AA)
    return image





skip_num = 5


# Loop through the video frames
def record_video(path, out_vidos, model):
    root, name = path
    output_name = name if '.3gp' not in name else name.replace(".3gp", ".mp4")
    vidos = os.path.join(root, name)
    cap = cv2.VideoCapture(vidos)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    ret, frame0 = cap.read()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w, rgb = frame0.shape
    #out = cv2.VideoWriter(os.path.join(out_vidos, name), cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
    out = cv2.VideoWriter(os.path.join(out_vidos, output_name), cv2.VideoWriter_fourcc(*'MJPG'), duration/skip_num, (min(w, h), min(w, h)))
    found = False
    while True:
        #print("WORKING")
        try:
            # Read a frame from the video
            for i in range(skip_num):
                success, frame = cap.read()

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.predict(frame[:, (w-h)//2:(w+h)//2, :],  conf=0.25) #persist=False,
            # labels = (results[0].boxes.cls.cpu().numpy().astype(dtype=int))
            # if labels:
            #     found = True
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            print(annotated_frame.shape)
            # Display the annotated frame
            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            out.write(annotated_frame)
            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break

        except Exception as e:
            print(e)
            out.release()
            cap.release()
            # if found:
            #     os.rename(os.path.join(out_vidos, output_name), os.path.join(out_vidos, "FOUND_" + output_name))
            break


def f(given):

    #model = YOLO("/home/popovpe/.pyenv/runs/detect/22.07/M_pasha_17-29.07+lesha+vst_syn/best.pt")
    root, file = given
    #for root, file in given:
    record_video((root, file), kuda_narisovat, model)

model = YOLO("/home/popovpe/.pyenv/runs/detect/22.07/M_yolo_gates_27.12/last.pt")

# model.export(
#     format="onnx"
# )
# exit()
#dohuya_videos = "/vol2/WATER/NEXTCLOUD/v2/izliv" #old waters
#dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/source" #old ksh
#dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ksh_407" #new kshs
#dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/ecn_2k/best"
dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_PROD/vstavka_fp_jestko"
#dohuya_videos = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/vstavkas_from_any_source"
kuda_narisovat = "/vol2/KSH/NEW/KSH/RUN_NXTCLD/RUN_27.12/base"
rootfile = []
for root, dirs, files in os.walk(dohuya_videos):
    for file in files:
        if (".mp4" in file or ".avi" in file or ".mkv" in file or ".3gp" in file):
            rootfile.append((root, file))
N = len(rootfile)

#rootfile = [('/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/source', '23253959683957.3gp')]

if __name__ == "__main__":


    #f(rootfile)

    with Pool(6) as p:
        p.map(f, rootfile[:1000])

    # P1 = Process(target=f, args=(rootfile[:N//3],))
    # P1.start()
    # print("P1 started!!")
    # P2 = Process(target=f, args=(rootfile[N //3:N //3*2],))
    # P2.start()
    # print("P2 started!!")
    # P3 = Process(target=f, args=(rootfile[N // 3*2:],))
    # P3.start()
    # print("P3 started!!")
    # P1.join()
    # P2.join()
    # P3.join()



