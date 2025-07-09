import os
import shutil
import cv2



translate_dict_vstavka = {"5": "7", #ksh",
                          "6": "8",
                  "1":"1", #"kran",
                  "2": "2", #vstavka2",
                  "3": "3", #"pipev2
                          }

label_dir = "/vol2/KSH/NEW/KSH/DATASET_TEST/23.12_gates/from_29.07/labels/val"

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file), "r+") as r:
        lines = r.readlines()
        classes = []
        bboxs = []
        for line in lines:
            bboxs.append(line.split())
            bboxs[-1][0] = translate_dict_vstavka[bboxs[-1][0]]


        r.seek(0)
        r.truncate()
        for bbox in bboxs:
            r.write(' '.join(bbox))
            r.write('\n')