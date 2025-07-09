import os
import shutil
import cv2


translate_dict_ksh = {"0": "0", #ksh_short_kran
                  "1":"1", #knot
                  "2": "100", #vstavka_pipe
                  "3": "100", #vstavka_2
                    "4": "100"
                  }
translate_dict_vstavka = {"0": "1", #ksh_short_kran
                  "1":"0", #knot
                  "2": "3", #pipe1end if vstavka
                  "3": "2", #vstavka
                 "4": "100" #pipe1end if ksh
                  }

def create_shit_dir(name):
    try:
        os.mkdir(name+"/images")
        os.mkdir(name + "/labels")
    except:
        shutil.rmtree(name+"/images")
        shutil.rmtree(name + "/labels")
        os.mkdir(name + "/images")
        os.mkdir(name + "/labels")

old_dataset = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/nxtcld_700ksh_labeled" #"/vol1/KSH/DATASET_OOPS/source/ksh_7-06/obj_train_data/images_ksh" #"/vol1/KSH/DATASET_UMAR/source/ksh/short_kran" #
new_dataset = "/vol2/KSH/NEW/KSH/DATASET_NXTCLD/700_kshs/labeled" #"/vol1/KSH/DATASET_OOPS/dataset/ksh" #"/vol1/KSH/DATASET_UMAR/test_dataset/ksh" #
mask = "nxtcld_700ksh"
create_shit_dir(new_dataset)

jpgs = 0
txts = 0
for root, dirs, files in os.walk(old_dataset):
    for file in files:
        if ".jpg" in file:
            txt = file.replace(".jpg", ".txt")
            bboxs = None
            try:
                with open(os.path.join(root,txt), 'r') as f:
                    bboxs = []
                    for line in f.readlines():
                        bboxs.append(line.split())
                        bboxs[-1][0] = translate_dict_vstavka[bboxs[-1][0]]
            except Exception as e:
                print(os.path.join(root,txt))
                print(e)
            if bboxs is not None:
                with open(os.path.join(new_dataset, 'labels',str(jpgs)+f"_{mask}.txt"), 'w') as f:
                    for bbox in bboxs:
                        if bbox[0] != "100":
                            print(" ".join(bbox), file=f)
                shutil.copy(os.path.join(root,file), os.path.join(new_dataset,'images',str(jpgs)+f"_{mask}.jpg"))
                jpgs+=1

print(f"TOTAL jpgs: {jpgs}")