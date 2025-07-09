import os
import shutil
from os.path import exists


root ='/vol1/KSH/dataset'
folders = [

# "PC PUMKA 7.07.25",
# "106 PUMKA 7.07.25",
"PC PUMKA 23.06.25",
"104 PUMKA 23.06.25",
# '116 KSH_gates 26.12',
# '114 KSH_gates 26.12',
# "105 arms 12.12",
# "UMAR KSH 12.12"
# "105 KSH_relabel 10.12",
# "UMAR KSH_relabel 10.12",
#"105 KSH_relabel 9.12",
# "PC KSH_relabel 9.12",
# "UMAR KSH_relabel 9.12"
# "105 KSH_arms 6.12",
# "PC KSH_arms 6.12",
# "PC PROC_ecn 5.12"
#"106 DEPTH 2.12",
# "PC DEPTH 2.12",
# "UMAR DEPTH 2.12",
# "UMAR PROC 28.11",
#"106 PROC 28.11",
# "PC PROC 28.11",

#"106 PROC 22.11_gis",
#"PC PROC 22.11_gis"
#"PC PROC 18.11",
#"106 PROC 16.11",
#"106 PROC 18.11",
# "PC KSH_redhands 16.10-0",
# "PC KSH_redhands 16.10-1",

#"PC KSH 14.10",
# "104 KSH 14.10-0",
# "104 KSH 14.10-1",
# "105 KSH 14.10-0",
# "105 KSH 14.10-1",
# "106 KSH 14.10-0",
# "106 KSH 14.10-1",
# "106 KSH 14.10-2",
# "106 KSH 14.10-3",#


#"104 KSH 10.10-0",
# "104 KSH 10.10-1",
# "105 KSH 10.10-0",
# "105 KSH 10.10-1",
# "106 KSH 10.10-0",
# "106 KSH 10.10-1",

# "104 KSH_hands 5.10-0",
# "104 KSH_hands 5.10-1",
# "104 KSH_hands 7.10-0",
# "104 KSH_hands 7.10-1",
# "105 KSH_hands 5.10-0",
# "105 KSH_hands 5.10-1",
# "105 KSH_hands 7.10-0",
# "105 KSH_hands 7.10-1",
# "106 KSH 7.10-0",
# "106 KSH 7.10-1",
]
merged_folder = '/vol1/KSH/dataset/syn_proc_rect_pumka_27.06.25' #pasha_syn_2.09'
for fld in folders:
    print(f'merging {fld} folder')
    imgs_source = os.path.join(root, fld, 'images', 'train')
    for smth in os.listdir(imgs_source):
        shutil.move(os.path.join(imgs_source, smth), os.path.join(merged_folder, 'images', 'train')) #, dirs_exist_ok=True)
    imgs_source = os.path.join(root, fld, 'images', 'val')
    for smth in os.listdir(imgs_source):
        shutil.move(os.path.join(imgs_source, smth), os.path.join(merged_folder, 'images', 'val')) #, dirs_exist_ok=True)

    lbls_source = os.path.join(root, fld, 'labels', 'train')
    for smth in os.listdir(lbls_source):
        shutil.move(os.path.join(lbls_source, smth), os.path.join(merged_folder, 'labels', 'train'))  # , dirs_exist_ok=True)
    lbls_source = os.path.join(root, fld, 'labels', 'val')
    for smth in os.listdir(lbls_source):
        shutil.move(os.path.join(lbls_source, smth), os.path.join(merged_folder, 'labels', 'val'))

    lbls_source = os.path.join(root, fld, 'labels_bbox', 'train')
    for smth in os.listdir(lbls_source):
        shutil.move(os.path.join(lbls_source, smth), os.path.join(merged_folder, 'labels_bbox', 'train'))
    lbls_source = os.path.join(root, fld, 'labels_bbox', 'val')
    for smth in os.listdir(lbls_source):
        shutil.move(os.path.join(lbls_source, smth), os.path.join(merged_folder, 'labels_bbox', 'val'))

