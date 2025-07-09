import os
import shutil
root = "./From 106 smoke/water_flow"
for dir in os.listdir(root):
	if dir != "images":
		if "labels_v2.txt" not in os.listdir(os.path.join(root, dir)):
			print(dir)
			f = open(os.path.join(root, dir, "labels_v2.txt"), "w")
			f.close()
	

