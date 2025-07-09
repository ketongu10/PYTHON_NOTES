import os
import shutil


for ttv in os.listdir("./DOWN"):
	for d in os.listdir("./DOWN/" +ttv):
		shutil.move(os.path.join("./DOWN", ttv, d), os.path.join("./UP", ttv, d+"_v1"))

