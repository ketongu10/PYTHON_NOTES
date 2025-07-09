import os
import shutil
n = 0
os.mkdir("./brief")
for dir in os.listdir("./images"):
	shutil.copy(f"./images/{dir}/0010.jpg", f"./brief/{dir}.jpg")

