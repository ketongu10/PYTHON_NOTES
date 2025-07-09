import os
import shutil
n = 0
for dir in os.listdir("./images"):
	if len(os.listdir(os.path.join("./images", dir))) < 18:
		shutil.rmtree(os.path.join("./images", dir))
		shutil.rmtree(os.path.join("./", dir))
		print(dir)
		n+=1
		
print(f"TOTAL TO BE REMOVED: {n}")
