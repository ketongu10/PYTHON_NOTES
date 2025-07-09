import os
import numpy as np
from matplotlib import pyplot as plt


path = "./prepared/"

statistics = {"true_labels": 0,
              "total_labels": 0,
              "labels": []}
for i in range(2138):
    with open(path+f"labels/{i}/labels.txt", "r") as f:
        for line in f.readlines():
            y = int(line.split()[1])
            if y > 100000:
                statistics["true_labels"]+=1
            statistics["total_labels"]+=1
            statistics["labels"].append(y/255/3)


print("True: ",statistics["true_labels"])
print("Total: ",statistics["total_labels"])
print("Percent: ",statistics["true_labels"]/statistics["total_labels"])
plt.hist(statistics["labels"], range=(0, 100000))
plt.show()