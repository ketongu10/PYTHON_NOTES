from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import time
import pandas as pd

def floatt(x):
    if x is None or x =='None':
        return None
    else:
        return float(x)
# data = pd.read_csv("/home/popovpe/Downloads/Telegram Desktop/output.txt")
# print(data.head())
# print(data.values)
# print(data.ix[0])

a = [[1,2], [3, 5, 4]]
print(a[:][0])
names = []
data = []
with open("/home/popovpe/Downloads/Telegram Desktop/winch_stats (2).txt", 'r') as f:
    lines = f.readlines()
    names = [x.replace('[', '').replace(']', '') for x in lines[0].split(',')[:-1]]
    for line in lines[1:]:
        splitted = line.split(',')[:-1]

        #print(splitted)
        l = [list(map(lambda x: floatt(x.replace('[', '').replace(']', '').replace(' ','')),y.split('; '))) for y in splitted]

        data.append(l)
sc = 1 #1080/640
ts = [line[0][0] for line in data]
pos1 = [line[1][0]*sc for line in data]
pos2 = [line[3][0]*sc for line in data]
pos3 = [line[5][0]*sc for line in data]
v1 = [line[2][0] for line in data]
v2 = [line[4][0] for line in data]
v3 = [line[6][0] for line in data]


print(min(pos3), max(pos3))
fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,sharex=True)
ax1.plot(ts, pos1)
ax1.plot(ts, pos2)
ax1.plot(ts, pos3)
#ax1.set_ylim(0.42, 0.75)
ax2.set_ylim(-100, 100)
ax2.plot(ts, v1)
ax2.plot(ts, v2)
ax2.plot(ts, v3)
ax2.set_xlim(0, 30)
fig.savefig("./geka1.png")