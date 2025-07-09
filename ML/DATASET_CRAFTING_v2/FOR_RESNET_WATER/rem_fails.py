import os
import shutil
n = 0
fails = [
112,
140,
180,
264,
292,
383,
419,
518,
598,
809,
815,
839,
865,
876,
883,
1003,
1032,
1061,
1115,
1138,
1340,
1371,
1387,
1454,
1498,
1511,
1513,
1598,
1641,
1647,
1692,
1705,
1756,
1984,
1990,
1996,
2043,
2107,
2165,
2375,
2575,
2614,
2727,
2843,
2858,
2872,
2954,
3033,
3044
]
for file in fails:
	try:
		shutil.rmtree(f"./{file}")
		print("label", file)
	except:
		print(f"label {file} not found")
	try:
		shutil.rmtree(f"./images/{file}")
		print("dir", file)
	except:
		print(f"dir {file} not found")

