import numpy as np
inds_list = [1, 2, 4]

a = np.array([[1, 3],[2, 4]])
cond = False
for psind in inds_list:
    cond |= (a == psind)
print(cond)
mask = np.isin(a, [1, 2]).astype(np.uint8)*255
print(mask)
result = np.where(mask, a, 0)
print(result)
