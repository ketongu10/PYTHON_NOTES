from time import time
a = [i for i in range(100000)]
b = 0
t0 = time()

for aa in a:
    b+=aa
print(b)
print(f"time = {(time()-t0):0.12f}")