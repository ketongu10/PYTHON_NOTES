from time import time, sleep

import numpy as np
import tqdm
from multiprocessing import Pool
# def sq(x):
#     sleep(0.0001)
#     return x*x
# N = 100000
# a = [i for i in range(N)]
# t0 = time()
# with Pool(6) as p:
#     ret = list(tqdm.tqdm(p.imap(sq, a), total=N))
# #print(ret)
# print(time()-t0)
# t0 = time()
# with Pool(6) as p:
#     ret = list(tqdm.tqdm(p.imap(sq, a), total=N))
# #print(ret)
# print(time()-t0)
# print(sum(ret))


# import numpy as np
#
# def is_intersected(bbox1, bbox2):
#     x1, y1, w1, h1, s1 = bbox1
#     x2, y2, w2, h2, s2 = bbox2
#     w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#     b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#     b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     print((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2), (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2))
#     return  ((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2 or b1_x1 <= b2_x2 <= b1_x2)
#              and (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2 or b1_y1 <= b2_y2 <= b1_y2))
#
# def unite_bboxes(bbox1, bbox2):
#     x1, y1, w1, h1, s1 = bbox1
#     x2, y2, w2, h2, s2 = bbox2
#     w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#     b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#     b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     b3_x1, b3_x2 = np.min([b1_x1, b2_x1]), np.max([b1_x2, b2_x2])
#     b3_y1, b3_y2 = np.min([b1_y1, b2_y1]), np.max([b1_y2, b2_y2])
#     x3, y3, w3, h3, s3 = (b3_x1 + b3_x2)/2, (b3_y1 + b3_y2)/2, (b3_x2 - b3_x1), (b3_y2 - b3_y1), s1+s2
#     return [x3, y3, w3, h3, s3]
#
# bbox1 = [ 0.59921875, 0.22135416666666666, 0.0390625, 0.109375, 0]
# bbox2 = [0.61328125, 0.4876302083333333, 0.028125, 0.06901041666666667, 0]
# bbox3 = [0.596484375, 0.150390625, 0.03203125, 0.05859375, 0]
#
# print(is_intersected(bbox1, bbox3), unite_bboxes(bbox1, bbox3))


a = np.ndarray(shape=(2 ,100), dtype=np.float32)
b = np.ndarray(shape=(2 ,200), dtype=np.float32)
print(np.hstack((a[0], b[0])).shape)