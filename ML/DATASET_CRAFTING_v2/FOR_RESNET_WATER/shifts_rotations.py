import cv2
import numpy as np
import albumentations as A


def shift(img, x=0, y=0):
    h, w = img.shape[:2]
    shift_matrix = np.float32([[1, 0, x], [0, 1, y]])
    new = cv2.warpAffine(img, shift_matrix, (w, h))
    return new

def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(center, angle,1+abs(np.sin(angle/180*np.pi)))
    rotated = cv2.warpAffine(img, rot_mat, (w, h))

    return rotated[0:768, 256:1024]

img = cv2.imread('./0013.jpg')
img = A.shift_scale_rotate(img, 0, 1, 0.3, 0)
cv2.imshow("shifted",img)
cv2.imwrite("./shifted.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()