import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations import add_rain as add_rain

# ======== TESTING AUGS

img = cv2.imread('./0112.jpg')
img =  A.shift_scale_rotate(img, 0, 1,0.04, 0)
img = cv2.line(img, (256*4, 0), (256*4, 768), (255, 0, 0), 1)
img = cv2.line(img, (256, 0), (256, 768), (255, 0, 0), 1)
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img1 = cv2.imread('./0024.jpg')
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB standard deviation
f = A.Compose([#A.Sharpen(p=1),
            #A.MultiplicativeNoise(multiplier=(0.15, 1.15), per_channel=True, elementwise=True, always_apply=False, p=0.5),
            #A.Blur(blur_limit=3, p=1),
            #A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.25, hue = 0.1, p=1),
            #A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
            #A.RandomBrightness(always_apply=False, p=1, limit=(0.5,0.5)),
            #A.ColorJitter(always_apply=False, p=1, brightness=0, contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            ##A.InvertImg(always_apply=False, p=0.15),
            ##A.Sharpen(always_apply=False, p=0.1, alpha=(0.77, 1.0), lightness=(0.0, 6.38)),
            #A.RandomGamma(p=1),
            #A.Downscale(scale_min=0.25, scale_max=0.99, p=1),
            #A.ImageCompression(quality_lower=30, quality_upper=100, p=1),
            #A.HorizontalFlip(p=0.5),
            #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            A.RandomRain(always_apply=False, p=1, slant_lower=-20, slant_upper=20, drop_length=4, drop_width=1,
                         drop_color=(219, 224, 223), blur_value=1, brightness_coefficient=1.0, rain_type='drizzle'),
            #A.shift_scale_rotate(p=1, shift_limit_y=0, shift_limit_x=(-0.2, 0.2), rotate_limit=(-15, 15))
        ], additional_targets={'image1': 'image'})


#img = f(image=img, image1=img1)

def make_rain(images):
    slant = np.random.randint(-20, 20)
    drop_length = np.random.randint(5, 15)
    global_num_drops = np.random.randint(200, 1000)
    height, width, rgb = images[0].shape
    ret = []
    for image in images:
        rain_drops = []
        num_drops = global_num_drops+np.random.randint(-100, 100)
        for _i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = np.random.randint(slant, width)
            else:
                x = np.random.randint(0, width - slant)

            y = np.random.randint(0, height - drop_length)

            rain_drops.append((x, y))
        ret.append(add_rain(img=image,slant=slant,drop_length=drop_length,drop_width=1,drop_color=(155,182,199), blur_value=1, brightness_coefficient=1.0, rain_drops=rain_drops))
    return ret




#print(np.shape(img["image"]), img["image"].mean())
new = np.hstack(make_rain((img, img1)))#np.hstack((img["image"], img["image1"]))
print(np.shape(new))
cv2.imshow("LOH",new)
cv2.imwrite("./sam_loh.jpg", new)
cv2.waitKey(0)
cv2.destroyAllWindows()"""