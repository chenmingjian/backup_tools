import cv2 
import numpy as np
import matplotlib.pyplot as plt
from menpo import image


img_path = R"D:\labwork\ao's_menpo\menpo_test\lfpw\ceph_trainset\001.bmp"
img = cv2.imread(img_path)
patch = [15, 23]

# img = image.Image(img)
# img = img.crop_to_landmarks_proportion(0.2)
# d = img.diagonal()
# if d > 400:
#     img = img.rescale(400.0 / d)

d = np.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1])
scale = 400 / d
img = cv2.resize(img, (int(img.shape[1] *scale),  int(img.shape[0] *scale)))
for i in patch:
    cv2.rectangle(img, (200,200), (200+i, 200+i),(0,0,255),2)

plt.imshow(img)
plt.show()