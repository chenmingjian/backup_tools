import cv2
import os
import numpy as np

PATH = R'C:\Users\chen\Desktop\5img_dot\头影侧量'

dot_img_resolution_list = []
raw_img_resolution_list = []

for root, dirs, files in os.walk(PATH):
    for f in files:
        if f[-5:] == '点.jpg':
            dot_img_path = os.path.join(root, f)
            dot_img = cv2.imdecode(np.fromfile(dot_img_path, dtype = np.uint8), 1)
            # dot_img_resolution_list.append(dot_img_path)
            dot_img_resolution_list.append(dot_img.shape)
        elif f[-5:] == '线.jpg':
            pass
        else: 
            raw_img_path = os.path.join(root, f)
            raw_img = cv2.imdecode(np.fromfile(raw_img_path, dtype = np.uint8), 1)
            # raw_img_resolution_list.append(raw_img_path)
            raw_img_resolution_list.append(raw_img.shape)

dot_img_resolution_np = np.array(dot_img_resolution_list)
raw_img_resolution_np = np.array(raw_img_resolution_list)
print(dot_img_resolution_np[0], raw_img_resolution_np[1])

scale = (dot_img_resolution_np / raw_img_resolution_np)[0:,0:2]
print (scale[0])
print(dot_img_resolution_np[0][0:2] / scale[0])
txts_path = R'C:\Users\chen\Desktop\5img_dot\tmp\1\label'

for root, dirs, files in os.walk(txts_path): 
    for i, f in enumerate(files): 
        if f[0] != 'd':
            txt_path = os.path.join(root, f)
            landmark = np.loadtxt(txt_path, comments='\n', delimiter=',')
            landmark = landmark / scale[i]
            np.savetxt(R'C:\Users\chen\Desktop\5img_dot\tmp\1\label\d'+str(i)+'.txt', landmark, fmt='%d', delimiter=',', newline='\r\n')

# print(scale[1][0:2])
# print(scale[1][0:2][::-1])