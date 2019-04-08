import cv2 
import os
import numpy as np


RED_DOT = 1
if RED_DOT:
    # TXT_DIR = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot/'
    # NORM_DIR =R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_norm/'
    TXT_DIR = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1/'
    NORM_DIR = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1_norm/"
else: 
    TXT_DIR = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label/'
    NORM_DIR =R'C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label_norm/'
IMG_DIR = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\img/"


txt_list = os.listdir(TXT_DIR)
txt_path_list = [TXT_DIR + x for x in txt_list]

img_path_list = []
for txt in txt_list:
    img_path_list.append(os.path.join(IMG_DIR, txt[:-3]+'jpg'))


target_path_list = []
for txt in txt_list:
    target_path_list.append(os.path.join(NORM_DIR, txt))

for i in range(len(txt_path_list)):
    (y, x) = cv2.imread(img_path_list[i],0).shape
    normed_list = []
    with open(txt_path_list[i]) as f:
        lines = f.readlines()
    for line in lines :
        tmp = np.array(line.split(','),dtype =int)
        normed = tmp / np.array([x, y])
        normed_list.append(normed)
    normed_list = np.array(normed_list)
    with open(target_path_list[i],'w') as f: 
        for norm in normed_list:
            tmp = '%16.7e %15.7e'%(norm[0], 1-norm[1])
            f.write(tmp + '\n')

            
    

