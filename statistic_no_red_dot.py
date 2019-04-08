import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

pre_label_path = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label"
red_dot_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot'
final_label_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\final_label'

resolutoin_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\resolution'
img_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\img'
img_red_dot_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\img_red_dot'
can_use_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\can_use'

pre_label_s = os.listdir(pre_label_path)
red_dot_s = os.listdir(red_dot_path)

import time 
begin = time.time()


# 940 is img size, 40 is the ear circle diameter in 940 img.
# the_ear_circle_scale is means a reasonable error.
the_ear_circle_img_resolution = 940

# try in multiresolution, below numbers is the l2 circle's diameter
threshold_scale_list = [20 , 40, 60, 100]
# threshold_scale_list = [40]

totol_img_num = 519

include_Ptm = 1

statistic_dict = {}

def get_resolution_to_txt():
    img_name_s = os.listdir(img_path)
    for img_name in img_name_s:
        img_file = os.path.join(img_path, img_name)
        img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), 1)
        save_path = os.path.join(resolutoin_path, img_name[:-3] + 'txt')
        print(save_path, np.array([img.shape[0]]))
        np.savetxt(save_path, np.array([img.shape[0]]))

def l2_dist(x, y):
    return sum((x-y) * (x-y))

for threshold in threshold_scale_list:
    statistic_dict[str(threshold)] = {}
    statistic_dict[str(threshold)]['can use'] = 0
    for pre_label_name in pre_label_s:
        can_use = True

        pre_label = os.path.join(pre_label_path, pre_label_name)
        pre_label_np = np.loadtxt(pre_label)

        red_dot = os.path.join(red_dot_path, pre_label_name)
        red_dot_np = np.loadtxt(red_dot, delimiter=',', comments='\n')

        # begin {get the resolution_local}
        resolution_file = os.path.join(resolutoin_path, pre_label_name)
        resolution_local = np.loadtxt(resolution_file)
        # end {get the resolution_local}

        threshold_local = threshold * (resolution_local / the_ear_circle_img_resolution)
        threshold_local = threshold_local * threshold_local

        for point in pre_label_np:
            dist_list = [l2_dist(point, red_dot) for red_dot in red_dot_np]
            dist_np = np.array(dist_list)
            
            if min(dist_np) > threshold_local:
                
                if include_Ptm :
                    can_use = False
                elif np.where(pre_label_np == point)[0][0] != 29:
                    can_use = False


                index = np.where(pre_label_np == point)[0][0]
                if str(index) not in statistic_dict[str(threshold)]:
                    statistic_dict[str(threshold)][str(index)] = 1
                else:
                    statistic_dict[str(threshold)][str(index)] += 1
                continue
            else:
                index = np.where(dist_np == min(dist_np))[0][0]
                red_dot_np = np.delete(red_dot_np, index, axis=0)
        
        if can_use:
            statistic_dict[str(threshold)]['can use'] += 1
            src_path = os.path.join(img_red_dot_path, pre_label_name[:-3]+ 'jpg')
            dest_path = os.path.join(can_use_path)
            # shutil.copy(src_path , dest_path)
            

print(statistic_dict)

import json

json_path = R'C:\Users\chen\Desktop\statistic_re.json'
json = json.dumps(statistic_dict)

with open(json_path, 'w') as f:
    f.write(json)

print('totol used ',time.time() - begin, 's')