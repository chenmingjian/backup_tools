import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt
import shutil
import time
begin = time.time()

landmark_num2name_map = ["G'","N'","Prn","Sn","Ls","St","Li","Si","Pog'","Gn'","Me'","N","Or","ANS","A","UIA","SPr","UI","LI","Id","LIA","B","Pog","Gn","Me","U6","L6","Go","PNS","Ptm","Ar","Co","S","Ba","P"]


red_dot_img_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\img_red_dot'
final_lable_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\final_label'
# final_lable_dir = R'C:\Users\chen\Desktop\copy\lfpw\text'
red_dot_location_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot'
save_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\check_imgs'

# use blue dot label the red dot in img.
# use green dot label the maybe_final_label to show.
# use green text to show the lable name for person easy to see.

# check_list = ['187.txt','276.txt','326.txt','343.txt','344.txt','357.txt','358.txt','360.txt','361.txt','362.txt','363.txt','372.txt','376.txt','392.txt','399.txt','403.txt','433.txt','437.txt','465.txt','493.txt','192.txt','195.txt','196.txt','200.txt','202.txt','203.txt','204.txt','233.txt','234.txt','236.txt','240.txt','247.txt','259.txt','276.txt','280.txt','283.txt','314.txt','327.txt','336.txt','340.txt','341.txt','345.txt']
# check_list = ['1.txt','100.txt','102.txt','103.txt','104.txt','105.txt','106.txt','107.txt','108.txt','109.txt','110.txt','111.txt','112.txt','114.txt','115.txt','117.txt','118.txt','119.txt','12.txt','187.txt','200.txt','203.txt','204.txt','233.txt','234.txt','236.txt','240.txt','247.txt','259.txt','276.txt','280.txt','283.txt','314.txt','326.txt','327.txt','336.txt','340.txt','343.txt','344.txt','357.txt','360.txt','361.txt','362.txt','363.txt','372.txt','376.txt','392.txt','399.txt','403.txt','433.txt','437.txt','465.txt','493.txt','d0.txt','d1.txt','d2.txt','d3.txt','d4.txt']
red_dot_img_name_s = os.listdir(red_dot_img_dir)
for red_dot_img_name in red_dot_img_name_s:
# for red_dot_img_name in check_list:
# read img_red_dot 
    red_dot_img_path = os.path.join(red_dot_img_dir, red_dot_img_name[:-3] + 'jpg')
    print(red_dot_img_path)
    red_dot_img = cv2.imread(red_dot_img_path, 1)
# draw blue dot
    txt_name = red_dot_img_name[:-3] + 'txt'
    # red_dot_location = os.path.join(red_dot_location_dir, txt_name) 
    # red_dot_location_np = np.loadtxt(red_dot_location, dtype=int,comments='\n', delimiter=',')
    # for loc in red_dot_location_np:
    #     cv2.circle(red_dot_img,(loc[0],loc[1]), 1,[255,0,0], -1 )
# draw green dot and text
    final_lable = os.path.join(final_lable_dir, txt_name) 
    # shutil.copy(final_lable, save_dir)
    final_lable_np = np.loadtxt(final_lable, dtype=int,comments='\n', delimiter=',')
    for i, loc in enumerate(final_lable_np):
        cv2.circle(red_dot_img,(loc[0],loc[1]), 4, [0, 255, 0], -1 )
        cv2.putText(red_dot_img, landmark_num2name_map[i],(loc[0],loc[1]), cv2.FONT_HERSHEY_SIMPLEX,2,[0, 255, 0],3)
# save img in save_dir
    save_path = os.path.join(save_dir, red_dot_img_name[:-3]+'jpg')
    cv2.imwrite(save_path, red_dot_img)
    # cv2.imshow('tmp', red_dot_img)
    # cv2.waitKey(10000)
    # plt.imshow(red_dot_img)
    # plt.show()


print('totol used ', time.time(), 's')

