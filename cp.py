import os
import shutil

check_list = ['187.txt','276.txt','326.txt','343.txt','344.txt','357.txt','358.txt','360.txt','361.txt','362.txt','363.txt','372.txt','376.txt','392.txt','399.txt','403.txt','433.txt','437.txt','465.txt','493.txt','192.txt','195.txt','196.txt','200.txt','202.txt','203.txt','204.txt','233.txt','234.txt','236.txt','240.txt','247.txt','259.txt','276.txt','280.txt','283.txt','314.txt','327.txt','336.txt','340.txt','341.txt','345.txt']

txt_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\check_imgs\label'
# img_dir =R'C:\Users\chen\Desktop\img_above_1k_resolutoin\img'
img_dir = R'C:\Users\chen\Desktop\copy\lfpw\ceph_trainset'
# dest_dir = R'C:\Users\chen\Desktop\copy\lfpw\ceph_trainset'
dest_dir = R'C:\Users\chen\Desktop\confuse'
txts = os.listdir(txt_dir)

for txt in txts:
    img = txt[0:-3] + 'jpg'
    path = os.path.join(img_dir, img)
    shutil.copy(path, dest_dir)