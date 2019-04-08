import os
import numpy as np
import cv2

# AO = 0
# TRAINING_DATA = 0
# ALL_RED_DOT = 0

# if AO:
#     if TRAINING_DATA:
#         img_dir = R"D:\labwork\ao's_menpo\landmark_dateset\RawImage\TrainingData"
#         output_dir = R"D:\labwork\ao's_menpo\menpo_test\lfpw\ceph_trainset_senior"
#     else:
#         img_dir = R"D:\labwork\ao's_menpo\landmark_dateset\RawImage\Test1Data"
#         output_dir = R"D:\labwork\ao's_menpo\landmark_dateset\AddRedDotImage"
#     red_dot_dir = R"D:\labwork\ao's_menpo\landmark_dateset\AnnotationsByMD\chen_format"
#     suffix = 'bmp'
# else:
#     if TRAINING_DATA:
#         img_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_trainset_raw"
#         output_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_trainset"
#         red_dot_dir = R"C:\Users\chen\Desktop\copy\lfpw\text"
#     else:
#         img_dir = R"C:\Users\chen\Desktop\img"
#         output_dir = R"C:\Users\chen\Desktop"
#         red_dot_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot"
#     suffix = 'jpg'


img_dir = R"C:\Users\chen\Desktop\250\output"
output_dir = R"C:\Users\chen\Desktop\250\raw_add_rd"
red_dot_dir = R"C:\Users\chen\Desktop\250\coordinate"
suffix = 'jpg'
radius_for_2k_resolution = 5  # 2k resolution
radius_scale = radius_for_2k_resolution / 2400

img_name_s = os.listdir(img_dir)
img_name_s = [img_name for img_name in img_name_s if img_name[-3:] == suffix]
for img_name in img_name_s:
    # read img_red_dot
    img_path = os.path.join(img_dir, img_name)
    print(img_path)
    img = cv2.imread(img_path, 1)
    radius = int(img.shape[0] * radius_scale)
    # draw blue dot
    txt_name = img_name[:-3] + 'txt'
    red_dot_location = os.path.join(red_dot_dir, txt_name)
    red_dot_location_np = np.loadtxt(
        red_dot_location, dtype=int, comments='\n', delimiter=',')
    for loc in red_dot_location_np:
        # cv2.circle(img,(loc[0],loc[1]), radius, [0,0,255], -1 )
        cv2.rectangle(img, (loc[0] - radius, loc[1] - radius),
                      (loc[0] + radius, loc[1] + radius), [0, 0, 255], -1)
# save img in save_dir
    save_path = os.path.join(output_dir, img_name[:-3] + suffix)
    cv2.imwrite(save_path, img)
