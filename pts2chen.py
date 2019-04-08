import os
import numpy as np

src_dir = R"D:\labwork\ao's_menpo\landmark_dateset\AnnotationsByMD\400_senior"
dest_dir = R"D:\labwork\ao's_menpo\landmark_dateset\AnnotationsByMD\chen_format"

txt_name_s = os.listdir(src_dir)

for txt_name in txt_name_s:
    txt_path = os.path.join(src_dir, txt_name)
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    useful_lines = lines[:-8]
    save_path = os.path.join(dest_dir, txt_name)
    with open(save_path, 'w') as f:
        f.writelines(useful_lines)