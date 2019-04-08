import os
import numpy as np 


src_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1'
dest_dir = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_2'

def l2_dist(y, x=np.array([6666, 0])):
    return sum((x-y) * (x-y))

src_name_s = os.listdir(src_dir)
for src_name in src_name_s:
    src_path = os.path.join(src_dir, src_name)
    src = np.loadtxt(src_path, comments='\n', delimiter=',')

    dist_list = [l2_dist(point) for point in src]
    dist_np = np.array(dist_list)
    min_dist = min(dist_list)
    index = np.where(dist_np == min_dist)[0][0]
    print(min_dist, index)

    src = np.delete(src, index, axis=0)
    dest_path = os.path.join(dest_dir, src_name)
    # np.savetxt(dest_path, src, fmt='%d', delimiter=',', newline='\r\n')
