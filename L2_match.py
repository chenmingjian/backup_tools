import os 
import numpy as np 

pre_label_path = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label"
red_dot_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot'
final_label_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\final_label'
resolutoin_path = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\resolution'

pre_label_s = os.listdir(pre_label_path)
red_dot_s = os.listdir(red_dot_path)

# 940 is img size, 40 is the ear circle diameter in 940 img.
# the_ear_circle_scale is means a reasonable error.
the_ear_circle_img_resolution = 940

# try in multiresolution, below numbers is the l2 circle's diameter
threshold_scale_list = [20 , 40, 60, 100]
threshold = 40



def l2_dist(x, y):
    tmp = (x-y) * (x-y)
    return sum(tmp)

# match_list = ['187.txt','276.txt','326.txt','343.txt','344.txt','357.txt','358.txt','360.txt','361.txt','362.txt','363.txt','372.txt','376.txt','392.txt','399.txt','403.txt','433.txt','437.txt','465.txt','493.txt','192.txt','195.txt','196.txt','200.txt','202.txt','203.txt','204.txt','233.txt','234.txt','236.txt','240.txt','247.txt','259.txt','276.txt','280.txt','283.txt','314.txt','327.txt','336.txt','340.txt','341.txt','345.txt']

neighbor_num = 2

for pre_label_name in pre_label_s[50:51]:
# for pre_label_name in match_list:
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


    paired_point = []
    point_list = []
    for point in pre_label_np:
        dist_list = [l2_dist(point, red_dot) for red_dot in red_dot_np]
        dist_np = np.array(dist_list)
        # if min(dist_np) > threshold_local:
        #     ture_point_list.append([0,0])
        #     continue

        index_list = []
        for _ in range(neighbor_num):
            index = np.where(dist_np == min(dist_np))[0][0]
            dist_np[index] = np.Infinity
            index_list.append(index)
            point_list.append(point)
        tmp = red_dot_np[index_list]
        paired_point.append(tmp)
    
    point_np = np.array(point_list)
    paired_point_np = np.array(paired_point) 
    paired_point_np.resize([pre_label_np.shape[0]*neighbor_num, 2])

    save_point_path = os.path.join(R"C:\Users\chen\Desktop\point", pre_label_name)
    np.savetxt(save_point_path, point_np,  fmt='%d', newline='\r\n',delimiter=',')
    save_paired_point_path = os.path.join(R"C:\Users\chen\Desktop\paired_point", pre_label_name)
    np.savetxt(save_paired_point_path, paired_point_np,  fmt='%d', newline='\r\n',delimiter=',')