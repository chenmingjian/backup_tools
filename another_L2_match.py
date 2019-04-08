import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# pre_label_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label"
pre_label_dir = R"C:\Users\chen\Desktop\250\predict"
red_dot_dir = R'C:\Users\chen\Desktop\250\coordinate'
final_label_dir = R'C:\Users\chen\Desktop\250\maybeLabel'
resolutoin_dir = R'C:\Users\chen\Desktop\250\resolution'
img_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\img"
img_output_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\check_img_2"

name_map = [
    "G'", "N'", "Prn", "Sn", "Ls", "St", "Li", "Si", "Pog'", "Gn'", "Me'", "N",
    "Or", "ANS", "A", "UIA", "SPr", "UI", "LI", "Id", "LIA", "B", "Pog", "Gn",
    "Me", "U6", "L6", "Go", "PNS", "Ptm", "Ar", "Co", "S", "Ba", "P"
]


def get_sift_feature_s(img, points):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    kp = [cv2.KeyPoint(p[0], p[1], 50) for p in points]
    kp, des = sift.compute(img, kp)
    return des


def denormalization(normed_point_s_np, resolution_np):
    return normed_point_s_np * resolution_np


def L2_distance(x, y):
    return np.sqrt(np.sum((x - y) * (x - y)))


def load_resolution(img_path):
    img = cv2.decode(np.fromfile(img_path, dtype=np.uint8), 0)
    return np.array(img.shape)


def load_point_set_after_prediction(txt_path):
    return np.loadtxt(txt_path, comments='\r\n', delimiter=' ')


def load_point_set_red_dot_location(txt_path):
    return np.loadtxt(txt_path, comments='\r\n', delimiter=',')


def get_name_list(dir_path):
    file_name_list = os.listdir(dir_path)
    file_name_list = [file_name[:-4] for file_name in file_name_list]
    return file_name_list


def find_another_point(distance, order):
    distance_sorted = sorted(distance)
    min_order_dist = distance_sorted[order]
    distance = np.array(distance)
    index = np.where(distance == min_order_dist)[0][0]
    return index, min_order_dist


def maybe_loot(all_distance_list, flag_point_occupid, index, min_distance,
               num):
    """
    接收
    """
    if flag_point_occupid[index][0] == -1:
        # 最近点未被占用
        flag_point_occupid[index][0] = num
        flag_point_occupid[index][1] = min_distance
    else:
        # 最近点被占用
        flaged_num = flag_point_occupid[index][0]
        flaged_dist = flag_point_occupid[index][1]

        if min_distance < flaged_dist:
            # 抢走
            flag_point_occupid[index][2] += 1
            order = flag_point_occupid[index][2]
            flag_point_occupid[index][0] = num
            flag_point_occupid[index][1] = min_distance
            another_index, min_order_dist = find_another_point(
                all_distance_list[flaged_num], order)
            maybe_loot(all_distance_list, flag_point_occupid, another_index,
                       min_order_dist, flaged_num)
        else:
            # 抢不过
            flag_point_occupid[index][2] += 1
            if flag_point_occupid[index][2] >= 3:
                return
            order = flag_point_occupid[index][2]
            another_index, min_order_dist = find_another_point(
                all_distance_list[num], order)
            maybe_loot(all_distance_list, flag_point_occupid, another_index,
                       min_order_dist, num)


def get_label(flag_point_occupid, point_red_dot, label_num=35):
    label = [np.array([0, 0]) for _ in range(label_num)]
    for i in range(len(flag_point_occupid)):
        if flag_point_occupid[i][0] != -1:
            num = flag_point_occupid[i][0]
            label[num] = point_red_dot[i]
    label = np.array(label)
    return label


def match(point_predicted, point_red_dot, resolution):
    flag_point_occupid = [[-1, 0, 0] for _ in range(len(point_red_dot))]
    # file_name = get_name_list(pre_label_path)
    all_distance_list = []
    for num, p in enumerate(point_predicted):
        distance_of_p_and_whole_point_red_dot = [
            L2_distance(p, point) for point in point_red_dot
        ]
        all_distance_list.append(distance_of_p_and_whole_point_red_dot)

        distance_of_p_and_whole_point_red_dot_sorted = sorted(
            distance_of_p_and_whole_point_red_dot)
        min_distance = min(distance_of_p_and_whole_point_red_dot_sorted)

        scale = resolution / 940 * 20
        if min_distance > scale:
            continue
        index = np.where(
            distance_of_p_and_whole_point_red_dot == min_distance)[0][0]
        maybe_loot(all_distance_list, flag_point_occupid, index, min_distance,
                   num)

        # distance_of_p_and_whole_point_red_dot[index]  # this is lable
    label = get_label(flag_point_occupid, point_red_dot)
    label = label.astype(np.int32)
    return label


def show(img, points):
    plt.imshow(img)
    pt = points.T
    plt.plot(pt[0], pt[1], 'r.')
    for i, p in enumerate(points):
        plt.text(p[0], p[1], name_map[i], color='red')
    plt.show()


def check_label_now(img_num):
    img_name = str(img_num)
    tmp_for_test_point_predicted_path = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\pre_pre_label/" + img_name + ".txt"
    tmp_for_test_point_red_dot_path = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1/" + img_name + ".txt"
    tmp_for_test_img_path = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\img_red_dot/" + img_name + ".jpg"

    point_predicted_np = np.loadtxt(
        tmp_for_test_point_predicted_path, comments='\n')
    point_red_dot_np = np.loadtxt(
        tmp_for_test_point_red_dot_path, comments='\n', delimiter=',')

    tmp_for_test_label = match(point_predicted_np, point_red_dot_np)

    tmp_for_test_img = cv2.imread(tmp_for_test_img_path, 1)
    show(tmp_for_test_img, tmp_for_test_label)


def save(save_path, label):
    np.savetxt(save_path, label, delimiter=',', newline='\r\n', fmt='%d')


def check_label_in_dir(img_path, save_path, label):
    img = cv2.imread(img_path, 1)
    # diagonal = np.sqrt(img.shape[0] * img.shape[0] +
    #                    img.shape[1] * img.shape[1])
    # scale = diagonal
    radius = 5
    for i, p in enumerate(label):
        cv2.circle(img, (p[0], p[1]), radius, (0, 0, 255), -1)
        cv2.putText(img, name_map[i], (p[0], p[1]), cv2.FONT_HERSHEY_PLAIN,
                    radius, (0, 0, 255), 4)
    cv2.imwrite(save_path, img)


def get_all_label():
    name_list = os.listdir(pre_label_dir)
    count = 0
    for name in name_list:
        pre_label_path = os.path.join(pre_label_dir, name)
        red_dot_path = os.path.join(red_dot_dir, name)
        save_path = os.path.join(final_label_dir, name)

        # img_path = os.path.join(img_dir, name[:-3] + "jpg")
        img_save_path = os.path.join(img_output_dir, name[:-3] + "jpg")
        print(img_save_path)
        pre_label_np = load_point_set_after_prediction(pre_label_path)
        red_dot_np = load_point_set_red_dot_location(red_dot_path)

        resolutoin_path = os.path.join(resolutoin_dir, name)
        resolution = np.loadtxt(resolutoin_path)
        label = match(pre_label_np, red_dot_np, resolution)

        save(save_path, label)
        # check_label_in_dir(img_path, img_save_path, label)

        count += 1
        print(count)


# get_all_label()
get_all_label()
