import os

import cv2
import menpo.io as mio
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from menpo import image
from menpo.feature import fast_dsift
from menpo.shape import PointDirectedGraph
from menpo.visualize import print_progress
from menpofit.aam import (LucasKanadeAAMFitter, PatchAAM,
                          WibergInverseCompositional)

os.system("")


def add_predict_dot(img, loc_np, img_name):
    output_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_validset\check_why"
    radius_for_2k_resolution = 6  # 2k resolution
    radius_scale = radius_for_2k_resolution / 2400
    radius = int(img.shape[0] * radius_scale)
    loc_np = loc_np.astype(np.int)
    for loc in loc_np:

        cv2.circle(img, (loc[0], loc[1]), radius, [255, 0, 0], -1)

    save_path = os.path.join(output_dir, img_name + '.jpg')
    cv2.imwrite(save_path, img)


def add_red_dot(r, img_dir, output_dir, red_dot_dir, suffix):

    radius_for_2k_resolution = r  # 2k resolution
    radius_scale = radius_for_2k_resolution / 2400
    img_name_s = os.listdir(img_dir)
    img_name_s = [
        img_name for img_name in img_name_s if img_name[-3:] == suffix
    ]
    for img_name in img_name_s:
        # read img_red_dot
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, 1)
        radius = int(img.shape[0] * radius_scale)
        # draw blue dot
        txt_name = img_name[:-3] + 'txt'
        red_dot_location = os.path.join(red_dot_dir, txt_name)
        red_dot_location_np = np.loadtxt(
            red_dot_location, dtype=int, comments='\n', delimiter=',')
        for loc in red_dot_location_np:
            cv2.rectangle(img, (loc[0] - radius, loc[1] - radius),
                          (loc[0] + radius, loc[1] + radius), [0, 0, 255], -1)
    # save img in save_dir
        save_path = os.path.join(output_dir, img_name[:-3] + suffix)
        print(save_path)
        cv2.imwrite(save_path, img)


def add_train_and_valid_red_dot(r=7):
    suffix = 'jpg'
    img_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_trainset_raw"
    output_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_trainset"
    red_dot_dir = R"C:\Users\chen\Desktop\copy\lfpw\text"
    add_red_dot(r, img_dir, output_dir, red_dot_dir, suffix)
    # img_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\img"
    # output_dir = R"C:/Users/chen/Desktop/test_img"
    # red_dot_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1"
    # add_red_dot(r, img_dir, output_dir, red_dot_dir, suffix)


def get_test_path():
    output_dir = R"C:/Users/chen/Desktop/test_img"
    red_dot_dir = R"C:\Users\chen\Desktop\img_above_1k_resolutoin\txt_red_dot_remove_topright_1"
    valid_name_list = [name[:-4] for name in os.listdir(output_dir)]
    valid_name_dir = output_dir
    red_dot_location_dir = red_dot_dir
    ground_truth_dir = R"C:\Users\chen\Desktop\copy\lfpw\ceph_validset\label"
    return valid_name_list, valid_name_dir, red_dot_location_dir, ground_truth_dir


def train():
    path_to_images = 'lfpw/ceph_trainset/'
    training_images = []
    for img in mio.import_images(path_to_images, verbose=True):
        # if img.n_channels == 3:
        # img = img.as_greyscale()
        img = img.crop_to_landmarks_proportion(0.2)
        d = img.diagonal()
        if d > 400:
            img = img.rescale(400.0 / d)
        training_images.append(img)
    # patch_aam = PatchAAM(training_images, group='PTS', patch_shape=[(15, 15), (23, 23)],
    #                      diagonal=200, scales=(0.5, 1.0), holistic_features=fast_dsift,
    #                      max_shape_components=60, max_appearance_components=200,
    #                      verbose=True)
    patch_aam = PatchAAM(
        training_images,
        group='PTS',
        patch_shape=[(16, 19), (19, 16)],
        diagonal=200,
        scales=(0.5, 1),
        holistic_features=fast_dsift,
        max_shape_components=74,
        max_appearance_components=175,
        verbose=True)
    fitter = LucasKanadeAAMFitter(
        patch_aam,
        lk_algorithm_cls=WibergInverseCompositional,
        n_shape=[10, 30],
        n_appearance=[40, 160])
    mio.export_pickle(fitter, '26_img_35_pnt.pkl', overwrite=True)
    return fitter


def comput_error(ground_truth_np, predict_value_np):
    dist = ground_truth_np - predict_value_np
    dist_list = [np.sqrt(i[0] * i[0] + i[1] * i[1]) for i in dist]
    dist_np = np.array(dist_list)
    return np.average(dist_np)


def get_coordinate(txt_path):
    coordinate = np.loadtxt(txt_path, comments='\n', delimiter=',')
    return coordinate


def get_bbx(txt_path):
    coordinate = get_coordinate(txt_path)
    y, x = coordinate.T
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y) - 18

    points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y],
                       [min_x, max_y]])
    adjacency_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                 [1, 0, 0, 0]])

    graph = PointDirectedGraph(points, adjacency_matrix)
    bbx = graph.bounding_box()
    return bbx


def test(fitter, mi0=28, mi1=24):
    # error_list = []
    # error_dict = {}
    name_list, valid_name_dir, red_dot_location_dir, ground_truth_dir = get_test_path(
    )
    for name in name_list:
        image_road = os.path.join(valid_name_dir, name + '.jpg')
        image = mio.import_image(image_road)
        # resolution = image.shape[1]

        txt_path = os.path.join(red_dot_location_dir, name + '.txt')
        bboxes = get_bbx(txt_path)

        # txt_path = os.path.join(ground_truth_dir, name + '.txt')
        # ground_truth_np = get_coordinate(txt_path)

        result = fitter.fit_from_bb(image, bboxes, max_iters=[mi0, mi1])

        pre_landmarks = result.final_shape.as_vector().copy()
        pre_landmarks.resize((35, 2))
        pre_landmarks[:, [0, 1]] = pre_landmarks[:, [1, 0]]
        root = R"C:\Users\chen\Desktop\test_pre_label"
        save_path = os.path.join(root, name + '.txt')
        np.savetxt(save_path, pre_landmarks, fmt='%d', delimiter=',', newline='\r\n')
    #     cv_img = cv2.imread(os.path.join(valid_name_dir, name + '.jpg'), 1)
    #     add_predict_dot(cv_img, pre_landmarks, name)

    #     error = comput_error(ground_truth_np, pre_landmarks)
    #     error_dict[name] = error
    #     error_list.append(error)
    # error_np = np.array(error_list)
    # average_error = np.average(error_np)
    # print('\n', error_dict, '\n', average_error)
    # return average_error


fitter = mio.import_pickle('26_img_35_pnt.pkl', )

# add_train_and_valid_red_dot()
# fitter = train()
error = test(fitter)


def waiting_to_max(mi0, mi1):
    # ps1 = int(ps1)
    # ps2 = int(ps2)
    # msc = int(msc)
    # mac = int(mac)
    # add_train_and_valid_red_dot()
    # fitter = train()
    error = test(fitter, mi0, mi1)
    return -error


# pbounds = {"mi0": (5, 60), "mi1": (5, 30)}

# optimizer = BayesianOptimization(
#     f=waiting_to_max,
#     pbounds=pbounds,
#     random_state=1,
# )

# logger = JSONLogger(path="./logs.json")
# optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# for i in range(5, 13):
#     optimizer.probe(
#         params=[i],
#         lazy=True,
#     )
# optimizer.maximize(init_points=0, n_iter=0)

# optimizer.probe(
#     params=[30, 5],
#     lazy=True,
# )
# optimizer.maximize(init_points=0, n_iter=0)

# optimizer.maximize(
#     init_points=2,
#     n_iter=3000,
# )

# from bayes_opt.util import load_logs

# new_optimizer = BayesianOptimization(
#     f=waiting_to_max,
#     pbounds=pbounds,
#     verbose=2,
#     random_state=7,
# )

# # New optimizer is loaded with previously seen points
# load_logs(new_optimizer, logs=["./logs.json"]);

# print(new_optimizer.max)