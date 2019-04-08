# coding=utf-8
import time
import os
import matplotlib.pyplot as plt
import menpo.io as mio
import numpy as np
from menpo import image
from menpo.feature import fast_dsift
from menpo.shape import PointDirectedGraph
from menpo.visualize import print_progress
from menpofit.aam import (LucasKanadeAAMFitter, PatchAAM,
                          WibergInverseCompositional)

begin = time.time()

TRAIN = 0

if TRAIN:
    path_to_images = 'lfpw/ceph_trainset/'
    training_images = []
    for img in print_progress(mio.import_images(path_to_images, verbose=True)):
        # convert to greyscale3
        # if img.n_channels == 3:
        # img = img.as_greyscale()
        # crop to landmarks bounding box with an extra 20% padding
        img = img.crop_to_landmarks_proportion(0.2)
        # rescale image if its diagonal is bigger than 400 pixels
        d = img.diagonal()
        if d > 400:
            img = img.rescale(400.0 / d)
        # append to list
        training_images.append(img)

    patch_aam = PatchAAM(
        training_images,
        group='PTS',
        patch_shape=[(16, 19), (19, 16)],
        diagonal=200,
        scales=(0.5, 1.0),
        holistic_features=fast_dsift,
        max_shape_components=60,
        max_appearance_components=200,
        verbose=True)
    fitter = LucasKanadeAAMFitter(
        patch_aam,
        lk_algorithm_cls=WibergInverseCompositional,
        n_shape=[10, 30],
        n_appearance=[40, 160])

adjacency_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                             [1, 0, 0, 0]])


def chen_get_bbx(txt_path):
    coordinate = np.loadtxt(txt_path, comments='\n', delimiter=',')
    y, x = coordinate.T
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y) - 18

    points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y],
                       [min_x, max_y]])
    graph = PointDirectedGraph(points, adjacency_matrix)
    bbx = graph.bounding_box()
    return bbx, coordinate


def chen_comput_error(ground_truth_np, predict_value_np):
    dist = ground_truth_np - predict_value_np
    dist_list = [np.sqrt(i[0] * i[0] + i[1] * i[1]) for i in dist]
    dist_np = np.array(dist_list)
    print(dist_np)
    print(np.average(dist_np))


def chen_comput_relative_error(ground_truth_np, predict_value_np):
    dist = ground_truth_np - predict_value_np
    dist_list = [np.sqrt(i[0] * i[0] + i[1] * i[1]) for i in dist]
    dist_np = np.array(dist_list)
    print(dist_np)
    print(np.average(dist_np))


if TRAIN:
    mio.export_pickle(fitter, '26_img_35_pnt.pkl', overwrite=True)
else:
    fitter = mio.import_pickle('26_img_35_pnt.pkl')

tmp = [
    187, 276, 326, 343, 344, 357, 358, 360, 361, 362, 363, 372, 376, 392, 399,
    403, 433, 437, 465, 493, 192, 195, 196, 200, 202, 203, 204, 233, 234, 236,
    240, 247, 259, 276, 280, 283, 314, 327, 336, 340, 341, 345
]

img_path = R"C:\Users\chen\Desktop\250\output"
coordinate_path = R"C:\Users\chen\Desktop\250\coordinate"

img_file_list = os.listdir(img_path)
for img in img_file_list:
    # t = input('image number:')

    image_road = os.path.join(img_path, img)
    print(image_road)
    # image_road = R"D:\labwork\ao's_menpo\landmark_dateset\AddRedDotImage/" +str(i)+'.bmp'
    # image_road = R'C:\Users\chen\Desktop\img_above_1k_resolutoin\img_red_dot/' + str(i) + '.jpg'
    image = mio.import_image(image_road)
    resolution = image.shape[1]
    # image = image.as_greyscale()

    # note_base_road = R"D:\labwork\ao's_menpo\landmark_dateset\AnnotationsByMD\chen_format"
    # current_note_road = note_base_road + '/' + str(i) + '.txt'
    # bboxes, ground_truth_np = chen_get_bbx(current_note_road)

    txt_path = os.path.join(coordinate_path, img.split(".")[0] + ".txt")
    print(txt_path)
    bboxes, _ = chen_get_bbx(txt_path)

    _, ground_truth_np = chen_get_bbx(txt_path)

    result = fitter.fit_from_bb(image, bboxes, max_iters=[30, 5])

    pre_landmarks = result.final_shape.as_vector().copy()
    pre_landmarks.resize((35, 2))
    pre_landmarks[:, [0, 1]] = pre_landmarks[:, [1, 0]]

    # chen_comput_error(ground_truth_np, pre_landmarks)

    # scale = 940 / resolution
    # chen_comput_relative_error(ground_truth_np * scale, pre_landmarks * scale)

    # # save
    save_path = R'C:\Users\chen\Desktop\250\predict'
    tmp_path = os.path.join(save_path, img.split(".")[0] + ".txt")
    print(tmp_path)
    np.savetxt(tmp_path, pre_landmarks, fmt='%d', newline='\r\n')

    print('total time', time.time() - begin, 's')

    # # plt.subplot(131)
    # image.view()
    # bboxes.view(line_width=3, render_markers=False)
    # plt.gca().set_title('Bounding box')
    # plt.show()

    # # # plt.subplot(132)
    # image.view()
    # result.initial_shape.view(marker_size=4)
    # plt.gca().set_title('Initial shape')
    # plt.show()

    # # plt.subplot(133)
    image.view()
    result.final_shape.view(marker_size=4, figure_size=(15, 13))
    plt.gca().set_title('Final shape')
    plt.show()
