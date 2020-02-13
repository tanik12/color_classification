import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import glob
import os

from data_load import data_load, extract_color_info 
from visualize import plot, plot_hist
from model import train, dir_check, load_model, inference

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

def main():
    #label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}
    label_dict = {"pedestrian_signal_blue":0, "pedestrian_signal_red":1, 
                  "traffic_signal_blue":2, "traffic_signal_red":3, "traffic_signal_yellow":4, 
                  "pedestrian_signal_unknown":5, "traffic_signal_unknown":6}

    #data_dir = "data/sample_trim/*"
    data_dir = "data/trim_img/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
    res_data = extract_color_info(data_list)

    ################
    #res_data -> [[赤色抽出した後のhsv, 青色抽出した後のhsv, 緑色抽出した後のhsv, 黄色抽出した後のhsv,
    #              rgbhsvのそれぞれの平均値, maskしたあとの画像の平均値, color hist, 正解ラベル, 画像path], [...], ..., [...]]

    #label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, 
    #              "vehicle_signal_red":3, "vehicle_signal_yellow":4}
    ################

    print("img_path: ", res_data[56][8])
    print("(r, g, b, h, s, v): ", res_data[56][4])
    cv2.imwrite("red_masked_img.png", res_data[56][0])
    cv2.imwrite("bule_masked_img.png", res_data[56][1])
    cv2.imwrite("green_masked_img.png", res_data[56][2])
    cv2.imwrite("yellow_masked_img.png", res_data[56][3])

    plot(res_data)
    lot_hist(res_data)

    #################
    #mashine learning
    current_path = os.getcwd()
    model_dirpath = current_path + "/model"

    res = pd.DataFrame(res_data, 
                       columns=['hsv_after_red', 'hsv_after_blue', 'hsv_after_green','hsv_after_yellow',
                                'avg_rgbhsv', 'avg_after_img', 'hist_color', 'label', 'img_path']
                        )

    mass_data = res[['avg_rgbhsv', 'avg_after_img', 'label']]
    train(mass_data['avg_rgbhsv'], mass_data['label'], model_dirpath)

    test_x = np.array([64.4052, 85.112, 87.6772, 102.0968, 64.3176, 89.7904])
    test_y = np.array([0])
    inference(test_x, test_y,  model_dirpath) 
    ################

if __name__ == "__main__":
    main()
