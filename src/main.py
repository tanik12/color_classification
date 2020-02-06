import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import glob
import os

from data_load import data_load, extract_color_info 
from visualize import plot
from model import train, dir_check, load_model, inference

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

def main():
    label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    data_dir = "data/sample_trim/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
    res_data = extract_color_info(data_list)

    #res_data -> [[赤色抽出した後のhsv, 青色抽出した後のhsv, rgbhsvのそれぞれの平均値, maskしたあとの画像の平均値, 正解ラベル, 画像path], [...], ..., [...]]
    #label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    print("img_path: ", res_data[0][5])
    print("(r, g, b, h, s, v): ", res_data[0][2])
    cv2.imwrite("red_masked_img.png", res_data[4][0])
    cv2.imwrite("bule_masked_img.png", res_data[4][1])

    plot(res_data)

    #################
    #mashine learning
    current_path = os.getcwd()
    model_dirpath = current_path + "/model"

    res = pd.DataFrame(res_data, columns=['hsv_after_red', 'hsv_after_blue', 'avg_rgbhsv', 'avg_after_img', 'label', 'img_path'])
    mass_data = res[['avg_rgbhsv', 'avg_after_img', 'label']]
    train(mass_data['avg_rgbhsv'], mass_data['label'], model_dirpath)

    test_x = np.array([64.4052, 85.112, 87.6772, 102.0968, 64.3176, 89.7904])
    test_y = np.array([0])
    inference(test_x, test_y,  model_dirpath) 
    ################

if __name__ == "__main__":
    main()
