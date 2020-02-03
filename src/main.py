import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import glob
import os

from data_load import data_load, extract_color_info 
from visualize import plot

def main():
    label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    data_dir = "data/sample_trim/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
    res_data = extract_color_info(data_list)

    #res_data -> [[赤色抽出した後のhsv, 青色抽出した後のhsv, rgbhsvのそれぞれの平均値, 正解ラベル, 画像path], [...], ..., [...]]
    #label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    print("img_path: ", res_data[0][4])
    print("(r, g, b, h, s, v): ", res_data[0][2])
    cv2.imwrite("red_masked_img.png", res_data[9][0])
    cv2.imwrite("bule_masked_img.png", res_data[9][1])

    plot(res_data)

if __name__ == "__main__":
    main()
