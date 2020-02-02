import cv2
import numpy as np
import glob
import os

def data_load(pathes, label_dict):
    data_list = []

    for path in pathes:
        data_dict = {}
        img_path = os.path.join(path, '*')
        img_path = glob.glob(img_path)
        label = path.split('/')[-1]
        label_num = label_dict[label]
        
        data_dict["img_pathes"] = img_path
        data_dict["label"] = label_num
        
        data_list.append(data_dict)

    return data_list

if __name__ == "__main__":
    label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    data_dir = "data/sample_trim/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
