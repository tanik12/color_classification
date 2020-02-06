import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pandas as pd
import glob
import os

from data_load import data_load, extract_color_info
from visualize import plot

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train(x_train, y_train):
    x_train = x_train.values
    y_train = y_train.values

    init_train = np.zeros((63, 6))
    for idx, chunk in enumerate(x_train):
        init_train[idx] = chunk
    
    x_train = init_train

    print(x_train.shape, y_train.shape)

    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)

    # モデルの学習。fit関数で行う。
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

if __name__ == "__main__":
    #res_data -> [[赤色抽出後のhsv, 青色抽出後のhsv, rgbhsvの各々平均値, mask後の画像の平均値, 正解ラベル, 画像path], [...], ..., [...]]
    #label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    data_dir = "data/sample_trim/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
    res_data = extract_color_info(data_list)

    res = pd.DataFrame(res_data, columns=['hsv_after_red', 'hsv_after_blue', 'avg_rgbhsv', 'avg_after_img', 'label', 'img_path'])
    mass_data = res[['avg_rgbhsv', 'avg_after_img', 'label']]
    train(mass_data['avg_rgbhsv'], mass_data['label'])
