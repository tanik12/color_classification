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
import pickle

def train(x_train, y_train, model_dirpath):
    x_train = x_train.values
    y_train = y_train.values

    init_train = np.zeros((63, 6))
    for idx, chunk in enumerate(x_train):
        init_train[idx] = chunk
    
    x_train = init_train

    #print(x_train.shape, y_train.shape)
    #print(x_train[0], y_train[0])

    model = SVC(kernel='rbf', gamma=0.001)

    # モデルの学習。fit関数で行う。
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

    dir_check(model_dirpath)
    
    with open(model_dirpath + "/model.pickle", mode='wb') as fp:
        print("start seva model")
        pickle.dump(model, fp)
        print("Model was successfully saved!")

def dir_check(model_dirpath):
    if os.path.exists(model_dirpath):
        print("Directory exists to save model file!!!")
    else:
        print("Directory did not exist to save model file...")
        print("Make directory to save model file...")
        os.mkdir(model_dirpath)
        
        print("Made directory to save model file!!!")

def load_model(model_dirpath):
    try:
        with open(model_dirpath + "/model.pickle", mode='rb') as fp:
            clf = pickle.load(fp)
            return clf
    except FileNotFoundError as e:
        print("Do not exist model file! Please make model file.", e)
        sys.exit()

def inference(x_train, y_train, model_dirpath):
    x_train = x_train.reshape(1, -1)
    y_train = y_train.reshape(1, -1)

    clf = load_model(model_dirpath)
    pred = clf.predict(x_train)

    print("予想ラベル出力: ", pred)

if __name__ == "__main__":
    current_path = os.getcwd()
    model_dirpath = current_path + "/model"
    
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

    res = pd.DataFrame(res_data, 
                       columns=['hsv_after_red', 'hsv_after_blue', 'hsv_after_green','hsv_after_yellow',
                                'avg_rgbhsv', 'avg_after_img', 'hist_color', 'label', 'img_path']
                        )

    mass_data = res[['avg_rgbhsv', 'avg_after_img', 'label']]
    train(mass_data['avg_rgbhsv'], mass_data['label'], model_dirpath)

    test_x = np.array([64.4052, 85.112, 87.6772, 102.0968, 64.3176, 89.7904])
    test_y = np.array([0])
    inference(test_x, test_y,  model_dirpath)
