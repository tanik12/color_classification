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

# 赤色の検出
def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,144,153])
    hsv_max = np.array([8,194,252])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

# 青色の検出
def detect_blue_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSVの値域1
    hsv_min = np.array([84, 163, 108])
    hsv_max = np.array([104, 203, 155])

    # 青色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

def extract_color_info(data_list):
    tmpA = []
    for item in data_list:
        img_pathes = item["img_pathes"]
        img_label = item["label"]
        for img_path in img_pathes:
            tmpB = []
            
            img = cv2.imread(img_path)
            img =  cv2.resize(img,(200, 200)) #あとで直す

            red_mask, red_masked_img = detect_red_color(img)
            blue_mask, bule_masked_img = detect_blue_color(img)
            
            tmpB.append(red_masked_img)
            tmpB.append(bule_masked_img)
            tmpB.append(img_label)
            tmpB.append(img_path)
            tmpA.append(tmpB)
    return tmpA

if __name__ == "__main__":
    label_dict = {"pedestrian_signs_blue":0, "pedestrian_signs_red":1, "vehicle_signal_blue":2, "vehicle_signal_red":3, "vehicle_signal_yellow":4}

    data_dir = "data/sample_trim/*"
    current_dir = os.chdir('../')
    current_dir = os.getcwd()

    target_pathes = os.path.join(current_dir, data_dir)
    path_list = glob.glob(target_pathes)

    data_list = data_load(path_list, label_dict)
    res_data = extract_color_info(data_list)

    print(res_data[3])
    cv2.imwrite("red_masked_img.png", res_data[0][0])
    cv2.imwrite("bule_masked_img.png", res_data[0][1])
