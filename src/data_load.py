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
def detect_red_color(img, hsv):
    tmp = np.array([])

    # 赤色のHSVの値域1
    hsv_min = np.array([0,144,153])
    hsv_max = np.array([8,194,252])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

# 青色の検出
def detect_blue_color(img, hsv):
    tmp = np.array([])

    # 青色のHSVの値域1
    hsv_min = np.array([84, 163, 108])
    hsv_max = np.array([104, 203, 155])

    # 青色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = masked_img.T[0].flatten().mean()
    g = masked_img.T[1].flatten().mean()
    r = masked_img.T[2].flatten().mean()

    tmp = np.append(tmp, r)
    tmp = np.append(tmp, g)
    tmp = np.append(tmp, b)

    return mask, masked_img, tmp

def color_info(img_path):
    color_arr = np.array([])         

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img =  cv2.resize(img,(200, 200)) #あとで直す

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得 
    b = img.T[0].flatten().mean()
    g = img.T[1].flatten().mean()
    r = img.T[2].flatten().mean()
    
    # BGRからHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSV平均値を取得
    # flattenで一次元化しmeanで平均を取得 
    h = hsv.T[0].flatten().mean()
    s = hsv.T[1].flatten().mean()
    v = hsv.T[2].flatten().mean()

    color_arr = np.append(color_arr, r)
    color_arr = np.append(color_arr, g)
    color_arr = np.append(color_arr, b)
    color_arr = np.append(color_arr, h)
    color_arr = np.append(color_arr, s)
    color_arr = np.append(color_arr, v)

    return color_arr, hsv, img 

def extract_color_info(data_list):
    tmpA = []
    for item in data_list:
        img_pathes = item["img_pathes"]
        img_label = item["label"]
        for img_path in img_pathes:
            tmpB = []
            
            color_arr, hsv, img = color_info(img_path)

            red_mask, red_masked_img, avg_red_masked_img = detect_red_color(img, hsv)
            blue_mask, bule_masked_img, avg_blue_masked_img = detect_blue_color(img, hsv)
            
            sum_array = avg_red_masked_img + avg_blue_masked_img   

            tmpB.append(red_masked_img)
            tmpB.append(bule_masked_img)
            tmpB.append(color_arr)
            tmpB.append(sum_array)
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
