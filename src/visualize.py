import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import *


def test_value():
    hsv = np.random.rand(10, 6)
    
    label_arr = np.array([])    
    for i in range(10):
        label = randint(2)
        label_arr = np.append(label_arr, label)

    return hsv, label_arr

def plot(color_info):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)
    
    #軸にラベルを付けたいときは書く
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation Chroma")
    ax.set_zlabel("Value Brightness")
   
    for row, info in enumerate(color_info):
        label = info[3]
        
        if label == 0 or label == 2:
            ax.scatter(info[2][3], info[2][4], info[2][5], s = 40, c = "blue")
        elif label == 1 or label == 3:
            ax.scatter(info[2][3], info[2][4], info[2][5], s = 40, c = "red")
        else:
            ax.scatter(info[2][3], info[2][4], info[2][5], s = 40, c = "yellow")

    plt.show()

if __name__ == "__main__":
    test_val, test_label = test_value()
    plot(test_val, test_label)
