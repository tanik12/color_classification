import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import *

#RGB情報の可視化(3D)
def plot(color_info):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)
   
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    ax.set_zlabel("blue")

    for row, info in enumerate(color_info):
        label = info[7]
        if label == 0:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "blue")
        elif label == 1:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "red")
        elif label == 2:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "yellow")
        else:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "black")

    plt.show()

#color histの作成f
def plot_hist(color_info):
    hist = color_info[10][6]
    # グラフの作成
    plt.xlim(0, 255)
    plt.plot(hist[0, :], "-r", label="Red")
    plt.plot(hist[1, :], "-g", label="Green")
    plt.plot(hist[2, :], "-b", label="Blue")
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    pass
