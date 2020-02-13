import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import *

def plot(color_info):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)
    
    #軸にラベルを付けたいときは書く
    #ax.set_xlabel("Hue")
    #ax.set_ylabel("Saturation Chroma")
    #ax.set_zlabel("Value Brightness")
   
    ax.set_xlabel("red")
    ax.set_ylabel("green")
    ax.set_zlabel("blue")

    for row, info in enumerate(color_info):
        label = info[7]
        
        if label == 0 or label == 2:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "blue")
        elif label == 1 or label == 3:
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "red")
        else:
            print(info[7], row)
            ax.scatter(info[5][0], info[5][1], info[5][2], s = 10, c = "yellow")

    plt.show()

if __name__ == "__main__":
    pass
