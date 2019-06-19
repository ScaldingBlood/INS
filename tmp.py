import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
from utils import *

if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 8))
    plt.axis([0, 1008, 1009, 0])
    bgimg = img.imread('data/f7.png')
    imgplot = plt.imshow(bgimg)

    pos_xt = [291, 300, 653, 643, 291]
    pos_yt = [404, 617, 602, 393, 404]
    plt.plot(pos_xt, pos_yt, 'black', ls='--', label='Ref')
    plt.legend()
    plt.show()