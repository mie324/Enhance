import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from read_pgm import *
import skimage.transform as skt

if __name__ == '__main__':
    imageyale = read_pgm("CroppedYale\yaleB01\yaleB01_P00A+000E+00.pgm")
    plt.subplot(3,1,1)
    plt.imshow(imageyale, plt.cm.gray)
    print(imageyale.shape)

    plt.subplot(3,1,2)
    imageyaleresized = skt.resize(imageyale, (112,92))
    plt.imshow(imageyaleresized, plt.cm.gray)
    print(imageyaleresized.shape)

    plt.subplot(3,1,3)
    imageyaleresized = skt.resize(imageyale, (67,55))
    plt.imshow(imageyaleresized, plt.cm.gray)
    print(imageyaleresized.shape)

    plt.show()

