import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from read_pgm import *
import skimage.transform as skt

importantpath = "CroppedYale/yaleB"

for i in range(40):
    path = importantpath + stri(i) + '/'
    print(path)