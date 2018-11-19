import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from read_pgm import *
import skimage.transform as skt
import os

importantpath = "C:/Users/Saad/Documents/MIEProject/Enhance/CroppedYale/yaleB"

counter = 0
for i in range(1,40):
    if (i == 14):
        continue
    if (i < 10):
        path = importantpath + '0' + str(i) + '/'
    else:
        path = importantpath + str(i) + '/'
    for filename in os.listdir(path):
        if filename.endswith(".pgm"):

            try:
                #print(os.path.join(path, filename))
                imageyale = read_pgm(os.path.join(path, filename))
                imageyalecropped = skt.resize(imageyale, (112, 92))
                imageyalelowres = skt.resize(imageyale, (56, 46))

                counter += 1

                highrespath = 'HighResCroppedYale/'+str(counter) +'.png'
                lowrespath = 'LowResCroppedYale/'+str(counter) +'.png'
                sp.misc.imsave(highrespath, imageyalecropped)
                sp.misc.imsave(lowrespath, imageyalelowres)
                print(counter)
                #print('Cropped shape',imageyalecropped.shape)
                #print('Low res shape', imageyalelowres.shape)
            except:
                pass
            continue
        else:
            continue