import numpy as np
import matplotlib.pyplot as plt

numyale = 2447
numATT = 390
totalimages = numyale + numATT

highres = np.zeros((totalimages, 112, 92))
lowres = np.zeros((totalimages, 67, 55))

yaleHRfolder = 'HighResCroppedYale/'
yaleLRfolder = 'LowResCroppedYale/'

attHRfolder = 'Original_images_ATT/'
attLRfolder = 'Resized_images_ATT/'

for i in range(numyale):
    print('Iteration = ', i)
    highresfile = yaleHRfolder+str(i+1)+'.png'
    lowresfile = yaleLRfolder +str(i+1)+'.png'

    hrImage = plt.imread(highresfile)
    lrImage = plt.imread(lowresfile)

    hrArray = np.array(hrImage)
    lrArray = np.array(lrImage)

    highres[i] = hrArray
    lowres[i] = lrArray

for i in range(numATT):
    print('Iteration = ', numyale+i)
    highresfile = attHRfolder+str(i+1)+'.png'
    lowresfile = attLRfolder +str(i+1)+'.png'

    hrImage = plt.imread(highresfile)
    lrImage = plt.imread(lowresfile)

    hrArray = np.array(hrImage)
    lrArray = np.array(lrImage)

    highres[numyale+i] = hrArray
    lowres[numyale+i] = lrArray

print('OurHighRes =', highres[totalimages-1])
print('OurLowRes =',lowres[totalimages-1])

np.save("imagedataset/highresimages.npy", highres)
np.save("imagedataset/lowresimages.npy", lowres)
