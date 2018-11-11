import os
import fnmatch
import numpy
from read_pgm import read_pgm
from skimage.transform import resize
import scipy.misc


'''
For future use change path to folders and initial index values. 
The folders must be made before the code is executed

'''

c = 0
index1 = 1
for path,dirs,files in os.walk('.'):
    for f in fnmatch.filter(files,'*.pgm'):
        fullname = os.path.abspath(os.path.join(path,f))
        p = os.path.join(path,f)
        try:
            image = read_pgm(p)
            img_resized = resize(image, (int(112 * 0.60), int(92 * 0.60)))
            filename = str(index1) + '.png'
            path_to_orig = os.path.join('Original_images_ATT', filename )
            path_to_resized = os.path.join('Resized_images_ATT', filename )
            scipy.misc.imsave(path_to_orig, image)
            scipy.misc.imsave(path_to_resized, img_resized)
            print(numpy.shape(img_resized))
            index1 += 1
            c += 1
        except:
            pass


print(c)



