from skimage.transform import resize
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os


start_path = 'results/'
counter = 0
for filename in os.listdir(start_path):
    if filename.startswith('real'):
        image_path = os.path.join(start_path, filename)
        image = io.imread(image_path, as_grey=True)
        print(image.shape)
        lowres_image = resize(image, (56,46))
        upscaled_image = resize(lowres_image, (112,92))
        print(filename)
        print('Low res image size', lowres_image.shape)
        print('Upscaled image size', upscaled_image.shape)
        print(start_path+'lowres_samples_epoch_%03d.png'%(counter))
        sp.misc.imsave(start_path+'lowres_samples_epoch_%03d.png'%(counter), lowres_image)
        sp.misc.imsave(start_path+'interpolated_samples_epoch_%03d.png'%(counter), upscaled_image)
        print(counter)
        counter += 1