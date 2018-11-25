import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.transform import resize
from skimage.measure import compare_psnr


def evaluate(lowres_images, generated_images, highres_images):
    batch_size = highres_images.shape[0]
    interpolated_images = np.zeros(highres_images.shape)

    psnrint = np.zeros(batch_size)
    psnrgen = np.zeros(batch_size)

    #perform interpolation
    for i in range(batch_size):
        interpolated_images[i] = resize(lowres_images[i], (112, 92))

    #calculate psnr of batch
    for i in range(batch_size):
        psnrint[i] = compare_psnr(highres_images[i], interpolated_images[i])
        psnrgen[i] = compare_psnr(highres_images[i], generated_images[i])

    total_psnrint = np.sum(psnrint)
    total_psnrgen = np.sum(psnrgen)

    return total_psnrgen, total_psnrint