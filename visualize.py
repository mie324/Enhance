import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.measure import compare_psnr
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=42)
    args = parser.parse_args()

    path_real = 'results/real_samples_epoch_%03d.png'%(args.epoch)
    path_fake = 'results/fake_samples_epoch_%03d.png'%(args.epoch)
    path_lowres = 'results/lowres_samples_epoch_%03d.png'%(args.epoch)
    path_interpolated = 'results/interpolated_samples_epoch_%03d.png'%(args.epoch)

    real = np.array(io.imread(path_real, as_grey= True))
    fake = np.array(io.imread(path_fake, as_grey= True))
    lowres = np.array(io.imread(path_lowres, as_grey= True))
    interpolated = np.array(io.imread(path_interpolated, as_grey= True))

    psnrint = compare_psnr(real, interpolated/255)
    psnrgen = compare_psnr(real, fake)

    fig = plt.figure()

    plt.subplot(1,4,1)
    plt.imshow(lowres, cmap = 'gray')
    plt.title('Low resolution image')

    plt.subplot(1,4,2)
    plt.imshow(fake, cmap = 'gray')
    plt.title('Generated image')
    plt.xlabel('PSNR value = %0.3f'%(psnrgen))

    plt.subplot(1,4,3)
    plt.imshow(interpolated, cmap = 'gray')
    plt.title('Interpolated image')
    plt.xlabel('PSNR value = %0.3f'%(psnrint))

    plt.subplot(1,4,4)
    plt.imshow(real, cmap = 'gray')
    plt.title('High resolution image (Ground Truth)')

    plt.show()

    fig.savefig('Visualization%03d.png'%(args.epoch), dpi=fig.dpi)