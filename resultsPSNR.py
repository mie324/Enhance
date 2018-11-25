import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from skimage.measure import compare_psnr
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=25)

    args = parser.parse_args()

    start_path = 'results/'

    highres = np.zeros((args.images, 112, 92))
    interpolated = np.zeros((args.images, 112, 92))
    generated = np.zeros((args.images, 112, 92))

    hrcount = 0
    intcount = 0
    gencount = 0
    for filename in os.listdir(start_path):
        if (filename.startswith('real') and filename.endswith('%03d.png'%(hrcount))):
            if hrcount >= args.images:
                continue
            highres[hrcount] = np.array(io.imread(os.path.join(start_path, filename), as_grey = True))
            print(os.path.join(start_path, filename), hrcount)
            hrcount += 1
        elif (filename.startswith('fake') and filename.endswith('%03d.png'%(gencount))):
            if gencount >= args.images:
                continue
            generated[gencount] = np.array(io.imread(os.path.join(start_path, filename), as_grey = True))
            print(os.path.join(start_path, filename), gencount)
            gencount += 1
        elif (filename.startswith('interpolated') and filename.endswith('%03d.png'%(intcount))):
            if intcount >= args.images:
                continue
            interpolated[intcount] = np.array(io.imread(os.path.join(start_path, filename), as_grey = True))
            print(os.path.join(start_path, filename), intcount)
            intcount += 1

    psnrint = np.zeros(args.images)
    psnrgen = np.zeros(args.images)
    print(np.max(highres))
    print(np.max(interpolated)/255)
    print(np.max(generated))

    for i in range(args.images):
        psnrint[i] = compare_psnr(highres[i], interpolated[i]/255)
        psnrgen[i] = compare_psnr(highres[i], generated[i])

    fig = plt.figure()
    plt.plot(np.array([i+1 for i in range(args.images)]),psnrint)
    plt.plot(np.array([i+1 for i in range(args.images)]), psnrgen)
    plt.legend(['PSNR of Interpolated', 'PSNR of Generated'])
    plt.title('PSNR Value per Epoch for Select Sample')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (High is better image)')
    plt.show()
    fig.savefig('PSNROverEpochs.png', dpi = fig.dpi)