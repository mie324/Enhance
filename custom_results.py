import torch
import numpy as np
from matplotlib import pyplot as plt
from dataset import ImgDataset
import torchvision.utils as vutils
from torch.autograd import Variable
import argparse
from skimage.transform import resize
from skimage import io
from skimage.measure import compare_psnr

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', type=str, default='SaadFace.png')
    parser.add_argument('--model', type=str, default = 'Generator.pt')
    args = parser.parse_args()

    image = io.imread(args.imagepath, as_grey= True)

    lowres = resize(image, (56, 46))
    real = resize(image, (112, 92))
    interpolated = resize(lowres, (112,92))

    newname = args.imagepath[0:-4]
    path_name = newname+'Visualization.png'

    Generator = torch.load(args.model, map_location=device)
    Generator = Generator.to(device)
    lowres = torch.from_numpy(lowres)
    myimage = Variable(lowres)
    myimage = myimage.to(device)
    myimage = myimage.unsqueeze(dim=0)
    myimage = myimage.unsqueeze(dim=0)
    fake = Generator(myimage.float())
    fake = fake.squeeze(0)
    fake = fake.squeeze(0)

    fake = np.array(fake.detach())
    print(np.shape(fake))
    print(np.shape(real))

    psnrint = compare_psnr(real, interpolated)
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

    fig.savefig(path_name, dpi=fig.dpi)

