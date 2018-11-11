from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import *



def run():
    torch.multiprocessing.freeze_support()
    print('loop')



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_GAN(lr):

    ######

    # 3.4 YOUR CODE HERE
    netG = G()
    loss_fnc = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    ######

    return netG, loss_fnc, optimizerG

def load_DISC(lr):

    netD = D()
    loss_fnc = torch.nn.BCELoss()
    optimizerD = optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    return netD, loss_fnc, optimizerD

if __name__ == '__main__':
    run()
    batchSize = 64
    imageSize = 64
    lr = 0.0002
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                           0.5)), ])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)

    netG, criterion, optimizerG = load_GAN(lr=lr)
    netG.apply(weights_init)
    netD, criterion, optimizerD = load_DISC(lr=lr)
    netD.apply(weights_init)

    for epoch in range(25):
        print('Epoch', epoch)
        for i, data in enumerate(dataloader, 0):
            print('step', i)
            netD.zero_grad()

            real, _ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)

            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))

            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)