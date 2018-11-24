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
from model_test_2 import *
import torch.utils.data as data_utils
from dataset import ImgDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision


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
    netG = G()
    loss_fnc = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    ######

    return netG, loss_fnc, optimizerG


def load_DISC(lr):
    netD = D()
    loss_fnc = torch.nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    return netD, loss_fnc, optimizerD


def load_data(batch_size, training_set_feat, training_set_labels, validation_set_feat, validation_set_labels):
    train_dataset = ImgDataset(training_set_feat, training_set_labels)
    validation_dataset = ImgDataset(validation_set_feat, validation_set_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run()
    batchSize = 64
    imageSize = 64
    lr = 0.0002
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                           0.5)), ])  # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
    netG, criterion, optimizerG = load_GAN(lr=lr)
    netG = netG.to(device)
    netG.apply(weights_init)

    netD, criterion, optimizerD = load_DISC(lr=lr)
    netD = netD.to(device)
    netD.apply(weights_init)


    #feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    #feature_extractor = feature_extractor.to(device)

    content_criterion = nn.MSELoss()
    adverserial_criterion = nn.BCELoss()

    ''' Get numpy arrays into one array, split amongst training and validation, pass into 
    load _data function and create dataloader classes for training and validation'''

    labels = np.load('./imagedataset/highresimages.npy')
    features = np.load('./imagedataset/lowresimages.npy')


    seed = 0
    training_set_feat, validation_set_feat, training_set_labels, validation_set_labels \
        = train_test_split(features, labels, test_size=0.2, random_state=seed)

    training_loader, validation_loader = load_data(batchSize, training_set_feat, training_set_labels,
                                         validation_set_feat, validation_set_labels)

    corr = 0
    corr2 = 0
    for epoch in range(200):
        print('Epoch', epoch)
        for i, batch in enumerate(training_loader):

            #Training Discriminator

            netD.zero_grad()
            lowres, real = batch

            real = Variable(real)
            ones_const = Variable(torch.ones(real.size()[0]))
            real = real.unsqueeze(1)

            real = real.to(device)
            output = netD(real.float())
            output = output.view(-1)

            corr2 += int(sum(output > 0.5))

            ones_const = ones_const.to(device)
            errD_real = criterion(output, ones_const)

            noise = Variable(lowres)
            noise = noise.unsqueeze(1)

            noise = noise.to(device)
            fake = netG(noise.float())

            zero_const = Variable(torch.zeros(real.size()[0]))
            zero_const = zero_const.to(device)

            output = netD(fake.detach())
            output = output.view(-1)

            corr += int(sum(output<0.5))
            errD_fake = criterion(output, zero_const)
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()


            #Training Generator
            netG.zero_grad()
            
            real = real.to(device)
            fake = fake.to(device)

            generator_content_loss = content_criterion(fake, real.float())

            output = netD(fake.detach())
            output = output.view(-1)

            ones_const = Variable(torch.ones(real.size()[0]))
            ones_const =  ones_const.to(device)

            generator_adversarial_loss = adverserial_criterion(output, ones_const.float())
            errG = generator_content_loss + 1e-3 * generator_adversarial_loss

            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
            epoch, 25, i, len(training_loader), errD.data[0], errG.data[0]))

        print("Epoch Ended")
        print("Fake image accuracy(Discriminator)", corr/len(training_loader.dataset))

        print("Real Image Accuracy (Discriminator)" , corr2/len(training_loader.dataset))

        print("Combined Accuracy (Discriminator)", ((corr + corr2)/(2*len(training_loader.dataset))))

        corr = 0
        corr2 = 0


        vutils.save_image(real[0], '%s/real_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)
        fake = netG(noise.float())
        vutils.save_image(fake[0], '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)
