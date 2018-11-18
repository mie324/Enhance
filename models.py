from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F




def swish(x):
    return x * F.sigmoid(x)



class residualBlock(nn.Module):
    def __init__(self, in_channels=16, k=3, n=16, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class G(nn.Module):
    def __init__(self, n_residual_blocks=1, upsample_factor=2):
        super(G, self).__init__()

        self.conv1 = nn.ConvTranspose2d(1, 16, 3, stride=1, padding=4)
        self.conv2 = nn.ConvTranspose2d(16, 16, 5, stride=2, padding=4)
        self.conv3 = nn.Conv2d(16, 1, (6,2), stride=1, padding=0)


    def forward(self, x):
        x = swish(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)

        #return(self.conv5(x))
        return x



























    '''
    
    
        def __init__(self, n_residual_blocks=4, upsample_factor=2):
        super(G, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 16, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        for i in range(int(self.upsample_factor / 2)):
            self.add_module('upsample' + str(i + 1), upsampleBlock(16, 64))

        self.conv3 = nn.Conv2d(16, 1, 9, stride=1, padding=0)
        self.conv4 = nn.Conv2d(1, 1, 9, stride=1, padding=0)
        self.conv5 = nn.Conv2d(1, 1, (7, 3), stride=1, padding=0)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor / 2)):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        #return x
        x = self.conv3(x)
        x = self.conv4(x)
        return(self.conv5(x))

    
    
    
    '''




    '''def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 1, 12, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 12, 2, 1, bias=False),

            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output'''


'''
            nn.ConvTranspose2d(1, 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()



'''


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()


        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, (3, 3), 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), bias = False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, (2, 1), bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        output = self.main(input)
        #output = self.conv1(input)

        return output