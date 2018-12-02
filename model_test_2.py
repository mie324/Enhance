import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


torch.manual_seed(0)



class G(nn.Module):
    def __init__(self, blocks=16, upsample_factor=2): 
        super(G, self).__init__()

        '''Number of residual blocks to be added'''
        self.blocks = blocks

        '''Our upsampling factor, upscale both of the picture's dimensions by this amount'''
        self.upsample_factor = upsample_factor

        '''Blow up the 1 channel grayscale into 20 channels'''
        self.conv1 = nn.Conv2d(1, 20, 9, stride=1, padding=4)

        ''' Add residual blocks, which learn the features we use to construct the image'''
        for i in range(self.blocks):
            self.add_module('residual_block' + str(i+1), residual())

        '''Final convolutions'''
        self.conv2 = nn.Conv2d(20, 20, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(20)

        ''' Upsample the image'''
        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample_layer' + str(i+1), upsampleBlock(20, 80))

        self.conv3 = nn.Conv2d(20, 1, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample_layer' + str(i+1))(x)

        x = self.conv3(x)
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        '''Standard discrminator architecture'''
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        '''1 to 1 channel outputs learn the features'''
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        '''Swish between each layer'''
        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

def swish(x):
    return x * F.sigmoid(x)


class residual(nn.Module):
    def __init__(self, in_channels=20, k=3, n=20, s=1):
        super(residual, self).__init__()

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
