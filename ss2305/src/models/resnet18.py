"""
Created on Sun Feb 21 2021
@author: ynomura
"""

import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_first_resblock=False):
        super(ResidualBlock, self).__init__()
        self.is_ch_changed = (in_channels != out_channels)

        if self.is_ch_changed:
            if is_first_resblock:
                stride = 1
            else:
                stride = 2
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            stride = 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                               stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                               stride=1)

    def forward(self, x):
        shortcut = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.is_ch_changed:
            shortcut = self.shortcut(x)

        out += shortcut
        return out


class ResNet18(nn.Module):

    def __init__(self, in_channels=3, out_channels=2):

        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 64)
        
        self.res_block3 = ResidualBlock(64, 128)
        self.res_block4 = ResidualBlock(128, 128)

        self.res_block5 = ResidualBlock(128, 256)
        self.res_block6 = ResidualBlock(256, 256)

        self.res_block7 = ResidualBlock(256, 512)
        self.res_block8 = ResidualBlock(512, 512)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_channels)
        self.drop_out_fc = nn.Dropout(0.5)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.res_block6(out)
        out = self.res_block7(out)
        out = self.res_block8(out)        
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop_out_fc(out)
        #out = F.softmax(out, dim=1)

        return out