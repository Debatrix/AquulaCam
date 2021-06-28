import torch
from torch.functional import chain_matmul
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision.models.shufflenetv2 import channel_shuffle


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class MobileNetV2_encoder(nn.Module):
    def __init__(self, dct=False, pretrained=True):
        super(MobileNetV2_encoder, self).__init__()
        self.dct = dct
        model = torchvision.models.mobilenet_v2(pretrained)
        self.feature = model.features
        if dct:
            self.feature[0][0] = nn.Conv2d(64,
                                           32,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=(1, 1),
                                           bias=False)
        else:
            self.feature[0][0] = nn.Conv2d(1,
                                           32,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=(1, 1),
                                           bias=False)
        self.low_feature = model.features[:5]
        self.high_feature = model.features[5:]

        init.kaiming_normal_(self.feature[0][0].weight, mode='fan_out')

    def forward(self, input):
        input = F.interpolate(input,
                              scale_factor=1 / 8,
                              recompute_scale_factor=True)
        out1 = self.low_feature(input)
        out2 = self.high_feature(out1)
        return out1, out2


class Resnet18_encoder(nn.Module):
    def __init__(self, dct=False, downsample=True, pretrained=True):
        super(Resnet18_encoder, self).__init__()
        self.dct = dct
        self.downsample = downsample
        model = torchvision.models.resnet18(pretrained)
        if not dct:
            self.conv1 = nn.Conv2d(1,
                                   64,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(3, 3),
                                   bias=False)
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, input):
        if self.downsample:
            input = F.interpolate(input, scale_factor=1 / 8)
        out1 = self.relu(self.conv1(input))
        out2 = self.layer1(self.maxpool(out1))
        out2 = self.layer2(out2)
        out3 = self.layer4(self.layer3(out2))
        return out1, out2, out3


class UNet_encoder(nn.Module):
    def __init__(self, dct=False, channel=32):
        super(UNet_encoder, self).__init__()
        filters = [channel, channel * 2, channel * 4]
        in_ch = 64 if dct else 1

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

    def forward(self, input):

        out1 = self.Conv1(input)

        out2 = self.Maxpool1(out1)
        out2 = self.Conv2(out2)

        out3 = self.Maxpool2(out2)
        out3 = self.Conv3(out3)
        return out1, out2, out3


class UNet_decoder(nn.Module):
    def __init__(self, channel=32):
        super(UNet_decoder, self).__init__()

        filters = [channel, channel * 2, channel * 4]
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], 2, 1, 1, 0)

    def forward(self, out1, out2, out3):
        dout3 = self.Up3(out3)
        dout3 = F.interpolate(dout3, (out2.shape[2], out2.shape[3]),
                              mode='bilinear',
                              align_corners=True)
        dout3 = torch.cat((out2, dout3), dim=1)
        dout3 = self.Up_conv3(dout3)

        dout2 = self.Up2(dout3)
        dout2 = F.interpolate(dout2, (out1.shape[2], out1.shape[3]),
                              mode='bilinear',
                              align_corners=True)
        dout2 = torch.cat((out1, dout2), dim=1)
        dout2 = self.Up_conv2(dout2)

        dout = self.Conv(dout2)

        return dout
