import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from model.encoder import MobileNetV2_encoder, UNet_encoder, UNet_decoder
from model.convRnn import ConvLSTMCell
from model.framework import Module


class MobileNetLstm(nn.Module):
    def __init__(self, dct=False):
        super(MobileNetLstm, self).__init__()
        self.encoder = MobileNetV2_encoder(dct)
        self.rnn = ConvLSTMCell(1280)
        self.linear = nn.Sequential()

        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=1280, out_features=1, bias=True))

        init.normal_(self.linear[1].weight, std=0.001)
        init.constant_(self.linear[1].bias, 0)

    def forward(self, input, hidden):
        _, feature = self.encoder(input)
        hidden = self.rnn(feature, hidden)
        output = F.adaptive_avg_pool2d(hidden[0], (1, 1))
        output = output.view(hidden[0].shape[0], -1)
        output = self.linear(output)
        return output, hidden


class LRASPPV2(nn.Module):
    """Lite R-ASPP"""
    def __init__(self, nclass=2):
        super(LRASPPV2, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(1280, 128, 1, bias=False),
                                nn.BatchNorm2d(128), nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1280, 128, 1, bias=False),
            nn.Sigmoid(),
        )

        self.project = nn.Conv2d(128, nclass, 1)
        self.shortcut = nn.Conv2d(32, nclass, 1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y):
        size = x.shape[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        x = self.project(x)
        y = self.shortcut(y)
        out = F.adaptive_avg_pool2d(y, size) + x
        return out


class MobileNetwAtt(nn.Module):
    def __init__(self, dct=False):
        super(MobileNetwAtt, self).__init__()
        self.encoder = MobileNetV2_encoder(dct)
        self.decoder = LRASPPV2()
        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=1280, out_features=1, bias=True))

        init.normal_(self.linear[1].weight, std=0.001)
        init.constant_(self.linear[1].bias, 0)

    def forward(self, input):
        low_fea, high_fea = self.encoder(input)

        mask = self.decoder(high_fea, low_fea)
        att_mask = torch.unsqueeze(torch.softmax(mask, 1)[:, 1, :, :], 1)
        out_mask = F.interpolate(mask,
                                 (input.shape[2] // 8, input.shape[3] // 8),
                                 mode='bilinear',
                                 align_corners=True)

        output = torch.sum(high_fea * att_mask, dim=(2, 3)) / (
            torch.sum(att_mask, dim=(2, 3)) + 1e-8)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output, out_mask


class UNetLike(Module):
    def __init__(self,
                 dct=False,
                 downsample=False,
                 channel=32,
                 use_lstm=True,
                 trainable={
                     'encoder': True,
                     'decoder': True,
                     'rnn': True
                 }):
        super(UNetLike, self).__init__()
        self.use_lstm = use_lstm
        self.downsample = downsample

        self.encoder = UNet_encoder(dct, channel)
        self.decoder = UNet_decoder(channel)
        self.rnn = ConvLSTMCell(channel * 4)
        self.linear = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=channel * 4, out_features=1, bias=True))

        if 'encoder' in trainable and not trainable['encoder']:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if 'decoder' in trainable and not trainable['decoder']:
            for p in self.decoder.parameters():
                p.requires_grad = False
        if 'rnn' in trainable and not trainable['rnn']:
            for p in self.rnn.parameters():
                p.requires_grad = False
            for p in self.linear.parameters():
                p.requires_grad = False

        self.init_params()

    def forward(self, input, hidden=None):
        if self.downsample:
            input = F.interpolate(input, scale_factor=1 / 8)

        if not self.use_lstm:
            hidden = None
        out1, out2, out3 = self.encoder(input)
        mask = self.decoder(out1, out2, out3)
        hidden = self.rnn(out3, hidden)

        att_mask = F.interpolate(mask, (out3.shape[2], out3.shape[3]),
                                 mode='bilinear',
                                 align_corners=True)
        att_mask = torch.unsqueeze(torch.softmax(att_mask, 1)[:, 1, :, :], 1)
        # mask = torch.softmax(mask, 1)[:, 1, :, :]

        output = torch.sum(hidden[0] * att_mask, dim=(2, 3)) / (
            torch.sum(att_mask, dim=(2, 3)) + 1e-8)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output, mask, hidden


if __name__ == "__main__":
    # import torch.utils.data as data
    # from dataset import HutSeqDataset, hut_collate_wrapper
    # from model.framework import Interaction
    # from model.loss import *
    # dataset = HutSeqDataset('dataset/HutIris-Blur', 'train', max_frame=2)
    # dataloader = data.DataLoader(
    #     dataset,
    #     1,
    #     True,
    #     collate_fn=hut_collate_wrapper,
    # )
    # criterion = SegTraceLoss()
    # model = Interaction(UNetLike(downsample=False), criterion=criterion)
    # model = model.to_device('0')

    # torch.cuda.empty_cache()
    # model.eval()
    # model.init_val()
    # for input in dataloader:
    #     # loss = model.train_epoch(input)
    #     # print(loss)
    #     with torch.no_grad():
    #         val_save = model.val_epoch(input)
    #     print(val_save.keys())
    import cv2
    import time
    from tqdm import tqdm
    import numpy as np
    model = UNetLike(downsample=False, channel=16).cuda()
    all_time = 0
    t = 100
    for _ in tqdm(range(t)):
        frame = np.random.random((3072, 4080, 1))
        start = time.time()
        frame = cv2.resize(frame, (640, 480))
        frame = torch.from_numpy(frame[np.newaxis, np.newaxis, :, :]).to(
            torch.float32).cuda()
        with torch.no_grad():
            output, mask, hidden = model(frame)
        all_time += time.time() - start
    print('{} fps'.format(t / all_time))
