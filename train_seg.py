import os
from scipy import stats
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from base_train import train
from util import LoadConfig
from dataset import HUTDataset
from model.network import UNetLike
from model.framework import Segmentation
from model.loss import *


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = ""
        self.train_name = "UNetLike_enhance_Segmentation"

        self.dataset_path = 'dataset/HutIris-Blur'
        self.cp_path = "checkpoints/1030_194858_UNetLike/80_2.2903e-03.pth"
        self.cp_path = ""
        self.visible = True
        self.log_interval = 5
        self.save_interval = 5
        self.less_data = False
        self.debug = False

        self.use_dct = False
        self.model_channel = 16
        self.mask_down = 8

        self.load_to_ram = False
        self.batchsize = 32
        self.device = [0, 1, 2, 3]
        self.num_workers = 0
        self.seed = np.random.randint(9999)

        self.max_epochs = 500
        self.lr = 8e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.apply()


def get_dataloaders(config):
    train_data = HUTDataset(path=config['dataset_path'],
                            mode='train',
                            less_data=config['less_data'],
                            lrud_move=True)
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=config['num_workers'])
    val_data = HUTDataset(path=config['dataset_path'],
                          mode='val',
                          less_data=False)
    val_data_loader = DataLoader(val_data,
                                 config['batchsize'],
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=config['num_workers'])
    return train_data_loader, val_data_loader


def evaluation(val_save, val_num):
    pred_loss = val_save['pred_loss'] / val_num
    mask_loss = val_save['mask_loss'] / val_num
    position = np.concatenate(val_save['position'], axis=0)
    offset = np.concatenate(val_save['offset'], axis=0)
    srocc = stats.spearmanr(offset.reshape(-1), position.reshape(-1))[0]
    lcc = stats.pearsonr(offset.reshape(-1), position.reshape(-1))[0]
    return {
        "Val_pred_loss": pred_loss,
        "Val_mask_loss": mask_loss,
        "SROCC": srocc,
        "LCC": lcc
    }


def val_plot(log_writer, epoch, val_save):
    idx = torch.randint(val_save['image'].shape[0], (1, )).item()
    image = nn.functional.interpolate(
        val_save['image'],
        (val_save['mask'][idx].shape[-2], val_save['mask'][idx].shape[-1]),
        mode='bilinear',
        align_corners=True)
    mask = val_save['mask'][idx] + image[idx]
    heatmap = val_save['heatmap'][idx] + image[idx]
    image = torch.clamp(torch.cat((heatmap, mask, image[idx]), dim=0), 0, 1)
    log_writer.add_image('Val/image', image, epoch)


if __name__ == "__main__":
    # set config
    config = Config()

    # data
    print('Loading Data')
    dataloaders = get_dataloaders(config)

    # model and criterion
    criterion = SegmentationLoss()
    model = UNetLike(dct=config['use_dct'],
                     channel=config['model_channel'],
                     downsample=False,
                     trainable={
                         'encoder': True,
                         'decoder': True,
                         'rnn': False
                     })
    model = Segmentation(model, criterion)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.5,
        patience=config['log_interval'] * 2,
        verbose=True)
    optimizers = (optimizer, scheduler)

    # train
    train(config, dataloaders, model, optimizers, evaluation, val_plot)
