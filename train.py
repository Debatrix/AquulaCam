from pickle import FALSE
import re
from scipy import stats
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from base_train import train
from util import LoadConfig
from dataset import HutSeqDataset, hut_collate_wrapper
from model.network import UNetLike
from model.framework import Interaction
from model.loss import *


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()
        self.info = ""
        self.log_name = "UNetLike"

        self.dataset_path = 'dataset/HutIris-Blur'
        self.cp_path = "checkpoints/0305_230737_UNetLike/27_1.1311e-01.pth"
        self.visible = True
        self.log_interval = 10
        self.save_interval = -1
        self.less_data = False
        self.debug = False

        self.model_channel = 16
        self.move = ['random', 'random']
        self.frames = 8
        self.use_dct = False
        self.use_lstm = True
        self.mask_down = 8
        self.loss = ['l1', 'ce', 100]

        self.load_to_ram = False
        self.batchsize = 4
        self.device = [2, 0, 1, 3]
        self.num_workers = 0
        self.seed = np.random.randint(9999)

        self.max_epochs = 500
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.apply()


def get_dataloaders(config):
    train_data = HutSeqDataset(path=config['dataset_path'],
                               mode='train',
                               less_data=config['less_data'],
                               topk=0.05,
                               fb_move=config['move'][0],
                               lrud_move=config['move'][1],
                               max_fb_move=0.3,
                               max_lrud_move=(500, 500),
                               max_frame=config['frames'],
                               mask_down=config['mask_down'],
                               load_to_ram=config['load_to_ram'])
    train_data_loader = DataLoader(train_data,
                                   config['batchsize'],
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=hut_collate_wrapper,
                                   num_workers=config['num_workers'])
    val_data = HutSeqDataset(path=config['dataset_path'],
                             mode='val',
                             less_data=config['less_data'],
                             topk=0.05,
                             fb_move=config['move'][0],
                             lrud_move=config['move'][1],
                             max_fb_move=0.2,
                             max_lrud_move=(500, 500),
                             max_frame=config['frames'],
                             mask_down=config['mask_down'],
                             load_to_ram=config['load_to_ram'])
    val_data_loader = DataLoader(val_data,
                                 len(config['device']),
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,
                                 collate_fn=hut_collate_wrapper,
                                 num_workers=config['num_workers'])
    return train_data_loader, val_data_loader


def evaluation(val_save, val_num):
    pred_loss = val_save['pred_loss'] / val_num
    mask_loss = val_save['mask_loss'] / val_num
    position = np.concatenate(val_save['position'], axis=0)
    position = position[:, 1:] - position[:, :-1]
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
    image = val_save['image'][idx]
    mask = val_save['mask'][idx] + image[:, 0, :, :]
    pred_mask = torch.softmax(val_save['pred_mask'][idx],
                              1)[:, 1, :, :] + image[:, 0, :, :]
    images = torch.cat((image, image, image), dim=1)
    mask = torch.stack((pred_mask, mask, image[:, 0, :, :]), dim=1)
    log_writer.add_images('Val/steps', torch.clamp(images, 0, 1), epoch)
    log_writer.add_images('Val/mask', torch.clamp(mask, 0, 1), epoch)
    offset = val_save['offset'][-1][idx]
    position = val_save['position'][-1][idx]
    info = 'offset: ' + str(offset) + '\nposition: ' + str(position)
    log_writer.add_text('Val/steps', info, epoch)


if __name__ == "__main__":
    # set config
    config = Config()

    # data
    print('Loading Data')
    dataloaders = get_dataloaders(config)

    # model and criterion
    criterion = SegTraceLoss(config['loss'])
    model = UNetLike(config['use_dct'],
                     channel=config['model_channel'],
                     use_lstm=config['use_lstm'],
                     trainable={
                         'encoder': True,
                         'decoder': True,
                         'rnn': True
                     })
    model = Interaction(model, criterion)

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