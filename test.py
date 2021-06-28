import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util import *
from dataset import HutSeqDataset, hut_collate_wrapper
from model.network import UNetLike
from model.framework import Interaction
from model.loss import *


class Config(LoadConfig):
    def __init__(self) -> None:
        super(Config, self).__init__()

        self.dataset_path = 'dataset/HutIris-Blur'
        self.cp_path = "checkpoints/0308_160732_UNetLike/90_1.8029e-01.pth"
        self.test_name = 'test_UNetLike'

        self.model_channel = 16
        self.move = ['None', 'None']
        self.frames = 15
        self.use_dct = False
        self.use_lstm = False
        self.mask_down = 8

        self.less_data = True
        self.load_to_ram = False
        self.device = [2]
        self.num_workers = 0
        self.seed = np.random.randint(9999)

        self.apply()


class Teng(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1,
                                out_channels=2,
                                kernel_size=3,
                                stride=1,
                                padding=0,
                                bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0],
                           [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0],
                           [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.eval()

    def forward(self, img, mask):
        B, F, C, H, W = img.shape
        img = img.reshape(-1, C, H, W) * 255
        # mask = mask.reshape(-1, H, W)[:, 1:-1, 1:-1]
        mask = torch.softmax(mask.reshape(-1, 2, H, W), dim=1)
        mask = mask[:, 1, 1:-1, 1:-1]
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1)
        score = (x * mask).sum(-1).sum(-1) / mask.sum(-1).sum(-1)
        score = score.reshape(B, F)
        return score


def fmeasure(img, mask):
    """
    Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.
    """
    B, F, C, H, W = img.shape
    img = np.round(img.reshape(-1, H, W).numpy() * 255).astype(np.double)
    mask = torch.softmax(mask.reshape(-1, 2, H, W), dim=1)
    mask = (mask[:, 1, :, :] > mask[:, 0, :, :]).numpy().astype(np.double)
    fm_score = np.zeros((B * F))
    for i in range(img.shape[0]):
        gaussianX = cv2.Sobel(img[i], cv2.CV_64F, 1, 0)
        gaussianY = cv2.Sobel(img[i], cv2.CV_64F, 0, 1)
        fm_score[i] = (
            (gaussianX**2 + gaussianY**2) * mask[i]).sum() / mask[i].sum()
    fm_score = fm_score.reshape(B, F)

    return fm_score


if __name__ == "__main__":
    # set config
    config = Config()
    test_name = config['test_name']

    # data
    print('Loading Data')
    test_data = HutSeqDataset(path=config['dataset_path'],
                              mode='test',
                              less_data=config['less_data'],
                              topk=0.05,
                              fb_move=config['move'][0],
                              lrud_move=config['move'][1],
                              max_fb_move=0.1,
                              max_lrud_move=(100, 50),
                              max_frame=config['frames'],
                              mask_down=config['mask_down'],
                              load_to_ram=config['load_to_ram'])
    test_data_loader = DataLoader(test_data,
                                  1,
                                  shuffle=True,
                                  drop_last=False,
                                  pin_memory=False,
                                  collate_fn=hut_collate_wrapper,
                                  num_workers=config['num_workers'])

    # model and criterion
    criterion = SegTraceLoss()
    model = UNetLike(config['use_dct'],
                     channel=config['model_channel'],
                     use_lstm=config['use_lstm'])
    model = Interaction(model, criterion)
    teng = Teng()

    # configure
    # print(config)
    set_random_seed(config['seed'])
    cp_config = model.load_checkpoint(config['cp_path'])
    model.to_device(config['device'])

    # test
    model.eval()
    model.init_val()
    score = []
    with torch.no_grad():
        for test_data in tqdm(test_data_loader):
            test_save = model.val_epoch(test_data)
            score.append(fmeasure(test_save['image'], test_save['pred_mask']))
    score = np.concatenate(score, 0)
    ele_list = np.concatenate(test_save['position'], 0) * 530
    ele_list = np.round(ele_list).astype(np.int)

    idx = 0
    image = test_save['image'][idx]
    mask = test_save['mask'][idx] + image[:, 0, :, :]
    pred_mask = torch.softmax(test_save['pred_mask'][idx], dim=1)
    pred_mask = pred_mask[:, 1, :, :] + image[:, 0, :, :]
    mask = torch.stack((image[:, 0, :, :], mask, pred_mask), dim=-1)
    image = np.clip(torch.squeeze(image).cpu().numpy() * 255, 0,
                    255).astype(np.uint8)
    mask = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)

    print('Saving Result')
    cp_dir_path = os.path.normcase(os.path.join('checkpoints', test_name))
    mkdir(cp_dir_path)
    # np.savez(os.path.join(cp_dir_path, 'result.npz'),
    #          fm_list=score,
    #          ele_list=ele_list)
    for i in tqdm(range(mask.shape[0])):
        name = '_{}_{}_{}.png'.format(i, ele_list[-1, i], int(score[-1, i]))
        cv2.imwrite(os.path.join(cp_dir_path, 'mask' + name), mask[i])
        cv2.imwrite(os.path.join(cp_dir_path, 'image' + name), image[i])
