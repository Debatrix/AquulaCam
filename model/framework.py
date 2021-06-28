import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def init_params(self, scale=1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class base_model(nn.Module):
    def __init__(self, model, criterion):
        super(base_model, self).__init__()
        self.model = model
        self.val_save = {}
        self.criterion = criterion

    def to_device(self, device=['cpu']):
        if isinstance(device, str):
            device = [device]

        if torch.cuda.is_available() and device[0] is not 'cpu':
            # torch.cuda.set_device('cuda:{}'.format(device[0]))
            _device = torch.device('cuda:0')
            self.is_cpu = False
        else:
            if not torch.cuda.is_available():
                print("hey man, buy a GPU!")
            _device = torch.device('cpu')
            self.is_cpu = True

        self.model = self.model.to(_device)
        self.criterion = self.criterion.to(
            _device) if self.criterion is not None else None
        if len(device) > 1:
            self.model = nn.DataParallel(self.model)

        return self

    def load_checkpoint(self, cp_path=None):
        cp_config = None
        if cp_path:
            cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
            try:
                self.model.load_state_dict(cp_data['model'])
            except Exception as e:
                self.model.load_state_dict(cp_data['model'], strict=False)
                print(e)
            cp_config = '' if 'config' not in cp_data else cp_data['config']
            # print('Load checkpoint {}'.format(
            #     {x.split('=')[0]: x
            #      for x in cp_config.split('\n')}['train_name']))
        return cp_config

    def save_checkpoint(self, save_path, info=None):
        if isinstance(self.model, nn.DataParallel) or isinstance(
                self.model, DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        cp_data = dict(
            cfg=info,
            model=state_dict,
        )
        torch.save(cp_data, save_path)

    def load_state_dict(self, checkpoint):
        return self.model.load_state_dict(checkpoint)

    def _feed_data(self, input):
        raise NotImplementedError

    def init_val(self):
        raise NotImplementedError

    def train_epoch(self, input):
        raise NotImplementedError

    def val_epoch(self, input):
        raise NotImplementedError


class Single(base_model):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(Single, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['loss'] = 0
        self.val_save['positions'] = []
        self.val_save['prediction'] = []

    def _feed_data(self, input):
        if self.is_cpu:
            imgs = input[0][:, 0, :, :, :]
            positions = input[1]
        else:
            imgs = input[0][:, 0, :, :, :].cuda()
            positions = input[1].cuda()
        return imgs, positions

    def train_epoch(self, input):
        imgs, positions = self._feed_data(input)
        pred = self.model(imgs)
        return self.criterion(pred, positions)

    def val_epoch(self, input):
        imgs, positions = self._feed_data(input)
        pred = self.model(imgs)
        self.val_save['loss'] += self.criterion(pred, positions)
        self.val_save['positions'].append(positions.cpu().numpy().reshape(
            (-1)))
        self.val_save['prediction'].append(pred.cpu().numpy().reshape((-1)))
        return pred


class Interaction(base_model):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(Interaction, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['mask_loss'] = 0
        self.val_save['offset'] = []
        self.val_save['position'] = []
        self.val_save['image'] = []
        self.val_save['mask'] = []
        self.val_save['pred_mask'] = []

    def _feed_data(self, envs, offsets, positions):
        imgs, masks, positions, move = envs(offsets, positions)
        if not self.is_cpu:
            imgs = imgs.cuda()
            masks = masks.cuda()
            positions = positions.cuda()
            move = move.cuda()
        return imgs, masks, positions, move

    def train_epoch(self, input):
        envs, step = input
        pos, hidden = None, None
        positions, offset, movement = [], [], []
        masks, pred_masks = [], []
        for _ in range(envs.max_frame):
            img, mask, pos, move = self._feed_data(envs, step, pos)
            step, pred_mask, hidden = self.model(img, hidden)
            positions.append(pos)
            offset.append(step.view(-1))
            movement.append(move)
            masks.append(mask)
            pred_masks.append(pred_mask)
        positions = torch.stack(positions, dim=1)
        movement = torch.stack(movement, dim=1)
        offset = torch.stack(offset[:-1], dim=1)
        masks = torch.stack(masks, dim=1)
        pred_masks = torch.stack(pred_masks, dim=1)
        loss, _ = self.criterion(offset, positions, movement, masks,
                                 pred_masks)
        return loss

    def val_epoch(self, input):
        envs, step = input
        pos, hidden = None, None
        offset, positions, movement, images = [], [], [], []
        masks, pred_masks = [], []
        for _ in range(envs.max_frame):
            img, mask, pos, move = self._feed_data(envs, step, pos)
            step, pred_mask, hidden = self.model(img, hidden)
            offset.append(step.view(-1))
            positions.append(pos)
            movement.append(move)
            masks.append(mask)
            pred_masks.append(pred_mask)
            images.append(img)
        offset = torch.stack(offset[:-1], dim=1)
        movement = torch.stack(movement, dim=1)
        positions = torch.stack(positions, dim=1)
        images = torch.stack(images, dim=1)
        masks = torch.stack(masks, dim=1)
        pred_masks = torch.stack(pred_masks, dim=1)
        _, loss_dict = self.criterion(offset, positions, movement, masks,
                                      pred_masks)
        self.val_save['pred_loss'] += loss_dict['pred_loss']
        self.val_save['mask_loss'] += loss_dict['mask_loss']
        self.val_save['offset'].append(offset.cpu().numpy())
        self.val_save['position'].append(positions.cpu().numpy())
        self.val_save['image'] = images.cpu()
        self.val_save['mask'] = masks.cpu()
        self.val_save['pred_mask'] = pred_masks.cpu()
        return self.val_save


class Segmentation(base_model):
    def __init__(self, model, criterion=nn.MSELoss()):
        super(Segmentation, self).__init__(model, criterion)

    def init_val(self):
        self.val_save['pred_loss'] = 0
        self.val_save['mask_loss'] = 0
        self.val_save['offset'] = []
        self.val_save['position'] = []
        self.val_save['prediction'] = []
        self.val_save['image'] = []
        self.val_save['mask'] = []
        self.val_save['heatmap'] = []

    def _feed_data(self, input):
        img, mask, positions = input
        if not self.is_cpu:
            img = img.cuda()
            positions = positions.cuda()
            mask = mask.cuda()
        return img, mask, positions

    def train_epoch(self, input):
        img, mask, positions = self._feed_data(input)
        output = self.model(img)
        offset, out_mask = output[:2]
        offset = offset.view(-1)
        loss, _ = self.criterion(offset, positions, mask, out_mask)
        return loss

    def val_epoch(self, input):
        img, mask, positions = self._feed_data(input)
        output = self.model(img)
        offset, out_mask = output[:2]
        offset = offset.view(-1)
        _, loss_dict = self.criterion(offset, positions, mask, out_mask)
        self.val_save['pred_loss'] += loss_dict['pred_loss']
        self.val_save['mask_loss'] += loss_dict['mask_loss']
        self.val_save['offset'].append(offset.cpu().numpy())
        self.val_save['position'].append(positions.cpu().numpy())
        self.val_save['image'] = img.cpu()
        self.val_save['mask'] = mask.cpu()
        self.val_save['heatmap'] = out_mask.cpu()
        return self.val_save