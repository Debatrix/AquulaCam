from model.framework import Segmentation
import torch
import torch.nn as nn


class TraceLoss(nn.Module):
    def __init__(self, type='l2') -> None:
        super(TraceLoss, self).__init__()
        self.type = type
        if type == 'l1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, offset, positions, movement):
        target = -1 * (positions[:, :-1] + movement[:, 1:, -1])
        loss = self.loss(target, offset)
        return loss


class SegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super(SegmentationLoss, self).__init__()
        self.pred_loss = nn.MSELoss()
        self.mask_loss = nn.SmoothL1Loss()

    def forward(self, offset, positions, mask, out_mask):
        pred_loss = self.pred_loss(offset, positions)
        mask_loss = self.mask_loss(out_mask, mask)
        return mask_loss, {
            'pred_loss': pred_loss.cpu().item(),
            'mask_loss': mask_loss.cpu().item()
        }


class SegTraceLoss(nn.Module):
    def __init__(self, type=['l1', 'ce', 1]) -> None:
        super(SegTraceLoss, self).__init__()
        self.type = type
        if type[0] == 'l1':
            self.pred_loss = nn.L1Loss()
        elif type[0] == 'smooth_l1':
            self.pred_loss = nn.SmoothL1Loss()
        else:
            self.pred_loss = nn.MSELoss()
        if type[1] == 'mse':
            self.mask_loss = nn.MSELoss()
        elif type[1] == 'smooth_l1':
            self.mask_loss = nn.SmoothL1Loss()
        else:
            self.mask_loss = nn.CrossEntropyLoss()

    def forward(self, offset, positions, movement, mask, out_mask):
        target = -1 * (positions[:, :-1] + movement[:, 1:, -1])
        if self.type[1] == 'ce':
            _, _, C, H, W = out_mask.shape
            out_mask = out_mask.reshape((-1, C, H, W))
            mask = mask.reshape((-1, H, W))
        pred_loss = self.pred_loss(target, offset)
        mask_loss = self.mask_loss(out_mask, mask)
        return pred_loss + self.type[-1] * mask_loss, {
            'pred_loss': pred_loss.cpu().item(),
            'mask_loss': mask_loss.cpu().item()
        }