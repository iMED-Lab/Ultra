# -*- coding: utf-8 -*-
import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class PixelLocalConnectivityLoss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, nk_masking: bool = False):
        """
        Implemented based on nnUNet's compound loss function: DC_and_BCE_loss
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/compound_losses.py
        """
        super(PixelLocalConnectivityLoss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'
        if nk_masking:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        self.nk_masking = nk_masking

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, nk_mask: torch.Tensor = None):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)

        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        elif self.nk_masking:
            if nk_mask is not None:
                ce_loss = (self.ce(net_output, target_regions) * nk_mask).sum() / torch.clip(nk_mask.sum(), min=1e-8)
            else:
                raise ValueError('nk_masking is True but nk_mask is None')
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
