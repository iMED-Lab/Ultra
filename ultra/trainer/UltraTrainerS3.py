# -*- coding: utf-8 -*-
import pydoc
import numpy as np
import torch
from typing import Tuple, Union, List
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from ultra.core.losses.pixel_local_connectivitty_loss import PixelLocalConnectivityLoss
from ultra.utilities.to_neighbor_connectivity import to_nk_maps
from ultra.core.models.ultra_network import Ultra

torch.set_float32_matmul_precision('high')


class UltraTrainerS3(nnUNetTrainer):
    # the neighborhood scale S
    S = 3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-2
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.enable_deep_supervision = True

        self.nk_kernels = [2 * (s + 1) + 1 for s in range(self.S)]
        self.nk_masking = False
        self.w1 = 1.0
        self.w2 = 1.0
        self.lambda_plc = 1.0
        self.loss_s1, self.plc = self._build_plc_loss()
        self.print_to_log_file([], also_print_to_console=False, add_timestamp=False)

    @classmethod
    def build_network_architecture(cls,
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   is_training: bool = True) -> nn.Module:
        neighborhood_scale = cls.S
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        n_stages = architecture_kwargs['n_stages'] - 1

        coarse_model = nnUNetTrainer.build_network_architecture(
            architecture_class_name=architecture_class_name,
            arch_init_kwargs=arch_init_kwargs,
            arch_init_kwargs_req_import=arch_init_kwargs_req_import,
            num_input_channels=num_input_channels,
            num_output_channels=1,
            enable_deep_supervision=enable_deep_supervision
        )
        coarse_model.decoder.deep_supervision = True
        network = Ultra(
            coarse_model=coarse_model,
            coarse_out_channels=1,
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            num_pool=n_stages,
            neighbor_scale=neighborhood_scale,
            hidden_dims=[32, 32, 32],
            dropout_rate=0.0,
            deep_supervision=enable_deep_supervision,
            is_training=is_training,
        )
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.coarse_model.decoder.deep_supervision = True
        mod.deep_supervision = enabled
        mod.is_training = enabled

    def _build_plc_loss(self):
        loss_bin = DC_and_BCE_loss(
            {},
            {'batch_dice': self.configuration_manager.batch_dice,
             'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        plc_loss = PixelLocalConnectivityLoss(
            {},
            {'batch_dice': self.configuration_manager.batch_dice,
             'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
            weight_ce=1.0,
            weight_dice=1.0,
            use_ignore_label=self.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss,
            nk_masking=self.nk_masking
        )
        if self._do_i_compile():
            loss_bin.dc = torch.compile(loss_bin.dc)
            plc_loss.dc = torch.compile(plc_loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss_bin = DeepSupervisionWrapper(loss_bin, weights)
            plc_loss = DeepSupervisionWrapper(plc_loss, weights)

        return loss_bin, plc_loss

    def train_step(self, batch: dict) -> dict:
        """
        Our training step based on nnUNetV2 framework.
        """
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            bin_target = [(i > 0).float() for i in target]  # for stage 1
            bin_target = [i.to(self.device, non_blocking=True) for i in bin_target]
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            bin_target = (target > 0).float()
            bin_target = bin_target.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

        nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
        nk_maps = [i.to(self.device, non_blocking=True) for i in nk_maps]
        nk_masks = [i.to(self.device, non_blocking=True) for i in nk_masks]

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            coarse, output, nk = self.network(data)
            target = target[:len(output)] if isinstance(target, list) else target
            nk_maps = nk_maps[:len(output)] if isinstance(nk_maps, list) else nk_maps
            nk_masks = nk_masks[:len(output)] if isinstance(nk_masks, list) else nk_masks

            l1 = self.loss_s1(coarse, bin_target)
            l2 = self.loss(output, target)
            l_plc = self.plc(nk, nk_maps, nk_masks)
            l = self.w1 * l1 + self.w2 * l2 + self.lambda_plc * l_plc

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            bin_target = [(i > 0).float() for i in target]  # for stage 1
        else:
            target = target.to(self.device, non_blocking=True)
            bin_target = (target > 0).float()

        nk_maps, nk_masks = to_nk_maps(target, kernel_sizes=self.nk_kernels)  # [B, total_neighbors, H*W]
        nk_maps = [i.to(self.device, non_blocking=True) for i in nk_maps]
        nk_masks = [i.to(self.device, non_blocking=True) for i in nk_masks]

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            coarse, output, nk = self.network(data)
            target = target[:len(output)] if isinstance(target, list) else target
            nk_maps = nk_maps[:len(output)] if isinstance(nk_maps, list) else nk_maps
            nk_masks = nk_masks[:len(output)] if isinstance(nk_masks, list) else nk_masks

            l1 = self.loss_s1(coarse, bin_target)
            l2 = self.loss(output, target)
            l_plc = self.plc(nk, nk_maps, nk_masks)
            l = self.w1 * l1 + self.w2 * l2 + self.lambda_plc * l_plc

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
