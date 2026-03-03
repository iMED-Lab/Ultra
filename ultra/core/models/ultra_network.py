# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class StackedConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 num_layers=2,
                 kernel_size=3,
                 activation=nn.LeakyReLU,
                 norm_layer=nn.InstanceNorm2d):
        super(StackedConvBlock, self).__init__()
        self.conv_ops = nn.ModuleList()
        in_dim = in_channels
        out_dim = out_channels
        for i in range(num_layers):
            self.conv_ops.append(nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size // 2))
            self.conv_ops.append(norm_layer(out_dim))
            self.conv_ops.append(activation(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        for layer in self.conv_ops:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class SPACE(nn.Module):
    """
    SPACE: Structural Prior Auxiliary Contextual Encoding layer
    """

    def __init__(self, f1_channels, f2_channels, output_channels, num_layers=2, kernel_size=3):
        super(SPACE, self).__init__()
        self.layers = StackedConvBlock(
            in_channels=f1_channels + f2_channels,
            out_channels=output_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=nn.LeakyReLU,
            norm_layer=nn.InstanceNorm2d
        )

    def forward(self, x1, x2):  # x1: coarse stage segmentation, x2: fine stage feature map
        x = torch.cat([x1, x2], dim=1)
        return self.layers(x)


class CpuBranch(nn.Module):
    """
    CPU: Connectivity Prediction Unit, i.e., CPU branch
    """

    def __init__(self, in_channels, kernel_size=3, hidden_dims=[32, 32, 32], dropout_rate=0.0):
        super(CpuBranch, self).__init__()
        self.k = kernel_size
        self.neighbor_dim = kernel_size * kernel_size - 1
        # We use the kxk kernel to fuse localized features around pixels.
        # The original F.unfold operation was too computationally intensive.
        self.unfold = nn.Conv2d(in_channels, in_channels, self.k, padding=self.k // 2)
        self.mlp = MLP(in_channels, self.neighbor_dim, hidden_dims, dropout_rate)

    def forward(self, x):
        b, c, h, w = x.shape
        patches = self.unfold(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # [b,h*w,c*k*k]
        conn_maps = self.mlp(patches)
        conn_maps = conn_maps.permute(0, 2, 1).contiguous().view(b, self.neighbor_dim, h, w)
        return torch.sigmoid(conn_maps)


class CPU(nn.Module):
    """
    Connectivity Prediction Union
    Input: feature tensor x
    Output: connectivity patterns(maps)
    """

    def __init__(self, channels, kernel_sizes=[3, 5, 7], hidden_dims=[32, 32], dropout_rate=0.0):
        super(CPU, self).__init__()
        self.num_experts = len(kernel_sizes)
        self.cpu_branches = nn.ModuleList()
        for k in kernel_sizes:
            self.cpu_branches.append(
                CpuBranch(in_channels=channels, kernel_size=k, hidden_dims=hidden_dims,
                          dropout_rate=dropout_rate))

    def forward(self, x):
        # 　Compute connectivity patterns according to the neighborhood kernel sizes
        conn_maps = []
        for cpu in self.cpu_branches:
            conn_maps.append(cpu(x))
        return conn_maps


class NICER(nn.Module):
    """
    NICER: Neighborhood Connectivity Regularization Layer
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list = [3, 5, 7],
                 hidden_dims: list = [32, 32],
                 dropout_rate: float = 0.0,
                 activation: nn.Module = nn.LeakyReLU,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 reduction: int = 4):
        super(NICER, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_neighbors = sum([k * k - 1 for k in kernel_sizes])
        self.num_cpu_branches = len(kernel_sizes)
        self.base_block = StackedConvBlock(in_channels, out_channels,
                                           2, 3, activation, norm_layer)
        self.cpu_branches = CPU(out_channels, kernel_sizes, hidden_dims, dropout_rate)
        self.gamma_beta_op = nn.Sequential(nn.Conv2d(self.total_neighbors, out_channels, 1),
                                           activation(inplace=True),
                                           nn.Conv2d(out_channels, out_channels * 2, 1))

    def forward(self, x):
        x_base = self.base_block(x)
        b, c, h, w = x_base.shape

        # The CPU stream
        conn_maps = self.cpu_branches(x_base)
        conn_maps = torch.cat(conn_maps, dim=1)

        conn_feats = self.gamma_beta_op(conn_maps)
        gamma, beta = torch.split(conn_feats, self.out_channels, dim=1)
        x_base = gamma * x_base + beta

        return x_base, conn_maps.view(b, self.total_neighbors, -1)  # [b, total_neighbors, h*w]


class RefineNet(nn.Module):
    MAX_FILTERS = 480
    BASE_FILTERS = 32

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 coarse_out_channels: int,
                 num_pool: int,
                 reg_kernels: list = [3, 5, 7],
                 hidden_dims: list = [32, 32, 32],
                 dropout_rate: float = 0.0):
        super(RefineNet, self).__init__()
        self.max_filters = self.MAX_FILTERS
        self.base_filters = self.BASE_FILTERS
        self.kernel_sizes = reg_kernels
        self.hidden_dims = hidden_dims

        self.encoders = []
        self.decoders = []
        self.down_layers = []
        self.up_layers = []
        self.seg_outputs = []

        output_features = self.base_filters
        input_features = in_channels

        self.filters_per_stage = []
        for d in range(num_pool):
            self.encoders.append(
                SPACE(coarse_out_channels, input_features, output_features, 2, 3))
            self.down_layers.append(nn.MaxPool2d(2, 2))
            self.filters_per_stage.append(output_features)
            input_features = output_features
            output_features = min(int(output_features * 2), self.max_filters)

        final_features = output_features
        # There is no need to perform the regularization in the bottleneck layer.
        self.bottleneck = StackedConvBlock(input_features, final_features, 3, 3)

        self.filters_per_stage = self.filters_per_stage[::-1]  # reverse the list
        for u in range(num_pool):
            features_from_down = final_features
            features_from_skip = self.filters_per_stage[u]
            features_after_concat = features_from_skip * 2
            self.up_layers.append(nn.ConvTranspose2d(features_from_down, features_from_skip, 2, 2))
            self.decoders.append(
                NICER(features_after_concat, features_from_skip, reg_kernels, hidden_dims, dropout_rate))
            final_features = features_from_skip
            # we concat the coarse segmentation mask with the refined segmentation mask.
            self.seg_outputs.append(nn.Conv2d(final_features, num_classes, 1, 1, 0, 1, 1))

        self.fusion_outputs = nn.Conv2d(num_classes * len(self.seg_outputs), num_classes, 1, 1, 0, 1, 1)

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x, coarse_masks):
        skips = []
        seg_outputs = []
        nk_outputs = []
        valid_masks = []

        for i in range(len(self.encoders)):
            x = self.encoders[i](x, coarse_masks[i])
            skips.append(x)
            valid_masks.append(coarse_masks[i])
            x = self.down_layers[i](x)

        x = self.bottleneck(x)

        for i in range(len(self.decoders)):
            x = self.up_layers[i](x)
            x = torch.cat([x, skips[-i - 1]], dim=1)
            x, nk = self.decoders[i](x)
            nk_outputs.append(nk)
            seg_outputs.append(self.seg_outputs[i](x))

        fusion_seg = [F.interpolate(seg, size=x.shape[2:], mode="bilinear", align_corners=False)
                      for seg in seg_outputs]
        fusion_seg = self.fusion_outputs(torch.cat(fusion_seg, dim=1))
        seg_outputs[-1] = fusion_seg

        seg_outputs, nk_outputs = seg_outputs[::-1], nk_outputs[::-1]
        return seg_outputs, nk_outputs


class Ultra(nn.Module):
    """
        Ultra: Expert-Guided Feature Learning and Neighborhood Connectivity Modeling
         for Retinal Artery-Vein Segmentation
        The coarse stage outputs the binary mask, whatever the refinement mask is
    """

    def __init__(self,
                 coarse_model: nn.Module,
                 coarse_out_channels: int,
                 in_channels: int,
                 num_classes: int,
                 num_pool: int,
                 neighbor_scale: int = 3,
                 hidden_dims: list = [32, 32, 32],
                 dropout_rate: float = 0.0,
                 deep_supervision: bool = True,
                 is_training: bool = False):
        super(Ultra, self).__init__()
        self.deep_supervision = deep_supervision
        self.S = neighbor_scale
        self.reg_kernels = [2 * (s + 1) + 1 for s in range(self.S)]
        self.is_training = is_training

        self.coarse_model = coarse_model
        self.refine_model = RefineNet(in_channels=in_channels,
                                      num_classes=num_classes,
                                      coarse_out_channels=coarse_out_channels,
                                      num_pool=num_pool,
                                      reg_kernels=self.reg_kernels,
                                      hidden_dims=hidden_dims,
                                      dropout_rate=dropout_rate)

        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        coarse_masks = self.coarse_model(x)  # nnUNet outputs

        refine_masks, nk_maps = self.refine_model(x, coarse_masks)

        if self.deep_supervision:
            seg_results, nk_results = refine_masks, nk_maps
        else:
            seg_results, nk_results = refine_masks[0], nk_maps[0]

        if self.is_training:
            return coarse_masks, seg_results, nk_results
        else:
            return seg_results
