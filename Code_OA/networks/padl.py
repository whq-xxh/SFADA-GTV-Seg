# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC and https://github.com/Merrical/PADL
Reference paper: https://arxiv.org/pdf/2111.13410.pdf
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from torch.distributions import Normal, Independent


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=False):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return x


class PADL_Module(nn.Module):

    def __init__(self, in_channels=16,
                 num_classes=2, rater_num=6):
        super(PADL_Module, self).__init__()
        self.rater_num = rater_num
        self.num_classes = num_classes

        self.global_mu_head = nn.Conv2d(in_channels, self.num_classes, 1)
        self.global_sigma_head_reduction = nn.Sequential(
            nn.Conv2d(in_channels, self.num_classes, 1),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(),
        )
        self.global_sigma_head_output = nn.Conv2d(
            self.num_classes * 2, self.num_classes, 1)

        self.rater_residual_heads_reduction = list()
        self.rater_residual_heads_output = list()
        self.rater_sigma_heads_reduction = list()
        self.rater_sigma_heads_output = list()
        for i in range(self.rater_num):
            self.rater_residual_heads_reduction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.num_classes, 1),
                    nn.BatchNorm2d(self.num_classes),
                    nn.ReLU(),
                )
            )
            self.rater_residual_heads_output.append(
                nn.Conv2d(self.num_classes * 2, self.num_classes, 1)
            )

            self.rater_sigma_heads_reduction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.num_classes, 1),
                    nn.BatchNorm2d(self.num_classes),
                    nn.ReLU(),
                )
            )
            self.rater_sigma_heads_output.append(
                nn.Conv2d(self.num_classes * 2, self.num_classes, 1)
            )

        self.rater_residual_heads_reduction = nn.ModuleList(
            self.rater_residual_heads_reduction)
        self.rater_residual_heads_output = nn.ModuleList(
            self.rater_residual_heads_output)
        self.rater_sigma_heads_reduction = nn.ModuleList(
            self.rater_sigma_heads_reduction)
        self.rater_sigma_heads_output = nn.ModuleList(
            self.rater_sigma_heads_output)

    def forward(self, head_input, training=True):
        global_mu = self.global_mu_head(head_input)
        global_mu_sigmoid = torch.sigmoid(global_mu)
        global_entropy_map = -global_mu_sigmoid * torch.log2(global_mu_sigmoid+1e-6) - \
            (1 - global_mu_sigmoid) * torch.log2(1 - global_mu_sigmoid+1e-6)
        global_entropy_map = global_entropy_map.detach()
        global_sigma_reduction = self.global_sigma_head_reduction(head_input)
        global_sigma_input = (1 + global_entropy_map) * global_sigma_reduction
        global_sigma_input = torch.cat([global_sigma_input, global_mu], dim=1)
        global_sigma = self.global_sigma_head_output(global_sigma_input)
        global_sigma = torch.abs(global_sigma)

        rater_residual_reduction_list = [self.rater_residual_heads_reduction[i](
            head_input) for i in range(self.rater_num)]
        rater_residual_input = [
            (1 + global_entropy_map) * rater_residual_reduction_list[i] for i in range(self.rater_num)]
        rater_residual_input = [torch.cat(
            [rater_residual_input[i], global_mu], dim=1) for i in range(self.rater_num)]
        rater_residual = [self.rater_residual_heads_output[i](
            rater_residual_input[i]) for i in range(self.rater_num)]

        rater_mu = [rater_residual[i] +
                    global_mu for i in range(self.rater_num)]

        rater_sigma_reduction_list = [self.rater_sigma_heads_reduction[i](
            head_input) for i in range(self.rater_num)]
        rater_sigma_input = [(1 + global_entropy_map) *
                             rater_sigma_reduction_list[i] for i in range(self.rater_num)]
        rater_sigma_input = [torch.cat(
            [rater_sigma_input[i], rater_mu[i]], dim=1) for i in range(self.rater_num)]
        rater_sigma = [self.rater_sigma_heads_output[i](
            rater_sigma_input[i]) for i in range(self.rater_num)]

        rater_sigmas = torch.stack(rater_sigma, dim=0)
        rater_sigmas = torch.abs(rater_sigmas)
        rater_mus = torch.stack(rater_mu, dim=0)
        rater_residuals = torch.stack(rater_residual, dim=0)
        rater_dists = list()
        for i in range(self.rater_num):
            rater_dists.append(Independent(
                Normal(loc=rater_mus[i], scale=rater_sigmas[i], validate_args=False), 1))
        global_dist = Independent(
            Normal(loc=global_mu, scale=global_sigma, validate_args=False), 1)

        if training:
            rater_samples = [dist.rsample() for dist in rater_dists]
            rater_samples = torch.stack(rater_samples, dim=0)
            global_samples = global_dist.rsample([self.rater_num])
        else:
            rater_samples = [dist.sample() for dist in rater_dists]
            rater_samples = torch.stack(rater_samples, dim=0)
            global_samples = global_dist.sample([self.rater_num])

        return global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals

    def close(self):
        for sf in self.sfs:
            sf.remove()


class UNet_PADL(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_PADL, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.bnout = nn.BatchNorm2d(params["feature_chns"][0])
        self.padl = PADL_Module(
            in_channels=params["feature_chns"][0], num_classes=params["class_num"], rater_num=3)

    def forward(self, x, training=True):
        feature = self.encoder(x)
        output = self.decoder(feature)
        head_input = self.bnout(output)
        global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = self.padl(
            head_input, training)
        return [global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals]


# unet_padl = UNet_PADL(1, 2).cuda()

# inputs = torch.rand((1, 1, 256, 256)).cuda()

# for x in unet_padl(inputs):
#     print(x.shape)
