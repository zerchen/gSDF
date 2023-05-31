#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :sdf_head.py
#@Date        :2022/04/09 16:57:10
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        Create a two-layers networks with relu activation.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class PosEncoder(nn.Module):
    '''Module to add positional encoding.'''
    def __init__(self, in_features=3):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class SDFHead(nn.Module):
    def __init__(self, sdf_latent, point_latent, dims, dropout, dropout_prob, norm_layers, latent_in, use_cls_hand=False, num_class=0):
        super(SDFHead, self).__init__()
        self.sdf_latent = sdf_latent
        self.point_latent = point_latent
        self.dims = [self.sdf_latent + self.point_latent] + dims + [1]
        self.num_layers = len(self.dims)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.use_cls_hand = use_cls_hand
        self.num_class = num_class

        self.relu = nn.ReLU()
        self.th = nn.Tanh()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = self.dims[layer + 1] - self.dims[0]
            else:
                out_dim = self.dims[layer + 1]

            if layer in self.norm_layers:
                setattr(self, "lin" + str(layer), nn.utils.weight_norm(nn.Linear(self.dims[layer], out_dim)),)
            else:
                setattr(self, "lin" + str(layer), nn.Linear(self.dims[layer], out_dim))

            # classifier
            if self.use_cls_hand and layer == self.num_layers - 2:
                self.classifier_head = nn.Linear(self.dims[layer], self.num_class)

    def forward(self, input, pose_results=None):
        latent = input
       
        predicted_class = None
        for layer in range(0, self.num_layers - 1):
            if self.use_cls_hand and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(latent)

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                latent = torch.cat([latent, input], 1)
            latent = lin(latent)

            if layer < self.num_layers - 2:
                latent = self.relu(latent)
                if layer in self.dropout:
                    latent = F.dropout(latent, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            latent = self.th(latent)

        return latent, predicted_class


class DeformableSDFHead(nn.Module):
    def __init__(self, dims, dropout, dropout_prob, norm_layers, latent_in, positional_encoding=False, use_bone_length=False, bone_latent_size=16, num_bones=16, global_pose_projection_size=0, actv='relu'):
        super(DeformableSDFHead, self).__init__()
        self.sdf_latent = 3
        if positional_encoding:
            self.posi_encoder = PosEncoder()
            self.point_latent = 3 + 3 * 2 * 10
        else:
            self.posi_encoder = None
            self.point_latent = 3
       
        self.global_pose_projection_size = global_pose_projection_size
        if global_pose_projection_size > 0:
            for i in range(num_bones):
                setattr(self, "global_proj" + str(i), nn.Linear(self.sdf_latent * num_bones, global_pose_projection_size))
            dims = [self.point_latent + global_pose_projection_size] + dims + [64]
        else:
            dims = [self.point_latent + self.sdf_latent * num_bones] + dims + [64]
       
        if use_bone_length:
            self.bone_latent_size = bone_latent_size
            if self.bone_latent_size > 0:
                dims[0] = dims[0] + self.bone_latent_size
                self.bone_encoder = TwoLayerNet(num_bones, 40, self.bone_latent_size)

        self.use_bone_length = use_bone_length
        self.num_bones = num_bones
        self.num_layers = len(dims)
        self.dims = dims
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        if actv == "siren":
            self.actv = Sine()
        else:
            self.actv = nn.ReLU()

        self.final_layer = nn.Linear(self.dims[-1] * num_bones, 1)

        # Part model
        for bone in range(self.num_bones):
            for layer in range(0, self.num_layers - 1):
                if layer + 1 in self.latent_in:
                    out_dim = self.dims[layer + 1] - self.dims[0]
                else:
                    out_dim = self.dims[layer + 1]
                out_dim = dims[layer + 1]

                if layer in self.norm_layers:
                    setattr(self, "lin" + str(bone) + "_" + str(layer), nn.utils.weight_norm(nn.Linear(self.dims[layer], out_dim)),)
                else:
                    setattr(self, "lin" + str(bone) + "_" + str(layer), nn.Linear(self.dims[layer], out_dim))

                if actv == "siren":
                    if layer == 0:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(first_layer_sine_init)
                    else:
                        getattr(self, "lin" + str(bone) + "_" + str(layer)).apply(sine_init)

    def forward(self, xyz, pose_results):
        input = xyz.reshape([xyz.size(0), -1, 3])[:, 1:, :]

        # Positional encoding
        if self.posi_encoder is not None:
            input = self.posi_encoder(input)

        global_latent = pose_results['joints'][:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]].clone().detach()
        global_latent -= global_latent[:, [0], :]
        global_latent = global_latent.reshape([global_latent.size(0), -1])
        batch_size = global_latent.size(0)
        num_sample_points = input.size(0) // batch_size
        global_latent = global_latent.unsqueeze(1).repeat(1, num_sample_points, 1).reshape([batch_size * num_sample_points, -1])

        # Compute global bone length encoding
        if self.use_bone_length and self.bone_latent_size > 0:
            bone_latent = self.bone_encoder(bone_lengths)

        # output = torch.zeros([input.size(0), self.num_bones], device=input.device)
        last_layer_latents = []

        for bone in range(self.num_bones):
            input_i = input[:, bone, :]
            x = input[:, bone, :]

            if self.use_bone_length:
                x = torch.cat([x, bone_lengths[:, bone].unsqueeze(-1)], axis=1)
                if self.bone_latent_size > 0:
                    x = torch.cat([x, bone_latent], axis=1)

            if self.global_pose_projection_size > 0:
                global_proj = getattr(self, "global_proj" + str(bone))
                projected_global_latent = global_proj(global_latent)
                x = torch.cat([x, projected_global_latent], 1)
            else:
                x = torch.cat([x, global_latent], 1)

            for layer in range(0, self.num_layers - 1):
                x_prev = x
                lin = getattr(self, "lin" + str(bone) + "_" + str(layer))
                if layer in self.latent_in:
                    x = torch.cat([x, input_i], 1)

                x_out = lin(x)

                if layer == self.num_layers - 2:
                    last_layer_latents.append(x_out)

                if layer < self.num_layers - 1:
                    if layer > 0:
                        x_out = x_out + x_prev
                    x_out = self.actv(x_out)
                    if self.dropout is not None and layer in self.dropout:
                        x_out = F.dropout(x_out, p=self.dropout_prob, training=self.training)

                x = x_out

        output = self.final_layer(torch.cat(last_layer_latents, dim=-1))
        if hasattr(self, "th"):
            output = self.th(output)

        return output, None


if __name__ == "__main__":
    import sys
    net = SDFHead(256, 6, [512, 512, 512, 512], [0, 1, 2, 3], 0.2, [0, 1, 2, 3], [2], True, 6)
    input_size = (2, 262)
    input_tensor = torch.randn(input_size)
    latent, pred_class = net(input_tensor)