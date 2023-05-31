#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :pose_utils.py
#@Date        :2022/11/29 14:58:27
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
from torch.nn import functional as F


def soft_argmax(cfg, heatmaps, num_joints):
    depth_dim = heatmaps.shape[1] // num_joints
    H_heatmaps = heatmaps.shape[2]
    W_heatmaps = heatmaps.shape[3]
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim * H_heatmaps * W_heatmaps))
    heatmaps = F.softmax(heatmaps, 2)
    confidence, _ = torch.max(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim, H_heatmaps, W_heatmaps))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(cfg.heatmap_size[1]).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(cfg.heatmap_size[0]).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(cfg.heatmap_size[2]).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out, confidence


def decode_volume(cfg, heatmaps, center3d, cam_intr):
    hm_pred = heatmaps.clone()
    hm_pred[:, :, 0] *= (cfg.image_size[1] // cfg.heatmap_size[1])
    hm_pred[:, :, 1] *= (cfg.image_size[0] // cfg.heatmap_size[0])
    hm_pred[:, :, 2] = (hm_pred[:, :, 2] / cfg.heatmap_size[2] * 2 - 1) * cfg.depth_dim + center3d[:, [2]]

    fx = cam_intr[:, 0, 0].unsqueeze(1)
    fy = cam_intr[:, 1, 1].unsqueeze(1)
    cx = cam_intr[:, 0, 2].unsqueeze(1)
    cy = cam_intr[:, 1, 2].unsqueeze(1)

    cam_x = ((hm_pred[:, :, 0] - cx) / fx * hm_pred[:, :, 2]).unsqueeze(2)
    cam_y = ((hm_pred[:, :, 1] - cy) / fy * hm_pred[:, :, 2]).unsqueeze(2)
    cam_z = hm_pred[:, :, [2]]
    cam_coords = torch.cat([cam_x, cam_y, cam_z], 2)

    return cam_coords


def decode_volume_abs(cfg, heatmaps, cam_intr):
    # please refer to the paper "Hand Pose Estimation via Latent 2.5D Heatmap Regression" for more details.
    norm_coords = heatmaps.clone()
    norm_coords[:, :, 0] *= (cfg.image_size[1] // cfg.heatmap_size[1])
    norm_coords[:, :, 1] *= (cfg.image_size[0] // cfg.heatmap_size[0])
    norm_coords[:, :, 2] = (norm_coords[:, :, 2] / cfg.heatmap_size[2] * 2 - 1) * cfg.depth_dim

    fx, fy = cam_intr[:, 0, 0], cam_intr[:, 1, 1]
    cx, cy = cam_intr[:, 0, 2], cam_intr[:, 1, 2]

    x_n, x_m = (norm_coords[:, 3, 0] - cx) / fx, (norm_coords[:, 2, 0] - cx) / fx
    y_n, y_m = (norm_coords[:, 3, 1] - cy) / fy, (norm_coords[:, 2, 1] - cy) / fy
    z_n, z_m = norm_coords[:, 3, 2], norm_coords[:, 2, 2]

    a = (x_n - x_m) ** 2 + (y_n - y_m) ** 2
    b = 2 * (x_n - x_m) * (x_n * z_n - x_m * z_m) + 2 * (y_n - y_m) * (y_n * z_n - y_m * z_m)
    c = (x_n * z_n - x_m * z_m) ** 2 + (y_n * z_n - y_m * z_m) ** 2 + (z_n - z_m) ** 2 - cfg.norm_factor ** 2

    z_root = 0.5 * (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (a + 1e-7)

    norm_coords[:, :, 2] += z_root.unsqueeze(1)
    cam_x = ((norm_coords[:, :, 0] - cx.unsqueeze(1)) / fx.unsqueeze(1) * norm_coords[:, :, 2]).unsqueeze(2)
    cam_y = ((norm_coords[:, :, 1] - cy.unsqueeze(1)) / fy.unsqueeze(1) * norm_coords[:, :, 2]).unsqueeze(2)
    cam_z = norm_coords[:, :, [2]]
    cam_coords = torch.cat([cam_x, cam_y, cam_z], 2)

    bone_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    pred_bones = torch.zeros((cam_coords.shape[0], 20)).to(cam_coords.device)
    for index, pair in enumerate(bone_pairs):
        pred_bones[:, index] = torch.norm(cam_coords[:, pair[0]] - cam_coords[:, pair[1]])
    
    if 'obman' in cfg.trainset_3d:
        bone_mean = torch.Tensor([0.04926945, 0.02837802, 0.02505871, 0.03195906, 0.0977657, 0.03123845, 0.02152403, 0.02244521, 0.10214221, 0.02953061, 0.02272312, 0.02512852, 0.09391599, 0.02677647, 0.02259798, 0.02372275, 0.08817818, 0.01826516, 0.01797429, 0.01902172]).to(cam_coords.device)
    else:
        bone_mean = torch.Tensor([0.03919412, 0.03161546, 0.02788814, 0.03607267, 0.0928468, 0.03420997, 0.02304366, 0.02415902, 0.09689835, 0.03286654, 0.02411255, 0.02707138, 0.08777174, 0.0301717, 0.02593414, 0.02469868, 0.08324047, 0.02167141, 0.0196476 , 0.02105321]).to(cam_coords.device)
    
    optim_scale = torch.sum(pred_bones * bone_mean, 1) / (torch.sum(pred_bones ** 2, 1) + 1e-7)
    cam_coords = cam_coords * optim_scale

    return cam_coords