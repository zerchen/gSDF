#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :sdf_utils.py
#@Date        :2022/12/02 22:31:25
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
from torch.nn import functional as F
from .transform import homoify, dehomoify


def pixel_align(cfg, input_xyz_points, num_points_per_scene, feature_maps, hand_center_3d, cam_intr):
    input_points = input_xyz_points.clone()
    input_points = input_points.reshape((-1, num_points_per_scene, 3))
    batch_size = input_points.shape[0]
    xyz = input_points * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)
    homo_xyz = homoify(xyz)
    homo_xyz_2d = torch.matmul(cam_intr, homo_xyz.transpose(1, 2)).transpose(1, 2)
    xyz_2d = (homo_xyz_2d[:, :, :2] / homo_xyz_2d[:, :, [2]]).unsqueeze(2)
    uv_2d = xyz_2d / cfg.image_size[0] * 2 - 1
    sample_feat = torch.nn.functional.grid_sample(feature_maps, uv_2d, align_corners=True)[:, :, :, 0].transpose(1, 2)
    uv_2d = uv_2d.squeeze(2).reshape((-1, 2))
    sample_feat = sample_feat.reshape((uv_2d.shape[0], -1))
    validity = (uv_2d[:, 0] >= -1.0) & (uv_2d[:, 0] <= 1.0) & (uv_2d[:, 1] >= -1.0) & (uv_2d[:, 1] <= 1.0)
    validity = validity.unsqueeze(1)

    if cfg.with_add_feats:
        depth_feat = xyz.reshape((-1, 3))[:, [-1]]
        view_dir_feat = F.normalize(xyz.reshape((-1, 3)), p=2, dim=1)
        sample_feat = torch.cat([sample_feat, depth_feat, view_dir_feat], axis=1)

    return sample_feat, validity


def kinematic_embedding(cfg, input_points, num_points_per_scene, pose_results, mode):
    if 'hand' in mode:
        assert cfg.hand_point_latent in [6, 21, 36, 51], 'please set a right hand embedding size'
    else:
        assert cfg.obj_point_latent in [6, 9, 69, 72], 'please set a right object embedding size'

    input_points = input_points.reshape((-1, num_points_per_scene, 3))
    batch_size = input_points.shape[0]
    try:
        inv_func = torch.linalg.inv
    except:
        inv_func = torch.inverse
    
    if 'hand' in mode:
        xyz = (input_points * 2 / cfg.recon_scale).unsqueeze(2)
        global_trans = pose_results['global_trans']

        xyz_mano = xyz.unsqueeze(2)
        homo_xyz_mano = homoify(xyz_mano)
        
        # inverse the hand global transformation
        inv_global_trans = inv_func(global_trans).unsqueeze(1)
        inv_homo_xyz_mano = torch.matmul(inv_global_trans, homo_xyz_mano.transpose(3, 4)).transpose(3, 4)
        inv_homo_xyz_mano = inv_homo_xyz_mano.squeeze(3)
        inv_xyz_mano = dehomoify(inv_homo_xyz_mano)
        if cfg.hand_point_latent == 6:
            inv_xyz_mano = inv_xyz_mano[:, :, [0], :]

        if cfg.hand_point_latent == 21:
            inv_xyz_mano = torch.cat([inv_xyz_mano[:, :, [0], :], inv_xyz_mano[:, :, [1, 4, 7, 10, 13], :]], dim=2)
        
        if cfg.hand_point_latent == 36:
            inv_xyz_mano = torch.cat([inv_xyz_mano[:, :, [0], :], inv_xyz_mano[:, :, [1, 4, 7, 10, 13], :], inv_xyz_mano[:, :, [2, 5, 8, 11, 14], :]], dim=2)
        
        if cfg.hand_point_latent == 51:
            inv_xyz_mano = torch.cat([inv_xyz_mano[:, :, [0], :], inv_xyz_mano[:, :, [1, 4, 7, 10, 13], :], inv_xyz_mano[:, :, [2, 5, 8, 11, 14], :], inv_xyz_mano[:, :, [3, 6, 9, 12, 15], :]], dim=2)
        
        point_embedding = torch.cat([xyz_mano.squeeze(2), inv_xyz_mano], 2)
        point_embedding = point_embedding.reshape((batch_size, num_points_per_scene, -1))
        point_embedding = point_embedding * cfg.recon_scale / 2
    else:
        xyz = (input_points * 2 / cfg.recon_scale)
        obj_trans = pose_results['global_trans']
        homo_xyz_obj = homoify(xyz)
        inv_obj_trans = inv_func(obj_trans)
        inv_homo_xyz_obj = torch.matmul(inv_obj_trans, homo_xyz_obj.transpose(2, 1)).transpose(2, 1)
        inv_xyz_obj = dehomoify(inv_homo_xyz_obj)
        try:
            hand_trans = pose_results['wrist_trans']
            xyz_mano = xyz
            homo_xyz_mano = homoify(xyz_mano)
            inv_hand_trans = inv_func(hand_trans)
        except:
            pass

        if cfg.obj_point_latent == 6:
            point_embedding = torch.cat([xyz, inv_xyz_obj], 2)
        
        if cfg.obj_point_latent == 9:
            inv_homo_xyz_mano = torch.matmul(inv_hand_trans, homo_xyz_mano.transpose(1, 2)).transpose(1, 2)
            inv_xyz_mano = dehomoify(inv_homo_xyz_mano)
            point_embedding = torch.cat([xyz, inv_xyz_obj, inv_xyz_mano], 2)

        if cfg.obj_point_latent == 69:
            inv_xyz_joint = [xyz, inv_xyz_obj]
            for i in range(21):
                inv_xyz_joint.append(xyz - pose_results['joints'][:, [i], :])
            point_embedding = torch.cat(inv_xyz_joint, 2)
        
        if cfg.obj_point_latent == 72:
            inv_homo_xyz_mano = torch.matmul(inv_hand_trans, homo_xyz_mano.transpose(1, 2)).transpose(1, 2)
            inv_xyz_mano = dehomoify(inv_homo_xyz_mano)
            inv_xyz_joint = [xyz, inv_xyz_obj, inv_xyz_mano]
            for i in range(21):
                inv_xyz_joint.append(xyz - pose_results['joints'][:, [i], :])
            point_embedding = torch.cat(inv_xyz_joint, 2)
                
        point_embedding = point_embedding.reshape((batch_size, num_points_per_scene, -1))
        point_embedding = point_embedding * cfg.recon_scale / 2

    return point_embedding