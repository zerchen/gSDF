#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :model.py
#@Date        :2022/04/09 16:48:16
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn import functional as F
from config import cfg
from networks.backbones.resnet import ResNetBackbone
from networks.necks.unet import UNet
from networks.heads.sdf_head import SDFHead
from networks.heads.mano_head import ManoHead
from networks.heads.fc_head import FCHead
from networks.heads.conv_head import ConvHead
from mano.inverse_kinematics import ik_solver_mano
from mano.rodrigues_layer import batch_rodrigues
from mano.rot6d import compute_rotation_matrix_from_ortho6d
from loss.ordinal_loss import HandOrdLoss, SceneOrdLoss
from utils.pose_utils import soft_argmax, decode_volume, decode_volume_abs


class model(nn.Module):
    def __init__(self, cfg, backbone, neck, volume_head, obj_rot_head):
        super(model, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.dim_backbone_feat = 2048 if cfg.backbone == 'resnet_50' else 512
        self.neck = neck
        self.volume_head = volume_head
        self.obj_rot_head = obj_rot_head

        self.loss_l2 = torch.nn.MSELoss()
        self.loss_hand_ord = HandOrdLoss()
        self.loss_scene_ord = SceneOrdLoss(cfg.obj_rot)
    
    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            backbone_feat = self.backbone(input_img)

            hm_feat = self.neck(backbone_feat)
            hm_pred = self.volume_head(hm_feat)
            num_joints = 22 if cfg.hand_branch and cfg.obj_branch else 21
            hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints)
            volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])

            if self.obj_rot_head is not None:
                rot_feat = self.obj_rot_head(backbone_feat.mean(3).mean(2))
                if cfg.rot_style == 'axisang':
                    obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                elif cfg.rot_style == '6d':
                    obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
            
            if cfg.hand_branch:
                hand_pose_results = {}
                hand_pose_results['volume_joints'] = volume_joint_preds[:, :21]
            else:
                hand_pose_results = None
            
            if cfg.obj_branch:
                obj_pose_results = {}
                obj_pose_results['center'] = volume_joint_preds[:, 21]
                if cfg.obj_rot:
                    obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                    obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds[:, [21]]
            else:
                obj_pose_results = None
       
            loss = {}
            if cfg.hand_branch and cfg.obj_branch:
                volume_joint_targets = torch.cat([targets['hand_joints_3d'], targets['obj_center_3d'].unsqueeze(1)], dim=1)
            else:
                volume_joint_targets = targets['hand_joints_3d']
            loss['volume_joint'] = cfg.volume_weight * self.loss_l2(volume_joint_preds, volume_joint_targets)

            if cfg.obj_rot and cfg.corner_weight > 0:
                loss['obj_corners'] = cfg.corner_weight * self.loss_l2(obj_pose_results['corners'], targets['obj_corners_3d'])
            
            if cfg.hand_ordinal_weight > 0:
                loss['hand_ord'] = cfg.hand_ordinal_weight * self.loss_hand_ord(hand_pose_results['volume_joints'], targets['hand_joints_3d'])

            if cfg.scene_ordinal_weight > 0:
                if cfg.obj_rot:
                    loss['scene_ord'] = cfg.scene_ordinal_weight * self.loss_scene_ord(hand_pose_results['volume_joints'], obj_pose_results['corners'], targets['hand_joints_3d'], targets['obj_corners_3d'])
                else:
                    loss['scene_ord'] = cfg.scene_ordinal_weight * self.loss_scene_ord(hand_pose_results['volume_joints'], obj_pose_results['center'].unsqueeze(1), targets['hand_joints_3d'], targets['obj_center_3d'].unsqueeze(1))
            
            return loss, hand_pose_results, obj_pose_results
        else:
            with torch.no_grad():
                input_img = inputs['img']
                backbone_feat = self.backbone(input_img)

                hm_feat = self.neck(backbone_feat)
                hm_pred = self.volume_head(hm_feat)
                num_joints = 22 if cfg.hand_branch and cfg.obj_branch else 21
                hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints)
                if cfg.norm_coords:
                    volume_joint_preds = decode_volume_abs(cfg, hm_pred, metas['cam_intr'])
                else:
                    volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])

                if self.obj_rot_head is not None:
                    rot_feat = self.obj_rot_head(backbone_feat.mean(3).mean(2))
                    if cfg.rot_style == 'axisang':
                        obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                    elif cfg.rot_style == '6d':
                        obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)

                hand_pose_results = {}
                if cfg.hand_branch:
                    hand_pose_results = ik_solver_mano(None, volume_joint_preds[:, :21])
                    hand_pose_results['volume_joints'] = volume_joint_preds[:, :21]
                else:
                    hand_pose_results = None

                if cfg.obj_branch:
                    obj_pose_results = {}
                    obj_pose_results['center'] = volume_joint_preds[:, 21]
                    if cfg.obj_rot:
                        obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                        obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds[:, [21]]
                else:
                    obj_pose_results = None

            return hand_pose_results, obj_pose_results


def get_model(cfg, is_train):
    num_resnet_layers = int(cfg.backbone.split('_')[-1])
    backbone = ResNetBackbone(num_resnet_layers)
    if is_train:
        backbone.init_weights()

    neck_inplanes = 2048 if num_resnet_layers == 50 else 512
    neck = UNet(neck_inplanes, 256, 3)
    
    if cfg.hand_branch and cfg.obj_branch:
        volume_head = ConvHead([256, 22 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
    else:
        volume_head = ConvHead([256, 21 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
    
    if cfg.obj_rot:
        if cfg.rot_style == 'axisang':
            obj_rot_head = FCHead(out_dim=3)
        elif cfg.rot_style == '6d':
            obj_rot_head = FCHead(out_dim=6)
    else:
        obj_rot_head = None
    
    ho_model = model(cfg, backbone, neck, volume_head, obj_rot_head)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)