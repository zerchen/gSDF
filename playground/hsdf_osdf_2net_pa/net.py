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
from mano.mano_preds import get_mano_preds
from mano.manolayer import ManoLayer
from mano.inverse_kinematics import ik_solver_mano
from mano.rodrigues_layer import batch_rodrigues
from mano.rot6d import compute_rotation_matrix_from_ortho6d
from utils.pose_utils import soft_argmax, decode_volume, decode_volume_abs
from utils.sdf_utils import kinematic_embedding, pixel_align


class pose_model(nn.Module):
    def __init__(self, cfg, backbone, neck, volume_head):
        super(pose_model, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.dim_backbone_feat = 2048 if self.cfg.backbone_pose == 'resnet_50' else 512
        self.neck = neck
        self.volume_head = volume_head
        for p in self.parameters():
           p.requires_grad = False
    
    def forward(self, inputs, metas=None):
        input_img = inputs['img']
        backbone_feat = self.backbone(input_img)
        hm_feat = self.neck(backbone_feat)
        hm_pred = self.volume_head(hm_feat)
        hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 21)
        if cfg.testset == 'core50':
            volume_joint_preds = decode_volume_abs(cfg, hm_pred, metas['cam_intr'])
        else:
            volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
        
        if self.cfg.hand_branch:
            hand_pose_results = ik_solver_mano(None, volume_joint_preds[:, :21])
            hand_pose_results['volume_joints'] = volume_joint_preds
        else:
            hand_pose_results = None
        
        return hand_pose_results


class model(nn.Module):
    def __init__(self, cfg, pose_model, backbone, neck, volume_head, rot_head, hand_sdf_head, obj_sdf_head):
        super(model, self).__init__()
        self.cfg = cfg
        self.pose_model = pose_model
        self.backbone = backbone
        self.neck = neck
        self.volume_head = volume_head
        self.rot_head = rot_head
        self.dim_backbone_feat = 2048 if self.cfg.backbone_shape == 'resnet_50' else 512
        self.hand_sdf_head = hand_sdf_head
        self.obj_sdf_head = obj_sdf_head

        self.backbone_2_sdf = UNet(self.dim_backbone_feat, 256, 1)
        if self.cfg.with_add_feats:
            self.sdf_encoder = nn.Linear(260, self.cfg.sdf_latent)
        else:
            self.sdf_encoder = nn.Linear(256, self.cfg.sdf_latent)
            
        self.loss_l1 = torch.nn.L1Loss(reduction='sum')
        self.loss_l2 = torch.nn.MSELoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            if self.cfg.hand_branch and self.cfg.obj_branch:
                sdf_data = torch.cat([targets['hand_sdf'], targets['obj_sdf']], 1)
                cls_data = torch.cat([targets['hand_labels'], targets['obj_labels']], 1)
                if metas['epoch'] < self.cfg.sdf_add_epoch:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.zeros(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                else:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
            elif self.cfg.hand_branch:
                sdf_data = targets['hand_sdf']
                cls_data = targets['hand_labels']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()
            elif self.cfg.obj_branch:
                sdf_data = targets['obj_sdf']
                cls_data = targets['obj_labels']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()

            sdf_data = sdf_data.reshape(self.cfg.train_batch_size * self.cfg.num_sample_points, -1)
            cls_data = cls_data.to(torch.long).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points)
            xyz_points = sdf_data[:, 0:-2]
            sdf_gt_hand = sdf_data[:, -2].unsqueeze(1)
            sdf_gt_obj = sdf_data[:, -1].unsqueeze(1)
            if self.cfg.hand_branch:
                sdf_gt_hand = torch.clamp(sdf_gt_hand, -self.cfg.clamp_dist, self.cfg.clamp_dist)
            if self.cfg.obj_branch:
                sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.cfg.clamp_dist, self.cfg.clamp_dist)

            with torch.no_grad():
                hand_pose_results = self.pose_model(inputs, metas)

            # go through backbone
            backbone_feat = self.backbone(input_img)

            # go through deconvolution
            if self.cfg.obj_branch:
                obj_pose_results = {}
                hm_feat = self.neck(backbone_feat)
                hm_pred = self.volume_head(hm_feat)
                hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)
                volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])

                obj_transform = torch.zeros((input_img.shape[0], 4, 4)).to(input_img.device)
                obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d']
                obj_transform[:, 3, 3] = 1
                if self.rot_head is not None:
                    rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                    if cfg.rot_style == 'axisang':
                        obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                    elif cfg.rot_style == '6d':
                        obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                    obj_transform[:, :3, :3] = obj_rot_matrix
                    obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                    obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                else:
                    obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                obj_pose_results['global_trans'] = obj_transform
                obj_pose_results['center'] = volume_joint_preds
                obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                obj_pose_results['joints'] = hand_pose_results['volume_joints'] - metas['hand_center_3d'].unsqueeze(1)
            else:
                obj_pose_results = None

            # generate features for the sdf head
            sdf_feat = self.backbone_2_sdf(backbone_feat)
            sdf_feat, _ = pixel_align(self.cfg, xyz_points, self.cfg.num_sample_points, sdf_feat, metas['hand_center_3d'], metas['cam_intr'])
            sdf_feat = self.sdf_encoder(sdf_feat)

            if self.hand_sdf_head is not None:
                if self.cfg.hand_encode_style == 'kine':
                    hand_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, hand_pose_results, 'hand')
                    hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                else:
                    hand_points = xyz_points.reshape((-1, self.cfg.hand_point_latent))
                hand_sdf_decoder_inputs = torch.cat([sdf_feat, hand_points], dim=1)
                sdf_hand, cls_hand = self.hand_sdf_head(hand_sdf_decoder_inputs)
                sdf_hand = torch.clamp(sdf_hand, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_hand = None
                cls_hand = None
        
            if self.obj_sdf_head is not None:
                if self.cfg.obj_encode_style == 'kine':
                    obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, obj_pose_results, 'obj')
                    obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                else:
                    obj_points = xyz_points.reshape((-1, self.cfg.obj_point_latent))
                obj_sdf_decoder_inputs = torch.cat([sdf_feat, obj_points], dim=1)
                sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
                sdf_obj = torch.clamp(sdf_obj, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_obj = None

            sdf_results = {}
            sdf_results['hand'] = sdf_hand
            sdf_results['obj'] = sdf_obj
            sdf_results['cls'] = cls_hand

            loss = {}
            if self.hand_sdf_head is not None:
                loss['hand_sdf'] = self.cfg.hand_sdf_weight * self.loss_l1(sdf_hand * mask_hand, sdf_gt_hand * mask_hand) / mask_hand.sum()

            if self.obj_sdf_head is not None:
                loss['obj_sdf'] = self.cfg.obj_sdf_weight * self.loss_l1(sdf_obj * mask_obj, sdf_gt_obj * mask_obj) / mask_obj.sum()
            
            if cfg.hand_branch and cfg.obj_branch:
                loss['volume_joint'] = cfg.volume_weight * self.loss_l2(obj_pose_results['center'], targets['obj_center_3d'].unsqueeze(1))

            if cfg.obj_rot and cfg.corner_weight > 0:
                loss['obj_corners'] = cfg.corner_weight * self.loss_l2(obj_pose_results['corners'], targets['obj_corners_3d'])

            if cls_hand is not None:
                if metas['epoch'] >= self.cfg.sdf_add_epoch:
                    loss['hand_cls'] = self.cfg.hand_cls_weight * self.loss_ce(cls_hand, cls_data)
                else:
                    loss['hand_cls'] = 0. * self.loss_ce(cls_hand, cls_data)

            return loss, sdf_results, hand_pose_results, obj_pose_results
        else:
            with torch.no_grad():
                input_img = inputs['img']
                hand_pose_results = self.pose_model(inputs, metas)
                # go through backbone
                backbone_feat = self.backbone(input_img)

                if self.cfg.obj_branch:
                    obj_pose_results = {}
                    hm_feat = self.neck(backbone_feat)
                    hm_pred = self.volume_head(hm_feat)
                    hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)
                    if cfg.testset == 'core50':
                        volume_joint_preds = decode_volume(cfg, hm_pred, hand_pose_results['volume_joints'][:, 0, :], metas['cam_intr'])
                        obj_transform = torch.zeros((input_img.shape[0], 4, 4)).to(input_img.device)
                        obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - hand_pose_results['volume_joints'][:, 0, :]
                        obj_transform[:, 3, 3] = 1
                    else:
                        volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
                        obj_transform = torch.zeros((input_img.shape[0], 4, 4)).to(input_img.device)
                        obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d']
                        obj_transform[:, 3, 3] = 1

                    if self.rot_head is not None:
                        rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                        if cfg.rot_style == 'axisang':
                            obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                        elif cfg.rot_style == '6d':
                            obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                        obj_transform[:, :3, :3] = obj_rot_matrix
                        obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'].transpose(1, 2)).transpose(1, 2)
                        obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                    else:
                        obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                    obj_pose_results['global_trans'] = obj_transform
                    obj_pose_results['center'] = volume_joint_preds
                    obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                    obj_pose_results['joints'] = hand_pose_results['volume_joints'] - metas['hand_center_3d'].unsqueeze(1)
                else:
                    obj_pose_results = None

            return backbone_feat, hand_pose_results, obj_pose_results


def get_model(cfg, is_train):
    num_pose_resnet_layers = int(cfg.backbone_pose.split('_')[-1])
    num_shape_resnet_layers = int(cfg.backbone_shape.split('_')[-1])

    backbone_pose = ResNetBackbone(num_pose_resnet_layers)
    backbone_shape = ResNetBackbone(num_shape_resnet_layers)
    if is_train:
        backbone_pose.init_weights()
        backbone_shape.init_weights()

    neck_inplanes = 2048 if num_pose_resnet_layers == 50 else 512
    neck = UNet(neck_inplanes, 256, 3)
    if cfg.hand_branch:
        volume_head_hand = ConvHead([256, 21 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
    posenet = pose_model(cfg, backbone_pose, neck, volume_head_hand)

    if cfg.obj_branch:
        neck_shape = UNet(neck_inplanes, 256, 3)
        volume_head_obj = ConvHead([256, 1 * 64], kernel=1, stride=1, padding=0, bnrelu_final=False)
        if cfg.obj_rot:
            if cfg.rot_style == 'axisang':
                rot_head_obj = FCHead(out_dim=3)
            elif cfg.rot_style == '6d':
                rot_head_obj = FCHead(out_dim=6)
        else:
            rot_head_obj = None
    else:
        neck_shape = None
        volume_head_obj = None
        rot_head_obj = None

    if cfg.hand_branch:
        hand_sdf_head = SDFHead(cfg.sdf_latent, cfg.hand_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], cfg.hand_cls, cfg.sdf_head['num_class'])
    else:
        hand_sdf_head = None
    
    if cfg.obj_branch:
        obj_sdf_head = SDFHead(cfg.sdf_latent, cfg.obj_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False, cfg.sdf_head['num_class'])
    else:
        obj_sdf_head = None
    
    ho_model = model(cfg, posenet, backbone_shape, neck_shape, volume_head_obj, rot_head_obj, hand_sdf_head, obj_sdf_head)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)