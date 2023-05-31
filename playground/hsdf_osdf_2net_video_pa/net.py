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
from networks.backbones.transformer import VideoTransformer, FactorizedVideoTransformer
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
        hand_pose_results_frames = []
        for i in range(len(input_img)):
            backbone_feat = self.backbone(input_img[i])
            hm_feat = self.neck(backbone_feat)
            hm_pred = self.volume_head(hm_feat)
            hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 21)
            volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'][i], metas['cam_intr'][i])
        
            if self.cfg.hand_branch:
                hand_pose_results = ik_solver_mano(None, volume_joint_preds[:, :21])
                hand_pose_results['volume_joints'] = volume_joint_preds
            else:
                hand_pose_resuts = None
            hand_pose_results_frames.append(hand_pose_results)
        
        return hand_pose_results_frames


class model(nn.Module):
    def __init__(self, cfg, pose_model, backbone, neck, volume_head, rot_head, hand_sdf_head, obj_sdf_head, feat_transformer):
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
        self.feat_transformer = feat_transformer
        self.backbone_2_sdf = UNet(self.dim_backbone_feat, 256, 1)
        if self.cfg.with_add_feats:
            self.sdf_encoder = nn.Linear(260, self.cfg.sdf_latent)
        else:
            self.sdf_encoder = nn.Linear(256, self.cfg.sdf_latent)

        self.loss_l1 = torch.nn.L1Loss(reduction='sum')
        self.loss_l2 = torch.nn.MSELoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_frames = inputs['img']
            num_frames = len(input_frames)

            if self.cfg.hand_branch and self.cfg.obj_branch:
                sdf_data_frames = []
                for i in range(num_frames):
                    sdf_data_frames.append(torch.cat([targets['hand_sdf'][i], targets['obj_sdf'][i]], 1))
                if metas['epoch'] < self.cfg.sdf_add_epoch:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.zeros(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                else:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.ones(targets['hand_sdf'][0].size()[:2]), torch.ones(targets['obj_sdf'][0].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
            elif self.cfg.hand_branch:
                sdf_data_frames = targets['hand_sdf']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()
            elif self.cfg.obj_branch:
                sdf_data_frames = targets['obj_sdf']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()

            xyz_points_frames, sdf_gt_hand_frames, sdf_gt_obj_frames = [], [], []
            for i in range(num_frames):
                sdf_data_frames[i] = sdf_data_frames[i].reshape(self.cfg.train_batch_size * self.cfg.num_sample_points, -1)
                xyz_points_frames.append(sdf_data_frames[i][:, 0:-2])
                sdf_gt_hand = sdf_data_frames[i][:, -2].unsqueeze(1)
                sdf_gt_obj = sdf_data_frames[i][:, -1].unsqueeze(1)
                if self.cfg.hand_branch:
                    sdf_gt_hand = torch.clamp(sdf_gt_hand, -self.cfg.clamp_dist, self.cfg.clamp_dist)
                    sdf_gt_hand_frames.append(sdf_gt_hand)
                if self.cfg.obj_branch:
                    sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.cfg.clamp_dist, self.cfg.clamp_dist)
                    sdf_gt_obj_frames.append(sdf_gt_obj)

            with torch.no_grad():
                hand_pose_results_frames = self.pose_model(inputs, metas)
            
            sdf_feat_frames, obj_pose_results_frames = [], []
            for i in range(num_frames):
                # go through backbone
                backbone_feat = self.backbone(input_frames[i])
                if self.cfg.obj_branch:
                    obj_pose_results = {}
                    hm_feat = self.neck(backbone_feat)
                    hm_pred = self.volume_head(hm_feat)
                    hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)
                    volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'][i], metas['cam_intr'][i])

                    obj_transform = torch.zeros((input_frames[i].shape[0], 4, 4)).to(input_frames[i].device)
                    obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d'][i]
                    obj_transform[:, 3, 3] = 1
                    if self.rot_head is not None:
                        rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                        if cfg.rot_style == 'axisang':
                            obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                        elif cfg.rot_style == '6d':
                            obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                        obj_transform[:, :3, :3] = obj_rot_matrix
                        obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'][i].transpose(1, 2)).transpose(1, 2)
                        obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                    else:
                        obj_transform[:, :3, :3] = torch.eye(3).to(input_frames[i].device)
                    obj_pose_results['global_trans'] = obj_transform
                    obj_pose_results['center'] = volume_joint_preds
                    obj_pose_results['wrist_trans'] = hand_pose_results_frames[i]['global_trans'][:, 0]
                    obj_pose_results['joints'] = hand_pose_results_frames[i]['volume_joints'] - metas['hand_center_3d'][i].unsqueeze(1)
                else:
                    obj_pose_results = None
                obj_pose_results_frames.append(obj_pose_results)

                sdf_feat = self.backbone_2_sdf(backbone_feat)
                sdf_feat_frames.append(sdf_feat)
            
            sdf_refined_feat_frames = [sdf_feat_frames[i].unsqueeze(1) for i in range(num_frames)]
            sdf_refined_feat_frames = torch.cat(sdf_refined_feat_frames, axis=1)
            sdf_refined_feat_frames = self.feat_transformer(sdf_refined_feat_frames)
            
            sdf_point_feat_frames = []
            for i in range(num_frames):
                sdf_point_feat, _ = pixel_align(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, sdf_refined_feat_frames[i], metas['hand_center_3d'][i], metas['cam_intr'][i])
                sdf_point_feat = self.sdf_encoder(sdf_point_feat)
                sdf_point_feat_frames.append(sdf_point_feat)
                
            if self.hand_sdf_head is not None:
                sdf_hand_frames = []
                for i in range(num_frames):
                    if self.cfg.hand_encode_style == 'kine':
                        hand_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, hand_pose_results_frames[i], 'hand')
                        hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                    else:
                        hand_points = xyz_points_frames[i].reshape((-1, self.cfg.hand_point_latent))
                    hand_sdf_decoder_inputs = torch.cat([sdf_point_feat_frames[i], hand_points], dim=1)
                    sdf_hand, _ = self.hand_sdf_head(hand_sdf_decoder_inputs)
                    sdf_hand = torch.clamp(sdf_hand, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
                    sdf_hand_frames.append(sdf_hand)
            else:
                sdf_hand_frames = None
        
            if self.obj_sdf_head is not None:
                sdf_obj_frames = []
                for i in range(num_frames):
                    if self.cfg.obj_encode_style == 'kine':
                        obj_points = kinematic_embedding(self.cfg, xyz_points_frames[i], self.cfg.num_sample_points, obj_pose_results_frames[i], 'obj')
                        obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                    else:
                        obj_points = xyz_points_frames[i].reshape((-1, self.cfg.obj_point_latent))
                    obj_sdf_decoder_inputs = torch.cat([sdf_point_feat_frames[i], obj_points], dim=1)
                    sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
                    sdf_obj = torch.clamp(sdf_obj, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
                    sdf_obj_frames.append(sdf_obj)
            else:
                sdf_obj = None

            sdf_results = {}
            sdf_results['hand'] = sdf_hand_frames
            sdf_results['obj'] = sdf_obj_frames
            sdf_results['cls'] = None

            loss = {}
            if self.hand_sdf_head is not None:
                loss_hand_frames = []
                for i in range(num_frames):
                    loss_hand_frames.append(self.cfg.hand_sdf_weight * self.loss_l1(sdf_hand_frames[i] * mask_hand, sdf_gt_hand_frames[i] * mask_hand) / mask_hand.sum())
                loss['hand_sdf'] = sum(loss_hand_frames) / num_frames

            if self.obj_sdf_head is not None:
                loss_obj_frames = []
                for i in range(num_frames):
                    loss_obj_frames.append(self.cfg.obj_sdf_weight * self.loss_l1(sdf_obj_frames[i] * mask_obj, sdf_gt_obj_frames[i] * mask_obj) / mask_obj.sum())
                loss['obj_sdf'] = sum(loss_obj_frames) / num_frames

            if cfg.hand_branch and cfg.obj_branch:
                loss_volume_frames =[]
                for i in range(num_frames):
                    loss_volume_frames.append(self.cfg.volume_weight * self.loss_l2(obj_pose_results_frames[i]['center'], targets['obj_center_3d'][i].unsqueeze(1)))
                loss['volume_joint'] = sum(loss_volume_frames) / num_frames

            return loss, sdf_results, hand_pose_results_frames, obj_pose_results_frames
        else:
            with torch.no_grad():
                input_frames = inputs['img']
                num_frames = len(input_frames)
                hand_pose_results_frames = self.pose_model(inputs, metas)

                sdf_feat_frames, obj_pose_results_frames = [], []
                for i in range(num_frames):
                    # go through backbone
                    backbone_feat = self.backbone(input_frames[i])
                    if self.cfg.obj_branch:
                        obj_pose_results = {}
                        hm_feat = self.neck(backbone_feat)
                        hm_pred = self.volume_head(hm_feat)
                        hm_pred, hm_conf = soft_argmax(cfg, hm_pred, 1)
                        volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'][i], metas['cam_intr'][i])

                        obj_transform = torch.zeros((input_frames[i].shape[0], 4, 4)).to(input_frames[i].device)
                        obj_transform[:, :3, 3] = volume_joint_preds.squeeze(1) - metas['hand_center_3d'][i]
                        obj_transform[:, 3, 3] = 1
                        if self.rot_head is not None:
                            rot_feat = self.rot_head(backbone_feat.mean(3).mean(2))
                            if cfg.rot_style == 'axisang':
                                obj_rot_matrix = batch_rodrigues(rot_feat).view(rot_feat.shape[0], 3, 3)
                            elif cfg.rot_style == '6d':
                                obj_rot_matrix = compute_rotation_matrix_from_ortho6d(rot_feat)
                            obj_transform[:, :3, :3] = obj_rot_matrix
                            obj_corners_3d = torch.matmul(obj_rot_matrix, metas['obj_rest_corners_3d'][i].transpose(1, 2)).transpose(1, 2)
                            obj_pose_results['corners'] = obj_corners_3d + volume_joint_preds
                        else:
                            obj_transform[:, :3, :3] = torch.eye(3).to(input_frames[i].device)
                        obj_pose_results['global_trans'] = obj_transform
                        obj_pose_results['center'] = volume_joint_preds
                        obj_pose_results['wrist_trans'] = hand_pose_results_frames[i]['global_trans'][:, 0]
                        obj_pose_results['joints'] = hand_pose_results_frames[i]['volume_joints'] - metas['hand_center_3d'][i].unsqueeze(1)
                    else:
                        obj_pose_results = None
                    obj_pose_results_frames.append(obj_pose_results)

                    # generate features for the sdf head
                    sdf_feat = self.backbone_2_sdf(backbone_feat)
                    sdf_feat_frames.append(sdf_feat)

            return sdf_feat_frames, hand_pose_results_frames, obj_pose_results_frames


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

    if cfg.fa_trans:
        feat_transformer = FactorizedVideoTransformer(image_size=16, num_frames=cfg.num_frames, depth=8, dim_head=256)
    else:
        feat_transformer = VideoTransformer(image_size=16, num_frames=cfg.num_frames, depth=8, dim_head=256)

    if cfg.hand_branch:
        hand_sdf_head = SDFHead(cfg.sdf_latent, cfg.hand_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], cfg.hand_cls, cfg.sdf_head['num_class'])
    else:
        hand_sdf_head = None
    
    if cfg.obj_branch:
        obj_sdf_head = SDFHead(cfg.sdf_latent, cfg.obj_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False, cfg.sdf_head['num_class'])
    else:
        obj_sdf_head = None
    
    ho_model = model(cfg, posenet, backbone_shape, neck_shape, volume_head_obj, rot_head_obj, hand_sdf_head, obj_sdf_head, feat_transformer)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)