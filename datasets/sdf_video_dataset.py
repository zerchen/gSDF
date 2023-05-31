#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :sdf_dataset.py
#@Date        :2022/04/05 16:58:35
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import time
import cv2
import torch
import lmdb
import json
import copy
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from utils.camera import PerspectiveCamera
from sdf_dataset import SDFDataset
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


class SDFVideoDataset(SDFDataset):
    def __init__(self, db, cfg, mode='train'):
        super(SDFVideoDataset, self).__init__(db, cfg, mode)
        self.num_frames = cfg.num_frames
        
    def __getitem__(self, index):
        sample_video_data = copy.deepcopy(self.db[index])
        length_video = len(sample_video_data)
        if self.mode == 'train':
            chosen_frames = np.random.choice(length_video, self.num_frames, replace=False)
        else:
            chosen_frames = [i for i in range(length_video)]

        sample_key_frames, img_path_frames, seg_path_frames, bbox_frames, video_masks = [], [], [], [], []
        for i in chosen_frames:
            sample_key_frames.append(sample_video_data[i]['id'])
            img_path_frames.append(sample_video_data[i]['img_path'])
            seg_path_frames.append(sample_video_data[i]['seg_path'])
            bbox_frames.append(sample_video_data[i]['bbox'])
            video_masks.append(torch.from_numpy(sample_video_data[i]['video_mask']).unsqueeze(0))

        hand_joints_3d_frames, hand_poses_frames, hand_shapes_frames, obj_center_3d_frames, obj_corners_3d_frames, obj_rest_corners_3d_frames, obj_transform_frames = [], [], [], [], [], [], []
        for i in chosen_frames:
            hand_joints_3d_frames.append(torch.from_numpy(sample_video_data[i]['hand_joints_3d']))
            hand_poses_frames.append(torch.from_numpy(sample_video_data[i]['hand_poses']))
            hand_shapes_frames.append(torch.from_numpy(sample_video_data[i]['hand_shapes']))
            obj_center_3d_frames.append(torch.from_numpy(sample_video_data[i]['obj_center_3d']))
            obj_corners_3d_frames.append(torch.from_numpy(sample_video_data[i]['obj_corners_3d']))
            obj_rest_corners_3d_frames.append(torch.from_numpy(sample_video_data[i]['obj_rest_corners_3d']))
            obj_transform_frames.append(torch.from_numpy(sample_video_data[i]['obj_transform']))

        if self.mode == 'train':
            sdf_hand_path_frames, sdf_obj_path_frames, sdf_scale_frames, sdf_offset_frames = [], [], [], []
            for i in chosen_frames:
                sdf_hand_path_frames.append(sample_video_data[i]['sdf_hand_path'])
                sdf_obj_path_frames.append(sample_video_data[i]['sdf_obj_path'])
                sdf_scale_frames.append(torch.from_numpy(sample_video_data[i]['sdf_scale']))
                sdf_offset_frames.append(torch.from_numpy(sample_video_data[i]['sdf_offset']))

        img_frames = []
        for i in range(len(sample_key_frames)):
            if self.use_lmdb and self.mode == 'train':
                img_frames.append(self.load_img_lmdb(self.img_env, sample_key_frames[i], (3, self.input_image_size[0], self.input_image_size[1])))
            else:
                img_frames.append(self.load_img(img_path_frames[i]))
        
        camera_frames = []
        for i in chosen_frames:
            camera_frames.append(PerspectiveCamera(sample_video_data[i]['fx'], sample_video_data[i]['fy'], sample_video_data[i]['cx'], sample_video_data[i]['cy']))

        if self.mode == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config(self.dataset_name)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]

        for i in range(len(img_frames)):
            if self.use_lmdb and self.mode == 'train':
                img_frames[i], _ = self.generate_patch_image(img_frames[i], [0, 0, self.input_image_size[1], self.input_image_size[0]], self.input_image_size, do_flip, scale, rot)
            else:
                bbox_frames[i][0] = bbox_frames[i][0] + bbox_frames[i][2] * trans[0]
                bbox_frames[i][1] = bbox_frames[i][1] + bbox_frames[i][3] * trans[1]
                img_frames[i], _ = self.generate_patch_image(img_frames[i], bbox_frames[i], self.input_image_size, do_flip, scale, rot)

            for j in range(3):
                img_frames[i][:, :, j] = np.clip(img_frames[i][:, :, j] * color_scale[j], 0, 255)
        
        rot_cam_extr_frames, rot_aug_mat_frames, cam_intr_frames, cam_extr_frames = [], [], [], []
        for i in range(len(camera_frames)):
            camera_frames[i].update_virtual_camera_after_crop(bbox_frames[i])
            camera_frames[i].update_intrinsics_after_resize((bbox_frames[i][-1], bbox_frames[i][-2]), self.input_image_size)
            rot_cam_extr_frames.append(torch.from_numpy(camera_frames[i].extrinsics[:3, :3].T))
            cam_intr_frames.append(torch.from_numpy(camera_frames[i].intrinsics))
            cam_extr_frames.append(torch.from_numpy(camera_frames[i].extrinsics))
            if self.mode == 'train':
                rot_aug_mat_frames.append(rot_aug_mat @ rot_cam_extr_frames[i])

        for i in range(len(img_frames)):
            img_frames[i] = self.image_transform(img_frames[i])
        
        if self.mode == 'train':
            if self.hand_branch and self.obj_branch:
                num_sample_points = self.num_sample_points // 2
            else:
                num_sample_points = self.num_sample_points
        
            hand_samples_frames, obj_samples_frames = [], []
            # get points to train sdf
            for i in range(len(sample_key_frames)):
                if self.hand_branch:
                    if self.use_lmdb:
                        hand_samples, _ = self.unpack_sdf_lmdb(self.hand_env, sample_key_frames[i], self.hand_meta, num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                    else:
                        hand_samples, _ = self.unpack_sdf(sdf_hand_path_frames[i], num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                    hand_samples[:, 0:3] = hand_samples[:, 0:3] / sdf_scale_frames[i] - sdf_offset_frames[i]
                    if 'ho3d' in self.dataset_name:
                        hand_samples[:, 1:3] *= -1
                else:
                    hand_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                hand_samples_frames.append(hand_samples)

                if self.obj_branch:
                    if self.use_lmdb:
                        obj_samples, _ = self.unpack_sdf_lmdb(self.obj_env, sample_key_frames[i], self.obj_meta, num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                    else:
                        obj_samples, _ = self.unpack_sdf(sdf_obj_path_frames[i], num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                    obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale_frames[i] - sdf_offset_frames[i]
                    if 'ho3d' in self.dataset_name:
                        obj_samples[:, 1:3] *= -1
                else:
                    obj_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                obj_samples_frames.append(obj_samples)

            hand_center_3d_frames = []
            for i in range(len(sample_key_frames)):
                hand_samples_frames[i][:, 0:3] = torch.mm(rot_aug_mat_frames[i], hand_samples_frames[i][:, 0:3].transpose(1, 0)).transpose(1, 0)
                obj_samples_frames[i][:, 0:3] = torch.mm(rot_aug_mat_frames[i], obj_samples_frames[i][:, 0:3].transpose(1, 0)).transpose(1, 0)
                hand_joints_3d_frames[i][:, 0:3] = torch.mm(rot_aug_mat_frames[i], hand_joints_3d_frames[i][:, 0:3].transpose(1, 0)).transpose(1, 0)
                hand_poses_frames[i][:3] = rotation_matrix_to_angle_axis(rot_aug_mat_frames[i] @ angle_axis_to_rotation_matrix(hand_poses_frames[i][:3].unsqueeze(0))).squeeze(0)
                obj_corners_3d_frames[i][:, 0:3] = torch.mm(rot_aug_mat_frames[i], obj_corners_3d_frames[i][:, 0:3].transpose(1, 0)).transpose(1, 0)
                obj_center_3d_frames[i] = torch.mm(rot_aug_mat_frames[i], obj_center_3d_frames[i].unsqueeze(1)).squeeze()
                trans_with_rot = torch.zeros((4, 4))
                trans_with_rot[:3, :3] = rot_aug_mat_frames[i]
                trans_with_rot[3, 3] = 1.
                obj_transform_frames[i] = torch.mm(trans_with_rot, obj_transform_frames[i])
                obj_transform_frames[i][:3, 3] = obj_transform_frames[i][:3, 3] - hand_joints_3d_frames[i][0]

                hand_center_3d_frames.append(hand_joints_3d_frames[i][0])
                hand_samples_frames[i][:, 0:3] = (hand_samples_frames[i][:, 0:3] - hand_center_3d_frames[i]) * self.recon_scale
                hand_samples_frames[i][:, 3:] = hand_samples_frames[i][:, 3:] / sdf_scale_frames[i] * self.recon_scale
                hand_samples_frames[i][:, 0:5] = hand_samples_frames[i][:, 0:5] / 2

                obj_samples_frames[i][:, 0:3] = (obj_samples_frames[i][:, 0:3] - hand_center_3d_frames[i]) * self.recon_scale
                obj_samples_frames[i][:, 3:] = obj_samples_frames[i][:, 3:] / sdf_scale_frames[i] * self.recon_scale
                obj_samples_frames[i][:, 0:5] = obj_samples_frames[i][:, 0:5] / 2

            input_iter = dict(img=img_frames, masks=video_masks)
            label_iter = dict(hand_sdf=hand_samples_frames, obj_sdf=obj_samples_frames, hand_joints_3d=hand_joints_3d_frames, obj_center_3d=obj_center_3d_frames, obj_corners_3d=obj_corners_3d_frames)
            meta_iter = dict(cam_intr=cam_intr_frames, cam_extr=cam_extr_frames, id=sample_key_frames, hand_center_3d=hand_center_3d_frames, hand_poses=hand_poses_frames, hand_shapes=hand_shapes_frames, obj_rest_corners_3d=obj_rest_corners_3d_frames, obj_transform=obj_transform_frames)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d_frames = []
            for i in range(len(sample_key_frames)):
                hand_center_3d_frames.append(torch.mm(rot_cam_extr_frames[i], hand_joints_3d_frames[i][:, 0:3].transpose(1, 0)).transpose(1, 0)[0])
                trans_with_rot = torch.zeros((4, 4))
                trans_with_rot[:3, :3] = rot_cam_extr_frames[i]
                trans_with_rot[3, 3] = 1.
                obj_transform_frames[i] = torch.mm(trans_with_rot, obj_transform_frames[i])
                obj_transform_frames[i][:3, 3] = obj_transform_frames[i][:3, 3] - hand_center_3d_frames[i]

            input_iter = dict(img=img_frames, masks=video_masks)
            meta_iter = dict(cam_intr=cam_intr_frames, cam_extr=cam_extr_frames, id=sample_key_frames, hand_center_3d=hand_center_3d_frames, hand_poses=hand_poses_frames, hand_shapes=hand_shapes_frames, obj_rest_corners_3d=obj_rest_corners_3d_frames, obj_transform=obj_transform_frames)

            return input_iter, meta_iter


if __name__ == "__main__":
    from obman.obman import obman
    obman_db = obman('train_30k')
