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
from base_dataset import BaseDataset
from utils.camera import PerspectiveCamera
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


class PoseDataset(BaseDataset):
    def __init__(self, db, cfg, mode='train'):
        super(PoseDataset, self).__init__(db, cfg, mode)
        self.norm_coords = cfg.norm_coords
        self.norm_factor = cfg.norm_factor

    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        sample_key = sample_data['id']
        img_path = sample_data['img_path']
        seg_path = sample_data['seg_path']
        bbox = sample_data['bbox']
        hand_side = torch.from_numpy(sample_data['hand_side'])
        try:
            hand_joints_3d = torch.from_numpy(sample_data['hand_joints_3d'])
        except:
            hand_joints_3d = torch.zeros((21, 3))

        try:
            hand_poses = torch.from_numpy(sample_data['hand_poses'])
        except:
            hand_poses = torch.zeros((48))

        try:
            hand_shapes = torch.from_numpy(sample_data['hand_shapes'])
        except:
            hand_shapes = torch.zeros((10))
        
        try:
            obj_center_3d = torch.from_numpy(sample_data['obj_center_3d'])
        except:
            obj_center_3d = torch.zeros(3)

        try:
            obj_corners_3d = torch.from_numpy(sample_data['obj_corners_3d'])
        except:
            obj_corners_3d = torch.zeros((8, 3))

        try:
            obj_rest_corners_3d = torch.from_numpy(sample_data['obj_rest_corners_3d'])
        except:
            obj_rest_corners_3d = torch.zeros((8, 3))

        try:
            obj_transform = torch.from_numpy(sample_data['obj_transform'])
        except:
            obj_transform = torch.zeros((4, 4))

        if self.use_lmdb and self.mode == 'train':
            img = self.load_img_lmdb(self.img_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
        else:
            img = self.load_img(img_path)

        camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])

        if self.mode == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config(self.dataset_name)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]

        if self.use_lmdb and self.mode == 'train':
            img, _ = self.generate_patch_image(img, [0, 0, self.input_image_size[1], self.input_image_size[0]], self.input_image_size, do_flip, scale, rot)
        else:
            bbox[0] = bbox[0] + bbox[2] * trans[0]
            bbox[1] = bbox[1] + bbox[3] * trans[1]
            img, _ = self.generate_patch_image(img, bbox, self.input_image_size, do_flip, scale, rot)
        
        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)

        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        if self.mode == 'train':
            rot_aug_mat = rot_aug_mat @ rot_cam_extr

        img = self.image_transform(img)
        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)

        if self.mode == 'train':
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]

            if self.norm_coords:
                hand_joints_3d = hand_joints_3d / torch.norm(hand_joints_3d[3] - hand_joints_3d[2]) * self.norm_factor

            hand_center_3d = hand_joints_3d[0]

            input_iter = dict(img=img)
            label_iter = dict(hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d, obj_corners_3d=obj_corners_3d, hand_betas=hand_shapes)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d

            input_iter = dict(img=img)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, meta_iter

        
if __name__ == "__main__":
    from obman.obman import obman
    obman_db = obman('train_30k')
