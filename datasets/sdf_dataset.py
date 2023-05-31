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
from base_dataset import BaseDataset
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


class SDFDataset(BaseDataset):
    def __init__(self, db, cfg, mode='train'):
        super(SDFDataset, self).__init__(db, cfg, mode)
        self.num_sample_points = cfg.num_sample_points
        self.recon_scale = cfg.recon_scale
        self.clamp = cfg.clamp_dist
        
        if self.use_lmdb and self.mode == 'train':
            if self.hand_branch:
                self.hand_env = lmdb.open(db.sdf_hand_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(db.sdf_hand_source + '.lmdb', 'meta_info.json'), 'r') as f:
                   self.hand_meta = json.load(f)

            if self.obj_branch:
                self.obj_env = lmdb.open(db.sdf_obj_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(db.sdf_obj_source + '.lmdb', 'meta_info.json'), 'r') as f:
                   self.obj_meta = json.load(f)

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

        if self.mode == 'train':
            sdf_hand_path = sample_data['sdf_hand_path']
            sdf_obj_path = sample_data['sdf_obj_path']
            sdf_scale = torch.from_numpy(sample_data['sdf_scale'])
            sdf_offset = torch.from_numpy(sample_data['sdf_offset'])

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

        if self.mode == 'train' and self.use_inria_aug and random.random() < 0.5:
            random_idx = np.random.randint(low=1, high=1492, size=1, dtype='l')
            inria_key = str(random_idx[0]).rjust(4, '0')
            if self.use_lmdb:
                seg = self.load_seg_lmdb(self.seg_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
                bg = self.load_img_lmdb(self.inria_env, inria_key, (3, self.input_image_size[0], self.input_image_size[1]))
            else:
                seg = self.load_seg(seg_path, sample_data['ycb_id'])
                bg = self.load_img(os.path.join(self.inria_aug_source, inria_key + '.jpg'))
            
            seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, 1.0, rot)
            seg = np.sum(seg, axis=-1, keepdims=True) > 0
            img = seg * img + (1 - seg) * bg
            img = img.astype(np.uint8)

        img = self.image_transform(img)
        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)

        if self.mode == 'train':
            if self.hand_branch and self.obj_branch:
                num_sample_points = self.num_sample_points // 2
            else:
                num_sample_points = self.num_sample_points
        
            # get points to train sdf
            if self.hand_branch:
                if self.use_lmdb:
                    hand_samples, hand_labels = self.unpack_sdf_lmdb(self.hand_env, sample_key, self.hand_meta, num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                else:
                    hand_samples, hand_labels = self.unpack_sdf(sdf_hand_path, num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                hand_samples[:, 0:3] = hand_samples[:, 0:3] / sdf_scale - sdf_offset
                hand_labels = hand_labels.long()
                if 'ho3d' in self.dataset_name:
                    hand_samples[:, 1:3] *= -1
            else:
                hand_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                hand_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            if self.obj_branch:
                if self.use_lmdb:
                    obj_samples, obj_labels = self.unpack_sdf_lmdb(self.obj_env, sample_key, self.obj_meta, num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                else:
                    obj_samples, obj_labels = self.unpack_sdf(sdf_obj_path, num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale - sdf_offset
                obj_lablels = obj_labels.long()
                if 'ho3d' in self.dataset_name:
                    obj_samples[:, 1:3] *= -1
            else:
                obj_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                obj_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            hand_samples[:, 0:3] = torch.mm(rot_aug_mat, hand_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]

            hand_center_3d = hand_joints_3d[0]
            hand_samples[:, 0:3] = (hand_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            hand_samples[:, 3:] = hand_samples[:, 3:] / sdf_scale * self.recon_scale
            hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2

            obj_samples[:, 0:3] = (obj_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_scale * self.recon_scale
            obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2

            input_iter = dict(img=img)
            label_iter = dict(hand_sdf=hand_samples, hand_labels=hand_labels, obj_sdf=obj_samples, obj_labels=obj_labels, hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d, obj_corners_3d=obj_corners_3d)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d

            input_iter = dict(img=img)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, meta_iter

        
    def unpack_sdf_lmdb(self, env, key, meta, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples stored in the lmdb dataset.
        ---------
        @param: lmdb env, sample key, lmdb meta, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf_lmdb(tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

        index = meta['keys'].index(key)
        npz = np.array(np.frombuffer(buf, dtype=np.float32))
        pos_num = meta['pos_num'][index]
        neg_num = meta['neg_num'][index]
        feat_dim = meta['dim']
        total_num  = pos_num + neg_num
        npz = npz.reshape((-1, feat_dim))[:total_num, :]

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz[:pos_num, :]))
            neg_tensor = remove_nans(torch.from_numpy(npz[pos_num:, :]))
        except Exception as e:
            print("fail to load {}, {}".format(key, e))

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor = filter_invalid_sdf_lmdb(pos_tensor, 2.0)
            neg_tensor = filter_invalid_sdf_lmdb(neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples_and_labels = torch.cat([sample_pos, sample_neg], 0)
        samples = samples_and_labels[:, :-1]
        labels = samples_and_labels[:, -1]

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels

    def unpack_sdf(self, data_path, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples.
        ---------
        @param: sdf data path, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf(tensor, lab_tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :], lab_tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        npz = np.load(data_path)

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
            pos_sdf_other = torch.from_numpy(npz["pos_other"])
            neg_sdf_other = torch.from_numpy(npz["neg_other"])
            if hand:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
            else:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
        except Exception as e:
            print("fail to load {}, {}".format(data_path, e))

        if hand:
            pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
            neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
        else:
            xyz_pos = pos_tensor[:, :3]
            sdf_pos = pos_tensor[:, 3].unsqueeze(1)
            pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

            xyz_neg = neg_tensor[:, :3]
            sdf_neg = neg_tensor[:, 3].unsqueeze(1)
            neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
            neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        # label
        sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
        sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

        # hand part label
        # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
        hand_part_pos = sample_pos_lab[:, 0]
        hand_part_neg = sample_neg_lab[:, 0]
        samples = torch.cat([sample_pos, sample_neg], 0)
        labels = torch.cat([hand_part_pos, hand_part_neg], 0)

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels
        

if __name__ == "__main__":
    from obman.obman import obman
    obman_db = obman('train_30k')
