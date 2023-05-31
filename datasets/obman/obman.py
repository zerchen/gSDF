#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :obman.py
#@Date        :2022/04/02 15:14:24
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import numpy as np
import time
import pickle
import json
import trimesh
from tqdm import tqdm
from loguru import logger
from pycocotools.coco import COCO
from scipy.spatial import cKDTree as KDTree


class obman:
    def __init__(self, data_split, start_point=None, end_point=None, video_mode=False, num_frames=3):
        self.name = 'obman'
        self.data_split = data_split
        self.start_point = start_point
        self.end_point = end_point
        self.video_mode = video_mode

        self.stage_split = data_split.split('_')[0]
        self.cur_dir = os.path.dirname(__file__)
        with open(os.path.join(self.cur_dir, 'splits', data_split + '.json'), 'r') as f:
            self.split = json.load(f)
        self.split = [int(idx) for idx in self.split]

        self.anno_file = os.path.join(self.cur_dir, f'{self.name}_{self.stage_split}.json')
        self.img_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'rgb')
        self.seg_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'segm')
        self.inria_aug_source = os.path.join(self.cur_dir, '..', 'inria_holidays', 'rgb')
        self.image_size = (256, 256)

        self.sdf_hand_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'sdf_hand')
        self.sdf_obj_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'sdf_obj')

        self.mesh_hand_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'mesh_hand')
        self.mesh_obj_source = os.path.join(self.cur_dir, 'data', self.stage_split, 'mesh_obj')

        self.joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')
        self.data = self.load_data()
        if self.video_mode:
            seq_data = [[] for i in range(len(self.data))]
            for idx, _ in enumerate(seq_data):
                for i in range(num_frames):
                    seq_data[idx].append(self.data[idx])
            self.data = seq_data

    def load_data(self):
        db = COCO(self.anno_file)
        data = []
        if self.start_point is None:
            self.start_point = 0

        if self.end_point is None:
            self.end_point = len(self.split)
        
        for aid in self.split[self.start_point:self.end_point]:
            sample = dict()
            ann = db.anns[aid]
            img_data = db.loadImgs(ann['image_id'])[0]

            sample['id'] = img_data['file_name']
            sample['video_id'] = sample['id']
            sample['ycb_id'] = 0
            sample['img_path'] = os.path.join(self.img_source, img_data['file_name'] + '.jpg')
            sample['seg_path'] = os.path.join(self.seg_source, img_data['file_name'] + '.png')
            sample['mesh_hand_path'] = os.path.join(self.mesh_hand_source, img_data['file_name'] + '.obj')
            sample['mesh_obj_path'] = os.path.join(self.mesh_obj_source, img_data['file_name'] + '.obj')
            sample['sdf_hand_path'] = os.path.join(self.sdf_hand_source, img_data['file_name'] + '.npz')
            sample['sdf_obj_path'] = os.path.join(self.sdf_obj_source, img_data['file_name'] + '.npz')
            sample['fx'] = 480.0
            sample['fy'] = 480.0
            sample['cx'] = 128.0
            sample['cy'] = 128.0

            sample['bbox'] = ann['bbox']
            sample['hand_side'] = np.array(1)
            sample['hand_joints_3d'] = np.array(ann['hand_joints_3d'], dtype=np.float32)
            sample['hand_poses'] = np.array(ann['hand_poses'], dtype=np.float32)
            sample['hand_shapes'] = np.zeros(10, dtype=np.float32)
            sample['obj_transform'] = np.array(ann['obj_transform'], dtype=np.float32)
            sample['obj_corners_3d'] = np.array(ann['obj_corners_3d'], dtype=np.float32)
            sample['obj_rest_corners_3d'] = np.array(ann['obj_rest_corners_3d'], dtype=np.float32)
            sample['obj_center_3d'] = np.array(ann['obj_center_3d'], dtype=np.float32)
            sample['video_mask'] = np.array(False)

            if self.stage_split == 'train':
                sample['sdf_scale'] = np.array(ann['sdf_scale'], dtype=np.float32)
                sample['sdf_offset'] = np.array(ann['sdf_offset'], dtype=np.float32)

            data.append(sample)
        
        return data
    
    def _evaluate(self, output_path, idx):
        sample = self.data[idx]
        sample_idx = sample['id']

        pred_hand_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_hand.ply')
        gt_hand_mesh_path = os.path.join(self.mesh_hand_source, sample_idx + '.obj')
        if not os.path.exists(pred_hand_mesh_path):
            chamfer_hand = None
            fscore_hand_1 = None
            fscore_hand_5 = None
        else:
            pred_hand_mesh = trimesh.load(pred_hand_mesh_path, process=False)
            gt_hand_mesh = trimesh.load(gt_hand_mesh_path, process=False)
            pred_hand_points, _ = trimesh.sample.sample_surface(pred_hand_mesh, 30000)
            gt_hand_points, _ = trimesh.sample.sample_surface(gt_hand_mesh, 30000)
            pred_hand_points *= 100.
            gt_hand_points *= 100.

            # one direction
            gen_points_kd_tree = KDTree(pred_hand_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_hand_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            # other direction
            gt_points_kd_tree = KDTree(gt_hand_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_hand_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_hand = gt_to_gen_chamfer + gen_to_gt_chamfer

            threshold = 0.1 # 1 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        pred_obj_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_obj.ply')
        gt_obj_mesh_path = os.path.join(self.mesh_obj_source, sample_idx + '.obj')
        if not os.path.exists(pred_obj_mesh_path):
            chamfer_obj = None
            fscore_obj_5 = None
            fscore_obj_10 = None
        else:
            pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False)
            gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
            pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
            gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
            pred_obj_points *= 100.
            gt_obj_points *= 100.

            # one direction
            gen_points_kd_tree = KDTree(pred_obj_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            # other direction
            gt_points_kd_tree = KDTree(gt_obj_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 1.0 # 10 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        
        pred_mano_pose_path = os.path.join(output_path, 'hand_pose', sample_idx + '.json')
        if not os.path.exists(pred_mano_pose_path):
            mano_joint_err = None
        else:
            with open(pred_mano_pose_path, 'r') as f:
                pred_hand_pose = json.load(f)
            try:
                pred_mano_joint = np.array(pred_hand_pose['joints'])
                mano_joint_err = np.mean(np.linalg.norm(pred_mano_joint - sample['hand_joints_3d'], axis=1)) * 1000.
            except:
                mano_joint_err = None

        pred_obj_pose_path = os.path.join(output_path, 'obj_pose', sample_idx + '.json')
        if not os.path.exists(pred_obj_pose_path):
            obj_center_err = None
            obj_corner_err = None
        else:
            with open(pred_obj_pose_path, 'r') as f:
                pred_obj_pose = json.load(f)
            try:
                pred_obj_center = np.array(pred_obj_pose['center'])
                obj_center_err = np.linalg.norm(pred_obj_center - sample['obj_center_3d']) * 1000.
            except:
                obj_center_err = None

            try:
                pred_obj_corners = np.array(pred_obj_pose['corners'])
                obj_corner_err = np.mean(np.linalg.norm(pred_obj_corners - sample['obj_corners_3d'], axis=1)) * 1000.
            except:
                obj_corner_err = None
        
        error_dict = {}
        error_dict['id'] = sample_idx
        error_dict['chamfer_hand'] = chamfer_hand
        error_dict['fscore_hand_1'] = fscore_hand_1
        error_dict['fscore_hand_5'] = fscore_hand_5
        error_dict['chamfer_obj'] = chamfer_obj
        error_dict['fscore_obj_5'] = fscore_obj_5
        error_dict['fscore_obj_10'] = fscore_obj_10
        error_dict['mano_joint'] = mano_joint_err
        error_dict['obj_center'] = obj_center_err
        error_dict['obj_corner'] = obj_corner_err

        return error_dict
        

if __name__ == "__main__":
    db = obman('test_6k')
