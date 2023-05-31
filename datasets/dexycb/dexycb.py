#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :dexycb.py
#@Date        :2022/07/05 18:51:49
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
from turtle import color
import numpy as np
import time
import pickle
import json
import trimesh
import torch
from tqdm import tqdm
from loguru import logger
from pycocotools.coco import COCO
from scipy.spatial import cKDTree as KDTree
from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS, _SERIALS, _YCB_CLASSES 


class dexycb:
    def __init__(self, data_split, start_point=None, end_point=None, video_mode=False, num_frames=3, use_whole_video_test=False):
        self.name = 'dexycb'
        self.data_split = data_split
        self.start_point = start_point
        self.end_point = end_point
        self.video_mode = video_mode
        self.use_whole_video_test = use_whole_video_test

        self.stage_split = '_'.join(data_split.split('_')[:-1])
        split = self.stage_split.split('_')[-1]
        self.cur_dir = os.path.dirname(__file__)
        self.img_source = os.path.join(self.cur_dir, 'data', f'rgb_{split}')
        self.seg_source = os.path.join(self.cur_dir, 'data', f'seg_{split}')
        self.sdf_hand_source = os.path.join(self.cur_dir, 'data', f'sdf_hand_{split}')
        self.sdf_obj_source = os.path.join(self.cur_dir, 'data', f'sdf_obj_{split}')
        self.num_frames = num_frames
        
        with open(os.path.join(self.cur_dir, 'splits', self.data_split + '.json'), 'r') as f:
            self.split = json.load(f)
        self.split = [int(idx) for idx in self.split]

        self.anno_file = os.path.join(self.cur_dir, f'{self.name}_{self.stage_split}.json')
        self.inria_aug_source = os.path.join(self.cur_dir, '..', 'inria_holidays', 'rgb')
        self.image_size = (480, 640)
        self.subject = _SUBJECTS
        self.camera = _SERIALS

        self.joints_name = ('wrist', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index1', 'index2', 'index3', 'index4', 'middle1', 'middle2', 'middle3', 'middle4', 'ring1', 'ring2', 'ring3', 'ring4', 'pinky1', 'pinky2', 'pinky3', 'pinky4')
        if self.video_mode:
            self.data = self.load_video_data()
            if 'test' in self.stage_split:
                if self.use_whole_video_test == False:
                    self.data = self.construct_clips()
                else:
                    self.data = self.data[self.start_point:self.end_point]
        else:
            self.data = self.load_data()
    
    def construct_clips(self):
        clip_data = []
        for i in range(len(self.data)):
            per_video_data = self.data[i]

            num_missing_frames = len(per_video_data) % self.num_frames
            num_groups = len(per_video_data) // self.num_frames
            for m in range(num_groups):
                sub_clip_data = per_video_data[m * self.num_frames: (m + 1) * self.num_frames]
                clip_data.append(sub_clip_data)
            
            if num_missing_frames > 0:
                sub_clip_data = per_video_data[num_groups * self.num_frames:]
                if num_groups > 0:
                    sub_clip_data = per_video_data[num_groups * self.num_frames - (self.num_frames - num_missing_frames): num_groups * self.num_frames] + sub_clip_data
                clip_data.append(sub_clip_data)
                
        return clip_data[self.start_point:self.end_point]
    
    def load_video_data(self):
        db = COCO(self.anno_file)
        data = {}

        for aid in self.split:
            sample = dict()
            ann = db.anns[aid]
            img_data = db.loadImgs(ann['image_id'])[0]

            sample['id'] = img_data['file_name']
            sample['subject_id'] = _SUBJECTS[int(sample['id'].split('_')[0]) - 1]
            sample['video_id'] = '_'.join(sample['id'].split('_')[1:3])
            sample['cam_id'] = sample['id'].split('_')[-2]
            sample['frame_id'] = sample['id'].split('_')[-1].rjust(6, '0')
            sample['ycb_id'] = ann['ycb_id']

            sample['img_path'] = os.path.join(self.cur_dir, 'data', sample['subject_id'], sample['video_id'], sample['cam_id'], 'color_' + sample['frame_id'] + '.jpg')
            sample['seg_path'] = os.path.join(self.cur_dir, 'data', sample['subject_id'], sample['video_id'], sample['cam_id'], 'labels_' + sample['frame_id'] + '.npz')
            if 'train' in self.stage_split:
                sample['sdf_hand_path'] = os.path.join(self.cur_dir, 'data', 'sdf_data', 'sdf_hand', sample['id'] + '.npz')
                sample['sdf_obj_path'] = os.path.join(self.cur_dir, 'data', 'sdf_data', 'sdf_obj', sample['id'] + '.npz')

            sample['fx'] = ann['fx']
            sample['fy'] = ann['fy']
            sample['cx'] = ann['cx']
            sample['cy'] = ann['cy']

            sample['bbox'] = ann['bbox']
            sample['hand_side'] = np.array(1)
            sample['hand_joints_3d'] = np.array(ann['hand_joints_3d'], dtype=np.float32)
            sample['hand_poses'] = np.array(ann['hand_poses'], dtype=np.float32)
            sample['hand_shapes'] = np.array(ann['hand_shapes'], dtype=np.float32)
            sample['hand_trans'] = np.array(ann['hand_trans'], dtype=np.float32)
            sample['obj_transform'] = np.array(ann['obj_transform'], dtype=np.float32)
            sample['obj_corners_3d'] = np.array(ann['obj_corners_3d'], dtype=np.float32)
            sample['obj_rest_corners_3d'] = np.array(ann['obj_rest_corners_3d'], dtype=np.float32)
            sample['obj_center_3d'] = np.array(ann['obj_center_3d'], dtype=np.float32)
            sample['video_mask'] = np.array(False)

            if 'train' in self.stage_split:
                sample['sdf_scale'] = np.array(ann['sdf_scale'], dtype=np.float32)
                sample['sdf_offset'] = np.array(ann['sdf_offset'], dtype=np.float32)

            seq_id = '_'.join(sample['id'].split('_')[:-1])
            if seq_id not in data.keys():
                data[seq_id] = []
                data[seq_id].append(sample)
            else:
                data[seq_id].append(sample)
            
        seq_data = []
        for seq_id in data.keys():
            data[seq_id].sort(key=lambda k: (k.get('frame_id'), 0))
            if len(data[seq_id]) >= self.num_frames and 'train' in self.stage_split:
                seq_data.append(data[seq_id])
            elif 'test' in self.stage_split:
                seq_data.append(data[seq_id])

        return seq_data

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
            sample['subject_id'] = _SUBJECTS[int(sample['id'].split('_')[0]) - 1]
            sample['video_id'] = '_'.join(sample['id'].split('_')[1:3])
            sample['cam_id'] = sample['id'].split('_')[-2]
            sample['frame_id'] = sample['id'].split('_')[-1].rjust(6, '0')
            sample['ycb_id'] = ann['ycb_id']

            sample['img_path'] = os.path.join(self.cur_dir, 'data', sample['subject_id'], sample['video_id'], sample['cam_id'], 'color_' + sample['frame_id'] + '.jpg')
            sample['seg_path'] = os.path.join(self.cur_dir, 'data', sample['subject_id'], sample['video_id'], sample['cam_id'], 'labels_' + sample['frame_id'] + '.npz')
            if 'train' in self.stage_split:
                sample['sdf_hand_path'] = os.path.join(self.cur_dir, 'data', 'sdf_data', 'sdf_hand', sample['id'] + '.npz')
                sample['sdf_obj_path'] = os.path.join(self.cur_dir, 'data', 'sdf_data', 'sdf_obj', sample['id'] + '.npz')

            sample['fx'] = ann['fx']
            sample['fy'] = ann['fy']
            sample['cx'] = ann['cx']
            sample['cy'] = ann['cy']

            sample['bbox'] = ann['bbox']
            sample['hand_side'] = np.array(1)
            sample['hand_joints_3d'] = np.array(ann['hand_joints_3d'], dtype=np.float32)
            sample['hand_poses'] = np.array(ann['hand_poses'], dtype=np.float32)
            sample['hand_shapes'] = np.array(ann['hand_shapes'], dtype=np.float32)
            sample['hand_trans'] = np.array(ann['hand_trans'], dtype=np.float32)
            sample['obj_transform'] = np.array(ann['obj_transform'], dtype=np.float32)
            sample['obj_corners_3d'] = np.array(ann['obj_corners_3d'], dtype=np.float32)
            sample['obj_rest_corners_3d'] = np.array(ann['obj_rest_corners_3d'], dtype=np.float32)
            sample['obj_center_3d'] = np.array(ann['obj_center_3d'], dtype=np.float32)

            if 'train' in self.stage_split:
                sample['sdf_scale'] = np.array(ann['sdf_scale'], dtype=np.float32)
                sample['sdf_offset'] = np.array(ann['sdf_offset'], dtype=np.float32)

            data.append(sample)
        
        return data
    
    def _evaluate(self, output_path, idx):
        sample = self.data[idx]
        sample_idx = sample['id']

        pred_mano_pose_path = os.path.join(output_path, 'hand_pose', sample_idx + '.json')
        with open(pred_mano_pose_path, 'r') as f:
            pred_hand_pose = json.load(f)
        cam_extr = np.array(pred_hand_pose['cam_extr'])
        try:
            pred_mano_joint = (cam_extr @ np.array(pred_hand_pose['joints']).transpose(1, 0)).transpose(1, 0)
            mano_joint_err = np.mean(np.linalg.norm(pred_mano_joint - sample['hand_joints_3d'], axis=1)) * 1000.
        except:
            mano_joint_err = None

        pred_obj_pose_path = os.path.join(output_path, 'obj_pose', sample_idx + '.json')
        with open(pred_obj_pose_path, 'r') as f:
            pred_obj_pose = json.load(f)
        cam_extr = np.array(pred_obj_pose['cam_extr'])
        try:
            pred_obj_center = (cam_extr @ np.array(pred_obj_pose['center']).reshape((1, 3)).transpose(1, 0)).squeeze()
            obj_center_err = np.linalg.norm(pred_obj_center - sample['obj_center_3d']) * 1000.
        except:
            obj_center_err = None
        try:
            pred_obj_corners = (cam_extr @ np.array(pred_obj_pose['corners']).transpose(1, 0)).transpose(1, 0)
            obj_corner_err = np.mean(np.linalg.norm(pred_obj_corners - sample['obj_corners_3d'], axis=1)) * 1000.
        except:
            obj_corner_err = None

        pred_hand_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_hand.ply')
        gt_hand_mesh_path = os.path.join(self.cur_dir, 'data', 'mesh_data', 'mesh_hand', sample_idx + '.obj')
        try:
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
        except:
            chamfer_hand = None
            fscore_hand_1 = None
            fscore_hand_5 = None

        pred_obj_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_obj.ply')
        gt_obj_mesh_path = os.path.join(self.cur_dir, 'data', 'mesh_data', 'mesh_obj', sample_idx + '.obj')
        try:
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
        except:
            chamfer_obj = None
            fscore_obj_5 = None
            fscore_obj_10 = None
        
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
        
    def _evaluate_video(self, output_path, idx):
        sample = self.data[idx]
        
        if self.video_mode and self.use_whole_video_test:
            sample_idx = '_'.join(sample[0]['id'].split('_')[:-1])
            ycb_id = _YCB_CLASSES[int(sample[0]['ycb_id'])]
        else:
            sample_idx = sample['id']
            ycb_id = _YCB_CLASSES[int(sample['ycb_id'])]

        pred_obj_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_obj.ply')
        gt_obj_mesh_path = os.path.join(self.cur_dir, 'data', 'models', ycb_id, 'textured_simple.obj')
        try:
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

            threshold = 0.1 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 0.5 # 10 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        except:
            chamfer_obj = None
            fscore_obj_1 = None
            fscore_obj_5 = None
        
        error_dict = {}
        error_dict['id'] = sample_idx
        error_dict['object_name'] = ycb_id
        error_dict['chamfer_obj'] = chamfer_obj
        error_dict['fscore_obj_1'] = fscore_obj_1
        error_dict['fscore_obj_5'] = fscore_obj_5

        return error_dict

if __name__ == "__main__":
    db = dexycb('train_s0_29k', video_mode=True)
