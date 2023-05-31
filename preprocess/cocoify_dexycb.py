#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :cocoify_dexycb.py
#@Date        :2022/04/20 16:18:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

from distutils.log import debug
import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
from fire import Fire
import json
import pickle
import shutil
import trimesh
from scipy.spatial import cKDTree as KDTree
import sys
from glob import glob
import cv2
import lmdb
sys.path.insert(0, '../common')
from mano.manolayer import ManoLayer
from utils.img_utils import generate_patch_image, process_bbox
sys.path.insert(0, '..')
from datasets.dexycb.toolkit.factory import get_dataset
from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS, _SERIALS


def preprocess(data_root='../datasets/dexycb', split='s0', mode='test', side='right'):
    sdf_data_root = os.path.join(data_root, 'data', 'sdf_data')
    if mode == 'test':
        hand_mesh_data_root = os.path.join(data_root, 'data', 'mesh_data', 'mesh_hand')
        obj_mesh_data_root = os.path.join(data_root, 'data', 'mesh_data', 'mesh_obj')
        os.makedirs(hand_mesh_data_root, exist_ok=True)
        os.makedirs(obj_mesh_data_root, exist_ok=True)

    dataset_name = f'{split}_{mode}'
    dataset = get_dataset(dataset_name)
    selected_ids = []

    with open(f'{data_root}/dexycb_{split}_{mode}_t.json', 'w') as json_data:
        coco_file = dict()
        data_images = []
        data_annos = []

        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            img_info = dict()
            anno_info = dict()
            sample_id = i

            if sample['mano_side'] in side:
                img_path = sample['color_file']
                subject = int(img_path.split('/')[6].split('-')[-1])
                video_id = img_path.split('/')[7]
                sub_video_id = img_path.split('/')[8]
                frame_idx = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])

                if frame_idx % 5 != 0:
                    continue

                img_info['id'] = sample_id
                img_info['file_name'] = '_'.join([str(subject), video_id, sub_video_id, str(frame_idx)])

                if (os.path.exists(os.path.join(sdf_data_root, 'norm', img_info['file_name'] + '.npz')) and mode == 'train') or mode == 'test':
                    anno_info['id'] = sample_id
                    anno_info['image_id'] = sample_id
                    anno_info['fx'] = sample['intrinsics']['fx']
                    anno_info['fy'] = sample['intrinsics']['fy']
                    anno_info['cx'] = sample['intrinsics']['ppx']
                    anno_info['cy'] = sample['intrinsics']['ppy']

                    label = np.load(sample['label_file'])
                    mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=sample['mano_side'], mano_root='../common/mano/assets/', use_pca=True)
                    betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
                    hand_verts, _, hand_poses, _, _ = mano_layer(torch.from_numpy(label['pose_m'][:, 0:48]), betas, torch.from_numpy(label['pose_m'][:, 48:51]))
                    hand_verts = hand_verts[0].numpy().tolist()

                    anno_info['hand_joints_3d'] = label['joint_3d'][0].tolist()
                    anno_info['hand_poses'] = hand_poses[0].numpy().tolist()
                    anno_info['hand_trans'] = label['pose_m'][0, 48:].tolist()
                    anno_info['hand_shapes'] = sample['mano_betas']

                    grasp_obj_id = sample['ycb_ids'][sample['ycb_grasp_ind']]
                    anno_info['ycb_id'] = grasp_obj_id
                    obj_rest_mesh = trimesh.load(dataset.obj_file[grasp_obj_id], process=False)
                    offset = (obj_rest_mesh.vertices.min(0) + obj_rest_mesh.vertices.max(0)) / 2
                    obj_rest_corners = trimesh.bounds.corners(obj_rest_mesh.bounds) - offset
                    pose_y = label['pose_y'][sample['ycb_grasp_ind']]
                    R, t = pose_y[:3, :3], pose_y[:, 3:]
                    new_t = R @ offset.reshape(3, 1) + t
                    obj_affine_transform = np.concatenate([np.concatenate([R, new_t], axis=1), np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
                    obj_corners = (R @ obj_rest_corners.transpose(1, 0) + new_t).transpose(1, 0)

                    homo_obj_verts = np.ones((obj_rest_mesh.vertices.shape[0], 4))
                    homo_obj_verts[:, :3] = obj_rest_mesh.vertices
                    obj_verts = np.dot(pose_y, homo_obj_verts.transpose(1, 0)).transpose(1, 0)
                    anno_info['obj_transform'] = obj_affine_transform.tolist()
                    anno_info['obj_center_3d'] = new_t.squeeze().tolist()
                    anno_info['obj_corners_3d'] = obj_corners.tolist()
                    anno_info['obj_rest_corners_3d'] = obj_rest_corners.tolist()
                
                    hand_joints_2d = label['joint_2d'][0]
                    if np.all(hand_joints_2d) == -1.0:
                        continue
                    obj_corners_2d = np.zeros((8, 2))
                    obj_corners_2d[:, 0] = anno_info['fx'] * obj_corners[:, 0] / obj_corners[:, 2] + anno_info['cx']
                    obj_corners_2d[:, 1] = anno_info['fy'] * obj_corners[:, 1] / obj_corners[:, 2] + anno_info['cy']
                    tl = np.min(np.concatenate([hand_joints_2d, obj_corners_2d], axis=0), axis=0)
                    br = np.max(np.concatenate([hand_joints_2d, obj_corners_2d], axis=0), axis=0)
                    box_size = br - tl
                    bbox = np.concatenate([tl-10, box_size+20],axis=0)
                    bbox = process_bbox(bbox)
                    anno_info['bbox'] = bbox.tolist()

                    if mode == 'train':
                        sdf_norm_data = np.load(os.path.join(sdf_data_root, 'norm', img_info['file_name'] + '.npz'))
                        anno_info['sdf_scale'] = sdf_norm_data['scale'].tolist()
                        anno_info['sdf_offset'] = sdf_norm_data['offset'].tolist()
                    elif mode == 'test':
                        hand_points_kd_tree = KDTree(hand_verts)
                        obj2hand_distances, _ = hand_points_kd_tree.query(obj_verts)
                        if obj2hand_distances.min() > 0.005:
                            continue
                        
                        hand_faces = np.load('../common/mano/assets/closed_fmano.npy')
                        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
                        obj_faces = obj_rest_mesh.faces
                        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
                        mesh_filename = '_'.join([str(subject), video_id, sub_video_id, str(frame_idx)]) + '.obj'
                        hand_mesh.export(os.path.join(hand_mesh_data_root, mesh_filename))
                        obj_mesh.export(os.path.join(obj_mesh_data_root, mesh_filename))
                    
                    selected_ids.append(str(sample_id).rjust(8, '0'))
                    data_images.append(img_info)
                    data_annos.append(anno_info)

        coco_file['images'] = data_images
        coco_file['annotations'] = data_annos
        json.dump(coco_file, json_data, indent=2)

        with open(f'{data_root}/splits/{split}_{mode}.json', 'w') as f:
            json.dump(selected_ids, f, indent=2)


def create_lmdb(data_root='../datasets/dexycb', mode='train', split='s0'):
    opt = dict()
    opt['image'] = dict()
    opt['image']['name'] = f'dexycb_rgb_{mode}_{split}'
    opt['image']['data_folder'] = f'{data_root}/data'
    opt['image']['lmdb_save_path'] = f'{data_root}/data/rgb_{split}.lmdb'
    opt['image']['commit_interval'] = 100
    opt['image']['split'] = split

    opt['seg'] = dict()
    opt['seg']['name'] = f'dexycb_seg_{mode}_{split}'
    opt['seg']['data_folder'] = f'{data_root}/data'
    opt['seg']['lmdb_save_path'] = f'{data_root}/data/seg_{split}.lmdb'
    opt['seg']['commit_interval'] = 100
    opt['seg']['split'] = split

    opt['sdf_hand'] = dict()
    opt['sdf_hand']['name'] = f'dexycb_sdf_hand_{mode}_{split}'
    opt['sdf_hand']['data_folder'] = f'{data_root}/data/sdf_data/sdf_hand'
    opt['sdf_hand']['lmdb_save_path'] = f'{data_root}/data/sdf_hand_{split}.lmdb'
    opt['sdf_hand']['commit_interval'] = 100
    opt['sdf_hand']['is_hand'] = True
    opt['sdf_hand']['split'] = split

    opt['sdf_obj'] = dict()
    opt['sdf_obj']['name'] = f'dexycb_sdf_obj_{mode}_{split}'
    opt['sdf_obj']['data_folder'] = f'{data_root}/data/sdf_data/sdf_obj'
    opt['sdf_obj']['lmdb_save_path'] = f'{data_root}/data/sdf_obj_{split}.lmdb'
    opt['sdf_obj']['commit_interval'] = 100
    opt['sdf_obj']['is_hand'] = False
    opt['sdf_obj']['split'] = split

    general_image_folder(opt['image'])
    general_seg_folder(opt['seg'])
    general_sdf_folder(opt['sdf_hand'])
    general_sdf_folder(opt['sdf_obj'])


def general_image_folder(opt):
    split = opt['split']
    img_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    print('Reading image path list ...')
    with open(f'../datasets/dexycb/dexycb_train_{split}.json', 'r') as f:
        json_data = json.load(f)
    anno_data = json_data['annotations']
    img_data = json_data['images']

    # create lmdb environment
    # estimate the space of the file
    data_size_per_img = np.zeros((256, 256, 3), dtype=np.uint8).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_data)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    for idx in tqdm(range(len(img_data))):
        key = img_data[idx]['file_name']
        key_byte = key.encode('ascii')
        subject_id = _SUBJECTS[int(img_data[idx]['file_name'].split('_')[0]) - 1]
        video_id = '_'.join(img_data[idx]['file_name'].split('_')[1:3])
        cam_id = img_data[idx]['file_name'].split('_')[-2]
        frame_id = img_data[idx]['file_name'].split('_')[-1].rjust(6, '0')
        img_path = os.path.join(img_folder, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
        bbox = anno_data[idx]['bbox']
        original_img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        data, _ = generate_patch_image(original_img_data, bbox, (256, 256), 1, 0)
        
        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


def general_seg_folder(opt):
    split = opt['split']
    img_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    print('Reading segmentation path list ...')
    with open(f'../datasets/dexycb/dexycb_train_{split}.json', 'r') as f:
        json_data = json.load(f)
    anno_data = json_data['annotations']
    img_data = json_data['images']

    # create lmdb environment
    # estimate the space of the file
    data_size_per_img = np.zeros((256, 256, 1), dtype=np.uint8).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_data)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    for idx in tqdm(range(len(img_data))):
        key = img_data[idx]['file_name']
        key_byte = key.encode('ascii')
        subject_id = _SUBJECTS[int(img_data[idx]['file_name'].split('_')[0]) - 1]
        video_id = '_'.join(img_data[idx]['file_name'].split('_')[1:3])
        cam_id = img_data[idx]['file_name'].split('_')[-2]
        frame_id = img_data[idx]['file_name'].split('_')[-1].rjust(6, '0')
        seg_path = os.path.join(img_folder, subject_id, video_id, cam_id, 'labels_' + frame_id + '.npz')
        original_seg_data = np.load(seg_path)['seg'][:, :, None]
        bbox = anno_data[idx]['bbox']
        data, _ = generate_patch_image(original_seg_data, bbox, (256, 256), 1, 0)
        
        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


def general_sdf_folder(opt):
    split = opt['split']
    sdf_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    is_hand = opt['is_hand']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading sdf path list ...')
    with open(f'../datasets/dexycb/dexycb_train_{split}.json', 'r') as f:
        json_data = json.load(f)
    split_data = json_data['images']

    all_sdf_list = []
    for i in range(len(split_data)):
        all_sdf_list.append(os.path.join(sdf_folder, split_data[i]['file_name'] + '.npz'))
    all_sdf_list = sorted(all_sdf_list)

    keys = []
    for sdf_path in all_sdf_list:
        keys.append(os.path.basename(sdf_path).split('.')[0])
    
    # create lmdb environment
    # estimate the space of the file
    data_size_per_sdf = np.zeros((20000, 6), dtype=np.float32).nbytes
    print('data size per sdf is: ', data_size_per_sdf)
    data_size = data_size_per_sdf * len(all_sdf_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_sdf_list, keys)), total=len(all_sdf_list), leave=False)
    pos_num = []
    neg_num = []
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = np.load(path)

        pos_data = data['pos']
        pos_other_data = data['pos_other']
        if is_hand:
            lab_pos_data = data['lab_pos'][:, [0]]
        else:
            lab_pos_data = data['lab_pos_other'][:, [0]]

        neg_data = data['neg']
        neg_other_data = data['neg_other']
        if is_hand:
            lab_neg_data = data['lab_neg'][:, [0]]
        else:
            lab_neg_data = data['lab_neg_other'][:, [0]]

        pos_num.append(pos_data.shape[0])
        neg_num.append(neg_data.shape[0])

        if is_hand:
            pos_sample = np.concatenate((pos_data, pos_other_data, lab_pos_data), axis=1)
            neg_sample = np.concatenate((neg_data, neg_other_data, lab_neg_data), axis=1)
            data_sample_real = np.concatenate((pos_sample, neg_sample), axis=0)
            data_sample = np.zeros((20000, 6), dtype=np.float32)
            data_sample[:pos_data.shape[0] + neg_data.shape[0], :] = data_sample_real
        else:
            pos_sample = np.concatenate((pos_data[:, :3], pos_other_data, pos_data[:, [3]], lab_pos_data), axis=1)
            neg_sample = np.concatenate((neg_data[:, :3], neg_other_data, neg_data[:, [3]], lab_neg_data), axis=1)
            data_sample_real = np.concatenate((pos_sample, neg_sample), axis=0)
            data_sample = np.zeros((20000, 6), dtype=np.float32)
            data_sample[:pos_data.shape[0] + neg_data.shape[0], :] = data_sample_real
        
        txn.put(key_byte, data_sample)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    meta_info['pos_num'] = pos_num
    meta_info['neg_num'] = neg_num
    meta_info['dim'] = 6
    meta_info['keys'] = keys
    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


if __name__ == '__main__':
    # Fire(preprocess)
    Fire(create_lmdb)