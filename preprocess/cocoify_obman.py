#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :cocoify_obman.py
#@Date        :2022/04/20 16:18:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import numpy as np
import torch
from torch import nn
import os
import os.path as osp
from tqdm import tqdm
from fire import Fire
import json
import pickle
import scipy.linalg
import trimesh
import sys
from glob import glob
import cv2
import lmdb
sys.path.insert(0, '../common')
from mano.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R


mano_faces = np.load('../common/mano/assets/closed_fmano.npy')
mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=True, flat_hand_mean=True)


def get_obj_corners(mesh_path, obj_transform):
    mesh = trimesh.load(mesh_path, process=False)
    verts = mesh.vertices

    inv_obj_transform = np.linalg.inv(obj_transform)
    homo_verts = np.concatenate([verts, np.ones((verts.shape[0], 1))], 1)
    homo_rest_verts = np.dot(inv_obj_transform, homo_verts.transpose(1, 0)).transpose(1, 0)
    rest_verts = homo_rest_verts[:, :3] / homo_rest_verts[:, [3]]

    min_verts = rest_verts.min(0)
    max_verts = rest_verts.max(0)

    obj_rest_corners = np.zeros((9, 3), dtype=np.float32)
    obj_rest_corners[0] = np.array([0., 0., 0.])
    obj_rest_corners[1] = np.array([min_verts[0], min_verts[1], min_verts[2]])
    obj_rest_corners[2] = np.array([min_verts[0], max_verts[1], min_verts[2]])
    obj_rest_corners[3] = np.array([max_verts[0], min_verts[1], min_verts[2]])
    obj_rest_corners[4] = np.array([max_verts[0], max_verts[1], min_verts[2]])
    obj_rest_corners[5] = np.array([min_verts[0], min_verts[1], max_verts[2]])
    obj_rest_corners[6] = np.array([min_verts[0], max_verts[1], max_verts[2]])
    obj_rest_corners[7] = np.array([max_verts[0], min_verts[1], max_verts[2]])
    obj_rest_corners[8] = np.array([max_verts[0], max_verts[1], max_verts[2]])

    homo_obj_rest_corners = np.concatenate([obj_rest_corners, np.ones((obj_rest_corners.shape[0], 1))], 1)
    homo_obj_corners = np.dot(obj_transform, homo_obj_rest_corners.transpose(1, 0)).transpose(1, 0)
    obj_corners = homo_obj_corners[:, :3] / homo_obj_corners[:, [3]]

    return obj_corners, obj_rest_corners


def preprocess(data_root='../datasets/obman', mode='train'):
    data_path = osp.join(data_root, 'data', mode)
    meta_path = osp.join(data_path, 'meta')
    norm_path = osp.join(data_path, 'norm')
    mesh_obj_path = osp.join(data_path, 'mesh_obj')
    cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)

    with open(f'{data_root}/obman_{mode}.json', 'w') as json_data:
        coco_file = dict()
        data_images = []
        data_annos = []
        for filename in tqdm(os.listdir(meta_path)):
            filename_prefix = filename.split('.')[0]
            if mode == 'train' and not osp.exists(osp.join(norm_path, filename_prefix + '.npz')):
                continue

            img_info = dict()
            anno_info = dict()

            meta_data_file = osp.join(meta_path, filename)
            with open(meta_data_file, 'rb') as f:
                meta_data = pickle.load(f)

            if mode == 'train':
                norm_data = np.load(osp.join(norm_path, filename_prefix + '.npz'))
            
            sample_id = int(filename_prefix)
            img_info['id'] = sample_id
            img_info['file_name'] = filename_prefix
            data_images.append(img_info)

            anno_info['id'] = sample_id
            anno_info['image_id'] = sample_id
            # format: [x0, y0, w, h]
            anno_info['bbox'] = [0, 0, 256, 256]
            anno_info['hand_joints_3d'] = np.dot(cam_extr, meta_data['coords_3d'].transpose(1, 0)).transpose(1, 0).tolist()
            hand_verts_3d = np.dot(cam_extr, meta_data['verts_3d'].transpose(1, 0)).transpose(1, 0).tolist()

            # approximate the hand wrist rotation
            hand_poses = np.zeros(48)
            mano_pose = torch.zeros((1, 48))
            mano_pca = torch.from_numpy(meta_data['pca_pose']).float()
            mano_pose[:, 3:] = mano_pca
            verts, joints, full_hand_pose, global_trans, rot_center = mano_layer(mano_pose, root_palm=False)
            canonical_verts = verts.squeeze(0).numpy()
            wrist_matrix, _, _ = trimesh.registration.procrustes(canonical_verts, np.array(hand_verts_3d), reflection=False, scale=False)
            wrist_rot = R.from_matrix(wrist_matrix[:3, :3]).as_rotvec()
            hand_poses[:3] = wrist_rot
            hand_poses[3:] = full_hand_pose[0, 3:].squeeze(0).numpy()
            anno_info['hand_poses'] = hand_poses.tolist()

            obj_affine_transform = meta_data['affine_transform']
            obj_affine_transform[:3, :3] = scipy.linalg.polar(obj_affine_transform[:3, :3])[0]
            obj_affine_transform[1, :] *= -1
            obj_affine_transform[2, :] *= -1
            anno_info['obj_transform'] = obj_affine_transform.tolist()
            obj_corners_3d, obj_rest_corners_3d = get_obj_corners(osp.join(mesh_obj_path, filename_prefix + '.obj'), obj_affine_transform)
            anno_info['obj_center_3d'] = obj_corners_3d[0, :].tolist()
            anno_info['obj_corners_3d'] = obj_corners_3d[1:, :].tolist()
            anno_info['obj_rest_corners_3d'] = obj_rest_corners_3d[1:, :].tolist()
            if mode == 'train':
                anno_info['sdf_scale'] = norm_data['scale'].tolist()
                anno_info['sdf_offset'] = norm_data['offset'].tolist()
            data_annos.append(anno_info)
        
        coco_file['images'] = data_images
        coco_file['annotations'] = data_annos
        json.dump(coco_file, json_data, indent=2)
    

def create_lmdb(data_root='../datasets/obman', mode='train'):
    opt = dict()
    opt['image'] = dict()
    opt['image']['name'] = f'obman_rgb_{mode}'
    opt['image']['data_folder'] = f'{data_root}/data/train/rgb'
    opt['image']['lmdb_save_path'] = f'{data_root}/data/train/rgb.lmdb'
    opt['image']['commit_interval'] = 100

    opt['seg'] = dict()
    opt['seg']['name'] = f'obman_seg_{mode}'
    opt['seg']['data_folder'] = f'{data_root}/data/train/segm'
    opt['seg']['lmdb_save_path'] = f'{data_root}/data/train/segm.lmdb'
    opt['seg']['commit_interval'] = 100

    opt['sdf_hand'] = dict()
    opt['sdf_hand']['name'] = f'obman_sdf_hand_{mode}'
    opt['sdf_hand']['data_folder'] = f'{data_root}/data/train/sdf_hand'
    opt['sdf_hand']['lmdb_save_path'] = f'{data_root}/data/train/sdf_hand.lmdb'
    opt['sdf_hand']['commit_interval'] = 100
    opt['sdf_hand']['is_hand'] = True

    opt['sdf_obj'] = dict()
    opt['sdf_obj']['name'] = f'obman_sdf_obj_{mode}'
    opt['sdf_obj']['data_folder'] = f'{data_root}/data/train/sdf_obj'
    opt['sdf_obj']['lmdb_save_path'] = f'{data_root}/data/train/sdf_obj.lmdb'
    opt['sdf_obj']['commit_interval'] = 100
    opt['sdf_obj']['is_hand'] = False

    general_image_folder(opt['image'])
    general_seg_folder(opt['seg'])
    general_sdf_folder(opt['sdf_hand'])
    general_sdf_folder(opt['sdf_obj'])


def general_image_folder(opt):
    img_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    print('Reading image path list ...')
    with open('../datasets/obman/splits/train_87k.json', 'r') as f:
        split_data = json.load(f)

    all_img_list = []
    for i in range(len(split_data)):
        all_img_list.append(os.path.join(img_folder, split_data[i] + '.jpg'))
    all_img_list = sorted(all_img_list)

    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path).split('.')[0])

    # create lmdb environment
    # estimate the space of the file
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
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
    img_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    print('Reading segmentation path list ...')
    with open('../datasets/obman/splits/train_87k.json', 'r') as f:
        split_data = json.load(f)

    all_img_list = []
    for i in range(len(split_data)):
        all_img_list.append(os.path.join(img_folder, split_data[i] + '.png'))
    all_img_list = sorted(all_img_list)

    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path).split('.')[0])

    # create lmdb environment
    # estimate the space of the file
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
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
    with open('../datasets/obman/splits/train_87k.json', 'r') as f:
        split_data = json.load(f)

    all_sdf_list = []
    for i in range(len(split_data)):
        all_sdf_list.append(os.path.join(sdf_folder, split_data[i] + '.npz'))
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
