#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :reconstruct.py
#@Date        :2022/04/25 21:53:05
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr


import numpy as np
import trimesh
import skimage.measure
import time
import torch
import os
import yaml
import json
from loguru import logger
from config import cfg
from utils.sdf_utils import kinematic_embedding, pixel_align
from utils.mesh import customized_export_ply
from utils.solver import icp_ts
from mano.manolayer import ManoLayer
from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS, _SERIALS


def reconstruct(cfg, filename, model, latent_vec, inputs, metas, hand_pose_results, obj_pose_results):
    model.eval()
    if cfg.hand_branch:
        hand_sdf_decoder = model.module.hand_sdf_head
    
    if cfg.obj_branch:
        obj_sdf_decoder = model.module.obj_sdf_head

    ply_filename_hand = filename[0] + "_hand"
    ply_filename_obj = filename[0] + "_obj"

    if cfg.hand_branch:
        hand_sdf_decoder.eval()
    
    if cfg.obj_branch:
        obj_sdf_decoder.eval()

    N = cfg.mesh_resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 5)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False

    head = 0
    max_batch = cfg.point_batch_size
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        if cfg.hand_encode_style == 'kine':
            if cfg.test_with_gt:
                mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
                gt_hand_pose_results = {}
                gt_hand_pose_results['global_trans'] = gt_global_trans
                gt_hand_pose_results['rot_center'] = gt_rot_center
                hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_hand_pose_results, 'hand')
            else:
                hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], hand_pose_results, 'hand')
        elif cfg.hand_encode_style == 'gt_kine':
            mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
            _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
            gt_hand_pose_results = {}
            gt_hand_pose_results['global_trans'] = gt_global_trans
            gt_hand_pose_results['rot_center'] = gt_rot_center
            hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_hand_pose_results, 'hand')
        else:
            hand_sample_subset = sample_subset
        hand_sample_subset = hand_sample_subset.reshape((-1, cfg.hand_point_latent))

        if cfg.obj_encode_style == 'kine':
            if cfg.test_with_gt:
                gt_obj_pose_results = {}
                gt_obj_pose = metas['obj_transform']
                if not cfg.obj_rot:
                    gt_obj_pose[:, :3, :3] = torch.eye(3)
                gt_obj_pose_results['global_trans'] = gt_obj_pose
                obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
            else:
                obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], obj_pose_results, 'obj')
        elif cfg.obj_encode_style == 'gt_trans':
            gt_obj_pose_results = {}
            gt_obj_pose = metas['obj_transform']
            gt_obj_pose[:, :3, :3] = torch.eye(3)
            gt_obj_pose_results['global_trans'] = gt_obj_pose
            obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
        elif cfg.obj_encode_style == 'gt_transrot':
            gt_obj_pose_results = {}
            gt_obj_pose = metas['obj_transform']
            gt_obj_pose_results['global_trans'] = gt_obj_pose
            obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
        else:
            obj_sample_subset = sample_subset
        obj_sample_subset = obj_sample_subset.reshape((-1, cfg.obj_point_latent))

        latent_vec_feat = model.module.backbone_2_sdf(latent_vec)
        latent_vec_feat, _ = pixel_align(cfg, sample_subset, sample_subset.shape[0], latent_vec_feat, metas['hand_center_3d'], metas['cam_intr'])
        latent_vec_sdf = model.module.sdf_encoder(latent_vec_feat)

        if cfg.hand_branch:
            sdf_hand, _ = decode_sdf(hand_sdf_decoder, latent_vec_sdf, hand_sample_subset, 'hand')
            samples[head : min(head + max_batch, num_samples), 3] = sdf_hand.squeeze(1).detach().cpu()

        if cfg.obj_branch:
            sdf_obj = decode_sdf(obj_sdf_decoder, latent_vec_sdf, obj_sample_subset, 'obj')
            samples[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()

        head += max_batch

    if cfg.hand_branch:
        sdf_values_hand = samples[:, 3]
        sdf_values_hand = sdf_values_hand.reshape(N, N, N)
    else:
        sdf_values_hand = None
    
    if cfg.obj_branch:
        sdf_values_obj = samples[:, 4]
        sdf_values_obj = sdf_values_obj.reshape(N, N, N)
    else:
        sdf_values_obj = None

    ###### high resolution ######
    new_voxel_size, new_origin = get_higher_res_cube(sdf_values_hand, sdf_values_obj, voxel_origin, voxel_size)
    samples_hr = torch.zeros(N ** 3, 5)

    # transform first 3 columns
    # to be the x, y, z index
    samples_hr[:, 2] = overall_index % N
    samples_hr[:, 1] = (overall_index.long() / N) % N
    samples_hr[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples_hr[:, 0] = (samples_hr[:, 0] * new_voxel_size) + new_origin[0]
    samples_hr[:, 1] = (samples_hr[:, 1] * new_voxel_size) + new_origin[1]
    samples_hr[:, 2] = (samples_hr[:, 2] * new_voxel_size) + new_origin[2]

    samples_hr.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples_hr[head : min(head + max_batch, num_samples), 0:3].cuda()

        if cfg.hand_encode_style == 'kine':
            if cfg.test_with_gt:
                mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
                gt_hand_pose_results = {}
                gt_hand_pose_results['global_trans'] = gt_global_trans
                gt_hand_pose_results['rot_center'] = gt_rot_center
                hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_hand_pose_results, 'hand')
            else:
                hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], hand_pose_results, 'hand')
        elif cfg.hand_encode_style == 'gt_kine':
            mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
            _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
            gt_hand_pose_results = {}
            gt_hand_pose_results['global_trans'] = gt_global_trans
            gt_hand_pose_results['rot_center'] = gt_rot_center
            hand_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_hand_pose_results, 'hand')
        else:
            hand_sample_subset = sample_subset
        hand_sample_subset = hand_sample_subset.reshape((-1, cfg.hand_point_latent))

        if cfg.obj_encode_style == 'kine':
            if cfg.test_with_gt:
                gt_obj_pose_results = {}
                gt_obj_pose = metas['obj_transform']
                if not cfg.obj_rot:
                    gt_obj_pose[:, :3, :3] = torch.eye(3)
                gt_obj_pose_results['global_trans'] = gt_obj_pose
                obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
            else:
                obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], obj_pose_results, 'obj')
        elif cfg.obj_encode_style == 'gt_trans':
            gt_obj_pose_results = {}
            gt_obj_pose = metas['obj_transform']
            gt_obj_pose[:, :3, :3] = torch.eye(3)
            gt_obj_pose_results['global_trans'] = gt_obj_pose
            obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
        elif cfg.obj_encode_style == 'gt_transrot':
            gt_obj_pose_results = {}
            gt_obj_pose = metas['obj_transform']
            gt_obj_pose_results['global_trans'] = gt_obj_pose
            obj_sample_subset = kinematic_embedding(cfg, sample_subset, sample_subset.shape[0], gt_obj_pose_results, 'obj')
        else:
            obj_sample_subset = sample_subset
        obj_sample_subset = obj_sample_subset.reshape((-1, cfg.obj_point_latent))

        latent_vec_feat = model.module.backbone_2_sdf(latent_vec)
        latent_vec_feat, _ = pixel_align(cfg, sample_subset, sample_subset.shape[0], latent_vec_feat, metas['hand_center_3d'], metas['cam_intr'])
        latent_vec_sdf = model.module.sdf_encoder(latent_vec_feat)

        if cfg.hand_branch:
            sdf_hand, predicted_class = decode_sdf(hand_sdf_decoder, latent_vec_sdf, hand_sample_subset, 'hand')
            samples_hr[head : min(head + max_batch, num_samples), 3] = sdf_hand.squeeze(1).detach().cpu()

        if cfg.obj_branch:
            sdf_obj = decode_sdf(obj_sdf_decoder, latent_vec_sdf, obj_sample_subset, 'obj')
            samples_hr[head : min(head + max_batch, num_samples), 4] = sdf_obj.squeeze(1).detach().cpu()

        head += max_batch

    sdf_values_hand = samples_hr[:, 3]
    sdf_values_hand = sdf_values_hand.reshape(N, N, N)
    sdf_values_obj = samples_hr[:, 4]
    sdf_values_obj = sdf_values_obj.reshape(N, N, N)

    voxel_size = new_voxel_size
    voxel_origin = new_origin.tolist()

    if cfg.hand_branch:
        vertices, mesh_faces, offset, scale = convert_sdf_samples_to_ply(sdf_values_hand.data.cpu(), voxel_origin, voxel_size, True, ply_filename_hand + ".ply", offset=None, scale=None)

    if cfg.obj_branch:
        convert_sdf_samples_to_ply(sdf_values_obj.data.cpu(), voxel_origin, voxel_size, False, ply_filename_obj + ".ply", offset, scale)


def decode_sdf(sdf_decoder, latent_vector, points, mode):
    # points: N x points_dim_embeddding
    inputs = torch.cat([latent_vector, points], 1)

    if mode == 'hand':
        sdf_val, predicted_class = sdf_decoder(inputs)
        return sdf_val, predicted_class
    else:
        sdf_val, _ = sdf_decoder(inputs)
        return sdf_val


def get_higher_res_cube(sdf_values_hand, sdf_values_obj, voxel_origin, voxel_size):
    N = cfg.mesh_resolution
    if cfg.hand_branch:
        indices = torch.nonzero(sdf_values_hand < 0).float()
        if indices.shape[0] == 0:
            min_hand = torch.Tensor([0., 0., 0.])
            max_hand = torch.Tensor([0., 0., 0.])
        else:
            x_min_hand = torch.min(indices[:,0])
            y_min_hand = torch.min(indices[:,1])
            z_min_hand = torch.min(indices[:,2])
            min_hand = torch.Tensor([x_min_hand, y_min_hand, z_min_hand])

            x_max_hand = torch.max(indices[:,0])
            y_max_hand = torch.max(indices[:,1])
            z_max_hand = torch.max(indices[:,2])
            max_hand = torch.Tensor([x_max_hand, y_max_hand, z_max_hand])

    if cfg.obj_branch:
        indices = torch.nonzero(sdf_values_obj < 0).float()
        if indices.shape[0] == 0:
            min_obj = torch.Tensor([0., 0., 0.])
            max_obj = torch.Tensor([0., 0., 0.])
        else:
            x_min_obj = torch.min(indices[:,0])
            y_min_obj = torch.min(indices[:,1])
            z_min_obj = torch.min(indices[:,2])
            min_obj = torch.Tensor([x_min_obj, y_min_obj, z_min_obj])

            x_max_obj = torch.max(indices[:,0])
            y_max_obj = torch.max(indices[:,1])
            z_max_obj = torch.max(indices[:,2])
            max_obj = torch.Tensor([x_max_obj, y_max_obj, z_max_obj])

    if not cfg.obj_branch:
        min_index = min_hand
        max_index = max_hand
    elif not cfg.hand_branch:
        min_index = min_obj
        max_index = max_obj
    else:
        min_index = torch.min(min_hand, min_obj)
        max_index = torch.max(max_hand, max_obj)

    # Buffer 2 voxels each side
    new_cube_size = (torch.max(max_index - min_index) + 4) * voxel_size

    new_voxel_size = new_cube_size / (N - 1)
    # [z,y,x]
    new_origin = (min_index - 2 ) * voxel_size - 1.0  # (-1,-1,-1) origin

    return new_voxel_size, new_origin


def convert_sdf_samples_to_ply(sdf_tensor, voxel_origin, voxel_size, is_hand, ply_filename_out, offset=None, scale=None):
    """
    Convert sdf samples to .ply
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    sdf_tensor = sdf_tensor.numpy()
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_tensor, level=0.0, spacing=[voxel_size] * 3)
        with open(os.path.join(cfg.hand_pose_result_dir, '_'.join(ply_filename_out.split('_')[:-1]) + '.json'), 'r') as f:
            data = json.load(f)
            cam_extr = np.array(data['cam_extr'], dtype=np.float32)
            verts = (cam_extr @ verts.transpose(1, 0)).transpose(1, 0)
    except:
        logger.warning("Cannot reconstruct mesh from '{}'".format(ply_filename_out))
        return None, None, np.array([0,0,0]), np.array([1 / cfg.recon_scale])

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset
    
    pred_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, process=False)
    split_mesh = trimesh.graph.split(pred_mesh)

    if len(split_mesh) > 1:
        max_area = -1
        final_mesh = split_mesh[0]
        for per_mesh in split_mesh:
            if per_mesh.area > max_area:
                max_area = per_mesh.area
                final_mesh = per_mesh
        pred_mesh = final_mesh

    trans = np.array([0, 0, 0])
    scale = np.array([1])
    if not is_hand:
        pred_mesh.export(os.path.join(cfg.sdf_result_dir, ply_filename_out))
        return verts, faces, trans, scale
    else:
        if cfg.chamfer_optim:
            if cfg.trainset_3d == 'obman':
                gt_mesh = trimesh.load(os.path.join(cfg.data_dir, cfg.testset_hand_source, ply_filename_out.split('_')[0] + '.obj'), process=False)
            elif cfg.trainset_3d == 'dexycb':
                gt_mesh = trimesh.load(os.path.join(cfg.data_dir, cfg.testset_hand_source, '_'.join(ply_filename_out.split('_')[:-1]) + '.obj'), process=False)
            elif cfg.testset == 'ho3dv3':
                if 's1' in cfg.testset_split:
                    gt_mesh = trimesh.load(os.path.join(cfg.data_dir, cfg.testset_hand_source, ply_filename_out.split('_')[0] + '.obj'), process=False)
                else:
                    pred_mesh.export(os.path.join(cfg.sdf_result_dir, ply_filename_out))
                    return verts, faces, trans, scale

            icp_solver = icp_ts(pred_mesh, gt_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter = 100)
            icp_solver.export_source_mesh(os.path.join(cfg.sdf_result_dir, ply_filename_out))
            trans, scale = icp_solver.get_trans_scale()
        else:
            pred_mesh.export(os.path.join(cfg.sdf_result_dir, ply_filename_out))

        return verts, faces, trans, scale


def write_verts_label_to_obj(xyz_tensor, label_tensor, obj_filename_out, offset=None, scale=None):
    mesh_points = xyz_tensor.data.numpy()
    label_tensor = label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset
   
    with open(obj_filename_out, 'w') as fp:
        for idx, v in enumerate(mesh_points):
            clr = numpy_label_tensor[idx] * 45.0
            fp.write('v %.4f %.4f %.4f %.2f %.2f %.2f\n' % ( v[0], v[1], v[2], clr, clr, clr) )


def write_verts_label_to_npz(xyz_tensor, label_tensor, npz_filename_out, offset=None, scale=None):
    mesh_points = xyz_tensor.data.numpy()
    label_tensor = label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset
    
    np.savez(npz_filename_out, points=mesh_points, labels=label_tensor)


def write_color_labeled_ply(xyz_tensor, faces, label_tensor, ply_filename_out, offset=None, scale=None):
    mesh_points = xyz_tensor.data.numpy()
    label_tensor = label_tensor.numpy()

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points * scale
    if offset is not None:
        mesh_points = mesh_points + offset
    
    part_color = np.array([[13, 212, 128], [250, 70, 42], [131, 66, 37], [78, 137, 54], [187, 246, 163], [67, 220, 74]]).astype(np.uint8)

    vertex_color = np.ones((mesh_points.shape[0], 3), dtype=np.uint8)
    vertex_color[:, 0:3] = part_color[label_tensor.astype(np.int32), :]

    customized_export_ply(outfile_name=ply_filename_out, v=mesh_points, f=faces, v_c=vertex_color)
