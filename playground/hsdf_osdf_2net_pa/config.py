#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :config.py
#@Date        :2022/04/08 09:47:39
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout


cfg = CN()

cfg.task = 'hsdf_osdf_2net_pa'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.sdf_result_dir = '.'
cfg.cls_sdf_result_dir = '.'
cfg.hand_pose_result_dir = '.'
cfg.obj_pose_result_dir = '.'
cfg.ckpt = '.'

## dataset
cfg.trainset_3d = 'obman'
cfg.trainset_3d_split = '87k'
cfg.testset = 'obman'
cfg.testset_split = '6k'
cfg.testset_hand_source = osp.join(cfg.testset, 'data/test/mesh_hand')
cfg.testset_obj_source = osp.join(cfg.testset, 'data/test/mesh_obj')
cfg.num_testset_samples = 6285
cfg.mesh_resolution = 128
cfg.point_batch_size = 2 ** 18
cfg.output_part_label = False
cfg.vis_part_label = False
cfg.chamfer_optim = True

## model setting
cfg.backbone_pose = 'resnet_18'
cfg.backbone_shape = 'resnet_18'
cfg.mano_pca_latent = 15
cfg.sdf_latent = 256
cfg.hand_point_latent = 3
cfg.obj_point_latent = 3
cfg.hand_encode_style = 'nerf'
cfg.obj_encode_style = 'nerf'
cfg.rot_style = '6d'
cfg.hand_branch = True
cfg.obj_branch = True
cfg.hand_cls = False
cfg.obj_rot = False
cfg.with_add_feats = True

cfg.sdf_head = CN()
cfg.sdf_head.layers = 5
cfg.sdf_head.dims = [512 for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.dropout = [i for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.norm_layers = [i for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.dropout_prob = 0.2
cfg.sdf_head.latent_in = [(cfg.sdf_head.layers - 1) // 2]
cfg.sdf_head.num_class = 6

## training config
cfg.image_size = (256, 256)
cfg.heatmap_size = (64, 64, 64)
cfg.depth_dim = 0.28
cfg.warm_up_epoch = 0
cfg.lr_dec_epoch = [600, 1200]
cfg.end_epoch = 1600
cfg.sdf_add_epoch = 1201
cfg.lr = 1e-4
cfg.lr_dec_style = 'step'
cfg.lr_dec_factor = 0.5
cfg.train_batch_size = 64
cfg.num_sample_points = 2000
cfg.clamp_dist = 0.05
cfg.recon_scale = 6.5
cfg.hand_sdf_weight = 0.5
cfg.obj_sdf_weight = 0.5
cfg.hand_cls_weight = 0.05
cfg.volume_weight = 0.5
cfg.corner_weight = 0.5
cfg.use_inria_aug = False
cfg.norm_factor = 0.02505871

## testing config
cfg.test_batch_size = 1
cfg.test_with_gt = False

## others
cfg.use_lmdb = False
cfg.num_threads = 6
cfg.gpu_ids = (0, 1, 2, 3)
cfg.num_gpus = 4
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 100

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.trainset_3d + cfg.trainset_3d_split, cfg.backbone_pose.replace('_', ''), cfg.backbone_shape.replace('_', ''), 'h' + str(int(cfg.hand_branch)), 'o' + str(int(cfg.obj_branch)), 'sdf' + str(cfg.sdf_head.layers), 'cls' + str(int(cfg.hand_cls)), 'rot' + str(int(cfg.obj_rot)), 'hand_' + cfg.hand_encode_style + '_' + str(cfg.hand_point_latent), 'obj_' + cfg.obj_encode_style + '_' + str(cfg.obj_point_latent), 'np' + str(cfg.num_sample_points), 'adf' + str(int(cfg.with_add_feats)), 'e' + str(cfg.end_epoch), 'ae' + str(cfg.sdf_add_epoch), 'scale' + str(cfg.recon_scale), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'hsw' + str(cfg.hand_sdf_weight), 'osw' + str(cfg.obj_sdf_weight), 'hcw' + str(cfg.hand_cls_weight), 'vw' + str(cfg.volume_weight)]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset, 'gt', str(int(cfg.test_with_gt))]))
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = osp.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = osp.join(cfg.result_dir, 'obj_pose')

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    else:
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset, 'gt', str(int(cfg.test_with_gt))]))
        cfg.log_dir = osp.join(cfg.output_dir, 'test_log')
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
