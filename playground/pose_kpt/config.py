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

cfg.task = 'pose_kpt'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.hand_pose_result_dir = '.'
cfg.obj_pose_result_dir = '.'

## dataset
cfg.trainset_3d = 'obman'
cfg.trainset_3d_split = '138k'
cfg.testset = 'obman'
cfg.testset_split = '6k'
cfg.num_testset_samples = 6285

## model setting
cfg.backbone = 'resnet_18'
cfg.hand_branch = True
cfg.obj_branch = True
cfg.obj_rot = True
# it could be 'axisang' - axis angles, '6d'- 6d rotation representation
cfg.rot_style = '6d'

## training config
cfg.image_size = (256, 256)
cfg.heatmap_size = (64, 64, 64)
cfg.depth_dim = 0.28
cfg.warm_up_epoch = 0
cfg.lr_dec_epoch = [40, 80, 120]
cfg.end_epoch = 100
cfg.lr = 1e-4
cfg.lr_dec_style = 'step'
cfg.lr_dec_factor = 0.5
cfg.train_batch_size = 64
cfg.volume_weight = 1.0
cfg.corner_weight = 1.0
cfg.hand_ordinal_weight = 0.0
cfg.scene_ordinal_weight = 1.0
cfg.use_inria_aug = False
cfg.norm_coords = False
cfg.norm_factor = 0.02505871

## testing config
cfg.test_batch_size = 1

## others
cfg.use_lmdb = False
cfg.num_threads = 6
cfg.gpu_ids = (0, 1, 2, 3)
cfg.num_gpus = 4
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 5

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.trainset_3d + cfg.trainset_3d_split, cfg.backbone.replace('_', ''), 'rot' + str(int(cfg.obj_rot)), cfg.rot_style, 'h' + str(int(cfg.hand_branch)), 'o' + str(int(cfg.obj_branch)), 'norm' + str(int(cfg.norm_coords)), 'e' + str(cfg.end_epoch), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'vw' + str(cfg.volume_weight), 'ocrw' + str(cfg.corner_weight), 'how' + str(cfg.hand_ordinal_weight), 'sow' + str(cfg.scene_ordinal_weight)]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset]))
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    else:
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset]))
        cfg.log_dir = osp.join(cfg.output_dir, 'test_log')
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.testset))