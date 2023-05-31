#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :train.py
#@Date :2022/05/03 16:40:33
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr


import os
import sys
import argparse
import yaml
from tqdm import tqdm
import torch
from loguru import logger
import _init_paths
from _init_paths import add_path, this_dir
from utils.dir_utils import export_pose_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-e', required=True, type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--test_epoch', default=0, type=int)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    task = args.cfg.split('/')[-3]
    add_path(os.path.join('../playground', task))
    from config import cfg, update_config
    update_config(cfg, args, mode='test')
    from base import Tester
    if args.test_epoch == 0:
        args.test_epoch = cfg.end_epoch - 1

    local_rank = args.local_rank
    device = 'cuda:%d' % local_rank
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    logger.info('Distributed Process %d, Total %d.' % (args.local_rank, world_size))

    tester = Tester(local_rank, args.test_epoch)
    tester._make_batch_generator()
    tester._make_model(local_rank)

    with torch.no_grad():
        for itr, (inputs, metas) in tqdm(enumerate(tester.batch_generator)):
            for k, v in inputs.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        inputs[k][i] = inputs[k][i].cuda(non_blocking=True)
                else:
                    inputs[k] = inputs[k].cuda(non_blocking=True)

            for k, v in metas.items():
                if k != 'id' and k != 'obj_id':
                    if isinstance(v, list):
                        for i in range(len(v)):
                            metas[k][i] = metas[k][i].cuda(non_blocking=True)
                    else:
                        metas[k] = metas[k].cuda(non_blocking=True)

            # forward
            if cfg.task in ['hsdf_osdf_1net', 'hsdf_osdf_2net', 'hsdf_osdf_2net_pa', 'hsdf_osdf_2net_video_pa']:
                sdf_feat, hand_pose_results, obj_pose_results = tester.model(inputs, targets=None, metas=metas, mode='test')
                export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
                export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)
                from recon import reconstruct
                if cfg.task == 'hsdf_osdf_2net_pa' or cfg.task == 'hsdf_osdf_2net_video_pa':
                    reconstruct(cfg, metas['id'], tester.model, sdf_feat, inputs, metas, hand_pose_results, obj_pose_results)
                else:
                    reconstruct(cfg, metas['id'], tester.model.module.hand_sdf_head, tester.model.module.obj_sdf_head, sdf_feat, metas, hand_pose_results, obj_pose_results)
            elif cfg.task == 'pose_kpt':
                hand_pose_results, obj_pose_results = tester.model(inputs, targets=None, metas=metas, mode='test')
                export_pose_results(cfg.hand_pose_result_dir, hand_pose_results, metas)
                export_pose_results(cfg.obj_pose_result_dir, obj_pose_results, metas)


if __name__ == "__main__":
    main()
