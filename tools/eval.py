#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :eval.py
#@Date        :2022/07/19 10:32:37
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import sys
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
from loguru import logger
import _init_paths
from _init_paths import add_path, this_dir
from multiprocessing import Process, Queue
import pandas as pd
import trimesh
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str)
    parser.add_argument('--num_proc', default=10, type=int)
    args = parser.parse_args()

    return args


def evaluate(queue, db, output_dir):
    for idx, sample in tqdm(enumerate(db.data)):
        error_dict = db._evaluate(output_dir, idx)
        queue.put([tuple(error_dict.values())])


def main():
    # argument parse and create log
    args = parse_args()
    testset = args.dir.strip('/').split('/')[-1].split('_')[1]
    exec(f'from datasets.{testset}.{testset} import {testset}')
    if testset == 'obman':
        data_root = '../datasets/obman/data/'
        rgb_source = '../datasets/obman/data/test/rgb/'
        mesh_hand_source = '../datasets/obman/data/test/mesh_hand/'
        mesh_obj_source = '../datasets/obman/data/test/mesh_obj/'
    elif testset == 'dexycb':
        data_root = '../datasets/dexycb/data/'
        from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS
        mesh_hand_source = '../datasets/dexycb/data/mesh_data/mesh_hand/'
        mesh_obj_source = '../datasets/dexycb/data/mesh_data/mesh_obj/'

    with open(os.path.join(args.dir, '../exp.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    start_points = []
    end_points = []
    division = cfg['num_testset_samples'] // args.num_proc
    for i in range(args.num_proc):
        start_point = i * division
        if i != args.num_proc - 1:
            end_point = start_point + division
        else:
            end_point = cfg['num_testset_samples']
        start_points.append(start_point)
        end_points.append(end_point)

    queue = Queue()
    process_list = []
    for i in range(args.num_proc):
        testset_db = eval(testset)('test_' + cfg['testset_split'], start_points[i], end_points[i])
        p = Process(target=evaluate, args=(queue, testset_db, args.dir))
        p.start()
        process_list.append(p)

    summary = []
    for p in process_list:
        while p.is_alive():
            while False == queue.empty():
                data = queue.get()
                summary = summary + data
    
    for p in process_list:
        p.join()

    summary = sorted(summary, reverse=False, key=lambda result: result[0])
    summary_filename = "eval_result.txt"

    with open(os.path.join(args.dir, summary_filename), "w") as f:
        eval_result = [[] for i in range(9)]
        name_list = ['sample_id', 'chamfer hand', 'fs_hand@1mm', 'fs_hand@5mm', 'chamfer obj', 'fs_obj@5mm', 'fs_obj@10mm', 'hand joint', 'obj center', 'obj corner']
        data_list = []
        for idx, result in enumerate(summary):
            data_sample = [result[0]]
            for i in range(9):
                if result[i + 1] is not None:
                    eval_result[i].append(result[i + 1])
                    data_sample.append(result[i + 1].round(3))
                else:
                    data_sample.append(result[i + 1])
            data_list.append(data_sample)
        f.write(pd.DataFrame(data_list, columns=name_list, index=[''] * len(summary), dtype=str).to_string())
        f.write('\n')

        for idx, _ in enumerate(eval_result):
            new_array = []
            for number in eval_result[idx]:
                if not np.isnan(number):
                    new_array.append(number)
            eval_result[idx] = new_array

        mean_chamfer_hand = "mean hand chamfer: {}\n".format(np.mean(eval_result[0]))
        median_chamfer_hand = "median hand chamfer: {}\n".format(np.median(eval_result[0]))
        fscore_hand_1 = "f-score hand @ 1mm: {}\n".format(np.mean(eval_result[1]))
        fscore_hand_5 = "f-score hand @ 5mm: {}\n".format(np.mean(eval_result[2]))
        mean_chamfer_obj = "mean obj chamfer: {}\n".format(np.mean(eval_result[3]))
        median_chamfer_obj = "median obj chamfer: {}\n".format(np.median(eval_result[3]))
        fscore_obj_1 = "f-score obj @ 5mm: {}\n".format(np.mean(eval_result[4]))
        fscore_obj_5 = "f-score obj @ 10mm: {}\n".format(np.mean(eval_result[5]))
        mean_mano_joint_err = "mean mano joint error: {}\n".format(np.mean(eval_result[6]))
        mean_obj_center_err = "mean obj center error: {}\n".format(np.mean(eval_result[7]))
        mean_obj_corner_err = "mean obj corner error: {}\n".format(np.mean(eval_result[8]))
        print(mean_chamfer_hand); f.write(mean_chamfer_hand)
        print(median_chamfer_hand); f.write(median_chamfer_hand)
        print(fscore_hand_1); f.write(fscore_hand_1)
        print(fscore_hand_5); f.write(fscore_hand_5)
        print(mean_chamfer_obj); f.write(mean_chamfer_obj)
        print(median_chamfer_obj); f.write(median_chamfer_obj)
        print(fscore_obj_1); f.write(fscore_obj_1)
        print(fscore_obj_5); f.write(fscore_obj_5)
        print(mean_mano_joint_err); f.write(mean_mano_joint_err)
        print(mean_obj_center_err); f.write(mean_obj_center_err)
        print(mean_obj_corner_err); f.write(mean_obj_corner_err)

        worst_hand_dir = os.path.join(args.dir, 'worst_hand'); os.makedirs(worst_hand_dir, exist_ok=True)
        best_hand_dir = os.path.join(args.dir, 'best_hand'); os.makedirs(best_hand_dir, exist_ok=True)
        best_obj_dir = os.path.join(args.dir, 'best_obj'); os.makedirs(best_obj_dir, exist_ok=True)
        worst_obj_dir = os.path.join(args.dir, 'worst_obj'); os.makedirs(worst_obj_dir, exist_ok=True)

        # begin to handle the hand case
        summary_hand = []
        for idx, sample in enumerate(summary):
            if sample[1] is not None:
                summary_hand.append(sample)
        if len(summary_hand) > 40:
            summary_hand = sorted(summary_hand, reverse=False, key=lambda result: result[1])
            for i in range(len(summary_hand)):
                sample_id = summary_hand[i][0]
                if i < 50 or i > len(summary_hand) - 51:
                    if i < 50:
                        sample_dir = os.path.join(best_hand_dir, sample_id)
                    else:
                        sample_dir = os.path.join(worst_hand_dir, sample_id)
                    os.makedirs(sample_dir, exist_ok=True)

                    if testset == 'obman':
                        shutil.copy2(os.path.join(rgb_source, sample_id + '.jpg'), sample_dir)
                        gt_mesh_hand = trimesh.load(os.path.join(mesh_hand_source, sample_id + '.obj'), process=False)
                        gt_mesh_hand.export(os.path.join(sample_dir, sample_id + '_gt_hand.glb'))
                        gt_mesh_obj = trimesh.load(os.path.join(mesh_obj_source, sample_id + '.obj'), process=False)
                        gt_mesh_obj.export(os.path.join(sample_dir, sample_id + '_gt_obj.glb'))
                        gt_mesh_fuse = trimesh.util.concatenate(gt_mesh_hand, gt_mesh_obj)
                        gt_mesh_fuse.export(os.path.join(sample_dir, sample_id + '_gt_fuse.glb'))
                    elif testset == 'dexycb':
                        subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
                        video_id = '_'.join(sample_id.split('_')[1:3])
                        cam_id = sample_id.split('_')[-2]
                        frame_id = sample_id.split('_')[-1].rjust(6, '0')
                        rgb_path = os.path.join(data_root, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
                        shutil.copy2(rgb_path, sample_dir)
                        gt_mesh_hand = trimesh.load(os.path.join(mesh_hand_source, sample_id + '.obj'), process=False)
                        gt_mesh_hand.export(os.path.join(sample_dir, sample_id + '_gt_hand.glb'))
                        gt_mesh_obj = trimesh.load(os.path.join(mesh_obj_source, sample_id + '.obj'), process=False)
                        gt_mesh_obj.export(os.path.join(sample_dir, sample_id + '_gt_obj.glb'))
                        gt_mesh_fuse = trimesh.util.concatenate(gt_mesh_hand, gt_mesh_obj)
                        gt_mesh_fuse.export(os.path.join(sample_dir, sample_id + '_gt_fuse.glb'))

                    mesh_hand = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_hand.ply'), process=False)
                    mesh_hand.export(os.path.join(sample_dir, sample_id + '_hand.glb'))
                    try:
                        mesh_obj = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_obj.ply'), process=False)
                        mesh_obj.export(os.path.join(sample_dir, sample_id + '_obj.glb'))
                        mesh_fuse = trimesh.util.concatenate(mesh_hand, mesh_obj)
                        mesh_fuse.export(os.path.join(sample_dir, sample_id + '_fuse.glb'))
                    except:
                        continue
               
        summary_obj = []
        for idx, sample in enumerate(summary):
            if sample[4] is not None:
                summary_obj.append(sample)
        if len(summary_obj) > 40:
            summary_obj = sorted(summary_obj, reverse=False, key=lambda result: result[1])
            for i in range(len(summary_obj)):
                sample_id = summary_obj[i][0]
                if i < 50 or i > len(summary_obj) - 51:
                    if i < 50:
                        sample_dir = os.path.join(best_obj_dir, sample_id)
                    else:
                        sample_dir = os.path.join(worst_obj_dir, sample_id)
                    os.makedirs(sample_dir, exist_ok=True)

                    if testset == 'obman':
                        shutil.copy2(os.path.join(rgb_source, sample_id + '.jpg'), sample_dir)
                        gt_mesh_hand = trimesh.load(os.path.join(mesh_hand_source, sample_id + '.obj'), process=False)
                        gt_mesh_hand.export(os.path.join(sample_dir, sample_id + '_gt_hand.glb'))
                        gt_mesh_obj = trimesh.load(os.path.join(mesh_obj_source, sample_id + '.obj'), process=False)
                        gt_mesh_obj.export(os.path.join(sample_dir, sample_id + '_gt_obj.glb'))
                        gt_mesh_fuse = trimesh.util.concatenate(gt_mesh_hand, gt_mesh_obj)
                        gt_mesh_fuse.export(os.path.join(sample_dir, sample_id + '_gt_fuse.glb'))
                    elif testset == 'dexycb':
                        subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
                        video_id = '_'.join(sample_id.split('_')[1:3])
                        cam_id = sample_id.split('_')[-2]
                        frame_id = sample_id.split('_')[-1].rjust(6, '0')
                        rgb_path = os.path.join(data_root, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
                        shutil.copy2(rgb_path, sample_dir)
                        gt_mesh_hand = trimesh.load(os.path.join(mesh_hand_source, sample_id + '.obj'), process=False)
                        gt_mesh_hand.export(os.path.join(sample_dir, sample_id + '_gt_hand.glb'))
                        gt_mesh_obj = trimesh.load(os.path.join(mesh_obj_source, sample_id + '.obj'), process=False)
                        gt_mesh_obj.export(os.path.join(sample_dir, sample_id + '_gt_obj.glb'))
                        gt_mesh_fuse = trimesh.util.concatenate(gt_mesh_hand, gt_mesh_obj)
                        gt_mesh_fuse.export(os.path.join(sample_dir, sample_id + '_gt_fuse.glb'))

                    mesh_obj = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_obj.ply'), process=False)
                    mesh_obj.export(os.path.join(sample_dir, sample_id + '_obj.glb'))
                    try:
                        mesh_hand = trimesh.load(os.path.join(args.dir, 'sdf_mesh', sample_id + '_hand.ply'), process=False)
                        mesh_hand.export(os.path.join(sample_dir, sample_id + '_hand.glb'))
                        mesh_fuse = trimesh.util.concatenate(mesh_hand, mesh_obj)
                        mesh_fuse.export(os.path.join(sample_dir, sample_id + '_fuse.glb'))
                    except:
                        continue

if __name__ == "__main__":
    main()
