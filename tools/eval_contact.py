#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :eval_contact.py
#@Date        :2022/11/15 16:52:45
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import argparse
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from multiprocessing import Process, Queue
import trimesh
from tqdm import tqdm
import igl as igl


def uniform_box_sampling(min_corner, max_corner, res = 0.005):
    x_min = min_corner[0] - res
    x_max = max_corner[0] + res
    y_min = min_corner[1] - res
    y_max = max_corner[1] + res
    z_min = min_corner[2] - res
    z_max = max_corner[2] + res

    h = int((x_max-x_min)/res)+1
    l = int((y_max-y_min)/res)+1
    w = int((z_max-z_min)/res)+1

    with torch.no_grad():
        xyz = x = torch.zeros(h, l, w, 3, dtype=torch.float32) + torch.tensor([x_min, y_min, z_min], dtype=torch.float32)
        for i in range(1,h):
            xyz[i,0,0] = xyz[i-1,0,0] + torch.tensor([res,0,0])
        for i in range(1,l):
            xyz[:,i,0] = xyz[:,i-1,0] + torch.tensor([0,res,0])
        for i in range(1,w):
            xyz[:,:,i] = xyz[:,:,i-1] + torch.tensor([0,0,res])
    return res, xyz


def bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1):
    min_x = max(min_corner0[0], min_corner1[0])
    min_y = max(min_corner0[1], min_corner1[1])
    min_z = max(min_corner0[2], min_corner1[2])

    max_x = min(max_corner0[0], max_corner1[0])
    max_y = min(max_corner0[1], max_corner1[1])
    max_z = min(max_corner0[2], max_corner1[2])

    if max_x > min_x and max_y > min_y and max_z > min_z:
        # print('Intersected bounding box size: %f x %f x %f'%(max_x - min_x, max_y - min_y, max_z - min_z))
        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    else:
        return np.zeros((1,3), dtype = np.float32), np.zeros((1,3), dtype = np.float32)


def contact_calc(queue, experiment_dir, start_point, end_point):
    pred_mesh_path = os.path.join(experiment_dir, 'sdf_mesh')
    hand_keys = [filename.split('_hand.ply')[0] for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]
    obj_keys = [filename.split('_obj.ply')[0] for filename in os.listdir(pred_mesh_path) if '_obj.ply' in filename]

    hand_keys_set = set(hand_keys)
    obj_keys_set = set(obj_keys)
    valid_keys = list(hand_keys_set & obj_keys_set)

    for key in tqdm(valid_keys[start_point:end_point]):
        hand_mesh = trimesh.load(os.path.join(pred_mesh_path, key + '_hand.ply'), process=False)
        obj_mesh = trimesh.load(os.path.join(pred_mesh_path, key + '_obj.ply'), process=False)

        S, I, C = igl.signed_distance(hand_mesh.vertices + 1e-10, obj_mesh.vertices, obj_mesh.faces, return_normals=False)

        mesh_mesh_distance = S.min()
        if mesh_mesh_distance > 0:
            queue.put([(key, 0, 0, 0)])
        else:
            min_corner_hand = np.array([hand_mesh.vertices[:,0].min(), hand_mesh.vertices[:,1].min(), hand_mesh.vertices[:,2].min()])
            max_corner_hand = np.array([hand_mesh.vertices[:,0].max(), hand_mesh.vertices[:,1].max(), hand_mesh.vertices[:,2].max()])
            min_corner_obj = np.array([obj_mesh.vertices[:,0].min(), obj_mesh.vertices[:,1].min(), obj_mesh.vertices[:,2].min()])
            max_corner_obj = np.array([obj_mesh.vertices[:,0].max(), obj_mesh.vertices[:,1].max(), obj_mesh.vertices[:,2].max()])
            min_corner_i, max_corner_i = bounding_box_intersection(min_corner_hand, max_corner_hand, min_corner_obj, max_corner_obj)

            if ((min_corner_i - max_corner_i)**2).sum() == 0:
                queue.put([(key, 0, 0, 0)])
            else:
                _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, 0.005)
                xyz = xyz.view(-1, 3)
                xyz = xyz.detach().cpu().numpy()

                S, I, C = igl.signed_distance(xyz, hand_mesh.vertices, hand_mesh.faces, return_normals=False)
                inside_sample_index = np.argwhere(S < 0.0)
                inside_samples = xyz[inside_sample_index[:,0], :]
                S, I, C = igl.signed_distance(inside_samples, obj_mesh.vertices, obj_mesh.faces, return_normals=False)
                inside_both_sample_index = np.argwhere(S < 0)
                i_v = inside_both_sample_index.shape[0] * (0.005**3)
                queue.put([(key, 1, abs(mesh_mesh_distance*1e2), i_v*1e6)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str)
    parser.add_argument('--num_proc', default=10, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()


    q = Queue()
    num_proc = args.num_proc

    pred_mesh_path = os.path.join(args.dir, 'sdf_mesh')
    hand_keys = [filename.split('_hand.ply')[0] for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]
    obj_keys = [filename.split('_obj.ply')[0] for filename in os.listdir(pred_mesh_path) if '_obj.ply' in filename]
    hand_keys_set = set(hand_keys)
    obj_keys_set = set(obj_keys)
    valid_keys = list(hand_keys_set & obj_keys_set)

    division = len(valid_keys) // num_proc
    start_points = []
    end_points = []
    for i in range(num_proc):
        start_point = i * division
        if i != num_proc - 1:
            end_point = start_point + division
        else:
            end_point = len(valid_keys)
        start_points.append(start_point)
        end_points.append(end_point)

    process_list = []
    for i in range(num_proc):
        p = Process(target=contact_calc, args=(q, args.dir, start_points[i], end_points[i]))
        p.start()
        process_list.append(p)

    summary = []
    for p in process_list:
        while p.is_alive():
            while False == q.empty():
                data = q.get()
                summary = summary + data
    
    for p in process_list:
        p.join()

    with open(os.path.join(args.dir, "contact.txt"), "w") as f:
        f.write("summary of chamfer_dist\n")
        contact_stat = []
        depth_stat = []
        volume_stat = []
        for idx, result in enumerate(summary):
            contact_stat.append(result[1])
            depth_stat.append(result[2])
            volume_stat.append(result[3])
            f.write("{}, {}, {}, {}\n".format(result[0], result[1], result[2], result[3]))
        
        contact_info = "Contact ratio:{}%\n".format(np.mean(contact_stat)*100) 
        depth_info = "mean penetration depth:{}\n".format(np.mean(depth_stat)) 
        volume_info = "mean intersection volume:{}\n".format(np.mean(volume_stat))
        print(contact_info)
        print(depth_info)
        print(volume_info)
        f.write(contact_info)
        f.write(depth_info)
        f.write(volume_info)
