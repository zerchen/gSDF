#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :solver.py
#@Date :2022/04/27 09:23:12
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr


import numpy as np
import trimesh
import copy
from sklearn.neighbors import KDTree


class icp_ts():
    """
    @description:
    icp solver which only aligns translation and scale
    """
    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source)**2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target)**2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter = 10, stop_error = 1e-3, stop_improvement = 1e-5, verbose=0):
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        previous_error = error
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            query_source_points = self.points_source * self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:, 0], :]

            # Find closest source point for each target point:
            query_target_points = (self.points_target - self.trans)/self.scale
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:, 0], :]
            closest_source_points = closest_source_points * self.scale + self.trans
            query_target_points = self.points_target

            # Compute current error:
            error = (((query_source_points - closest_target_points)**2).sum() + ((query_target_points - closest_source_points)**2).sum()) / (query_source_points.shape[0] + query_target_points.shape[0])
            error = error ** 0.5
            if verbose >= 1:
                print(i, "th iter, error: ", error)

            if previous_error - error < stop_improvement:
                break
            else:
                previous_error = error

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1, 1), self.points_source[closest_source_points_index[:, 0], :].reshape(-1, 1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3), dtype=np.float) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            b = np.vstack([closest_target_points.reshape(-1, 1), query_target_points.reshape(-1, 1)])
            x = np.linalg.lstsq(A, b, rcond=None)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

    def get_trans_scale(self):
        all_scale = self.scale_target * self.scale / self.scale_source 
        all_trans = self.trans + self.offset_target * self.scale - self.offset_source * self.scale_target * self.scale / self.scale_source
        return all_trans, all_scale

    def export_source_mesh(self, output_name):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        self.mesh_source.export(output_name)
