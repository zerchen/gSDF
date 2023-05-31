#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :camera.py
#@Date        :2022/08/11 16:58:46
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import numpy as np


class PerspectiveCamera:
    def __init__(self, fx, fy, cx, cy, R=np.eye(3), t=np.zeros(3)):
        self.K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]], dtype=np.float32)

        self.R = np.array(R, dtype=np.float32).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t, dtype=np.float32).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

    def update_virtual_camera_after_crop(self, bbox, option='same'):
        left, upper, width, height = bbox
        new_img_center = np.array([left + width / 2, upper + height / 2, 1], dtype=np.float32).reshape(3, 1)
        new_cam_center = np.linalg.inv(self.K[:3, :3]).dot(new_img_center)
        self.K[0, 2], self.K[1, 2] = width / 2, height / 2

        x, y, z = new_cam_center[0], new_cam_center[1], new_cam_center[2]
        sin_theta = -y / np.sqrt(1 + x ** 2 + y ** 2)
        cos_theta = np.sqrt(1 + x ** 2) / np.sqrt(1 + x ** 2 + y ** 2)
        R_x = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]], dtype=np.float32)
        sin_phi = x / np.sqrt(1 + x ** 2)
        cos_phi = 1 / np.sqrt(1 + x ** 2)
        R_y = np.array([[cos_phi, 0, sin_phi], [0, 1, 0], [-sin_phi, 0, cos_phi]], dtype=np.float32)
        self.R = R_y @ R_x

        # update focal length for virtual camera; please refer to the paper "PCLs: Geometry-aware Neural Reconstruction of 3D Pose with Perspective Crop Layers" for more details.
        if option == 'length':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2)
            self.K[1, 1] = self.K[1, 1] * np.sqrt(1 + x ** 2 + y ** 2)
        
        if option == 'scale':
            self.K[0, 0] = self.K[0, 0] * np.sqrt(1 + x ** 2 + y ** 2) * np.sqrt(1 + x ** 2)
            self.K[1, 1] = self.K[1, 1] * (1 + x ** 2 + y ** 2)/ np.sqrt(1 + x ** 2)

    def update_intrinsics_after_crop(self, bbox):
        left, upper, _, _ = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_intrinsics_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def intrinsics(self):
        return self.K

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])
