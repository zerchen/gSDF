#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :mano_head.py
#@Date        :2022/04/07 10:48:59
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
import numpy as np
from mano.manolayer import ManoLayer

class ManoHead(nn.Module):
    def __init__(self, ncomps=15, base_neurons=[512, 512, 512], center_idx=0, use_shape=True, use_pca=True, mano_root="../common/mano/assets/", depth=False):
        super(ManoHead, self).__init__()
        self.ncomps = ncomps
        self.base_neurons = base_neurons
        self.center_idx = center_idx
        self.use_shape = use_shape
        self.use_pca = use_pca
        self.depth = depth

        if self.use_pca:
            # pca comps + 3 global axis-angle params
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 comps per rot
            mano_pose_size = 16 * 9

        # Base layers
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(zip(self.base_neurons[:-1], self.base_neurons[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layers = nn.Sequential(*base_layers)

        # Pose layers
        self.pose_reg = nn.Linear(self.base_neurons[-1], mano_pose_size)

        # Shape layers
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(self.base_neurons[-1], 10))
        
        if self.depth:
            depth_layers = []
            trans_neurons = [512, 256]
            for layer_idx, (inp_neurons, out_neurons) in enumerate(zip(trans_neurons[:-1], trans_neurons[1:])):
                depth_layers.append(nn.Linear(inp_neurons, out_neurons))
                depth_layers.append(nn.ReLU())
            depth_layers.append(nn.Linear(trans_neurons[-1], 3))
            self.depth_layers = nn.Sequential(*depth_layers)
        
        # Mano layers
        self.mano_layer_right = ManoLayer(
            ncomps=self.ncomps,
            center_idx=self.center_idx,
            side="right",
            mano_root=mano_root,
            use_pca=self.use_pca,
            flat_hand_mean=False
        )

    def forward(self, inp):
        mano_features = self.base_layers(inp)
        pose = self.pose_reg(mano_features)

        scale_trans = None
        if self.depth:
            scale_trans = self.depth_layers(inp)

        if self.use_pca:
            mano_pose = pose
        else:
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)

        if self.use_shape:
            shape = self.shape_reg(mano_features)
        else:
            shape = None

        if mano_pose is not None and shape is not None:
            verts, joints, poses, global_trans, rot_center = self.mano_layer_right(mano_pose, th_betas=shape, root_palm=False)
        
        valid_idx = torch.ones((inp.shape[0], 1), device=inp.device).long()
        mean_pose = torch.from_numpy(np.array(self.mano_layer_right.smpl_data['hands_mean'], dtype=np.float32)).to(inp.device)

        results = {"verts": verts, "joints": joints, "shape": shape, "pcas": mano_pose, "pose": poses,  "global_trans":global_trans, "rot_center": rot_center, "scale_trans": scale_trans, "vis": valid_idx, "mean_pose": mean_pose}

        return results

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/zerchen/workspace/code/ho_recon/common')
    from mano.manolayer import ManoLayer
    net = ManoHead(depth=True)
    input_size = (2, 512)
    input_tensor = torch.randn(input_size)
    results = net(input_tensor)