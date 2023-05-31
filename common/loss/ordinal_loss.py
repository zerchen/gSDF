#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :ordinal_loss.py
#@Date        :2022/09/14 14:49:19
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import combinations, product


def partlevel_ordinal_relation(ppair: torch.Tensor, view_vecs: torch.Tensor):
    """
    Args:
        ppair: TENSOR (B, NPPAIRS, 6)
        view_vecs: TENSOR (B, NVIEWS, 3)
    Returns:
        ppair_ord: TENSOR (B, NPPAIRS, NVIEWS, 1)
    """

    nviews = view_vecs.shape[1]
    npairs = ppair.shape[1]
    ppair = ppair.unsqueeze(2).expand(-1, -1, nviews, -1)  # (B, NPPAIRS, NVIEWS, 6)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)  # (B, NPPAIRS, NVIEWS, 3)
    ppair_cross = torch.cross(ppair[..., :3], ppair[..., 3:])  # (B, NPPAIRS, NVIEWS, 3)
    ppair_ord = torch.einsum("bijk, bijk->bij", ppair_cross, view_vecs)  # (B, NPPAIRS, NVIEWS)

    return ppair_ord.unsqueeze(-1)  # (B, NPPAIRS, NVIEWS, 1)


def jointlevel_ordinal_relation(jpair: torch.Tensor, view_vecs: torch.Tensor):
    """
    Args:
        jpair: TENSOR (B, NPAIRS, 6)
        view_vecs: TENSOR (B, NVIEWS, 3)
    Returns:
        jpair_ord: TENSOR (B, NPAIRS, NVIEWS, 1)
    """

    nviews = view_vecs.shape[1]
    npairs = jpair.shape[1]
    jpair = jpair.unsqueeze(2).expand(-1, -1, nviews, -1)  # (B, NPAIRS, NVIEWS, 6)
    view_vecs = view_vecs.unsqueeze(1).expand(-1, npairs, -1, -1)  # (B, NPAIRS, NVIEWS, 3)
    jpair_diff = jpair[..., :3] - jpair[..., 3:]  # (B, NPAIRS, NVIEWS, 3)
    jpair_ord = torch.einsum("bijk, bijk->bij", jpair_diff, view_vecs)  # (B, NPAIRS, NVIEWS)

    return jpair_ord.unsqueeze(-1)  # (B, NPAIRS, NVIEWS, 1)


def sample_view_vectors(n_virtual_views=20):
    cam_vec = torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0)  # TENSOR (1, NVIEWS)
    theta = torch.rand(n_virtual_views) * 2.0 * np.pi  # TENSSOR (NVIEWS, )
    u = torch.rand(n_virtual_views)

    nv_x = torch.sqrt(1.0 - u ** 2) * torch.cos(theta)  # TENSSOR (NVIEWS, )
    nv_y = torch.sqrt(1.0 - u ** 2) * torch.sin(theta)  # TENSSOR (NVIEWS, )
    nv_z = u  # TENSSOR (NVIEWS, )

    nv = torch.cat([nv_x.unsqueeze(1), nv_y.unsqueeze(1), nv_z.unsqueeze(1)], dim=1)  # TENSSOR (NVIEWS, 3)
    nv = torch.cat([cam_vec, nv], dim=0)  # TENSOR (NVIEWS, 3)

    return nv


class HandOrdLoss(nn.Module):
    def __init__(self):
        super(HandOrdLoss, self).__init__()
        self.n_virtual_views = 20
        self.nviews = self.n_virtual_views + 1

        self.njoints = 21
        self.nparts = 20

        # crate joint pair index
        joints_idx = list(range(self.njoints))
        self.joint_pairs_idx = list(combinations(joints_idx, 2))

        # create part pair index
        parts_idx = list(range(self.nparts))
        self.parts_pairs_idx = list(combinations(parts_idx, 2))

    def joints_2_part_pairs(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joints: TENSOR (B, NJOINTS, 3)
        Returns:
            ppairs: TENSOR (B, NPAIRS, 6)
        """
        child_idx = list(range(self.njoints))
        parents_idx = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        parts_ = joints[:, child_idx, :] - joints[:, parents_idx, :]  # (B, NJOINTS, 3)
        parts = parts_[:, 1:, :]  # (B, NPARTS, 3)

        pairs_idx = np.array(self.parts_pairs_idx)  # (NPAIRS, 2)
        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_parts1 = parts[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_parts2 = parts[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        pparis = torch.cat([pairs_parts1, pairs_parts2], dim=2)  # (B, NPAIRS, 6)

        return pparis

    def joints_2_joint_pairs(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Converts joints3d into joint pairs. The pairing idx are defined in self.joint_pair_idx
        Args:
            joints: TENSOR (B, NJOINTS, 3)
        Returns:
            jpairs: TENSOR (B, NPAIRS, 6)
        """
        pairs_idx = np.array(self.joint_pairs_idx)  # (NPAIRS, 2)
        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_joints1 = joints[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_joints2 = joints[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        jpairs = torch.cat([pairs_joints1, pairs_joints2], dim=2)  # (B, NPAIRS, 6)

        return jpairs

    def forward(self, pred_joints, gt_joints):
        batch_size = pred_joints.shape[0]
        device = pred_joints.device

        view_vecs = sample_view_vectors(self.n_virtual_views).to(device)  # TENOSR (NVIEWS, 3)
        view_vecs = view_vecs.unsqueeze(0).expand(batch_size, -1, -1)  # TENOSR (B, NVIEWS, 3)

        # ============== JOINT LEVEL ORDINAL LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_jpairs = self.joints_2_joint_pairs(pred_joints)  # TENSOR (BATCH, NPAIRS, 6)
        gt_jpairs = self.joints_2_joint_pairs(gt_joints)  # TENSOR (BATCH, NPAIRS, 6)

        shuffle_idx = list(range(len(self.joint_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_jpairs = pred_jpairs[:, shuffle_idx, :]
        gt_jpairs = gt_jpairs[:, shuffle_idx, :]

        gt_jpairs_ord = jointlevel_ordinal_relation(gt_jpairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        gt_jpairs_sign = torch.sign(gt_jpairs_ord)
        pred_jpairs_ord = jointlevel_ordinal_relation(pred_jpairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        joint_ord_loss_ = F.relu(-1.0 * gt_jpairs_sign * pred_jpairs_ord)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        joint_ord_loss_ = torch.log(1.0 + joint_ord_loss_)
        joint_ord_loss = torch.mean(joint_ord_loss_)  # mean on batch, npairs, nviews

        # ============== PART LEVEL ORDINAL LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_ppairs = self.joints_2_part_pairs(pred_joints)  # TENSOR(BATCH, NPAIRS, 6)
        gt_ppairs = self.joints_2_part_pairs(gt_joints)

        shuffle_idx = list(range(len(self.parts_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_ppairs = pred_ppairs[:, shuffle_idx, :]
        gt_ppairs = gt_ppairs[:, shuffle_idx, :]

        gt_ppairs_ord = partlevel_ordinal_relation(gt_ppairs, view_vecs)
        gt_ppairs_sign = torch.sign(gt_ppairs_ord)  # G.T. sign
        pred_ppairs_ord = partlevel_ordinal_relation(pred_ppairs, view_vecs)
        part_ord_loss_ = F.relu(-1.0 * gt_ppairs_sign * pred_ppairs_ord)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        part_ord_loss = torch.mean(part_ord_loss_)  # mean on batch, npairs, nviews

        return joint_ord_loss + part_ord_loss


class SceneOrdLoss(nn.Module):
    def __init__(self, obj_rot=False):
        super(SceneOrdLoss, self).__init__()
        self.n_virtual_views = 40
        self.nviews = self.n_virtual_views + 1

        # crate joint | corners index
        joints_idx = list(range(21))  # [0, 1, ..., 20]
        if obj_rot:
            corners_idx = list(range(8))  # [0, 1, ..., 7]
        else:
            corners_idx = list(range(1))  # [0, 1, ..., 7]

        # create hand-object points pairs index
        self.ho_pairs_idx = list(product(joints_idx, corners_idx))

    def ho_joints_2_ho_pairs(self, joints: torch.Tensor, corners: torch.Tensor):
        pairs_idx = np.array(self.ho_pairs_idx)  # (NPAIRS, 2)

        pairs_idx1 = pairs_idx[:, 0]
        pairs_idx2 = pairs_idx[:, 1]

        pairs_joints = joints[:, pairs_idx1, :]  # (B, NPAIRS, 3)
        pairs_corners = corners[:, pairs_idx2, :]  # (B, NPAIRS, 3)

        ho_pairs = torch.cat([pairs_joints, pairs_corners], dim=2)  # (B, NPAIRS, 6)

        return ho_pairs

    def forward(self, pred_joints, pred_corners, gt_joints, gt_corners):
        batch_size = pred_joints.shape[0]
        device = pred_joints.device

        view_vecs = sample_view_vectors(self.n_virtual_views).to(device)  # TENOSR (NVIEWS, 3)
        view_vecs = view_vecs.unsqueeze(0).expand(batch_size, -1, -1)  # TENOSR (B, NVIEWS, 3)

        pred_ho_pairs = self.ho_joints_2_ho_pairs(pred_joints, pred_corners)
        gt_ho_pairs = self.ho_joints_2_ho_pairs(gt_joints,gt_corners )

        shuffle_idx = list(range(len(self.ho_pairs_idx)))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[: len(shuffle_idx) // 3]
        pred_ho_pairs = pred_ho_pairs[:, shuffle_idx, :]
        gt_ho_pairs = gt_ho_pairs[:, shuffle_idx, :]

        gt_ho_pairs_ord = jointlevel_ordinal_relation(gt_ho_pairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        gt_ho_pairs_sign = torch.sign(gt_ho_pairs_ord)
        pred_ho_pairs_ord = jointlevel_ordinal_relation(pred_ho_pairs, view_vecs)  # TENSOR (B, NPAIRS, NVIEWS, 1)
        scene_ord_loss_ = F.relu(-1.0 * gt_ho_pairs_sign * pred_ho_pairs_ord)
        scene_ord_loss_ = torch.log(1.0 + scene_ord_loss_)
        scene_ord_loss = torch.mean(scene_ord_loss_)

        return scene_ord_loss