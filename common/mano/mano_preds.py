#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :mano_preds.py
#@Date        :2022/04/18 11:36:43
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch


def recover_3d_proj(objpoints3d, camintr, est_scale, est_trans, off_z=0.4, input_res=(256, 256)):
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d


def get_mano_preds(mano_results, cfg, cam_intr, wrist_pos=None):
    if mano_results['scale_trans'] is not None:
        trans = mano_results['scale_trans'][:, 1:]
        scale = mano_results['scale_trans'][:, [0]]
        final_trans = trans * 100.0
        final_scale = scale * 0.0001
        cam_joints, center3d = recover_3d_proj(mano_results['joints'], cam_intr, final_scale, final_trans, input_res=cfg.image_size)
        cam_verts = center3d + mano_results['verts']
        mano_results['joints'] = cam_joints
        mano_results['verts'] = cam_verts
    else:
        center3d = wrist_pos.reshape((mano_results['joints'].shape[0], 1, 3))
        mano_results['joints'] = mano_results['joints'] + center3d
        mano_results['verts'] = mano_results['verts'] + center3d
    
    return mano_results