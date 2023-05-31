#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :img_utils.py
#@Date :2022/09/20 15:50:32
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr

import numpy as np
import cv2


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  get affine transform matrix
    ---------
    @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
    -------
    @Returns: affine transformation matrix
    -------
    """

    def rotate_2d(pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_patch_image(cvimg, bbox, input_shape, scale, rot):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  generate the patch image from the bounding box and other parameters.
    ---------
    @param: input image, bbox(x1, y1, h, w), dest image shape, do_flip, scale factor, rotation degrees.
    -------
    @Returns: processed image, affine_transform matrix to get the processed image.
    -------
    """

    img = cvimg.copy()
    img_height, img_width, _ = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    new_trans = np.zeros((3, 3), dtype=np.float32)
    new_trans[:2, :] = trans
    new_trans[2, 2] = 1

    return img_patch, new_trans

def merge_handobj_bbox(hand_bbox, obj_bbox):
    # the format of bbox: xyxy
    tl = np.min(np.concatenate([hand_bbox.reshape((2, 2)), obj_bbox.reshape((2, 2))], axis=0), axis=0)
    br = np.max(np.concatenate([hand_bbox.reshape((2, 2)), obj_bbox.reshape((2, 2))], axis=0), axis=0)
    box_size = br - tl
    bbox = np.concatenate([tl, box_size], axis=0)

    return bbox
    

def process_bbox(bbox):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 1.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox