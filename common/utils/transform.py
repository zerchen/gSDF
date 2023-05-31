#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :transform.py
#@Date        :2022/08/11 16:58:01
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import numpy as np


def homoify(points):
    """
    Convert a batch of points to homogeneous coordinates.
    Args:
        points: e.g. (B, N, 3) or (N, 3)
    Returns:
        homoified points: e.g., (B, N, 4)
    """
    points_dim = points.shape[:-1] + (1,)
    ones = points.new_ones(points_dim)

    return torch.cat([points, ones], dim=-1)


def dehomoify(points):
    """
    Convert a batch of homogeneous points to cartesian coordinates.
    Args:
        homogeneous points: (B, N, 4/3) or (N, 4/3)
    Returns:
        cartesian points: (B, N, 3/2)
    """
    return points[..., :-1] / points[..., -1:]
