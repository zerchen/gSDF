#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :_init_paths.py
#@Date        :2022/04/07 23:37:58
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

root_path = osp.join(this_dir, '..')
lib_path = osp.join(this_dir, '..', 'common')
add_path(root_path)
add_path(lib_path)