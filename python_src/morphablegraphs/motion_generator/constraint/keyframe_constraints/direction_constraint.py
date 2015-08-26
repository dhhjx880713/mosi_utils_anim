# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np

from ....animation_data.motion_editing import pose_orientation_quat
from keyframe_constraint_base import KeyframeConstraintBase


class DirectionConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(DirectionConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array(
            [self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir = self.target_dir / np.linalg.norm(self.target_dir)

    def evaluate_motion_sample(self, aligned_quat_frames):
        motion_dir = pose_orientation_quat(aligned_quat_frames[-1])

        error = abs(self.target_dir[0] - motion_dir[0]) + \
            abs(self.target_dir[1] - motion_dir[1])

        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        return error
