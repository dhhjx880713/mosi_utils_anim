# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np

from .....animation_data.motion_editing import pose_orientation_quat, \
                                               get_dir_from_2d_points, \
                                               extract_root_positions
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR


class DirectionConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(DirectionConstraint, self).__init__(constraint_desc, precision, weight_factor)

        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array(
            [self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir = self.target_dir / np.linalg.norm(self.target_dir)

    def evaluate_motion_sample(self, aligned_quat_frames):
        # motion_dir = pose_orientation_quat(aligned_quat_frames[self.canonical_keyframe])
        root_points = extract_root_positions(aligned_quat_frames)
        motion_dir = get_dir_from_2d_points(root_points)
        error = abs(self.target_dir[0] - motion_dir[0]) + \
            abs(self.target_dir[1] - motion_dir[1])
        # print "target direction: ", self.target_dir
        # print "motion dir: ", motion_dir
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        return error

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_length_of_residual_vector(self):
        return 1