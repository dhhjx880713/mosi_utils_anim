# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np
from .....animation_data.motion_editing import pose_orientation_quat
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
from math import acos


class Direction2DConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(Direction2DConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array(
            [self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir = self.target_dir / np.linalg.norm(self.target_dir)
        self.target_dir_len = np.linalg.norm(self.target_dir)

    def evaluate_motion_sample(self, aligned_quat_frames):
        motion_dir = pose_orientation_quat(aligned_quat_frames[self.canonical_keyframe])
        #TODO implement alternative constraint using trajectory direction instead of pose direction
        # root_points = extract_root_positions(aligned_quat_frames)
        # print root_points
        # motion_dir = get_trajectory_dir_from_2d_points(root_points)
        #error = abs(self.target_dir[0] - motion_dir[0]) + \
        #    abs(self.target_dir[1] - motion_dir[1])
        error = acos(np.dot(self.target_dir, motion_dir)/ (self.target_dir_len * np.linalg.norm(motion_dir)))
        # print "################################################"
        # print "target direction: ", self.target_dir
        # print "motion dir: ", motion_dir
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        return error

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_length_of_residual_vector(self):
        return 1