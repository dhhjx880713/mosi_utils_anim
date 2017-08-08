# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np
from ....animation_data.motion_concatenation import get_global_node_orientation_vector
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
from math import acos, degrees


class Direction2DConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(Direction2DConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array([self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir /= np.linalg.norm(self.target_dir)
        self.target_dir_len = np.linalg.norm(self.target_dir)

    def evaluate_motion_spline(self, aligned_spline):
        frame = aligned_spline.evaluate(self.canonical_keyframe)
        motion_dir = get_global_node_orientation_vector(self.skeleton, self.skeleton.aligning_root_node, frame, self.skeleton.aligning_root_dir)
        magnitude = self.target_dir_len * np.linalg.norm(motion_dir)
        cos_angle = np.dot(self.target_dir, motion_dir)/magnitude
        #print self.target_dir, motion_dir
        cos_angle = min(1,max(cos_angle,-1))
        angle = acos(cos_angle)
        error = abs(degrees(angle))
        #print "angle", error
        return error

    def evaluate_motion_sample(self, aligned_quat_frames):
        frame = aligned_quat_frames[self.canonical_keyframe]
        motion_dir = get_global_node_orientation_vector(self.skeleton, self.skeleton.aligning_root_node, frame, self.skeleton.aligning_root_dir)
        #TODO implement alternative constraint using trajectory direction instead of pose direction
        # root_points = extract_root_positions(aligned_quat_frames)
        # print root_points
        # motion_dir = get_trajectory_dir_from_2d_points(root_points)
        #error = abs(self.target_dir[0] - motion_dir[0]) + \
        #    abs(self.target_dir[1] - motion_dir[1])
        magnitude = self.target_dir_len * np.linalg.norm(motion_dir)
        cos_angle = np.dot(self.target_dir, motion_dir)/ magnitude
        cos_angle = min(1,max(cos_angle,-1))
        angle = acos(cos_angle)#
        # print "################################################"
        # print "target direction: ", self.target_dir
        # print "motion dir: ", motion_dir
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        #error = 0
        error = abs(degrees(angle))
        return error

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_length_of_residual_vector(self):
        return 1