
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_FEET


class FeetConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(FeetConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_FEET
        self.left = constraint_desc["left"]
        self.right = constraint_desc["right"]

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def evaluate_motion_sample(self, aligned_quat_frames):
        return self.evaluate_frame(aligned_quat_frames[self.canonical_keyframe])

    def evaluate_frame(self, frame):
        return sum(self.get_residual_vector(frame))

    def get_residual_vector(self, aligned_quat_frames):
        left_error = np.linalg.norm(self.left - self.skeleton.nodes["LeftFoot"].get_global_position(aligned_quat_frames))*self.weight_factor
        right_error = np.linalg.norm(self.right - self.skeleton.nodes["RightFoot"].get_global_position(aligned_quat_frames))*self.weight_factor
        print "foot sliding error", sum([left_error, right_error]),self.left,self.skeleton.nodes["LeftFoot"].get_global_position(aligned_quat_frames)
        return [left_error, right_error]

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_length_of_residual_vector(self):
        return 2