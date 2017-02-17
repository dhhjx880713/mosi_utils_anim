
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np
from ....external.transformations import quaternion_matrix, rotation_from_matrix
from ....animation_data.motion_editing import quaternion_from_vector_to_vector
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT


class LookAtConstraint(KeyframeConstraintBase):
    REFERENCE_VECTOR = [0, 0, 1, 1]

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(LookAtConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT
        self.joint_name = "Head"
        self.target_position = constraint_desc["position"]

    def _get_direction_vector_from_orientation(self, q):
        q /= np.linalg.norm(q)
        rotation_matrix = quaternion_matrix(q)
        vec = np.dot(rotation_matrix, self.REFERENCE_VECTOR)[:3]
        return vec/np.linalg.norm(vec)

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def evaluate_motion_sample(self, aligned_quat_frames):
        return self.evaluate_frame(aligned_quat_frames[self.canonical_keyframe])

    def evaluate_frame(self, frame):
        head_position = self.skeleton.nodes[self.skeleton.head_joint].get_global_position(frame, use_cache=False)
        target_direction = self.target_position - head_position
        target_direction /= np.linalg.norm(target_direction)

        head_orientation = self.skeleton.nodes[self.skeleton.head_joint].get_global_orientation_quaternion(frame, use_cache=True)
        head_direction = self._get_direction_vector_from_orientation(head_orientation)

        delta_q = quaternion_from_vector_to_vector(head_direction, target_direction)
        angle, _, _ = rotation_from_matrix(quaternion_matrix(delta_q))
        return abs(angle)

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_length_of_residual_vector(self):
        return 1