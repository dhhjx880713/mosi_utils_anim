
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:01:21 2015

@author: erhe01
"""

import numpy as np
from .....external.transformations import quaternion_matrix, rotation_from_matrix
from .....animation_data.motion_editing import quaternion_from_vector_to_vector
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT


class LookAtConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(LookAtConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT
        self.target_position = constraint_desc["position"]

    def _get_direction_vector_from_orientation(self, q):
        q /= np.linalg.norm(q)
        rotation_matrix = quaternion_matrix(q)
        ref_vector = [0, 0, 1, 1]
        vec = np.dot(rotation_matrix, ref_vector)[:3]
        return vec/np.linalg.norm(vec)

    def evaluate_motion_sample(self, aligned_quat_frames):
        pose_parameters = aligned_quat_frames[self.canonical_keyframe]
        head_position = self.skeleton.nodes[self.skeleton.head_joint].get_global_position(pose_parameters, use_cache=False)
        target_direction = head_position - self.target_position

        head_orientation = self.skeleton.nodes[self.skeleton.head_joint].get_global_orientation_quaternion(pose_parameters, use_cache=True)
        head_direction = self._get_direction_vector_from_orientation(head_orientation)

        delta_q = quaternion_from_vector_to_vector(head_direction, target_direction)
        angle, _, _ = rotation_from_matrix(quaternion_matrix(delta_q))
        return abs(angle)

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_motion_sample(aligned_quat_frames)]

    def get_length_of_residual_vector(self):
        return 1