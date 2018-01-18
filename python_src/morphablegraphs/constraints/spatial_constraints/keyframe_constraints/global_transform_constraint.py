# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:02:55 2015

@author: erhe01
"""

from math import sqrt
import numpy as np
from ....animation_data.utils import quaternion_to_euler, quaternion_rotate_vector, euler_to_quaternion, get_cartesian_coordinates_from_quaternion
from ....external.transformations import rotation_matrix, angle_between_vectors
from .keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION


class GlobalTransformConstraint(KeyframeConstraintBase):
    """
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """

    ROTATION_AXIS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ORIGIN = [0,0,0,1]

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(GlobalTransformConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
        self.skeleton = skeleton
        self.joint_name = constraint_desc["joint"]
        if "position" in list(constraint_desc.keys()):
            self.position = constraint_desc["position"]
        else:
            self.position = None
        if "orientation" in list(constraint_desc.keys()) and None not in constraint_desc["orientation"]:
            self.orientation = euler_to_quaternion(constraint_desc["orientation"])
        else:
            self.orientation = None
        self.n_canonical_frames = constraint_desc["n_canonical_frames"]


    def evaluate_motion_spline(self, aligned_spline):
        error = 0
        frame = aligned_spline.evaluate(self.canonical_keyframe)
        if self.position is not None:
            error += self._evaluate_joint_position(frame)
        if self.orientation is not None:
            error += self._evaluate_joint_orientation(frame)
        return error

    def evaluate_motion_sample(self, aligned_quat_frames):
        error = 0
        if self.position is not None:
            error += self._evaluate_joint_position(aligned_quat_frames[self.canonical_keyframe])
        if self.orientation is not None:
            error += self._evaluate_joint_orientation(aligned_quat_frames[self.canonical_keyframe])
        return error

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_residual_vector(self, aligned_frames):
        return [self.evaluate_motion_sample(aligned_frames)]

    def _evaluate_frame(self, frame):
        error = 0
        if self.position is not None:
            error += self._evaluate_joint_position(frame)
        if self.orientation is not None:
            error += self._evaluate_joint_orientation(frame)
        return error

    def _evaluate_joint_position(self, frame):
        joint_position = self.skeleton.nodes[self.joint_name].get_global_position(frame)
        return GlobalTransformConstraint._point_distance(self.position, joint_position)

    def _evaluate_joint_orientation(self, frame):
        joint_orientation = self.skeleton.nodes[self.joint_name].get_global_orientation_quaternion(frame, use_cache=True)
        return self._quaternion_distance(joint_orientation)

    def _quaternion_distance(self, joint_orientation):
        """
        Args:
            joint_orientation(Vec4f): quaternion (qw, qx, qy, qz)

        Returns:
            angle (float)
        """
        v1 = quaternion_rotate_vector(joint_orientation, self.ORIGIN)
        v2 = quaternion_rotate_vector(self.orientation, self.ORIGIN)
        return angle_between_vectors(v1, v2)

    def _orientation_distance(self, joint_orientation):
        joint_euler_angles = quaternion_to_euler(joint_orientation)
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in range(3):
            if self.orientation[i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(self.orientation[i]), self.ROTATION_AXIS[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(joint_euler_angles[i]), self.ROTATION_AXIS[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = GlobalTransformConstraint._vector_distance(np.ravel(rotmat_constraint), np.ravel(rotmat_target), 16)
        return rotation_distance

    @staticmethod
    def _point_distance(target_p, sample_p):
        """Returns the distance ignoring entries with None
        """
        d_sum = 0
        for i in range(3):
            if target_p[i] is not None:
                d_sum += (target_p[i]-sample_p[i])**2
        return sqrt(d_sum)

    @staticmethod
    def _vector_distance(a, b, length):
        """Returns the distance ignoring entries with None
        """
        d_sum = 0
        for i in range(length):
            if a[i] is not None and b[i] is not None:
                d_sum += (a[i]-b[i])**2
        return sqrt(d_sum)

    def get_length_of_residual_vector(self):
        return 1
