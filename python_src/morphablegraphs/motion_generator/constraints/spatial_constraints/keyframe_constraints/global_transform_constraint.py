# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:02:55 2015

@author: erhe01
"""

from math import sqrt
import numpy as np
from .....animation_data.motion_editing import quaternion_to_euler, get_cartesian_coordinates_from_quaternion
from .....external.transformations import rotation_matrix
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION

RELATIVE_HUERISTIC_RANGE = 0.00 #5 # used for setting the search range relative to the number of frames of motion primitive
CONSTRAINT_CONFLICT_ERROR = 100000  # returned when conflicting constraints were set


class GlobalTransformConstraint(KeyframeConstraintBase):
    """
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(GlobalTransformConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
        self.skeleton = skeleton
        self.joint_name = constraint_desc["joint"]
        if "position" in constraint_desc.keys():
            self.position = constraint_desc["position"]
        else:
            self.position = None
        if "orientation" in constraint_desc.keys():
            self.orientation = constraint_desc["orientation"]
        else:
            self.orientation = None
        self.n_canonical_frames = constraint_desc["n_canonical_frames"]
        if "n_canonical_frames" in constraint_desc.keys():
            self.frame_range = RELATIVE_HUERISTIC_RANGE*self.n_canonical_frames
        else:
            self.frame_range = 0
        self.start_keyframe = int(max(self.canonical_keyframe - self.frame_range, 0))
        self.stop_keyframe = int(min(self.canonical_keyframe + self.frame_range, self.n_canonical_frames))
        if self.start_keyframe == self.stop_keyframe:
            self.start_keyframe -= 1
        #print "RANGE", self.start_keyframe, self.stop_keyframe
        self.rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def evaluate_motion_sample(self, aligned_quat_frames):
        #min_error = CONSTRAINT_CONFLICT_ERROR
        # ignore a special case which should not happen in a single constraint
        #for frame in aligned_quat_frames[self.start_keyframe:self.stop_keyframe]:
        #    error = self._evaluate_frame(frame)
        #    if min_error > error:
        #        min_error = error
        #return min_error
        return self._evaluate_frame(aligned_quat_frames[self.canonical_keyframe])

    def get_residual_vector(self, aligned_frames):
        return [self.evaluate_motion_sample(aligned_frames)]

    def _evaluate_frame(self, frame):
        error = 0
        if self.position is not None:
            error += self._evaluate_joint_position(frame)
        if self.orientation is not None:
            error += self._evaluate_joint_orientation(frame)
        return error

    def _evaluate_joint_orientation(self, frame):
        joint_index = self.skeleton.node_name_frame_map[self.joint_name]
        joint_orientation = frame[joint_index:joint_index+4]
        return self._orientation_distance(joint_orientation)

    def _evaluate_joint_position(self, frame):
        #joint_position = self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame)
        joint_position = self.skeleton.joint_map[self.joint_name].get_global_position(frame)
        #print self.joint_name, joint_position, joint_position3, self.position
        return self._vector_distance(self.position, joint_position)

    def _orientation_distance(self, joint_orientation):
        joint_euler_angles = quaternion_to_euler(joint_orientation)
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(3):
            if self.orientation[i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(self.orientation[i]), self.rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(joint_euler_angles[i]), self.rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = self._vector_distance(np.ravel(rotmat_constraint), np.ravel(rotmat_target))
        return rotation_distance

    def _vector_distance(self, a, b):
        """Returns the distance ignoring entries with None
        """
        d_sum = 0
        for i in xrange(len(a)):
            if a[i] is not None and b[i] is not None:
                d_sum += (a[i]-b[i])**2
        return sqrt(d_sum)

    def get_length_of_residual_vector(self):
        return 1