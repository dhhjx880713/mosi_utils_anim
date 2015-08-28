# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:02:55 2015

@author: erhe01
"""

from math import sqrt
import numpy as np
from .....animation_data.motion_editing import get_cartesian_coordinates_from_quaternion,\
                                    quaternion_to_euler
from python_src.morphablegraphs.external.transformations import rotation_matrix
from keyframe_constraint_base import KeyframeConstraintBase


RELATIVE_HUERISTIC_RANGE = 0.10  # used for setting the search range relative to the number of frames of motion primitive
CONSTRAINT_CONFLICT_ERROR = 100000  # returned when conflicting constraints were set


class PositionAndRotationConstraint(KeyframeConstraintBase):
    """
    * skeleton: Skeleton
        Necessary for the evaluation of frames
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PositionAndRotationConstraint, self).__init__(constraint_desc, precision, weight_factor)
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
        self.relative_heuristic_range = RELATIVE_HUERISTIC_RANGE
        self.constrain_first_frame = constraint_desc["semanticAnnotation"]["firstFrame"]
        self.constrain_last_frame = constraint_desc["semanticAnnotation"]["lastFrame"]
        self._convert_annotation_to_indices()
        self.rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def evaluate_motion_sample(self, aligned_quat_frames):
        min_error = CONSTRAINT_CONFLICT_ERROR
        n_frames = len(aligned_quat_frames)
        # ignore a special case which should not happen in a single constraint
        if not (self.constrain_first_frame and self.constrain_last_frame):
            heuristic_range = int(self.relative_heuristic_range * n_frames)
            filtered_frames = aligned_quat_frames[-heuristic_range:]
            filtered_frame_nos = range(n_frames)            
            for frame_no, frame in zip(filtered_frame_nos, filtered_frames):
                error = self._evaluate_frame(frame)
                if min_error > error:
                    min_error = error
        return min_error

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
        joint_index = self.skeleton.node_name_map[self.joint_name]
        joint_orientation = frame[joint_index:joint_index+4]
        return self._orientation_distance(joint_orientation)

    def _evaluate_joint_position(self, frame):
        joint_position = get_cartesian_coordinates_from_quaternion(self.skeleton, self.joint_name, frame)
        return self._vector_distance(self.position, joint_position)

    def _orientation_distance(self, joint_orientation):
        joint_euler_angles = quaternion_to_euler(joint_orientation)
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(3):
            if self.orientation[i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(self.orientation[i]),
                                                 self.rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(joint_euler_angles[i]),
                                             self.rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = self._vector_distance(np.ravel(rotmat_constraint),
                                            np.ravel(rotmat_target))
        return rotation_distance

    def _vector_distance(self, a, b):
        """Returns the distance ignoring entries with None
        """
        d_sum = 0
        for i in xrange(len(a)):
            if a[i] is not None and b[i] is not None:
                d_sum += (a[i]-b[i])**2
        return sqrt(d_sum)

    def _convert_annotation_to_indices(self):
            start_stop_dict = {
                (None, None): (None, None),
                (True, None): (None, 1),
                (False, None): (1, None),
                (None, True): (-1, None),
                (None, False): (None, -1),
                (True, False): (None, 1),
                (False, True): (-1, None),
                (False, False): (1, -1)
            }
            self.start, self.stop = start_stop_dict[(self.constrain_first_frame, self.constrain_last_frame)]

    def get_length_of_residual_vector(self):
        return 1
