__author__ = 'herrmann'
import numpy as np

from splines.parameterized_spline import ParameterizedSpline
from spatial_constraint_base import SpatialConstraintBase
from ....animation_data.motion_editing import get_cartesian_coordinates_from_quaternion
from . import SPATIAL_CONSTRAINT_TYPE_TRAJECTORY

TRAJECTORY_DIM = 3  # spline in cartesian space


class TrajectoryConstraint(ParameterizedSpline, SpatialConstraintBase):
    def __init__(self, joint_name, control_points, min_arc_length, unconstrained_indices, skeleton, precision, weight_factor=1.0):
        ParameterizedSpline.__init__(self, control_points, TRAJECTORY_DIM)
        SpatialConstraintBase.__init__(self, precision, weight_factor)

        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TRAJECTORY
        self.joint_name = joint_name
        self.skeleton = skeleton
        self.min_arc_length = min_arc_length
        self.n_canonical_frames = 0
        self.arc_length = 0.0  # will store the full arc length after evaluation
        self.unconstrained_indices = unconstrained_indices

    def set_number_of_canonical_frames(self, n_canonical_frames):
        self.n_canonical_frames = n_canonical_frames

    def set_min_arc_length_from_previous_frames(self, previous_frames):
        """ Sets the minimum arc length of the constraint as the approximate arc length of the position of the joint
            in the last frame of the previous frames.
        :param previous_frames: list of quaternion frames.
        """
        point = self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, previous_frames[-1])
        closest_point, distance = self.find_closest_point(point, min_arc_length=self.min_arc_length)
        self.min_arc_length = self.get_absolute_arc_length_of_point(closest_point)[0]

    def evaluate_motion_sample(self, aligned_quat_frames):
        """
        :param aligned_quat_frames: list of quaternion frames.
        :return: average error
        """
        error = np.average(self.get_residual_vector(aligned_quat_frames))
        return error

    def get_residual_vector(self, aligned_quat_frames):
        """ Calculate distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :return: the residual vector
        """
        self.arc_length = self.min_arc_length
        last_joint_position = None
        errors = []
        for frame in aligned_quat_frames:
            joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
            if last_joint_position is not None:
                self.arc_length += np.linalg.norm(joint_position - last_joint_position)
            target = self.query_point_by_absolute_arc_length(self.arc_length)
            last_joint_position = joint_position
            #target[self.unconstrained_indices] = 0
            joint_position[self.unconstrained_indices] = 0
            errors.append(np.linalg.norm(joint_position-target))
        return errors

    def get_length_of_residual_vector(self):
        return self.n_canonical_frames
