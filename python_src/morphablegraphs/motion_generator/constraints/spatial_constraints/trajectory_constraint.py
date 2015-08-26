__author__ = 'herrmann'
import numpy as np

from splines.parameterized_spline import ParameterizedSpline
from spatial_constraint_base import SpatialConstraintBase
from ....animation_data.motion_editing import get_cartesian_coordinates_from_quaternion

TRAJECTORY_DIM = 3  # spline in cartesian space


class TrajectoryConstraint(ParameterizedSpline, SpatialConstraintBase):
    def __init__(self, joint_name, control_points, min_arc_length, unconstrained_indices, skeleton, precision, weight_factor=1.0):
        ParameterizedSpline.__init__(self, control_points, TRAJECTORY_DIM)
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.joint_name = joint_name
        self.skeleton = skeleton
        self.min_arc_length = min_arc_length
        self.arc_length = 0  # will store the full arc length after evaluation
        self.unconstrained_indices = unconstrained_indices

    def evaluate_motion_sample(self, aligned_quat_frames):
        """  Calculate sum of distances between discrete frames and samples with corresponding arc length from the trajectory
             unconstrained indices are ignored
        :param aligned_quat_frames:
        :return: error
        """
        error = 0
        self.arc_length = self.min_arc_length
        last_joint_position = None
        for frame in aligned_quat_frames:
            joint_position = get_cartesian_coordinates_from_quaternion(self.skeleton, self.joint_name, frame)
            if last_joint_position is not None:
                self.arc_length += np.linalg.norm(joint_position - last_joint_position)
            target = self.query_point_by_absolute_arc_length(self.arc_length)
            target[self.unconstrained_indices] = 0
            joint_position[self.unconstrained_indices] = 0
            error += np.linalg.norm(joint_position-target)
        return error

