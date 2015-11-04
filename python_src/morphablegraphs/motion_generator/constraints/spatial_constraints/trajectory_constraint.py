__author__ = 'herrmann'
import numpy as np

from splines.parameterized_spline import ParameterizedSpline
from spatial_constraint_base import SpatialConstraintBase
from ....animation_data.motion_editing import get_cartesian_coordinates_from_quaternion
from discrete_trajectory_constraint import DiscreteTrajectoryConstraint
from . import SPATIAL_CONSTRAINT_TYPE_TRAJECTORY

TRAJECTORY_DIM = 3  # spline in cartesian space

class TrajectoryConstraint(ParameterizedSpline, SpatialConstraintBase):
    def __init__(self, joint_name, control_points, spline_type, min_arc_length, unconstrained_indices, skeleton, precision, weight_factor=1.0,
                 closest_point_search_accuracy=0.001, closest_point_search_max_iterations=5000):
        ParameterizedSpline.__init__(self, control_points, spline_type,
                                     closest_point_search_accuracy=closest_point_search_accuracy,
                                     closest_point_search_max_iterations=closest_point_search_max_iterations)
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.semantic_annotation = dict()
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TRAJECTORY
        self.joint_name = joint_name
        self.skeleton = skeleton
        self.min_arc_length = min_arc_length
        self.n_canonical_frames = 0
        self.arc_length = 0.0  # will store the full arc length after evaluation
        self.unconstrained_indices = unconstrained_indices
        self.range_start = None
        self.range_end = None

    def create_discrete_trajectory(self, aligned_quat_frames):
        discrete_trajectory_constraint = DiscreteTrajectoryConstraint(self.joint_name, self.skeleton, self.precision, self.weight_factor)
        discrete_trajectory_constraint.init_from_trajectory(self, aligned_quat_frames, self.min_arc_length)
        return discrete_trajectory_constraint

    def set_active_range(self, range_start, range_end):
        self.range_start = range_start
        self.range_end = range_end

    def set_number_of_canonical_frames(self, n_canonical_frames):
        self.n_canonical_frames = n_canonical_frames

    def set_min_arc_length_from_previous_frames(self, previous_frames):
        """ Sets the minimum arc length of the constraint as the approximate arc length of the position of the joint
            in the last frame of the previous frames.
        :param previous_frames: list of quaternion frames.
        """
        if len(previous_frames > 0):
            point = self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, previous_frames[-1])
            closest_point, distance = self.find_closest_point(point, self.min_arc_length, -1)
            if closest_point is not None:
                self.min_arc_length = self.get_absolute_arc_length_of_point(closest_point)[0]
            else:
                self.min_arc_length = self.full_arc_length
        else:
            self.min_arc_length = 0.0


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
        #if self.arc_length is None:
        #    return np.zeros(len(aligned_quat_frames))

        last_joint_position = None
        errors = []
        for frame in aligned_quat_frames:
            joint_position = np.asarray(self.skeleton.get_cartesian_coordinates_from_quaternion(self.joint_name, frame))
            if last_joint_position is not None:
                self.arc_length += np.linalg.norm(joint_position - last_joint_position)
            if self.range_start is None or self.range_start <= self.arc_length <= self.range_end:
                target = self.query_point_by_absolute_arc_length(self.arc_length)
                last_joint_position = joint_position
                #target[self.unconstrained_indices] = 0
                joint_position[self.unconstrained_indices] = 0
                errors.append(np.linalg.norm(joint_position-target))
            else:
                errors.append(0.0)
        return errors

    def get_length_of_residual_vector(self):
        return self.n_canonical_frames
