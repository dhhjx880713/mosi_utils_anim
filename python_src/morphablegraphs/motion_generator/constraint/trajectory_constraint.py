__author__ = 'herrmann'
import numpy as np
from spatial_constraint_base import SpatialConstraintBase
from ...animation_data.motion_editing import get_cartesian_coordinates_from_quaternion


class TrajectoryConstraint(SpatialConstraintBase):
    def __init__(self, joint_name, trajectory, min_arc_length, skeleton, precision, weight_factor=1.0):
        SpatialConstraintBase.__init__(self, precision, weight_factor)
        self.joint_name = joint_name
        self.trajectory = trajectory
        self.skeleton = skeleton
        self.min_arc_length = min_arc_length
        self.arc_length = 0

    def evaluate_motion_sample(self, aligned_quat_frames):
        """  use min_arc_length as start and calculate distance of joint in discrete frames and sample with corresponding arc length from spline
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
            target = self.trajectory.query_point_by_absolute_arc_length(self.arc_length)
            error += np.linalg.norm(joint_position-target)

        return error

