# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 18:59:44 2015

@author: erhe01
"""

from math import sqrt
from .....animation_data.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
    align_point_clouds_2D,\
    transform_point_cloud,\
    calculate_point_cloud_distance
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE


class PoseConstraint(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PoseConstraint, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        return

    def evaluate_motion_sample(self, aligned_quat_frames):
        """ Evaluates the difference between the first frame of the motion
        and the frame constraint.

        Parameters
        ----------
        * aligned_quat_frames: np.ndarray
            Motion aligned to previous motion in quaternion format
        * frame_constraint: dict of np.ndarray
            Dict containing a position for each joint
        * skeleton: Skeleton
            Used for hierarchy information
        * node_name_map : dict
           Optional: Maps node name to index in frame vector ignoring "Bip" joints

        Returns
        -------
        * error: float
            Difference to the desired constraint value.
        """

        # get point cloud of first frame
        point_cloud = convert_quaternion_frame_to_cartesian_frame(
            self.skeleton, aligned_quat_frames[self.canonical_keyframe])

        constraint_point_cloud = []
        for joint in self.skeleton.node_name_map.keys():
            constraint_point_cloud.append(self.pose_constraint[joint])
        theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                          point_cloud,
                                                          self.skeleton.joint_weights)
        t_point_cloud = transform_point_cloud(
            point_cloud, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(
            constraint_point_cloud, t_point_cloud)

        return error

    def get_residual_vector(self, aligned_quat_frames):
        # get point cloud of first frame
        point_cloud = convert_quaternion_frame_to_cartesian_frame(
            self.skeleton, aligned_quat_frames[0])

        constraint_point_cloud = []
        for joint in self.skeleton.node_name_map.keys():
            constraint_point_cloud.append(self.pose_constraint[joint])
        theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                          point_cloud,
                                                          self.skeleton.joint_weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)
        residual_vector = []
        for i in xrange(len(t_point_cloud)):
            d = [constraint_point_cloud[i][0] - t_point_cloud[i][0],
                 constraint_point_cloud[i][1] - t_point_cloud[i][1],
                 constraint_point_cloud[i][2] - t_point_cloud[i][2]]
            residual_vector.append(sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        return residual_vector

    def get_length_of_residual_vector(self):
        return len(self.skeleton.node_name_map.keys())
