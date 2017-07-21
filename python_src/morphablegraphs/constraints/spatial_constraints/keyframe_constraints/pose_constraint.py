# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 18:59:44 2015

@author: erhe01
"""
import numpy as np
from math import sqrt
from ....animation_data.utils import convert_quaternion_frame_to_cartesian_frame,\
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
        self.velocity_constraint = constraint_desc["velocity_constraint"]
        print self.velocity_constraint.shape
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        self.node_names = constraint_desc["node_names"]
        self.weights = constraint_desc["weights"]

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def evaluate_motion_sample(self, aligned_quat_frames):
        """ Evaluates the difference between the pose of at the canonical frame of the motion and the pose constraint.

        Parameters
        ----------
        * aligned_quat_frames: np.ndarray
            Motion aligned to previous motion in quaternion format

        Returns
        -------
        * error: float
            Difference to the desired constraint value.
        """
        return self.evaluate_frame(aligned_quat_frames[self.canonical_keyframe])

    def evaluate_frame(self, frame):
        # get point cloud of first two frames
        point_cloud1 = np.array(self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame, self.node_names))
        point_cloud2 = np.array(self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame+1, self.node_names))
        velocity = point_cloud2.flatten()-point_cloud1.flatten()

        theta, offset_x, offset_z = align_point_clouds_2D(self.pose_constraint,
                                                          point_cloud1,
                                                          self.weights)
        t_point_cloud = transform_point_cloud(point_cloud1, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(self.pose_constraint, t_point_cloud)
        vel_error = np.linalg.norm(self.velocity_constraint - velocity)
        return error + vel_error

    def get_residual_vector_spline(self, aligned_spline):
        return self.get_residual_vector_frame(aligned_spline.evaluate(self.canonical_keyframe))

    def get_residual_vector(self, aligned_quat_frames):
        return self.get_residual_vector_frame(aligned_quat_frames[self.canonical_keyframe])

    def get_residual_vector_frame(self, frame):
        # get point cloud of first frame
        point_cloud = self.skeleton.convert_quaternion_frame_to_cartesian_frame(frame, self.node_names)

        theta, offset_x, offset_z = align_point_clouds_2D(self.pose_constraint,
                                                          point_cloud,
                                                          self.weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)
        residual_vector = []
        for i in xrange(len(t_point_cloud)):
            d = [self.pose_constraint[i][0] - t_point_cloud[i][0],
                 self.pose_constraint[i][1] - t_point_cloud[i][1],
                 self.pose_constraint[i][2] - t_point_cloud[i][2]]
            residual_vector.append(sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        return residual_vector

    def get_length_of_residual_vector(self):
        return len(self.skeleton.node_name_frame_map.keys())
