__author__ = 'hadu01'

from math import sqrt
from .....animation_data.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
    align_point_clouds_2D,\
    transform_point_cloud,\
    calculate_point_cloud_distance, \
    calculate_weighted_frame_distance_quat
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE

class PoseConstraintQuatFrame(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PoseConstraintQuatFrame, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        assert len(self.pose_constraint) == 79, ("pose_constraint is not quaternion frame")
        return

    def evaluate_motion(self, aligned_quat_frames):
        weights = self.skeleton.get_joint_weights()
        error = calculate_weighted_frame_distance_quat(self.pose_constraint,
                                                       aligned_quat_frames[0],
                                                       weights)
        return error