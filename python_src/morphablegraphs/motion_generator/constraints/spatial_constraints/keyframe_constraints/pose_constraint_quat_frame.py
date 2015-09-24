__author__ = 'hadu01'

from math import sqrt
from .....animation_data.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
    align_point_clouds_2D,\
    transform_point_cloud,\
    calculate_point_cloud_distance, \
    calculate_weighted_frame_distance_quat, \
    quat_distance
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
LEN_QUAT = 4
LEN_ROOT_POS = 3
LEN_QUAT_FRAME = 79

class PoseConstraintQuatFrame(KeyframeConstraintBase):

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(PoseConstraintQuatFrame, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
        assert len(self.pose_constraint) == LEN_QUAT_FRAME, ("pose_constraint is not quaternion frame")
        return

    def evaluate_motion_sample(self, aligned_quat_frames):
        weights = self.skeleton.get_joint_weights()
        error = calculate_weighted_frame_distance_quat(self.pose_constraint,
                                                       aligned_quat_frames[0],
                                                       weights)
        return error

    def get_residual_vector(self, aligned_quat_frames):
        weights = self.skeleton.get_joint_weights()
        residual_vector = []
        quat_frame_a = self.pose_constraint
        quat_frame_b = aligned_quat_frames[0]
        for i in xrange(len(weights) - 1):
            quat1 = quat_frame_a[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
            quat2 = quat_frame_b[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
            tmp = quat_distance(quat1, quat2)*weights[i]
            residual_vector.append(tmp)
        return residual_vector

    def get_length_of_residual_vector(self):
        return  LEN_QUAT_FRAME