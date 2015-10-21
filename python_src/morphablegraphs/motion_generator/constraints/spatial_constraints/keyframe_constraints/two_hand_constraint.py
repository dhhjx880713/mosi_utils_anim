__author__ = 'erhe01'
import numpy as np
from keyframe_constraint_base import KeyframeConstraintBase
from .. import SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION


LEN_TWO_HAND_CONSTRAINT_SET = 3


class TwoHandConstraintSet(KeyframeConstraintBase):
    """calculates the sum of three features:
    the center of both hands, the global orientation of the line between both hands and the distance between both hands

    """
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(TwoHandConstraintSet, self).__init__(constraint_desc, precision, weight_factor)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
        self.skeleton = skeleton
        self.target_center = constraint_desc["target_center"]
        self.target_delta = constraint_desc["target_delta"]
        self.target_delta = constraint_desc["target_orientation"]
        self.line = constraint_desc["line"]
        self.n_canonical_frames = constraint_desc["n_canonical_frames"]

    def _get_global_hand_positions(self, aligned_quat_frames):
        left_hand_position = self.skeleton.joint_map[self.joint_name].get_global_position(aligned_quat_frames[self.canonical_keyframe])
        right_hand_position = self.skeleton.joint_map[self.joint_name].get_global_position(aligned_quat_frames[self.canonical_keyframe])
        delta_vector = right_hand_position - left_hand_position
        return left_hand_position, right_hand_position, delta_vector

    def evaluate_motion_sample(self, aligned_quat_frames):
        return sum(self.get_residual_vector(aligned_quat_frames))

    def get_residual_vector(self, aligned_quat_frames):
        left_hand_position, right_hand_position, delta_vector = self._get_global_hand_positions(aligned_quat_frames)
        residual_vector = [0.0, 0.0, 0.0]
        #get distance to center
        residual_vector[0] = np.linalg.norm(self.target_center - (left_hand_position + 0.5 *delta_vector))
        #get difference to distance between hands
        delta = np.linalg.norm(delta_vector)
        residual_vector[1] += self.target_delta - delta
        #get difference of global orientation
        orientation = delta_vector/delta
        residual_vector[2] += abs(self.target_orientation[0] - orientation[0]) + \
                              abs(self.target_orientation[1] - orientation[1]) +\
                              abs(self.target_orientation[2] - orientation[2])

        return residual_vector

    def get_length_of_residual_vector(self):
        return LEN_TWO_HAND_CONSTRAINT_SET
