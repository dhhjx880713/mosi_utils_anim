__author__ = 'hadu01'

from .keyframe_constraint_base import KeyframeConstraintBase
import numpy as np
from ....external.transformations import euler_matrix, \
                                          quaternion_matrix
LEN_ROOT_POSITION = 3
LEN_QUAT = 4


class JointRotationConstraint(KeyframeConstraintBase):
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(JointRotationConstraint, self).__init__(constraint_desc,
                                                      precision,
                                                      weight_factor)
        self.skeleton = skeleton
        self.joint_name = constraint_desc['joint_name']
        self.rotation_type = constraint_desc['rotation_type']
        self.rotation_constraint = constraint_desc['rotation_constraint']
        self.frame_idx = constraint_desc['frame_index']
        if self.rotation_type == "euler":
            rad_angles = list(map(np.deg2rad, self.rotation_constraint))
            self.constraint_rotmat = euler_matrix(rad_angles[0],
                                                  rad_angles[1],
                                                  rad_angles[2],
                                                  axes='rxyz')
        elif self.rotation_type == "quaternion":
            quat = np.asarray(self.rotation_constraint)
            quat /= np.linalg.norm(quat)
            self.constraint_rotmat = quaternion_matrix(quat)
        else:
            raise ValueError('Unknown rotation type!')

    def evaluate_motion_sample(self, aligned_quat_frames):
        """
        Extract the rotation angle of given joint at certain frame, to compare with
        constrained rotation matrix

        """
        return self.evaluate_frame(aligned_quat_frames[self.frame_idx])

    def evaluate_motion_spline(self, aligned_spline):
        return self.evaluate_frame(aligned_spline.evaluate(self.frame_idx))

    def evaluate_frame(self, frame):
        joint_idx = list(self.skeleton.node_name_frame_map.keys()).index(self.joint_name)
        quat_value = frame[LEN_ROOT_POSITION + joint_idx*LEN_QUAT :
                     LEN_ROOT_POSITION + (joint_idx + 1) * LEN_QUAT]
        quat_value = np.asarray(quat_value)
        quat_value /= np.linalg.norm(quat_value)
        rotmat = quaternion_matrix(quat_value)
        diff_mat = self.constraint_rotmat - rotmat
        tmp = np.ravel(diff_mat)
        error = np.linalg.norm(tmp)
        return error

    def get_residual_vector(self, aligned_quat_frames):
        return [self.evaluate_frame(aligned_quat_frames[self.frame_idx])]

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_frame(aligned_spline.evaluate(self.frame_idx))]

    def get_length_of_residual_vector(self):
        return 1