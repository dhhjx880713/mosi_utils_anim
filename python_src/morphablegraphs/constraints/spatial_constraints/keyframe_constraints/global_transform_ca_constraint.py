import numpy as np
from .global_transform_constraint import GlobalTransformConstraint
from .. import SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT


class GlobalTransformCAConstraint(GlobalTransformConstraint):
    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0, step_idx=-1):
        super(GlobalTransformCAConstraint, self).__init__(skeleton, constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
        self.step_idx = step_idx

    def evaluate_motion_spline(self, aligned_spline):
        errors = np.zeros(self.n_canonical_frames)
        for i in range(self.n_canonical_frames):
            errors[i] = self._evaluate_joint_position(aligned_spline.evaluate(i))
        error = min(errors)
        print("ca constraint", error)
        return error#min(errors)

    def evaluate_motion_sample(self, aligned_quat_frames):
        errors = np.zeros(len(aligned_quat_frames))
        for i, frame in enumerate(aligned_quat_frames):
            errors[i] = self._evaluate_joint_position(frame)
        return min(errors)

    def get_residual_vector_spline(self, aligned_spline):
        return [self.evaluate_motion_spline(aligned_spline)]

    def get_residual_vector(self, aligned_frames):
        return [self.evaluate_motion_sample(aligned_frames)]

    def get_length_of_residual_vector(self):
        return 1
