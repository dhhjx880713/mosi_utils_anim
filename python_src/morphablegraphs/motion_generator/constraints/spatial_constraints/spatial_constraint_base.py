# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:36:36 2015

@author: erhe01
"""


class SpatialConstraintBase(object):

    def __init__(self, precision, weight_factor=1.0):
        self.precision = precision
        self.weight_factor = weight_factor

    def evaluate_motion_sample(self, aligned_quat_frames):
        pass

    def evaluate_motion_sample_with_precision(self, aligned_quat_frames):
        error = self.evaluate_motion_sample(aligned_quat_frames)
        if error < self.precision:
            success = True
        else:
            success = False
        return error, success

    def get_residual_vector(self, aligned_frames):
        pass