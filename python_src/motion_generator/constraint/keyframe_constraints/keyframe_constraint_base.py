# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:36:36 2015

@author: erhe01
"""


class KeyframeConstraintBase(object):
    def __init__(self,constraint_desc, precision):
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
        self.precision = precision

    def evaluate_motion_sample(self, aligned_quat_frames):
        pass
        
    def evaluate_motion_sample_with_precision(self, aligned_quat_frames):
        error = self.evaluate_motion_sample(aligned_quat_frames)
        if error < self.precision:
            success = True
        else:
            success = False
        return error, success
