# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:05:23 2015

@author: erhe01
"""


class SynthesisError(Exception):
    def __init__(self,  quat_frames, bad_samples):
        message = "Could not process input file"
        super(SynthesisError, self).__init__(message)
        self.bad_samples = bad_samples
        self.quat_frames = quat_frames


class PathSearchError(Exception):
    def __init__(self, parameters):
        self.search_parameters = parameters
        message = "Error in the navigation goal generation"
        super(PathSearchError, self).__init__(message)


class ConstraintError(Exception):
    def __init__(self,  bad_samples):
        message = "Could not reach constraint"
        super(ConstraintError, self).__init__(message)
        self.bad_samples = bad_samples
