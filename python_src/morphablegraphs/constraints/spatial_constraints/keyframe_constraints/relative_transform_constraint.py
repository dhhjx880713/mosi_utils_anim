# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 19:02:55 2015

@author: erhe01
"""


import numpy as np
from .global_transform_constraint import GlobalTransformConstraint
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION


class RelativeTransformConstraint(GlobalTransformConstraint):
    """
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """

    def __init__(self, skeleton, constraint_desc, precision, weight_factor=1.0):
        super(RelativeTransformConstraint, self).__init__(skeleton, constraint_desc, precision, weight_factor)
        self.constraint_type = SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION
        self.offset = constraint_desc["offset"]

    def _evaluate_joint_position(self, frame):
        global_m = self.skeleton.nodes[self.joint_name].get_global_matrix(frame)
        pos = np.dot(global_m, self.offset)[:3]
        d = np.linalg.norm(self.position-pos)
        return d

