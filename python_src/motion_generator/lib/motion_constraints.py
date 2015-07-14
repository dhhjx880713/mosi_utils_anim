# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:26:29 2015

@author: erhe01
"""

import copy
from lib.input_processing import extract_keyframe_annotations, transform_point_from_cad_to_opengl_cs       

class MotionConstraints(object):
    """
    Parameters
    ----------
    * mg_input : json data read from a file
        Contains elementary action list with constraints, start pose and keyframe annotations.
    * max_step : integer
        Sets the maximum number of graph walk steps to be performed. If less than 0
        then it is unconstrained
    """
    def __init__(self, mg_input, max_step=-1):
        self.max_step = max_step
        self.elementary_action_list = mg_input["elementaryActions"]
        self.keyframe_annotations = extract_keyframe_annotations(self.elementary_action_list)
        self.start_pose = mg_input["startPose"]
        self._transform_from_left_to_right_handed_cs()

            
    
    def _transform_from_left_to_right_handed_cs(self):
        """ Transform transition and rotation of the start pose from CAD to Opengl 
            coordinate system.
        """
        start_pose_copy = copy.copy(self.start_pose)
        self.start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
        self.start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
        