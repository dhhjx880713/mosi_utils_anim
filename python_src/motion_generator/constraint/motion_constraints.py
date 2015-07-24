# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:26:29 2015

@author: erhe01
"""

from copy import copy
from elementary_action_constraints import ElementaryActionConstraints
from constraint_extraction import transform_point_from_cad_to_opengl_cs

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
    def __init__(self, mg_input, morphable_graph):
        self.morphable_graph = morphable_graph
        self.elementary_action_list = mg_input["elementaryActions"]
        self.keyframe_annotations = self._extract_keyframe_annotations(self.elementary_action_list)
        self.start_pose = mg_input["startPose"]
        self._transform_from_left_to_right_handed_cs()
        self.action_index = 0
        self.n_actions = len(self.elementary_action_list)

            
    
    def _transform_from_left_to_right_handed_cs(self):
        """ Transform transition and rotation of the start pose from CAD to Opengl 
            coordinate system.
        """
        start_pose_copy = copy(self.start_pose)
        self.start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
        self.start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
        
        
    def get_next_elementary_action_constraints(self):
        """
        Returns:
        --------
        * action_constraints : ElementarActionConstraints
          Constraints for the next elementary action extracted from an input file.
        """
        if self.action_index < self.n_actions:
            action_constraints = ElementaryActionConstraints(self.action_index, self)
            self.action_index+=1
            return action_constraints
        else:
            return None
  

    def _extract_keyframe_annotations(self, elementary_action_list):
        """
        Returns
        ------
        * keyframe_annotations : a list of dicts
          Contains for every elementary action a dict that associates of events/actions with certain keyframes
        """
        keyframe_annotations = []
        for entry in elementary_action_list:
            print  "entry#################",entry
            if "keyframeAnnotations" in entry.keys():
                annotations = {}
           
                for annotation in entry["keyframeAnnotations"]:
                    key = annotation["keyframe"]
                    annotations[key] = annotation
                keyframe_annotations.append(annotations)
            else:
                keyframe_annotations.append({})
        return keyframe_annotations
  

        