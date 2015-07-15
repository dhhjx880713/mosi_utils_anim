# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""


class GraphWalkEntry(object):
    def __init__(self, action_name, motion_primitive_name, parameters, arc_length):
        self.action_name = action_name
        self.motion_primitive_name = motion_primitive_name
        self.parameters = parameters
        self.arc_length = arc_length


class AnnotatedMotion(object):
    def __init__(self):
        self.action_list = {}
        self.frame_annotation = {}
        self.frame_annotation['elementaryActionSequence'] = []
        self.graph_walk = []
        self.quat_frames = None
        self.step_count = 0
        self.n_frames = 0


    def update_action_list(self, new_action_list):
        """  merge the new actions list with the existing list.
        """
        self.action_list.update(new_action_list)
  
        
    def update_frame_annotation(self,action_name, start_frame, end_frame):
            #update frame annotation
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] =  start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)  


