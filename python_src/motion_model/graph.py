# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:58:50 2015

@author: erhe01
"""


NODE_TYPE_START = "start"
NODE_TYPE_STANDARD = "standard"
NODE_TYPE_END = "end"


class GraphEdge(object):
    """ Contains a transition model. 
    """
    def __init__(self,from_action,from_motion_primitive,to_action,to_motion_primitive,
                 transition_type=NODE_TYPE_STANDARD,transition_model=None):
        self.from_action = from_action
        self.to_action = to_action
        self.from_motion_primitive = from_motion_primitive
        self.to_motion_primitive = to_motion_primitive
        self.transition_type = transition_type
        self.transition_model = transition_model
