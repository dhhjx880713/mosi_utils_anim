# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:58:50 2015

@author: erhe01
"""

from . import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END


class MotionStateTransition(object):
    """ Defines a transition between actions or motion primitives and can contain a transition model. 
    """
    def __init__(self,from_node_key, to_node_key, transition_type=NODE_TYPE_STANDARD, transition_model=None):
        self.from_node_key = from_node_key
        self.to_node_key = to_node_key
        self.transition_type = transition_type
        self.transition_model = transition_model
