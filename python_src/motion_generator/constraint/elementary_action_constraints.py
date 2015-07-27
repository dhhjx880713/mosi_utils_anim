# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:41 2015

@author: erhe01
"""
import numpy as np


class ElementaryActionConstraints(object):
    def __init__(self):
        self.parent_constraint = None
        self.action_name = None
        self.keyframe_annotations = None
        self.constraints = None
        self.start_pose = None
        self.trajectory = None
        self.unconstrained_indices = None
        self.keyframe_constraints = None
        self._initialized = False


    def get_subgraph(self):
        return self.parent_constraint.morphable_graph.subgraphs[self.action_name]
        
    def get_skeleton(self):
        return self.parent_constraint.morphable_graph.skeleton

    
    def check_end_condition(self, prev_frames, travelled_arc_length, arc_length_offset):
        """
        Checks wether or not a threshold distance to the end has been reached.
        Returns
        -------
        True if yes and False if not
        """
        distance_to_end = np.linalg.norm(self.trajectory.get_last_control_point() - prev_frames[-1][:3])
    #    print "current distance to end: " + str(distance_to_end)
    #    print "travelled arc length is: " + str(travelled_arc_length)
    #    print "full arc length is; " + str(trajectory.full_arc_length)
    #    raw_input("go on...")
    
        continue_with_the_loop = distance_to_end > arc_length_offset/2 and \
                            travelled_arc_length < self.trajectory.full_arc_length - arc_length_offset
        return not continue_with_the_loop

