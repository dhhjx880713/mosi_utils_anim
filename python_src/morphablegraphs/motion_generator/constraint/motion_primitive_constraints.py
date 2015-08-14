# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
from animation_data.motion_editing import align_quaternion_frames

class MotionPrimitiveConstraints(object):
    """ Represents the input to the generate_motion_primitive_from_constraints
        method of the MotionPrimitiveGenerator class.
    Attributes
     -------
     * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    def __init__(self):
        self.pose_constraint_set = False
        self.motion_primitive_name = None
        self.settings = None
        self.constraints = []
        self.goal_arc_length = 0           
        self.use_optimization = False
        self.step_goal = None
        self.step_start = None
        self.start_pose = None
        self.skeleton = None
        self.precision = {"pos": 1.0, "rot": 1.0, "smooth": 1.0}
        self.verbose = False
        
    def print_status(self):
#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_motion_primitive,"is",goal,"at arc length",arc_length
        print "starting from: "
        print self.step_start
        print "the new goal for " + self.motion_primitive_name
        print self.step_goal
        print "arc length is: " + str(self.goal_arc_length)

    def evaluate(self, motion_primitive, sample, prev_frames, use_time_parameters=False):
        """
        Calculates the error of a list of constraints given a sample parameter value.
        
        Returns
        -------
        * sum_error : float
        \tThe sum of the errors for all constraints
    
        """
        #find aligned frames once for all constraints
        quat_frames = motion_primitive.back_project(sample, use_time_parameters=use_time_parameters).get_motion_vector()
        aligned_frames = align_quaternion_frames(quat_frames, prev_frames, self.start_pose)
    
        #evaluate constraints with the generated motion
        error_sum = 0
        for c in self.constraints:
            error_sum += c.weight_factor * c.evaluate_motion_sample(aligned_frames)
        return error_sum
        


